"""验证 config/models.yaml 中所有模型仓库在 HF 上真实可下载（防"找不到模型"）。

规则：
- 对每个 backend in (huggingface, vllm) 且 path 为远程仓库 id 的条目：
  - config.json 必在；
  - safetensors 权重必在（model.safetensors.index.json 或单文件 model.safetensors）；
  - 量化判定（仅 backend=vllm 且 quantization in awq/gptq/fp8 时启用）：
      满足其一即 PASS——① 仓库含 quantize_config.json；
      ② config.json 内含 quantization_config 且声明对应 quant_method（awq/gptq/fp8）。
    huggingface 后端的 4bit/8bit 是 bitsandbytes 运行时量化，不要求仓库自带量化元数据。
- path 为本地目录（os.path.isdir）跳过。

用法（云端，网络可达）：
    python -X utf8 scripts/verify_models.py
    # 输出全 PASS 即放心 preflight；有 FAIL 会提示可替换仓库。
"""
import argparse, json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_manager import ConfigManager

REQUIRED_FILES = {"config.json"}
INDEX_FILES = {"model.safetensors.index.json", "model.safetensors"}
VLLM_QUANT = ("awq", "gptq", "fp8")


def _make_api(timeout: int):
    from huggingface_hub import HfApi

    try:
        return HfApi(timeout=timeout)  # huggingface_hub<1.30 支持
    except TypeError:
        return HfApi()  # huggingface_hub>=1.30 移除了 timeout 参数


def _config_text(api, repo_id: str) -> str:
    try:
        p = api.hf_hub_download(repo_id=repo_id, filename="config.json")
        return open(p, encoding="utf-8").read()
    except Exception:
        return ""


def check_repo(repo_id: str, quant: str | None, vllm: bool, timeout: int = 20):
    api = _make_api(timeout)
    info = api.model_info(repo_id, files_metadata=True)
    names = {f.rfilename for f in info.siblings}
    notes = []
    missing = [f for f in REQUIRED_FILES if f not in names]
    if not (INDEX_FILES & names):
        missing.append("safetensors 权重缺失")
    need_quant = vllm and quant in VLLM_QUANT
    if need_quant:
        if "quantize_config.json" in names:
            notes.append("quant=quantize_config.json")
        else:
            cfg = _config_text(api, repo_id)
            try:
                qc = json.loads(cfg).get("quantization_config") or {}
                qm = str(qc.get("quant_method", "")).lower()
            except Exception:
                qm = ""
            if qm == quant or (qm in VLLM_QUANT and quant == "4bit"):
                notes.append(f"quant=config.json:quantization_config({qm})")
            else:
                missing.append(f"量化声明缺失(quantize_config.json 或 config.json 内 {quant})")
    return missing, len(names), notes


def main() -> int:
    cfg = ConfigManager()
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=20)
    args = ap.parse_args()
    fail = 0
    print(f"{'model':12} {'backend':11} {'quant':6} {'repo/file'}")
    print("-" * 90)
    for name, m in cfg.get("models").items():
        repo = m.get("path", "")
        if not repo or "/" not in repo or os.path.isdir(repo):
            print(f"{name:12} local/local dir 跳过（{repo}）")
            continue
        quant = m.get("quantization")
        vllm = m.get("backend") == "vllm"
        try:
            missing, nfiles, notes = check_repo(repo, quant, vllm, timeout=args.timeout)
        except Exception as exc:  # 401/404/网络等
            missing = [f"{type(exc).__name__}: {str(exc)[:120]}"]
            nfiles = 0
            notes = []
        status = "PASS" if not missing else "FAIL"
        if missing:
            fail += 1
        note = ("; ".join(notes) + " ") if notes else ""
        print(f"{name:12} {m.get('backend','?'):11} {str(quant):6} {repo}  files={nfiles}  "
              f"{note}[{status}]")
        for mm in missing[:3]:
            print(f"    missing: {mm}")
    print("-" * 90)
    if fail:
        print(f"[verify] FAIL={fail}：先处理 FAIL 条目再跑 preflight（见 models.yaml 顶部可用仓库说明）")
        return 1
    print("[verify] ALL PASS：可直接 preflight / 全量运行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
