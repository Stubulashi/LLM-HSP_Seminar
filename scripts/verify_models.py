"""Verify that every model repository in config/models.yaml really is downloadable on HF (guards against "model not found").

Rules:
- For every entry whose backend is in (huggingface, vllm) and whose path is a remote repository id:
  - config.json must exist;
  - safetensors weights must exist (model.safetensors.index.json or a single model.safetensors);
  - quantization check (enabled only when backend=vllm and quantization in awq/gptq/fp8):
      PASS if either ① the repository contains quantize_config.json, or
      ② config.json contains quantization_config declaring the matching quant_method (awq/gptq/fp8).
    The huggingface backend's 4bit/8bit is bitsandbytes runtime quantization, so the repository does
    not need to ship quantization metadata.
- Entries whose path is a local directory (os.path.isdir) are skipped.

Usage (on the cloud, with network access):
    python -X utf8 scripts/verify_models.py
    # All PASS means preflight is safe; any FAIL points at a replacement repository.
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
        return HfApi(timeout=timeout)  # supported by huggingface_hub<1.30
    except TypeError:
        return HfApi()  # huggingface_hub>=1.30 removed the timeout parameter


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
        missing.append("safetensors weights missing")
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
                missing.append(f"quantization declaration missing (quantize_config.json or {quant} inside config.json)")
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
            print(f"{name:12} local/local dir skipped ({repo})")
            continue
        quant = m.get("quantization")
        vllm = m.get("backend") == "vllm"
        try:
            missing, nfiles, notes = check_repo(repo, quant, vllm, timeout=args.timeout)
        except Exception as exc:  # 401/404/network etc.
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
        print(f"[verify] FAIL={fail}: fix the FAIL entries before the preflight (see the usable-repository notes at the top of models.yaml)")
        return 1
    print("[verify] ALL PASS: ready for the preflight / full runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
