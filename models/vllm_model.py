"""vLLM 后端模型（AutoDL 成本优化方案新增；scaf.md 6 节 + 裁决 C4 backend 语义扩展）。

用于单张 80/90 系卡（如 RTX 4090/4080 24G）压榨吞吐：
- 以 vLLM 引擎加载，支持 AWQ/GPTQ 量化（14B/32B 在 24G 上可行）；
- generate_batch：一批 prompts 一次引擎调用（continuous batching），配合
  enable_prefix_caching 复用同 story 相邻 step 的共享前缀，吞吐可提升一个量级；
- config 可调显存/长度旋钮（models.yaml 单模型 dict 内）：
    gpu_memory_utilization: 0.90~0.95（默认 0.90；OOM 时下调或降 max_model_len）
    max_model_len: 默认 8192（长 story 或 KV 压力大时下调，如 4096）
    max_num_seqs: 默认 32（并发批次上限；OOM 时下调）
    enable_prefix_caching: 默认 True
- 若环境无 vLLM（本地无 GPU 调试），自动回退 HFModel（bitsandbytes 4bit），保证可运行。

OOM 策略：load/generate 捕获 vLLM 的显存类错误并给出可操作指引（降 util/max_model_len/
max_num_seqs 或换更小量化），然后重新抛出，绝不静默吞错。
与 HFModel 保持同一 generate(prompt) -> str 契约（pipeline.md L750-793）。
"""

from __future__ import annotations

from models.base_model import BaseModel


def _oom_hint(config: dict, stage: str) -> str:
    return (
        f"[vllm:{config.get('path')}] {stage} 触发显存不足。请按序尝试：\n"
        "  1) 降低 gpu_memory_utilization（0.90 -> 0.85 -> 0.80）；\n"
        "  2) 降低 max_model_len（8192 -> 4096 -> 2048）；\n"
        "  3) 降低 max_num_seqs（32 -> 16 -> 8）；\n"
        "  4) 使用更小量化（bf16 -> AWQ/GPTQ；14B -> 7B）。\n"
        "  以上均在 config/models.yaml 对应模型条目中调整，改后重跑即可。"
    )


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("out of memory", "cuda out of memory", "oom",
                                  "not enough memory", "no available memory"))


def _resolve_dtype(config: dict) -> str:
    """vLLM 推理 dtype：AWQ/GPTQ 只支持 float16；其余默认 bfloat16；config 显式 dtype 优先。"""
    q = config.get("quantization", "none")
    default = "float16" if q in ("awq", "gptq") else "bfloat16"
    return config.get("dtype", default)


class VLLMModel(BaseModel):
    """vLLM 后端适配：优先 vLLM，缺失时回退 HF 4bit（见类文档）。"""

    def __init__(self, config: dict):
        self.config = config
        self.path = config["path"]
        self._impl = None  # 实际推理引擎（vLLM LLM 或 HFModel）
        self._using_vllm = False
        self._temperature = float(config.get("temperature", 0.7))
        self._max_tokens = int(config.get("max_tokens", 512))

    def load(self) -> None:
        """尝试 vLLM；不可用则回退 HFModel（4bit 量化按 config）。"""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            # vLLM 不可用（本地无 GPU/未安装），回退 HF 4bit，保持实验可运行
            from models.hf_model import HFModel

            self._impl = HFModel(self.config)
            self._using_vllm = False
            self._impl.load()
            return

        quantization = self.config.get("quantization", "none")
        llm_kwargs: dict = {
            "model": self.path,
            "dtype": _resolve_dtype(self.config),
            "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0.90),
            "max_model_len": self.config.get("max_model_len", 8192),
            "max_num_seqs": self.config.get("max_num_seqs", 32),
            "enable_prefix_caching": self.config.get("enable_prefix_caching", True),
        }
        if quantization in ("awq", "gptq", "fp8", "bitsandbytes"):
            llm_kwargs["quantization"] = quantization
        elif quantization == "4bit":  # 兼容旧值：按 GPTQ 处理（预量化权重）
            llm_kwargs["quantization"] = "gptq"
        elif quantization == "8bit":
            llm_kwargs["quantization"] = "awq"

        self._sampling = SamplingParams(
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        try:
            self._impl = LLM(**llm_kwargs)
        except Exception as exc:  # noqa: BLE001 - 显存/引擎错误统一转指引
            if _is_oom(exc):
                raise RuntimeError(_oom_hint(self.config, "load")) from exc
            msg = str(exc).lower()
            if any(k in msg for k in ("repository not found", "invalid repository",
                                      "401", "404", "network is unreachable",
                                      "getaddrinfo", "too many requests")):
                raise RuntimeError(
                    f"[vllm:{self.path}] 仓库不可下载/未找到（{str(exc)[:160]}）。\n"
                    "  1) 网络问题：开启 AutoDL 学术加速(source /etc/network_turbo)或设 "
                    "HF_ENDPOINT=https://hf-mirror.com 后重试；\n"
                    "  2) 仓库不存在/被墙：运行 scripts/verify_models.py 复验，并按 "
                    "config/models.yaml 顶部说明换成可用仓库（如 casperhansen/graelo 的 AWQ）。"
                ) from exc
            raise
        self._using_vllm = True

    def generate(self, prompt: str, temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        """输入 prompt，返回文本响应（pipeline.md L740-744）。

        兼容 HFModel 的扩展签名：judge 等调用方可传 temperature/max_tokens
        （如 temperature=0 贪心判定）；未传则用引擎默认采样参数。
        """
        if self._impl is None:
            raise RuntimeError("model not loaded; call load() first")
        if not self._using_vllm:  # 回退分支：HFModel
            return self._impl.generate(
                prompt,
                temperature=0.7 if temperature is None else temperature,
                max_tokens=512 if max_tokens is None else max_tokens,
            )
        from vllm import SamplingParams

        if temperature is None and max_tokens is None:
            params = self._sampling
        else:
            params = SamplingParams(
                temperature=self._temperature if temperature is None else temperature,
                max_tokens=self._max_tokens if max_tokens is None else max_tokens,
            )
        try:
            outputs = self._impl.generate([prompt], params)
            return outputs[0].outputs[0].text.strip()
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise RuntimeError(_oom_hint(self.config, "generate")) from exc
            raise

    def generate_batch(self, prompts: list[str], seeds: list[int] | None = None) -> list[str]:
        """批量生成：一批 prompts 一次引擎调用（vLLM continuous batching）。"""
        if self._impl is None:
            raise RuntimeError("model not loaded; call load() first")

        if not self._using_vllm:  # 回退分支：HFModel
            return [self._impl.generate(p) for p in prompts]

        from vllm import SamplingParams

        if seeds and len(seeds) == len(prompts):
            params = [
                SamplingParams(temperature=self._temperature, max_tokens=self._max_tokens,
                               seed=seeds[i])
                for i in range(len(prompts))
            ]
        else:
            params = self._sampling
        try:
            outputs = self._impl.generate(prompts, params)
            return [o.outputs[0].text.strip() for o in outputs]
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise RuntimeError(_oom_hint(self.config, "generate")) from exc
            raise

    def unload(self) -> None:
        """释放引擎与显存。"""
        if self._impl is None:
            return
        if hasattr(self._impl, "unload"):  # HFModel 分支
            self._impl.unload()
        del self._impl
        self._impl = None
        import gc

        gc.collect()
