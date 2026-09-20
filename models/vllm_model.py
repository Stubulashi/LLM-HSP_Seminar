"""vLLM backend model (added by the AutoDL cost optimisation plan; scaf.md section 6 + the backend semantics extension of ruling C4).

Designed to maximise throughput on a single 80/90-series card (e.g. RTX 4090/4080 24G):
- loaded through the vLLM engine, supporting AWQ/GPTQ quantization (14B/32B become feasible on 24G);
- generate_batch: one engine call per batch of prompts (continuous batching), combined with
  enable_prefix_caching to reuse the shared prefix of adjacent steps of one story;
  throughput improves by roughly an order of magnitude;
- memory/length knobs are configurable per model (models.yaml single-model dict):
    gpu_memory_utilization: 0.90~0.95 (default 0.90; lower it or max_model_len on OOM)
    max_model_len: default 8192 (lower for long stories or KV pressure, e.g. 4096)
    max_num_seqs: default 32 (concurrency cap; lower on OOM)
    enable_prefix_caching: default True
- if vLLM is unavailable (local debugging without a GPU), it falls back to HFModel
  (bitsandbytes 4bit) so the experiment can still run.

OOM policy: load/generate catch vLLM memory errors, attach actionable guidance (lower
util/max_model_len/max_num_seqs or use a smaller quantization), and then re-raise; errors
are never swallowed silently. The generate(prompt) -> str contract matches HFModel
(pipeline.md L750-793).
"""

from __future__ import annotations

from models.base_model import BaseModel


def _oom_hint(config: dict, stage: str) -> str:
    return (
        f"[vllm:{config.get('path')}] out of memory during {stage}. Try, in order:\n"
        "  1) lower gpu_memory_utilization (0.90 -> 0.85 -> 0.80);\n"
        "  2) lower max_model_len (8192 -> 4096 -> 2048);\n"
        "  3) lower max_num_seqs (32 -> 16 -> 8);\n"
        "  4) use a smaller quantization (bf16 -> AWQ/GPTQ; 14B -> 7B).\n"
        "  All of these are adjusted in the corresponding entry of config/models.yaml; rerun afterwards."
    )


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("out of memory", "cuda out of memory", "oom",
                                  "not enough memory", "no available memory"))


def _resolve_dtype(config: dict) -> str:
    """Inference dtype for vLLM: AWQ/GPTQ support float16 only; otherwise bfloat16 by default; an explicit dtype in the config takes precedence."""
    q = config.get("quantization", "none")
    default = "float16" if q in ("awq", "gptq") else "bfloat16"
    return config.get("dtype", default)


class VLLMModel(BaseModel):
    """vLLM backend adapter: prefer vLLM; fall back to HF 4bit when it is unavailable (see the class docstring)."""

    def __init__(self, config: dict):
        self.config = config
        self.path = config["path"]
        self._impl = None  # the actual inference engine (a vLLM LLM or an HFModel)
        self._using_vllm = False
        self._temperature = float(config.get("temperature", 0.7))
        self._max_tokens = int(config.get("max_tokens", 512))

    def load(self) -> None:
        """Try vLLM; fall back to HFModel when unavailable (4bit quantization per config)."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            # vLLM unavailable (no local GPU / not installed): fall back to HF 4bit so the experiment can still run
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
        elif quantization == "4bit":  # legacy value: treated as GPTQ (pre-quantized weights)
            llm_kwargs["quantization"] = "gptq"
        elif quantization == "8bit":
            llm_kwargs["quantization"] = "awq"

        self._sampling = SamplingParams(
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        try:
            self._impl = LLM(**llm_kwargs)
        except Exception as exc:  # noqa: BLE001 - memory/engine errors are converted to guidance uniformly
            if _is_oom(exc):
                raise RuntimeError(_oom_hint(self.config, "load")) from exc
            msg = str(exc).lower()
            if any(k in msg for k in ("repository not found", "invalid repository",
                                      "401", "404", "network is unreachable",
                                      "getaddrinfo", "too many requests")):
                raise RuntimeError(
                    f"[vllm:{self.path}] repository not downloadable / not found ({str(exc)[:160]}).\n"
                    "  1) network issue: enable the AutoDL academic accelerator (source /etc/network_turbo) or set "
                    "HF_ENDPOINT=https://hf-mirror.com and retry;\n"
                    "  2) repository missing or blocked: re-check with scripts/verify_models.py and switch to a "
                    "usable repository as explained at the top of config/models.yaml (e.g. the casperhansen/graelo AWQ builds)."
                ) from exc
            raise
        self._using_vllm = True

    def generate(self, prompt: str, temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        """Take a prompt and return the text response (pipeline.md L740-744).

        Compatible with the extended HFModel signature: callers such as the judge may pass
        temperature/max_tokens (e.g. temperature=0 for greedy judging); when omitted, the
        engine's default sampling parameters are used.
        """
        if self._impl is None:
            raise RuntimeError("model not loaded; call load() first")
        if not self._using_vllm:  # fallback branch: HFModel
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
        """Batch generation: one engine call per batch of prompts (vLLM continuous batching)."""
        if self._impl is None:
            raise RuntimeError("model not loaded; call load() first")

        if not self._using_vllm:  # fallback branch: HFModel
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
        """Release the engine and GPU memory."""
        if self._impl is None:
            return
        if hasattr(self._impl, "unload"):  # HFModel branch
            self._impl.unload()
        del self._impl
        self._impl = None
        import gc

        gc.collect()
