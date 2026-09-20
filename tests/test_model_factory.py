"""Model factory and backend dispatch regression tests (AutoDL cost optimisation + the VLLMModel inheritance contract).

Coverage:
- all five models in the list can be instantiated by factory.create and all satisfy the BaseModel contract (decision: VLLMModel must inherit from BaseModel).
- backend dispatch is correct: huggingface -> HFModel, vllm -> VLLMModel.
- the vllm-backend models carry the quantized weights declared in models.yaml (cost-optimisation convention, not hard-coded).
- calling generate before load raises RuntimeError (the scheduler's load-before-use contract).
"""

from config.config_manager import ConfigManager
from models import HFModel, ModelFactory, VLLMModel
from models.base_model import BaseModel

# the agreed source of facts in models.yaml (ruling C4 backend semantics + C14 size/training)
_ALL_MODELS = [
    ("qwen7b", "vllm", "instruction", "7B"),
    ("qwen14b", "vllm", "instruction", "14B"),
    ("deepseek7b", "vllm", "reasoning", "7B"),
    ("deepseek14b", "vllm", "reasoning", "14B"),
    ("deepseek32b", "vllm", "reasoning", "32B"),
]

# the 24G single-card strategy: all five official models use vllm (7B bf16; 14B/32B pre-quantized AWQ/GPTQ)
_VLLM_MODELS = [name for name, *_ in _ALL_MODELS]
# tiers that must be quantized on 24G
_LARGE_QUANT = ["qwen14b", "deepseek14b", "deepseek32b"]


def test_all_models_listed_and_instantiable():
    """models.yaml provides the five models; factory.create can instantiate all of them."""
    cfg = ConfigManager()
    models = cfg.get("models")
    factory = ModelFactory(cfg)
    for name, _backend, _training, _size in _ALL_MODELS:
        assert name in models, f"missing model {name} in models.yaml"
        inst = factory.create(name)
        assert inst.path == models[name]["path"]


def test_vllm_model_inherits_base_model():
    """Regression: VLLMModel must inherit from BaseModel (a missing inheritance once broke the contract)."""
    cfg = ConfigManager()
    factory = ModelFactory(cfg)
    for name in _VLLM_MODELS:
        inst = factory.create(name)
        assert isinstance(inst, BaseModel), f"{name} must inherit BaseModel"
        assert isinstance(inst, VLLMModel)


def test_backend_dispatch_models():
    """Backend dispatch: all five official models use vllm -> VLLMModel."""
    cfg = ConfigManager()
    factory = ModelFactory(cfg)
    for name, backend, _training, _size in _ALL_MODELS:
        inst = factory.create(name)
        assert isinstance(inst, VLLMModel), f"{name} should dispatch to VLLMModel"


def test_vllm_large_models_quantized_with_knobs():
    """vllm 14B/32B must carry pre-quantization (awq/gptq) and the memory knobs; 7B may stay bf16."""
    cfg = ConfigManager()
    models = cfg.get("models")
    for name, _b, _t, size in _ALL_MODELS:
        m = models[name]
        assert "gpu_memory_utilization" in m and "max_model_len" in m and "max_num_seqs" in m
        if name in _LARGE_QUANT:
            assert m.get("quantization") in ("awq", "gptq", "fp8", "4bit", "8bit"), \
                f"{name}({size}) needs pre-quantization on 24G"
            assert m.get("dtype") == "float16", f"{name} AWQ/GPTQ requires dtype=float16"
        else:
            assert m.get("quantization") in (None, "none", "awq", "gptq")


def test_vllm_resolve_dtype():
    """AWQ/GPTQ default to float16; without quantization bfloat16; an explicit dtype wins."""
    from models.vllm_model import _resolve_dtype

    assert _resolve_dtype({"quantization": "awq"}) == "float16"
    assert _resolve_dtype({"quantization": "gptq"}) == "float16"
    assert _resolve_dtype({"quantization": "none"}) == "bfloat16"
    assert _resolve_dtype({"quantization": "awq", "dtype": "bfloat16"}) == "bfloat16"  # explicit wins

def test_factory_unload_all_untouched_instances_safe():
    """unload_all on instances that were never loaded must not raise (with vllm instances in the factory cache)."""
    cfg = ConfigManager()
    factory = ModelFactory(cfg)
    for name, _backend, _training, _size in _ALL_MODELS:
        factory.create(name)
    factory.unload_all()  # nothing is loaded or unloaded; just clear the cache


def _make_vllm_instance(name):
    cfg = ConfigManager()
    return ModelFactory(cfg).create(name)


def test_hf_quantized_load_kwargs_use_bitsandbytes_config(tmp_path):
    """Regression: 4bit/8bit models must not pass a top-level load_in_4bit to Qwen2.from_pretrained.

    (transformers rejects the unexpected parameter for Qwen2; it must go through quantization_config=BitsAndBytesConfig.)
    """
    from models.hf_model import _model_load_kwargs

    cfg = ConfigManager()
    q4 = cfg.get("models", "qwen15b")  # huggingface + quantization:4bit
    kw4 = _model_load_kwargs(q4)
    # load_in_4bit must never appear at the top level
    assert "load_in_4bit" not in kw4
    assert "load_in_8bit" not in kw4
    qc = kw4.get("quantization_config")
    assert qc is not None and qc.load_in_4bit is True

    qnone = cfg.get("models", "qwen05b")  # huggingface, not quantized
    kwn = _model_load_kwargs(qnone)
    assert "quantization_config" not in kwn
    assert "torch_dtype" in kwn  # non-quantized models use torch_dtype


def test_generate_without_load_raises():
    """Regression: VLLMModel.generate before load raises RuntimeError (the load-before-use contract)."""
    inst = _make_vllm_instance("deepseek32b")
    try:
        inst.generate("probe")
    except RuntimeError:
        pass
    except Exception as exc:  # noqa: BLE001 - raising is the contract, but it must be RuntimeError
        raise AssertionError(
            f"deepseek32b generate-without-load raised {type(exc).__name__}, "
            f"expected RuntimeError"
        ) from exc
    else:
        raise AssertionError(
            "deepseek32b generate-without-load did not raise (loading was not called)"
        )


def test_generate_batch_fallback_contract():
    """generate_batch falls back to generate one by one by default (the BaseModel contract, verified with MockModel)."""
    from models import MockModel

    model = MockModel(config={"path": "mock"}, seed=1)
    model.load()
    outs = model.generate_batch(["a", "b", "c"], seeds=[1, 2, 3])
    assert len(outs) == 3 and all(isinstance(x, str) and x for x in outs)


def test_vllm_oom_hint_and_is_oom():
    """OOM detection and the guidance text exist (testable offline; no vllm load is triggered)."""
    from models.vllm_model import _is_oom, _oom_hint

    assert _is_oom(Exception("CUDA out of memory"))
    assert not _is_oom(Exception("some other error"))
    cfg = ConfigManager().get("models", "qwen14b")
    hint = _oom_hint(cfg, "load")
    assert "gpu_memory_utilization" in hint and "max_model_len" in hint
    assert "max_num_seqs" in hint

