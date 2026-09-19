"""模型工厂与后端分发回归测试（AutoDL 成本优化 + VLLMModel 继承契约）。

覆盖：
- 5 模型清单均可被 factory.create 实例化，且全部满足 BaseModel 契约（决策：VLLMModel 必须继承 BaseModel）。
- backend 分发正确：huggingface -> HFModel，vllm -> VLLMModel。
- vllm 后端模型在 models.yaml 中带 quantization=4bit（成本优化约定，未硬编码）。
- 未 load 即调用 generate 抛 RuntimeError（保持 scheduler 的 load 前置契约）。
"""

from config.config_manager import ConfigManager
from models import HFModel, ModelFactory, VLLMModel
from models.base_model import BaseModel

# models.yaml 中约定的事实源（裁决 C4 backend 语义 + C14 size/training）
_ALL_MODELS = [
    ("qwen7b", "vllm", "instruction", "7B"),
    ("qwen14b", "vllm", "instruction", "14B"),
    ("deepseek7b", "vllm", "reasoning", "7B"),
    ("deepseek14b", "vllm", "reasoning", "14B"),
    ("deepseek32b", "vllm", "reasoning", "32B"),
]

# 24G 单卡策略：5 个正式模型全部 vllm（7B bf16；14B/32B 预量化 AWQ/GPTQ）
_VLLM_MODELS = [name for name, *_ in _ALL_MODELS]
# 24G 上必须量化的档位
_LARGE_QUANT = ["qwen14b", "deepseek14b", "deepseek32b"]


def test_all_models_listed_and_instantiable():
    """models.yaml 提供 5 个模型，factory.create 均可实例化。"""
    cfg = ConfigManager()
    models = cfg.get("models")
    factory = ModelFactory(cfg)
    for name, _backend, _training, _size in _ALL_MODELS:
        assert name in models, f"missing model {name} in models.yaml"
        inst = factory.create(name)
        assert inst.path == models[name]["path"]


def test_vllm_model_inherits_base_model():
    """回归：VLLMModel 必须继承 BaseModel（此前因漏继承导致契约破坏）。"""
    cfg = ConfigManager()
    factory = ModelFactory(cfg)
    for name in _VLLM_MODELS:
        inst = factory.create(name)
        assert isinstance(inst, BaseModel), f"{name} must inherit BaseModel"
        assert isinstance(inst, VLLMModel)


def test_backend_dispatch_models():
    """backend 分发：正式 5 模型全部 vllm -> VLLMModel。"""
    cfg = ConfigManager()
    factory = ModelFactory(cfg)
    for name, backend, _training, _size in _ALL_MODELS:
        inst = factory.create(name)
        assert isinstance(inst, VLLMModel), f"{name} should dispatch to VLLMModel"


def test_vllm_large_models_quantized_with_knobs():
    """vllm 14B/32B 必须带预量化(awq/gptq)与显存旋钮；7B 可 bf16。"""
    cfg = ConfigManager()
    models = cfg.get("models")
    for name, _b, _t, size in _ALL_MODELS:
        m = models[name]
        assert "gpu_memory_utilization" in m and "max_model_len" in m and "max_num_seqs" in m
        if name in _LARGE_QUANT:
            assert m.get("quantization") in ("awq", "gptq", "fp8", "4bit", "8bit"), \
                f"{name}({size}) 在 24G 需预量化"
            assert m.get("dtype") == "float16", f"{name} AWQ/GPTQ 必须 dtype=float16"
        else:
            assert m.get("quantization") in (None, "none", "awq", "gptq")


def test_vllm_resolve_dtype():
    """AWQ/GPTQ 默认 float16；无量化默认 bfloat16；显式 dtype 优先。"""
    from models.vllm_model import _resolve_dtype

    assert _resolve_dtype({"quantization": "awq"}) == "float16"
    assert _resolve_dtype({"quantization": "gptq"}) == "float16"
    assert _resolve_dtype({"quantization": "none"}) == "bfloat16"
    assert _resolve_dtype({"quantization": "awq", "dtype": "bfloat16"}) == "bfloat16"  # 显式优先

def test_factory_unload_all_untouched_instances_safe():
    """未 load 的实例 unload_all 不应抛错（factory 缓存含 vllm 实例时）。"""
    cfg = ConfigManager()
    factory = ModelFactory(cfg)
    for name, _backend, _training, _size in _ALL_MODELS:
        factory.create(name)
    factory.unload_all()  # 不加载不卸载，纯清缓存


def _make_vllm_instance(name):
    cfg = ConfigManager()
    return ModelFactory(cfg).create(name)


def test_hf_quantized_load_kwargs_use_bitsandbytes_config(tmp_path):
    """回归：4bit/8bit 模型不得把顶层 load_in_4bit 传入 Qwen2.from_pretrained。

    (transformers 对 Qwen2 抛不可预料参数；必须经 quantization_config=BitsAndBytesConfig)。
    """
    from models.hf_model import _model_load_kwargs

    cfg = ConfigManager()
    q4 = cfg.get("models", "qwen15b")  # huggingface + quantization:4bit
    kw4 = _model_load_kwargs(q4)
    # 顶层绝不能出现 load_in_4bit
    assert "load_in_4bit" not in kw4
    assert "load_in_8bit" not in kw4
    qc = kw4.get("quantization_config")
    assert qc is not None and qc.load_in_4bit is True

    qnone = cfg.get("models", "qwen05b")  # huggingface 未量化
    kwn = _model_load_kwargs(qnone)
    assert "quantization_config" not in kwn
    assert "torch_dtype" in kwn  # 非量化走 torch_dtype


def test_generate_without_load_raises():
    """回归：VLLMModel 未 load 即 generate 抛 RuntimeError（load 前置契约）。"""
    inst = _make_vllm_instance("deepseek32b")
    try:
        inst.generate("probe")
    except RuntimeError:
        pass
    except Exception as exc:  # noqa: BLE001 - 契约是抛错，但必须是 RuntimeError
        raise AssertionError(
            f"deepseek32b generate-without-load raised {type(exc).__name__}, "
            f"expected RuntimeError"
        ) from exc
    else:
        raise AssertionError(
            "deepseek32b generate-without-load did not raise (loading was not called)"
        )


def test_generate_batch_fallback_contract():
    """generate_batch 默认逐条回退 generate（BaseModel 契约，MockModel 验证）。"""
    from models import MockModel

    model = MockModel(config={"path": "mock"}, seed=1)
    model.load()
    outs = model.generate_batch(["a", "b", "c"], seeds=[1, 2, 3])
    assert len(outs) == 3 and all(isinstance(x, str) and x for x in outs)


def test_vllm_oom_hint_and_is_oom():
    """OOM 识别与指引文案存在（离线可测，不触发 vllm 加载）。"""
    from models.vllm_model import _is_oom, _oom_hint

    assert _is_oom(Exception("CUDA out of memory"))
    assert not _is_oom(Exception("some other error"))
    cfg = ConfigManager().get("models", "qwen14b")
    hint = _oom_hint(cfg, "load")
    assert "gpu_memory_utilization" in hint and "max_model_len" in hint
    assert "max_num_seqs" in hint

