"""LLM 等价判定（judge）—— 替代/补充余弦相似度的 accuracy/emergence 判据。

背景（实测）：0.5B/1.5B 对单点 question 的回答常"语义邻近但措辞不同"，MiniLM 短句
余弦 0.7 阈值既会漏报(swmimp001 近对判 0)也会误报(sfp001 说 no faux pas 却 cosine 过线)。
本模块对 (question, model_answer, gold) 做"是否表达同一意思"的二值判定。

provider 三选一：
- "local"：本地缓存的 HF 模型（默认 qwen15b），懒加载 + 卸载；
- "deepseek" / "kimi"：OpenAI 兼容 API（密钥经 .env / 环境变量，绝不入代码或提交）。

失败策略：缺 key / 请求失败均显式 raise（ConfigError/RuntimeError），不静默；
解析不出 YES/NO 时返回 {'label': None}（聚合层计"无法判定"），而非当作 False。
fallback_provider 提供显式回退开关（默认 None = 不回退，避免静默降级混淆口径）。

用法：
    j = LLMEquivalenceJudge(provider="local", model_name="qwen15b")
    j = LLMEquivalenceJudge(provider="deepseek")   # 读 .env DEEPSEEK_*
    j.load()  # local 才真正加载模型；api 仅校验配置
    print(j.judge(question, answer, gold))  # {'label': True/False/None, 'rationale': str}
    j.unload()
"""
from __future__ import annotations

import os
import re

from models import ModelFactory
from models.base_model import BaseModel

_JUDGE_PROMPT = (
    "You are grading whether an assistant correctly answered a comprehension question.\n\n"
    "Question: {question}\n\n"
    "Reference answer (gold): {gold}\n\n"
    "Assistant answer: {answer}\n\n"
    "Decide whether the assistant answer conveys the SAME key meaning as the reference answer.\n"
    "Rules:\n"
    "- YES if the assistant states the same key conclusion / belief / intended meaning, "
    "even if it omits background or explanatory clauses that appear only in the reference.\n"
    "- NO only if the assistant's key conclusion is missing, contradicts, or is unrelated to the reference.\n"
    "- Judge the key answer content, not style or completeness of explanation.\n"
    "Reply with exactly one line starting YES or NO, then a short reason.\n"
)

_PROVIDER_ENV = {
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"),
    "kimi": ("KIMI_API_KEY", "KIMI_BASE_URL", "KIMI_MODEL"),
}
_DEFAULTS = {
    "deepseek": ("https://api.deepseek.com", "deepseek-chat"),
    "kimi": ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
}
LOCAL_PROVIDERS = ("local",)
API_PROVIDERS = tuple(_PROVIDER_ENV)
VALID_PROVIDERS = LOCAL_PROVIDERS + API_PROVIDERS


def load_dotenv_silent() -> None:
    """加载根目录 .env（可选依赖 python-dotenv；缺失则手写解析）。"""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())


def _env_or_raise(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise RuntimeError(
            f"{key} 未配置：请在 c:\\HSP\\.env 或环境变量中设置（示例见 docs 说明）。"
        )
    return val


def _api_config(provider: str):
    key_name, url_name, model_name = _PROVIDER_ENV[provider]
    default_url, default_model = _DEFAULTS[provider]
    api_key = _env_or_raise(key_name)
    base_url = os.environ.get(url_name, "").strip() or default_url
    model = os.environ.get(model_name, "").strip() or default_model
    return {"api_key": api_key, "base_url": base_url, "model": model}


def _parse_label(out: str) -> bool | None:
    """从 judge 输出解析 YES/NO；兼容加粗/破折号/前后缀噪声。"""
    if not out:
        return None
    # 去掉常见 markdown/装饰字符后取首个词
    clean = re.sub(r"[*_`#\-]+", "", out).strip()
    first = clean.splitlines()[0].strip().lstrip(".: ") if clean.splitlines() else clean
    head = first.upper()
    if head.startswith("YES"):
        return True
    if head.startswith("NO"):
        return False
    # 兜底：全文首个独立 yes/no 词（防止“… 回答是 Yes”等句式）
    m = re.search(r"\b(YES|NO)\b", clean.upper())
    return True if m and m.group(1) == "YES" else (False if m else None)


def _api_judge(provider: str, prompt: str, max_tokens: int,
               timeout: int, retries: int) -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai 未安装（API judge 需要）；请 pip install openai") from exc
    import random
    import time

    cfg = _api_config(provider)
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=timeout)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": "You grade semantic equivalence. Reply YES or NO only, then a brief reason."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            out = (resp.choices[0].message.content or "").strip()
            return {"label": _parse_label(out), "rationale": out[:200]}
        except Exception as exc:  # 超时/限流/网络等
            code = getattr(exc, "status_code", None)
            if code == 402:
                # 余额不足：重试无意义，立即报错并给指引
                raise RuntimeError(
                    "[judge] API 402 Insufficient Balance：请到 DeepSeek 控制台充值后重跑；"
                    "已成功判定的行在 judge_cache.jsonl 中，重跑不会重复计费。"
                ) from exc
            last_err = exc
            if attempt < retries:
                # 指数退避 + 抖动；429/5xx/连接错误给更长等待，避免持续撞限流
                base = 2.0 if code in (429, 500, 502, 503, 504) else 1.0
                delay = min(30.0, base * (2 ** attempt)) + random.uniform(0, 0.8)
                time.sleep(delay)
    raise RuntimeError(f"[judge:{provider}] 请求失败（重试{retries}次后）：{last_err}")


class LLMEquivalenceJudge:
    """语义等价判定器（local HF / deepseek / kimi API）。"""

    def __init__(self, provider: str = "local", model_name: str = "qwen15b",
                 config=None, max_tokens: int = 64, timeout: int = 30,
                 retries: int = 2, fallback_provider: str | None = None):
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"provider 须为 {VALID_PROVIDERS}，收到 {provider!r}")
        self.provider = provider
        self.model_name = model_name
        self._config = config
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.fallback_provider = fallback_provider
        self._model: BaseModel | None = None

    def load(self) -> None:
        load_dotenv_silent()
        if self.provider == "local":
            if self._model is None:
                if self._config is None:
                    from config.config_manager import ConfigManager

                    self._config = ConfigManager()
                self._model = ModelFactory(self._config).create(self.model_name)
                self._model.load()
        else:
            # API：仅校验配置可解析（不校验 key 有效性，留到首次请求）
            _api_config(self.provider)

    def unload(self) -> None:
        if self._model is not None:
            self._model.unload()
            self._model = None

    def judge(self, question: str, answer: str, gold: str) -> dict:
        """返回 {'label': True/False/None, 'rationale': str}。None=无法判定。"""
        prompt = _JUDGE_PROMPT.format(
            question=question or "(none)", gold=(gold or "").strip(), answer=(answer or "").strip())
        if self.provider != "local":
            try:
                return _api_judge(self.provider, prompt, self.max_tokens,
                                  self.timeout, self.retries)
            except RuntimeError:
                if self.fallback_provider == "local":
                    self.provider = "local"
                    self.load()
                else:
                    raise
        if self._model is None:
            raise RuntimeError("judge local model not loaded; call load() first")
        out = self._model.generate(prompt, temperature=0.0, max_tokens=self.max_tokens)
        return {"label": _parse_label(out), "rationale": (out or "").strip()[:200]}
