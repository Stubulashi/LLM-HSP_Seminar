"""LLM equivalence judge — an accuracy/emergence criterion that replaces or complements cosine similarity.

Background (measured): for single-point questions, 0.5B/1.5B answers are often "semantically close but differently worded";
the MiniLM cosine threshold of 0.7 both misses true matches (swmimp001: near-correct scored 0) and
accepts false ones (sfp001 said "no faux pas" yet passed the cosine line). This module makes a binary
"same meaning or not" decision over (question, model_answer, gold).

Three providers:
- "local": a locally cached HF model (qwen15b by default), lazy load + unload;
- "deepseek" / "kimi": OpenAI-compatible APIs (keys come from .env / environment variables and never enter the code or commits).

Failure policy: a missing key or a failed request raises explicitly (ConfigError/RuntimeError), never silently;
when YES/NO cannot be parsed it returns {'label': None} (aggregated as "undecidable") rather than False.
fallback_provider is an explicit fallback switch (default None = no fallback, so silent degradation never blurs the protocol).

Usage:
    j = LLMEquivalenceJudge(provider="local", model_name="qwen15b")
    j = LLMEquivalenceJudge(provider="deepseek")   # reads .env DEEPSEEK_*
    j.load()  # local really loads the model; api only validates the config
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
    """Load the root .env (python-dotenv is optional; fall back to a hand-written parser)."""
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
            f"{key} is not set: configure it in c:\\HSP\\.env or in the environment (see the docs for examples)."
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
    """Parse YES/NO from the judge output; tolerates bold, dashes and prefix/suffix noise."""
    if not out:
        return None
    # strip common markdown/decoration characters, then take the first word
    clean = re.sub(r"[*_`#\-]+", "", out).strip()
    first = clean.splitlines()[0].strip().lstrip(".: ") if clean.splitlines() else clean
    head = first.upper()
    if head.startswith("YES"):
        return True
    if head.startswith("NO"):
        return False
    # fallback: the first standalone yes/no word in the whole text (handles wording like "... the answer is Yes")
    m = re.search(r"\b(YES|NO)\b", clean.upper())
    return True if m and m.group(1) == "YES" else (False if m else None)


def _api_judge(provider: str, prompt: str, max_tokens: int,
               timeout: int, retries: int) -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is not installed (required for the API judge); run: pip install openai") from exc
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
        except Exception as exc:  # timeouts / rate limits / network errors etc.
            code = getattr(exc, "status_code", None)
            if code == 402:
                # insufficient balance: retrying is pointless; fail immediately with guidance
                raise RuntimeError(
                    "[judge] API 402 Insufficient Balance: top up in the DeepSeek console and rerun; "
                    "rows already judged are stored in judge_cache.jsonl, so a rerun is not billed twice."
                ) from exc
            last_err = exc
            if attempt < retries:
                # exponential backoff with jitter; longer waits for 429/5xx/connection errors to avoid hammering the rate limit
                base = 2.0 if code in (429, 500, 502, 503, 504) else 1.0
                delay = min(30.0, base * (2 ** attempt)) + random.uniform(0, 0.8)
                time.sleep(delay)
    raise RuntimeError(f"[judge:{provider}] request failed (after {retries} retries): {last_err}")


class LLMEquivalenceJudge:
    """Semantic equivalence judge (local HF / deepseek / kimi API)."""

    def __init__(self, provider: str = "local", model_name: str = "qwen15b",
                 config=None, max_tokens: int = 64, timeout: int = 30,
                 retries: int = 2, fallback_provider: str | None = None):
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"provider must be one of {VALID_PROVIDERS}; got {provider!r}")
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
            # API: only check that the config parses (key validity is checked on the first request)
            _api_config(self.provider)

    def unload(self) -> None:
        if self._model is not None:
            self._model.unload()
            self._model = None

    def judge(self, question: str, answer: str, gold: str) -> dict:
        """Returns {'label': True/False/None, 'rationale': str}. None means undecidable."""
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
