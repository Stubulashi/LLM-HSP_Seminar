"""模型基类（pipeline.md 九节 L738-793 / scaf.md 6 节 L677-701）。

所有模型必须实现 load / generate / unload；
实验系统只依赖 generate(prompt) -> str（低耦合，scaf.md Principle 1）。
可选扩展：generate_batch(prompts) 供 vLLM 等引擎批量并发（压榨 GPU 用），
默认实现为逐条循环调用 generate，保证任何后端都可用。
"""

from __future__ import annotations


class BaseModel:
    def load(self) -> None:
        """加载模型与 tokenizer。"""

    def generate(self, prompt: str) -> str:
        """输入 prompt，返回文本响应（pipeline.md L740-744）。"""
        raise NotImplementedError

    def generate_batch(self, prompts: list[str], seeds: list[int] | None = None) -> list[str]:
        """批量生成；默认逐条回退 generate。支持批量引擎（vLLM）覆写以压榨 GPU。"""
        return [self.generate(p) for p in prompts]

    def unload(self) -> None:
        """释放显存（scaf.md L748-754）。"""
