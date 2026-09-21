"""Abstract LLM interface.

Mọi backend (Ollama, OpenAI, OpenRouter, ...) phải implement BaseLLM.
Agent code chỉ gọi qua interface này, không phụ thuộc backend cụ thể.

Phase 2: thêm generate_async cho parallel execution.
"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Kết quả trả về từ LLM."""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    logprobs: list = None  # Chứa danh sách các token log probability nếu được yêu cầu



class BaseLLM(ABC):
    """Abstract LLM backend."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Sinh response (sync)."""
        pass

    async def generate_async(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Sinh response (async).

        Default: wrap sync generate trong executor.
        Subclass có thể override bằng native async nếu muốn.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(messages, **kwargs)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model})"
