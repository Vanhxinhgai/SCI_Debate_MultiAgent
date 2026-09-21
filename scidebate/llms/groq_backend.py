"""Groq backend.

Groq dùng LPU (Language Processing Unit) → tốc độ inference rất cao:
    - ~500-800 tokens/s (so với Ollama ~20-50 tok/s local)
    - Free tier: ~14,400 requests/day (tốt hơn OpenRouter 50/day)
    - API tương thích OpenAI → dùng openai SDK

Models miễn phí tiêu biểu:
    - llama-3.1-8b-instant     → nhanh nhất, tốt cho uncertainty sampling
    - llama-3.3-70b-versatile  → chất lượng cao nhất, tốt cho debate
    - gemma2-9b-it             → cân bằng tốc độ/chất lượng
    - mixtral-8x7b-32768       → context dài, tốt cho judge

Setup:
    export GROQ_API_KEY=gsk_...
    Hoặc truyền api_key= trực tiếp.
"""
import os
from openai import OpenAI
from .base import BaseLLM, LLMResponse


class GroqLLM(BaseLLM):
    """Groq backend (OpenAI-compatible API)."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str = None,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY env var "
                "or pass api_key= to GroqLLM()."
            )
        self._client = OpenAI(base_url=self.BASE_URL, api_key=key)

    def generate(self, messages: list[dict], **kwargs) -> LLMResponse:
        # Cho phép override temperature/max_tokens per-call
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        logprobs = kwargs.get("logprobs", None)
        top_logprobs = kwargs.get("top_logprobs", None)

        api_kwargs = {}
        if logprobs is not None:
            api_kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            api_kwargs["top_logprobs"] = top_logprobs

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **api_kwargs
            )
        except Exception as e:
            # Fallback if logprobs not supported by the model/endpoint
            if logprobs:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise e

        choice = response.choices[0]
        logprobs_data = None
        if hasattr(choice, "logprobs") and choice.logprobs is not None:
            logprobs_data = []
            if hasattr(choice.logprobs, "content") and choice.logprobs.content is not None:
                for t in choice.logprobs.content:
                    top_lp = []
                    if hasattr(t, "top_logprobs") and t.top_logprobs is not None:
                        for top in t.top_logprobs:
                            top_lp.append({
                                "token": top.token,
                                "logprob": top.logprob,
                                "bytes": getattr(top, "bytes", None)
                            })
                    logprobs_data.append({
                        "token": t.token,
                        "logprob": t.logprob,
                        "bytes": getattr(t, "bytes", None),
                        "top_logprobs": top_lp
                    })

        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=self.model,
            logprobs=logprobs_data,
        )
