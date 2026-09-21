"""Ollama backend.

Ollama có API tương thích OpenAI, nên dùng openai SDK gọi tới
http://localhost:11434/v1 với api_key giả.
"""

from openai import OpenAI
from .base import BaseLLM, LLMResponse

class OllamaLLM(BaseLLM):
    """Ollama backend."""

    def __init__ (
        self,
        model: str = "qwen2.5:3b",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        base_url: str = "http://localhost:11434/v1",
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.base_url = base_url
        self._client = OpenAI(base_url=self.base_url, api_key="ollama")
    
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
            # Fallback if logprobs not supported
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
