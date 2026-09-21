"""OpenRouter backend.

OpenRouter là aggregator API tương thích OpenAI, hỗ trợ nhiều model
(Llama, GPT-OSS, Qwen, Gemini...) qua 1 endpoint duy nhất.

Free tier: ~50 request/ngày cho các model có suffix ':free'.
Docs: https://openrouter.ai/docs
"""
import os
from openai import OpenAI
from .base import BaseLLM, LLMResponse


class OpenRouterLLM(BaseLLM):
    """LLM backend dùng OpenRouter cloud."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b:free",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: str = None,
        base_url: str = None,
    ):
        """
        Args:
            model: tên model trên OpenRouter, vd "meta-llama/llama-3.3-70b-instruct:free"
            api_key: nếu None, đọc từ env OPENROUTER_API_KEY
            base_url: nếu None, dùng default OpenRouter endpoint
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        # Resolve api_key
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY env variable or pass api_key=..."
            )

        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, messages: list[dict], **kwargs) -> LLMResponse:
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
            # Fallback if logprobs not supported by provider/model
            if logprobs:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise e

        # Debug: kiểm tra response structure
        if not response or not response.choices:
            raise ValueError(
                f"OpenRouter API trả về response không hợp lệ: {response}\n"
                f"Model: {self.model}\n"
                f"Có thể model đã bị xóa hoặc API key không hoạt động.\n"
                f"Kiểm tra: https://openrouter.ai/docs/models"
            )

        choice = response.choices[0]
        content = choice.message.content
        
        # Handle AI reasoning models (e.g., Baidu Cobuddy) - reasoning thay vì content
        if not content and hasattr(choice, 'message') and hasattr(choice.message, 'reasoning'):
            content = choice.message.reasoning
        
        if not content:
            raise ValueError(
                f"OpenRouter trả về content rỗng.\n"
                f"Model: {self.model}\n"
                f"Finish reason: {choice.finish_reason}\n"
                f"💡 Gợi ý: Tăng max_tokens hoặc dùng model khác.\n"
                f"Danh sách models: https://openrouter.ai/docs/models"
            )

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
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=self.model,
            logprobs=logprobs_data,
        )
