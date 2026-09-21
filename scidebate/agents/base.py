"""Abstract Agent — logic chung cho mọi agent trong debate.

Phase 2: thêm respond_async cho parallel execution.
"""
from abc import ABC
from scidebate.llms import BaseLLM, LLMResponse
from scidebate.memory import BaseMemory, ChatHistoryMemory


class BaseAgent(ABC):
    """Abstract debate agent."""

    name: str = "BaseAgent"
    system_prompt: str = ""

    def __init__(self, llm: BaseLLM, memory: BaseMemory = None):
        self.llm = llm
        self.memory = memory if memory is not None else ChatHistoryMemory()

        if self.system_prompt:
            self.memory.add("system", self.system_prompt)

    def respond(self, user_message: str, **llm_kwargs) -> str:
        """Sinh response (sync), update memory."""
        self.memory.add("user", user_message)

        response: LLMResponse = self.llm.generate(
            self.memory.get_messages(),
            **llm_kwargs,
        )

        self.memory.add("assistant", response.content)
        return response.content

    async def respond_async(self, user_message: str, **llm_kwargs) -> str:
        """Sinh response (async), update memory.

        Dùng cho parallel opening.
        """
        self.memory.add("user", user_message)

        response: LLMResponse = await self.llm.generate_async(
            self.memory.get_messages(),
            **llm_kwargs,
        )

        self.memory.add("assistant", response.content)
        return response.content

    def reset(self) -> None:
        self.memory.clear()
        if self.system_prompt:
            self.memory.add("system", self.system_prompt)

    def __repr__(self):
        return f"{self.name}(llm={self.llm.model}, memory_size={len(self.memory)})"
