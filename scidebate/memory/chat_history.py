"""Chat history memory — lưu nguyên văn toàn bộ messages.

Đơn giản nhất, phù hợp cho Phase 1.
Hạn chế: context có thể bị dài nếu debate nhiều round.
"""
from .base import BaseMemory


class ChatHistoryMemory(BaseMemory):
    """Memory lưu toàn bộ messages dưới dạng list."""

    def __init__(self):
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role: {role}. Must be system/user/assistant.")
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        # Return copy để tránh user vô tình mutate
        return [dict(m) for m in self._messages]

    def clear(self) -> None:
        self._messages = []

    def remove_last(self, n: int = 1) -> None:
        if len(self._messages) >= n:
            self._messages = self._messages[:-n]
        else:
            self._messages = []

    def __repr__(self):
        return f"ChatHistoryMemory(n_messages={len(self._messages)})"
