"""Abstract Memory interface.

Memory lưu trữ lịch sử messages của một agent.
Các biến thể (ChatHistory, Summary, Masked...) đều implement BaseMemory.
"""
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Abstract memory backend."""

    @abstractmethod
    def add(self, role: str, content: str) -> None:
        """Thêm 1 message vào memory.

        Args:
            role: "system" | "user" | "assistant"
            content: nội dung message
        """
        pass

    @abstractmethod
    def get_messages(self) -> list[dict]:
        """Lấy toàn bộ messages dưới dạng list dict.

        Returns:
            list[{"role": ..., "content": ...}] — format chuẩn cho LLM API
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Xóa toàn bộ memory."""
        pass

    def remove_last(self, n: int = 1) -> None:
        """Xóa n tin nhắn cuối cùng trong memory."""
        pass

    def __len__(self) -> int:
        return len(self.get_messages())
