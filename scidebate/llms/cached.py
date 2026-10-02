"""CachedLLM — bọc một backend bất kỳ với: cache SQLite + rate limit + retry.

Mục đích: chạy thực nghiệm quy mô lớn trên API miễn phí.
    - Cache: request giống hệt (model, messages, params) → trả kết quả đã lưu,
      không tốn quota. Chạy lại sau crash / chạy ablation gần như miễn phí.
    - Rate limit: giãn cách tối thiểu 60/rpm giây giữa các request cùng provider
      (dùng chung cho mọi instance → Pro/Con/Judge cùng Groq không vượt RPM).
    - Retry: exponential backoff + jitter khi gặp 429 / 5xx / lỗi mạng.

Sampling ở temperature > 0 (uncertainty) gửi nhiều request giống hệt nhau và cần
kết quả KHÁC nhau. Vì vậy key cache có thêm "occurrence index": lần gọi thứ k của
cùng một request trong một process map tới entry thứ k → vừa giữ được độ đa dạng,
vừa tái lập được khi chạy lại.

Usage:
    llm = CachedLLM(GroqLLM(model="llama-3.3-70b-versatile"), cache_path="cache.sqlite", rpm=30)
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path

from .base import BaseLLM, LLMResponse


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class QuotaExhaustedError(RuntimeError):
    """Provider báo phải chờ rất lâu (hết quota ngày) — nên dừng và chạy lại sau."""


class _RateLimiter:
    """Giãn cách tối thiểu giữa các request (thread-safe)."""

    def __init__(self, rpm: float):
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_time)
            self._next_time = start + self.min_interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)


# Một limiter cho mỗi provider, dùng chung giữa mọi CachedLLM trong process.
_LIMITERS: dict[str, _RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def _get_limiter(provider: str, rpm: float) -> _RateLimiter:
    with _LIMITERS_LOCK:
        if provider not in _LIMITERS:
            _LIMITERS[provider] = _RateLimiter(rpm)
        return _LIMITERS[provider]


def _is_retryable(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    if status is None:
        status = getattr(getattr(err, "response", None), "status_code", None)
    if status is not None:
        return status in RETRYABLE_STATUS
    name = type(err).__name__
    return any(k in name for k in ("RateLimit", "Timeout", "Connection", "InternalServer"))


def _retry_after_seconds(err: Exception) -> float | None:
    headers = getattr(getattr(err, "response", None), "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class _SqliteCache:
    """Key-value cache trên SQLite, dùng chung giữa các thread."""

    _instances: dict[str, "_SqliteCache"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()
        self._lock = threading.Lock()

    @classmethod
    def open(cls, path: str | Path) -> "_SqliteCache":
        resolved = str(Path(path).resolve())
        with cls._instances_lock:
            if resolved not in cls._instances:
                cls._instances[resolved] = cls(Path(resolved))
            return cls._instances[resolved]

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            self._conn.commit()


class CachedLLM(BaseLLM):
    """Wrapper: cache + rate limit + retry cho một BaseLLM bất kỳ."""

    def __init__(
        self,
        inner: BaseLLM,
        cache_path: str | Path | None = None,
        rpm: float = 0,
        max_retries: int = 6,
        base_delay: float = 4.0,
        max_delay: float = 90.0,
    ):
        super().__init__(model=inner.model, temperature=inner.temperature, max_tokens=inner.max_tokens)
        self.inner = inner
        self.cache = _SqliteCache.open(cache_path) if cache_path else None
        self.limiter = _get_limiter(type(inner).__name__, rpm)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.stats = {"cache_hits": 0, "api_calls": 0, "retries": 0}
        self._occurrences: dict[str, int] = defaultdict(int)
        self._occ_lock = threading.Lock()

    def _request_key(self, messages: list[dict], kwargs: dict) -> str:
        payload = {
            "backend": type(self.inner).__name__,
            "model": self.inner.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.inner.temperature),
            "max_tokens": kwargs.get("max_tokens", self.inner.max_tokens),
            "logprobs": kwargs.get("logprobs"),
            "top_logprobs": kwargs.get("top_logprobs"),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        base = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        with self._occ_lock:
            idx = self._occurrences[base]
            self._occurrences[base] += 1
        return f"{base}:{idx}"

    def generate(self, messages: list[dict], **kwargs) -> LLMResponse:
        key = self._request_key(messages, kwargs) if self.cache else None
        if key:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats["cache_hits"] += 1
                return LLMResponse(**hit)

        response = self._generate_with_retry(messages, **kwargs)

        if key and response.content:
            self.cache.set(key, {
                "content": response.content,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "model": response.model,
                "logprobs": response.logprobs,
            })
        return response

    def _generate_with_retry(self, messages: list[dict], **kwargs) -> LLMResponse:
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                self.stats["api_calls"] += 1
                return self.inner.generate(messages, **kwargs)
            except Exception as err:
                if attempt >= self.max_retries or not _is_retryable(err):
                    raise
                self.stats["retries"] += 1
                delay = _retry_after_seconds(err)
                if delay is not None and delay > 300:
                    # Hết quota ngày (TPD/RPD) — chờ không có ý nghĩa, dừng để chạy lại sau.
                    raise QuotaExhaustedError(
                        f"{self.model}: provider asks to wait {delay:.0f}s — daily quota likely exhausted."
                    ) from err
                if delay is None:
                    delay = min(self.max_delay, self.base_delay * (2 ** attempt))
                delay += random.uniform(0, 1.0)
                print(f"[CachedLLM] {self.model}: {type(err).__name__} — retry {attempt + 1}/{self.max_retries} in {delay:.1f}s")
                time.sleep(delay)
        raise RuntimeError("unreachable")

    def __repr__(self):
        return f"CachedLLM({self.inner!r})"
