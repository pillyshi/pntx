from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

anthropic = pytest.importorskip("anthropic")

from pntx.backends.anthropic import AnthropicBackend  # noqa: E402


@dataclass
class _FakeMessage:
    content: list[Any]


class _FakeMessagesResource:
    """Fake sync `client.messages` keyed by prompt text."""

    def __init__(self, responses_by_prompt: dict[str, str]) -> None:
        self._responses_by_prompt = responses_by_prompt
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeMessage:
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        text = self._responses_by_prompt[prompt]
        return _FakeMessage(content=[anthropic.types.TextBlock(text=text, type="text")])


class _FakeClient:
    def __init__(self, responses_by_prompt: dict[str, str]) -> None:
        self.messages = _FakeMessagesResource(responses_by_prompt)


@dataclass
class _ConcurrencyTracker:
    current: int = 0
    max_seen: int = field(default=0)


class _FakeAsyncMessagesResource:
    """Fake async `client.messages` keyed by prompt text, tracking peak concurrency."""

    def __init__(self, responses_by_prompt: dict[str, str], delay: float = 0.0) -> None:
        self._responses_by_prompt = responses_by_prompt
        self._delay = delay
        self.calls: list[dict[str, Any]] = []
        self.tracker = _ConcurrencyTracker()

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.calls.append(kwargs)
        self.tracker.current += 1
        self.tracker.max_seen = max(self.tracker.max_seen, self.tracker.current)
        await asyncio.sleep(self._delay)
        self.tracker.current -= 1
        prompt = kwargs["messages"][0]["content"]
        text = self._responses_by_prompt[prompt]
        return _FakeMessage(content=[anthropic.types.TextBlock(text=text, type="text")])


class _FakeAsyncClient:
    def __init__(self, responses_by_prompt: dict[str, str], delay: float = 0.0) -> None:
        self.messages = _FakeAsyncMessagesResource(responses_by_prompt, delay)


def test_complete_extracts_text_from_response() -> None:
    client = _FakeClient({"hello": "world"})
    backend = AnthropicBackend(
        model="claude-test",
        client=client,  # type: ignore[arg-type]
        async_client=_FakeAsyncClient({}),  # type: ignore[arg-type]
    )

    assert backend.complete("hello") == "world"
    assert client.messages.calls[0]["model"] == "claude-test"
    assert client.messages.calls[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_complete_passes_through_generation_params() -> None:
    client = _FakeClient({"p": "r"})
    backend = AnthropicBackend(
        model="claude-test",
        client=client,  # type: ignore[arg-type]
        async_client=_FakeAsyncClient({}),  # type: ignore[arg-type]
    )

    backend.complete("p", temperature=0.3, max_tokens=42, stop=["STOP"])

    call = client.messages.calls[0]
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 42
    assert call["stop_sequences"] == ["STOP"]


def test_complete_batch_empty_returns_empty_without_calling_client() -> None:
    async_client = _FakeAsyncClient({})
    backend = AnthropicBackend(
        model="claude-test",
        client=_FakeClient({}),  # type: ignore[arg-type]
        async_client=async_client,  # type: ignore[arg-type]
    )

    assert backend.complete_batch([]) == []
    assert async_client.messages.calls == []


def test_complete_batch_returns_results_in_input_order() -> None:
    responses = {"a": "resp-a", "b": "resp-b", "c": "resp-c"}
    async_client = _FakeAsyncClient(responses, delay=0.01)
    backend = AnthropicBackend(
        model="claude-test",
        client=_FakeClient({}),  # type: ignore[arg-type]
        async_client=async_client,  # type: ignore[arg-type]
    )

    results = backend.complete_batch(["a", "b", "c"])

    assert results == ["resp-a", "resp-b", "resp-c"]


def test_complete_batch_runs_concurrently_bounded_by_concurrency() -> None:
    prompts = [f"p{i}" for i in range(6)]
    responses = {p: f"r{p}" for p in prompts}
    async_client = _FakeAsyncClient(responses, delay=0.05)
    backend = AnthropicBackend(
        model="claude-test",
        concurrency=3,
        client=_FakeClient({}),  # type: ignore[arg-type]
        async_client=async_client,  # type: ignore[arg-type]
    )

    results = backend.complete_batch(prompts)

    assert results == [f"r{p}" for p in prompts]
    # With 6 prompts, a 0.05s delay each, and concurrency=3, some overlap
    # must have occurred (otherwise this would take 6x as long), but the
    # semaphore must never let more than 3 run at once.
    assert async_client.messages.tracker.max_seen > 1
    assert async_client.messages.tracker.max_seen <= 3


def test_concurrency_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        AnthropicBackend(
            model="claude-test",
            concurrency=0,
            client=_FakeClient({}),  # type: ignore[arg-type]
            async_client=_FakeAsyncClient({}),  # type: ignore[arg-type]
        )
