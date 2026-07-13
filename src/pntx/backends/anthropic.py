from __future__ import annotations

import asyncio
from typing import Any

try:
    import anthropic
except ImportError as e:
    raise ImportError(
        "AnthropicBackend requires the 'anthropic' extra. "
        "Install it with: pip install 'pntx[anthropic]'"
    ) from e

_DEFAULT_CONCURRENCY = 5


class AnthropicBackend:
    """Backend using the Anthropic Messages API.

    Implements only ``Backend`` (plain text completion): the Messages API
    doesn't expose token log-probabilities, so classification for this
    backend goes through PNTX's parse-based fallback rather than
    ``score_choices``. ``complete_batch`` implements ``BatchBackend`` by
    running requests concurrently via asyncio, bounded by ``concurrency``.
    """

    def __init__(
        self,
        model: str,
        *,
        concurrency: int = _DEFAULT_CONCURRENCY,
        client: anthropic.Anthropic | None = None,
        async_client: anthropic.AsyncAnthropic | None = None,
        **client_kwargs: Any,
    ) -> None:
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self.model = model
        self.concurrency = concurrency
        self._client = client if client is not None else anthropic.Anthropic(**client_kwargs)
        self._async_client = (
            async_client if async_client is not None else anthropic.AsyncAnthropic(**client_kwargs)
        )

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop or [],
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(message)

    def complete_batch(
        self,
        prompts: list[str],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Complete every prompt concurrently, bounded by ``self.concurrency``."""
        if not prompts:
            return []
        return asyncio.run(
            self._complete_batch_async(
                prompts, temperature=temperature, max_tokens=max_tokens, stop=stop
            )
        )

    async def _complete_batch_async(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> list[str]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def complete_one(prompt: str) -> str:
            async with semaphore:
                message = await self._async_client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop or [],
                    messages=[{"role": "user", "content": prompt}],
                )
                return _extract_text(message)

        return list(await asyncio.gather(*(complete_one(prompt) for prompt in prompts)))


def _extract_text(message: anthropic.types.Message) -> str:
    return "".join(
        block.text for block in message.content if isinstance(block, anthropic.types.TextBlock)
    )
