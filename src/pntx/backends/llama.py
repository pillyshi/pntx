from __future__ import annotations

from typing import Any

try:
    import llama_cpp
except ImportError as e:
    raise ImportError(
        "LlamaCppBackend requires the 'llama' extra. "
        "Install it with: pip install 'pntx[llama]'"
    ) from e


class LlamaCppBackend:
    """In-process backend backed by llama.cpp (via ``llama-cpp-python``).

    Implements both ``Backend`` and ``ScoringBackend``. ``score_choices``
    evaluates ``prompt`` once, then for each choice replays only that
    choice's tokens on top of the cached prompt KV state, rolling back to
    the prompt boundary between choices instead of re-evaluating the shared
    prefix each time.
    """

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        # score_choices needs per-token logits, which llama.cpp only keeps
        # around when logits_all=True.
        kwargs["logits_all"] = True
        self._llm = llama_cpp.Llama(model_path=model_path, **kwargs)

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> str:
        result = self._llm.create_completion(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or [],
        )
        if not isinstance(result, dict):
            raise TypeError(f"expected a non-streaming completion response, got {type(result)}")
        return result["choices"][0]["text"]

    def score_choices(self, prompt: str, choices: list[str]) -> list[float]:
        """Return the summed log-likelihood of each choice as a continuation of ``prompt``."""
        if not choices:
            return []

        prompt_tokens = self._llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        self._llm.reset()  # type: ignore[no-untyped-call]  # llama_cpp.Llama.reset() has no annotations
        self._llm.eval(prompt_tokens)
        prefix_len = self._llm.n_tokens

        scores: list[float] = []
        for choice in choices:
            choice_tokens = self._llm.tokenize(choice.encode("utf-8"), add_bos=False)
            if not choice_tokens:
                scores.append(0.0)
                continue
            # Llama.eval() trims the KV cache down to the current n_tokens
            # before appending, so rewinding n_tokens here discards the
            # previous choice's tokens while keeping the prompt's KV cache.
            self._llm.n_tokens = prefix_len
            self._llm.eval(choice_tokens)
            logits = self._llm.scores[prefix_len - 1 : prefix_len + len(choice_tokens) - 1]
            logprobs = self._llm.logits_to_logprobs(logits)
            scores.append(float(sum(logprobs[j, tok] for j, tok in enumerate(choice_tokens))))
        self._llm.n_tokens = prefix_len
        return scores
