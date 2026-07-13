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

    Implements ``Backend``, ``ScoringBackend`` and ``BatchScoringBackend``.
    Scoring methods reuse the KV cache of whatever prefix they share: rather
    than re-evaluating a shared prefix for every downstream token span, they
    eval it once and then, for each span, rewind ``n_tokens`` back to the
    prefix boundary before eval'ing that span's tokens (``Llama.eval()``
    trims the KV cache down to the current ``n_tokens`` before appending, so
    this discards only the previous span, not the shared prefix).
    """

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        # Scoring needs per-token logits, which llama.cpp only keeps around
        # when logits_all=True.
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
        prompt_tokens = self._tokenize(prompt, add_bos=True)
        self._reset()
        self._llm.eval(prompt_tokens)
        return self._score_choices_at(self._llm.n_tokens, choices)

    def score_choices_batch(
        self, prefix: str, queries: list[str], choices: list[str]
    ) -> list[list[float]]:
        """For each query, score ``choices`` as a continuation of ``prefix + query``.

        ``prefix`` (e.g. the few-shot exemplar block) is eval'd once and its
        KV cache reused across every query, instead of re-evaluating it per
        query as a naive per-item ``score_choices`` loop would.
        """
        if not queries:
            return []
        prefix_tokens = self._tokenize(prefix, add_bos=True)
        self._reset()
        self._llm.eval(prefix_tokens)
        base = self._llm.n_tokens

        results: list[list[float]] = []
        for query in queries:
            query_tokens = self._tokenize(query, add_bos=False)
            self._llm.n_tokens = base
            self._llm.eval(query_tokens)
            results.append(self._score_choices_at(self._llm.n_tokens, choices))
        self._llm.n_tokens = base
        return results

    def _tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos)

    def _reset(self) -> None:
        self._llm.reset()  # type: ignore[no-untyped-call]  # llama_cpp.Llama.reset() has no annotations

    def _score_choices_at(self, base_n_tokens: int, choices: list[str]) -> list[float]:
        """Score each choice freshly on top of the KV cache at ``base_n_tokens``,
        rewinding between choices so only the shared prefix up to
        ``base_n_tokens`` is reused."""
        scores = [
            self._score_tokens(base_n_tokens, self._tokenize(choice, add_bos=False))
            for choice in choices
        ]
        self._llm.n_tokens = base_n_tokens
        return scores

    def _score_tokens(self, base_n_tokens: int, tokens: list[int]) -> float:
        if not tokens:
            return 0.0
        self._llm.n_tokens = base_n_tokens
        self._llm.eval(tokens)
        logits = self._llm.scores[base_n_tokens - 1 : base_n_tokens + len(tokens) - 1]
        logprobs = self._llm.logits_to_logprobs(logits)
        return float(sum(logprobs[j, tok] for j, tok in enumerate(tokens)))
