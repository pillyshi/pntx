from __future__ import annotations

import warnings
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

    Every entry point also guards against prompts that don't fit in
    ``n_ctx``: rather than letting llama.cpp raise once eval is attempted,
    tokens are trimmed from the *front* of the prompt (warning when this
    happens) down to a budget that still leaves room for the response
    (``max_tokens``) or the scored choices. The front is what's dropped
    because prompts here are built exemplars-first, query/instruction-last
    (see ``prompts.py``), so trimming the front sheds the oldest few-shot
    material while preserving the part that's actually being asked about.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Load a GGUF model, either from a local path or the Hugging Face Hub.

        Exactly one of ``model_path`` or ``repo_id`` must be given. With
        ``repo_id`` (optionally narrowed to one file via ``filename``), the
        model is resolved through ``Llama.from_pretrained`` (downloaded and
        cached under the standard Hugging Face Hub cache).

        Remaining ``kwargs`` (e.g. ``n_ctx``, ``n_gpu_layers``, ``flash_attn``)
        are forwarded as-is to ``llama_cpp.Llama``.
        """
        if (model_path is None) == (repo_id is None):
            raise ValueError("exactly one of model_path or repo_id must be given")
        # Scoring needs per-token logits, which llama.cpp only keeps around
        # when logits_all=True.
        kwargs["logits_all"] = True
        if repo_id is not None:
            self._llm = llama_cpp.Llama.from_pretrained(
                repo_id=repo_id, filename=filename, **kwargs
            )
        else:
            assert model_path is not None  # guaranteed by the check above
            self._llm = llama_cpp.Llama(model_path=model_path, **kwargs)

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> str:
        prompt_tokens = self._fit_to_context(
            self._tokenize(prompt, add_bos=True), reserve=max_tokens
        )
        result = self._llm.create_completion(
            prompt_tokens,
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
        prompt_tokens = self._fit_to_context(
            self._tokenize(prompt, add_bos=True), reserve=self._max_choice_tokens(choices)
        )
        self._reset()
        self._llm.eval(prompt_tokens)
        return self._score_choices_at(self._llm.n_tokens, choices)

    def score_choices_batch(
        self, prefix: str, queries: list[str], choices: list[str]
    ) -> list[list[float]]:
        """For each query, score ``choices`` as a continuation of ``prefix + query``.

        ``prefix`` (e.g. the few-shot exemplar block) is eval'd once and its
        KV cache reused across every query, instead of re-evaluating it per
        query as a naive per-item ``score_choices`` loop would. ``prefix`` is
        trimmed to leave room for the *longest* query plus the longest
        choice, so that every query in the batch is guaranteed to fit
        against the shared, trimmed prefix.
        """
        if not queries:
            return []
        max_query_tokens = max(
            (len(self._tokenize(query, add_bos=False)) for query in queries), default=0
        )
        prefix_tokens = self._fit_to_context(
            self._tokenize(prefix, add_bos=True),
            reserve=max_query_tokens + self._max_choice_tokens(choices),
        )
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

    def count_tokens(self, text: str) -> int:
        """Return how many tokens ``text`` tokenizes to (no BOS).

        Meant to be handed to ``selection.BudgetSelector`` as its
        ``tokenizer_fn``, so exemplar selection can stay within the actual
        model's token accounting rather than an approximation.
        """
        return len(self._tokenize(text, add_bos=False))

    def _tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos)

    def _max_choice_tokens(self, choices: list[str]) -> int:
        return max((len(self._tokenize(choice, add_bos=False)) for choice in choices), default=0)

    def _fit_to_context(self, tokens: list[int], *, reserve: int) -> list[int]:
        """Trim ``tokens`` (dropping from the front) so ``len(tokens) + reserve``
        fits in ``n_ctx``, warning when a trim actually happens.

        ``reserve`` is however many tokens the caller still needs room for
        after ``tokens`` -- ``max_tokens`` for a completion, or the longest
        scored choice (plus, for a batch, the longest query) for scoring.
        """
        n_ctx = self._llm.n_ctx()
        budget = n_ctx - reserve
        if budget <= 0:
            raise ValueError(
                f"reserve ({reserve} tokens) alone leaves no room in the context "
                f"window ({n_ctx} tokens); reduce max_tokens/choices or increase n_ctx"
            )
        if len(tokens) <= budget:
            return tokens
        warnings.warn(
            f"prompt ({len(tokens)} tokens) exceeds the available context budget "
            f"({budget} of {n_ctx} tokens, after reserving {reserve} for the "
            "response); dropping the oldest exemplars to fit. Pass "
            "PNTX(max_exemplars=...) to select fewer exemplars deliberately "
            "instead of relying on this truncation.",
            UserWarning,
            stacklevel=3,
        )
        if budget <= 1:
            return tokens[-budget:] if budget else []
        # Every call site tokenizes with add_bos=True, so tokens[0] is BOS;
        # keep it and drop from just after it, rather than dropping BOS
        # itself along with the oldest exemplars.
        return tokens[:1] + tokens[-(budget - 1) :]

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
