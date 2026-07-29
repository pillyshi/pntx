from __future__ import annotations

import ctypes
import json
import logging
import warnings
from typing import Any

try:
    import llama_cpp
except ImportError as e:
    raise ImportError(
        "LlamaCppBackend requires the 'llama' extra. Install it with: pip install 'pntx[llama]'"
    ) from e

# Rough, model-agnostic allowance for the tokens a chat template adds on top
# of message content (role markers, turn boundaries, the trailing
# generation prompt, ...); we can't size this exactly without rendering the
# model's actual template, so it's a conservative fixed margin rather than a
# precise count.
_CHAT_TEMPLATE_OVERHEAD = 64

# ggml_log_level -> logging level. Level 5 (GGML_LOG_LEVEL_CONT) means "continue
# the previous line" and has no level of its own, so it inherits whatever the
# last non-continuation line mapped to.
_GGML_LOG_LEVEL_TO_LOGGING_LEVEL = {
    0: logging.CRITICAL,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.DEBUG,
    5: logging.DEBUG,
}

# llama-cpp-python registers its own ggml/llama.cpp log callback at import
# time (llama_cpp._logger), which gates on this same logger's level but then
# prints straight to stderr rather than through the Handler pipeline -- so a
# Handler attached to this logger is silently never invoked. Replace it with
# one that actually calls logger.log(...), so callers can redirect llama.cpp's
# native logs (e.g. to a file) the same way as any other stdlib logger,
# without pntx installing a Handler/destination of its own.
_native_logger = logging.getLogger("llama-cpp-python")
_last_ggml_log_level = _GGML_LOG_LEVEL_TO_LOGGING_LEVEL[0]


@llama_cpp.llama_log_callback  # type: ignore[untyped-decorator]  # ctypes.CFUNCTYPE result has no stub signature
def _llama_log_callback(level: int, text: bytes, user_data: ctypes.c_void_p) -> None:
    global _last_ggml_log_level
    mapped_level = (
        _GGML_LOG_LEVEL_TO_LOGGING_LEVEL[level] if level != 5 else _last_ggml_log_level
    )
    # Most lines carry a trailing '\n'; progress-report dots ('.') repeated
    # without one are the documented exception (see llama_cpp's own
    # llama_log_callback signature comment).
    msg = text.decode("utf-8", errors="replace").rstrip("\n")
    if msg:
        _native_logger.log(mapped_level, msg)
    _last_ggml_log_level = mapped_level


llama_cpp.llama_log_set(_llama_log_callback, ctypes.c_void_p(0))


class LlamaCppBackend:
    """In-process backend backed by llama.cpp (via ``llama-cpp-python``).

    Implements ``Backend``, ``ScoringBackend``, ``BatchScoringBackend`` and
    ``StructuredBackend`` (``complete_json``, used by ``pn2t``'s structured
    output layer to grammar-constrain generation to valid JSON instead of
    prompting-and-parsing free text).

    Scoring methods reuse the KV cache of whatever prefix they share: rather
    than re-evaluating a shared prefix for every downstream token span, they
    eval it once and then, for each span, rewind ``n_tokens`` back to the
    prefix boundary before eval'ing that span's tokens (``Llama.eval()``
    trims the KV cache down to the current ``n_tokens`` before appending, so
    this discards only the previous span, not the shared prefix). They call
    ``Llama.eval()`` directly rather than going through chat/completion
    endpoints, since choice-scoring needs raw prefix continuation -- a chat
    template would insert its own tokens between prefix and choice and break
    that.

    ``complete``/``complete_json`` route through ``create_chat_completion``
    instead (a single ``user``-role message wrapping the whole prompt),
    rather than treating the prompt as raw text to continue. This applies
    the model's own chat template (turn-boundary tokens the model was
    instruction-tuned against), which matters a lot for reliably stopping at
    the right place -- without it, generation is more prone to degenerating
    into repeating the same token instead of naturally converging, which
    grammar-constrained decoding (every token syntactically valid, nothing
    telling the model when to stop *early*) makes easier to fall into.

    Every entry point also guards against prompts that don't fit in
    ``n_ctx``, warning and trimming from the *front* when they don't --
    prompts here are built exemplars-first, query/instruction-last (see
    ``prompts.py``), so trimming the front sheds the oldest few-shot
    material while preserving the part that's actually being asked about.
    ``score_choices``/``score_choices_batch`` trim precisely (token IDs,
    never round-tripped through text). ``complete``/``complete_json`` can
    only trim approximately: ``create_chat_completion`` takes message text,
    not pre-tokenized input, so trimming happens on a text reconstructed
    from tokenizing-then-detokenizing the prompt, and the budget reserves
    ``_CHAT_TEMPLATE_OVERHEAD`` extra tokens for the template's own markers
    since their exact count isn't known without rendering the template.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        repeat_penalty: float = 1.1,
        **kwargs: Any,
    ) -> None:
        """Load a GGUF model, either from a local path or the Hugging Face Hub.

        Exactly one of ``model_path`` or ``repo_id`` must be given. With
        ``repo_id`` (optionally narrowed to one file via ``filename``), the
        model is resolved through ``Llama.from_pretrained`` (downloaded and
        cached under the standard Hugging Face Hub cache).

        ``repeat_penalty`` is applied to every ``complete``/``complete_json``
        call (matching llama.cpp's own CLI/server default of 1.1, rather than
        ``llama-cpp-python``'s ``create_completion`` default of 1.0, i.e. no
        penalty at all). Without it, generation -- especially inside a
        grammar-constrained JSON string, where "any character" stays a valid
        continuation indefinitely -- can collapse into repeating the same
        token until ``max_tokens`` cuts it off instead of naturally closing.

        Remaining ``kwargs`` (e.g. ``n_ctx``, ``n_gpu_layers``, ``flash_attn``)
        are forwarded as-is to ``llama_cpp.Llama``. ``verbose`` defaults to
        ``False`` here (``llama_cpp.Llama`` itself defaults to ``True``), so
        ggml/llama.cpp's native logs stay quiet unless the caller opts in.
        """
        if (model_path is None) == (repo_id is None):
            raise ValueError("exactly one of model_path or repo_id must be given")
        self._repeat_penalty = repeat_penalty
        # Scoring needs per-token logits, which llama.cpp only keeps around
        # when logits_all=True.
        kwargs["logits_all"] = True
        # llama-cpp-python defaults verbose=True, which prints ggml/llama.cpp's
        # native logs (and wraps loading in its own fd-suppression only when
        # verbose is False); quiet by default, overridable via kwargs.
        kwargs.setdefault("verbose", False)
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
        return self._complete(prompt, temperature=temperature, max_tokens=max_tokens, stop=stop)

    def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        temperature: float = 1.0,
        max_tokens: int = 512,
    ) -> str:
        """Implements ``StructuredBackend``: constrain decoding to ``schema``
        via llama.cpp's grammar-constrained sampling (a GBNF grammar derived
        from the JSON schema, so every sampled token is masked down to ones
        that keep the output on a path to valid JSON matching ``schema``).

        This guarantees JSON-syntax validity but not full schema conformance
        (e.g. numeric ranges/enums) -- ``pn2t._structured.complete_structured``
        still validates the result with pydantic on top of this.
        """
        grammar = llama_cpp.LlamaGrammar.from_json_schema(
            json.dumps(_inline_refs(schema)), verbose=False
        )
        return self._complete(
            prompt, temperature=temperature, max_tokens=max_tokens, grammar=grammar
        )

    def _complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None = None,
        grammar: Any = None,
    ) -> str:
        prompt = self._fit_prompt_to_context(prompt, reserve=max_tokens)
        result = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or [],
            grammar=grammar,
            repeat_penalty=self._repeat_penalty,
        )
        if not isinstance(result, dict):
            raise TypeError(
                f"expected a non-streaming chat completion response, got {type(result)}"
            )
        return result["choices"][0]["message"]["content"] or ""

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
            "t2pn.Classifier(max_exemplars=...) to select fewer exemplars deliberately "
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

    def _fit_prompt_to_context(self, prompt: str, *, reserve: int) -> str:
        """Trim ``prompt`` (dropping from the front) so it fits in ``n_ctx``
        as chat-message content, warning when a trim actually happens.

        Unlike ``_fit_to_context``, this can't work on exact pre-tokenized
        input: ``create_chat_completion`` renders ``messages`` through the
        model's chat template itself, so the tokens actually sent depend on
        that template, not just ``prompt``. This tokenizes ``prompt`` (no
        BOS -- the template adds its own) to measure and, if needed, trim
        it, then detokenizes back to text; the budget reserves
        ``_CHAT_TEMPLATE_OVERHEAD`` on top of ``reserve`` (``max_tokens``)
        as a margin for the template's own added tokens.
        """
        n_ctx = self._llm.n_ctx()
        budget = n_ctx - reserve - _CHAT_TEMPLATE_OVERHEAD
        if budget <= 0:
            raise ValueError(
                f"reserve ({reserve} tokens) alone leaves no room in the context "
                f"window ({n_ctx} tokens) after reserving {_CHAT_TEMPLATE_OVERHEAD} tokens "
                "for the chat template; reduce max_tokens or increase n_ctx"
            )
        tokens = self._tokenize(prompt, add_bos=False)
        if len(tokens) <= budget:
            return prompt
        warnings.warn(
            f"prompt ({len(tokens)} tokens) exceeds the available context budget "
            f"({budget} of {n_ctx} tokens, after reserving {reserve} for the response "
            f"and {_CHAT_TEMPLATE_OVERHEAD} for the chat template); dropping the oldest "
            "exemplars to fit. Pass t2pn.Classifier(max_exemplars=...) to select fewer "
            "exemplars deliberately instead of relying on this truncation.",
            UserWarning,
            stacklevel=3,
        )
        trimmed = tokens[-budget:] if budget else []
        return self._llm.detokenize(trimmed).decode("utf-8", errors="ignore")

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


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve ``$ref`` pointers against ``$defs``, returning a flat schema.

    ``pydantic``'s ``model_json_schema()`` emits ``$ref``/``$defs`` for any
    nested ``BaseModel`` field (e.g. ``pn2t``'s ``list[HardPositive]``), and
    llama.cpp's JSON-schema-to-grammar converter doesn't reliably resolve
    ``$ref`` in every version -- left unresolved, the referenced item type
    can end up under-constrained in the generated grammar (notably, list
    fields losing their element schema), which manifests as generation that
    never converges to a well-formed stop and runs to ``max_tokens`` instead.
    Must run before every ``complete_json`` call, not just ones known to
    reference nested models.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, _visiting: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref: str = node["$ref"]
                if ref.startswith("#/$defs/") and ref not in _visiting:
                    return resolve(defs[ref[len("#/$defs/") :]], _visiting | {ref})
                return node  # circular ref -- leave as-is
            return {k: resolve(v, _visiting) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item, _visiting) for item in node]
        return node

    return resolve(schema)  # type: ignore[no-any-return]
