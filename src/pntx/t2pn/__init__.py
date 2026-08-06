from __future__ import annotations

from typing import TYPE_CHECKING

from .prompting import LLMPromptingClassifier

if TYPE_CHECKING:
    from .finetuning import FineTuningClassifier

__all__ = ["LLMPromptingClassifier", "FineTuningClassifier"]


def __getattr__(name: str) -> object:
    # FineTuningClassifier needs the 'finetuning' extra (transformers/torch);
    # importing it lazily here means `from pntx.t2pn import LLMPromptingClassifier`
    # doesn't require that extra to be installed at all.
    if name == "FineTuningClassifier":
        from .finetuning import FineTuningClassifier

        return FineTuningClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
