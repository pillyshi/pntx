from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self


class LLMEstimatorMixin:
    """sklearn ``clone()`` compatibility for estimators wrapping a ``Backend``.

    ``sklearn.base.clone()`` normally reconstructs an estimator by deep-copying
    each ``__init__`` parameter. A ``Backend`` may wrap a non-deep-copyable
    object (e.g. ``llama_cpp.Llama``, which holds ctypes pointers), so the
    default behavior would break -- or silently try to reload the model.
    Overriding ``__sklearn_clone__`` reconstructs the estimator by re-running
    ``__init__`` with the same (not copied) parameter values instead, so
    ``cross_val_score``/``GridSearchCV`` share one loaded backend across folds.

    Must be combined with ``sklearn.base.BaseEstimator`` (which provides
    ``get_params``); mix this in *before* ``BaseEstimator`` in the MRO.
    """

    def __sklearn_clone__(self) -> Self:
        new = type(self)(**self.get_params(deep=False))  # type: ignore[attr-defined]
        # Preserve set_output()/metadata-routing config, mirroring what
        # BaseEstimator._clone_parametrized copies after construction.
        for attr in ("_sklearn_output_config", "_metadata_request"):
            if hasattr(self, attr):
                setattr(new, attr, copy.deepcopy(getattr(self, attr)))
        return new
