"""Predictive models behind one interface.

``build_forecaster`` is the factory used by the experiment runner; it maps the
``model.name`` / ``model.params`` block of a YAML config onto either a vendored
upstream neural model or the sklearn Random Forest baseline.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from elxai.models.base import FitReport, Forecaster
from elxai.models.forest import ALIASES as _FOREST_ALIASES
from elxai.models.forest import RandomForestForecaster

log = logging.getLogger(__name__)

#: Neural architectures with an adapter registered in
#: :mod:`elxai.models.torch_adapter`.
NEURAL_MODELS: tuple[str, ...] = ("iTransformer", "DLinear", "Autoformer", "Transformer")

__all__ = [
    "Forecaster",
    "FitReport",
    "RandomForestForecaster",
    "NEURAL_MODELS",
    "build_forecaster",
    "canonical_name",
    "filter_model_params",
    "model_param_keys",
]


def canonical_name(name: str) -> str:
    """Normalise a YAML model name to its class name."""
    if name in NEURAL_MODELS:
        return name
    return _FOREST_ALIASES.get(name.lower(), name)


def model_param_keys(name: str) -> set[str]:
    """The parameter names ``name`` understands, so foreign keys can be dropped.

    Needed because configs are composed by deep merge: combining an iTransformer
    config with a Random Forest config leaves ``d_model`` in ``model.params``, and
    the forest would receive it as a sklearn kwarg and raise. Dropping the foreign
    keys (loudly, in the log) is preferable to either crashing or silently passing
    an unknown argument to a model that happens to accept ``**kwargs``.
    """
    resolved = canonical_name(name)
    if resolved == "RandomForest":
        from elxai.models.forest import DEFAULTS

        return set(DEFAULTS)
    from elxai.models.torch_adapter import required_config_attrs

    return required_config_attrs(resolved)


def filter_model_params(name: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only the entries of ``params`` that ``name`` can consume."""
    params = dict(params or {})
    if not params:
        return {}
    allowed = model_param_keys(name)
    kept = {k: v for k, v in params.items() if k in allowed}
    dropped = sorted(set(params) - allowed)
    if dropped:
        log.info(
            "%s: ignoring %d parameter(s) not read by this model: %s",
            canonical_name(name),
            len(dropped),
            dropped,
        )
    return kept


def build_forecaster(
    name: str,
    *,
    seq_len: int,
    pred_len: int,
    n_channels: int,
    params: Mapping[str, Any] | None = None,
    train: Any = None,
) -> Forecaster:
    """Instantiate the model named by a config.

    ``params`` is filtered to the keys the selected model actually reads, then
    passed through: for neural models it becomes part of the upstream ``configs``
    namespace, for the forest it becomes sklearn kwargs.
    """
    resolved = canonical_name(name)
    usable = filter_model_params(resolved, params)

    if resolved == "RandomForest":
        return RandomForestForecaster(usable)

    if resolved in NEURAL_MODELS:
        from elxai.models.torch_adapter import TorchForecaster

        return TorchForecaster(
            resolved,
            seq_len=seq_len,
            pred_len=pred_len,
            n_channels=n_channels,
            model_params=usable,
            train=train,
        )

    raise KeyError(
        f"unknown model {name!r}; neural options: {NEURAL_MODELS}, "
        f"plus {sorted(set(_FOREST_ALIASES))} for the Random Forest baseline"
    )
