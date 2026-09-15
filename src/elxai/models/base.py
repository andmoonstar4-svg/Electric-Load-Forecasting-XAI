"""Forecaster interface shared by the neural adapters and the sklearn baseline.

The point of this abstraction is that the explainability layer and the training
loop must not know which model they are driving. In the original code base the
Random Forest baseline lived outside the framework entirely (so it could not be
explained with the same SHAP code) and the SHAP routine reached into
``DataLoader`` internals to build its inputs.

Contract
--------
``fit`` / ``predict`` operate on arrays whose last axis is the channel axis in
:mod:`elxai.data.schema` order, and ``predict`` returns ``(n, pred_len)`` target
values only — the paper's MS setting. Use :meth:`Forecaster.predict_all_channels`
when the upstream model's full output is needed.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from elxai.data import schema


@dataclass
class FitReport:
    """What happened during training — enough to audit a run."""

    n_train: int
    n_val: int
    best_epoch: int | None = None
    epochs_run: int = 0
    best_val_loss: float | None = None
    history: list[dict[str, float]] | None = None
    n_parameters: int | None = None
    device: str | None = None
    notes: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Forecaster(ABC):
    """Minimal, array-in/array-out forecasting interface."""

    #: Short identifier used in run directories and figures.
    name: str = "base"
    #: Whether the model normalises each input window internally. Documented
    #: because it stacks on top of the dataset-level scaler and changes what the
    #: reported (standardised) errors mean.
    uses_internal_normalisation: bool = False

    @abstractmethod
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitReport:
        """Train on ``(n, seq_len, C) -> (n, pred_len)``."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return ``(n, pred_len)`` predictions of the target channel."""

    def predict_all_channels(self, x: np.ndarray) -> np.ndarray:
        """Return ``(n, pred_len, C)``. Default: broadcast the target prediction."""
        pred = self.predict(x)
        out = np.zeros((*pred.shape, x.shape[-1]), dtype=np.float32)
        out[:, :, schema.TARGET_INDEX] = pred
        return out

    def feature_importance(self) -> np.ndarray | None:
        """Native importance scores per input channel, when the model has them."""
        return None

    def save(self, directory: str | Path) -> Path:
        """Persist whatever is needed to reload without retraining."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return (directory / "model.json").write_text(
            json.dumps({"name": self.name, "class": type(self).__name__}, indent=2),
            encoding="utf-8",
        )

    @property
    def predict_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """``(n, window*C) -> (n,)`` callable for one target step.

        Convenience for explainers that need a flat scalar-output function.
        """

        def fn(flat: np.ndarray) -> np.ndarray:
            return self.predict(flat).reshape(-1)

        return fn

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"{type(self).__name__}(name={self.name!r})"
