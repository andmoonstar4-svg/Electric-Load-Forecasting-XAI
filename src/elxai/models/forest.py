"""Random Forest baseline (paper S2.2).

The original project trained the forest outside the modelling framework, so it
could not participate in the same evaluation or the same attribution code. Here
it implements the identical :class:`~elxai.models.base.Forecaster` contract:

* input ``(n, seq_len, C)`` is flattened to ``(n, seq_len * C)``;
* the forest predicts all ``pred_len`` steps at once (direct multi-step), which
  is what the paper describes ("直接将输出设为 24 维的代表未来 24 小时负荷的张量");
* out-of-bag / impurity importances are exposed at both the flat and the
  channel level, so the tree baseline can be compared against SHAP for the
  neural models on the same axis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from elxai.data import schema
from elxai.models.base import FitReport, Forecaster

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "n_jobs": -1,
    "random_state": 2021,
    "oob_score": False,
}


class RandomForestForecaster(Forecaster):
    """Direct multi-step Random Forest regressor."""

    name = "RandomForest"
    uses_internal_normalisation = False

    def __init__(self, params: Mapping[str, Any] | None = None):
        self.params = {**DEFAULTS, **(params or {})}
        self._model = None
        self._seq_len: int | None = None
        self._n_channels: int | None = None

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def flatten(x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError(f"expected (n, seq_len, C), got shape {x.shape}")
        return x.reshape(x.shape[0], -1)

    def unflatten_importance(self, flat: np.ndarray) -> np.ndarray:
        """``(seq_len * C,) -> (seq_len, C)`` impurity importance grid."""
        if self._seq_len is None or self._n_channels is None:
            raise RuntimeError("model is not fitted")
        return flat.reshape(self._seq_len, self._n_channels)

    def channel_importance(self) -> np.ndarray | None:
        """Importance summed over lags, one score per input channel."""
        flat = self.feature_importance()
        if flat is None:
            return None
        grid = self.unflatten_importance(flat)
        total = grid.sum(axis=0)
        s = total.sum()
        return total / s if s > 0 else total

    # -- Forecaster API ---------------------------------------------------- #

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitReport:
        from sklearn.ensemble import RandomForestRegressor

        self._seq_len = x_train.shape[1]
        self._n_channels = x_train.shape[2]
        self._model = RandomForestRegressor(**self.params)
        self._model.fit(self.flatten(x_train), y_train)

        notes: dict[str, Any] = {"params": self.params}
        val_loss = None
        if x_val is not None and y_val is not None and len(x_val):
            pred = self.predict(x_val)
            val_loss = float(np.mean((pred - y_val) ** 2))
            notes["val_mse"] = val_loss
        if self.params.get("oob_score"):
            notes["oob_score"] = float(getattr(self._model, "oob_score_", np.nan))

        log.info(
            "RandomForest fitted: %d trees, %d features -> %d outputs",
            self.params["n_estimators"],
            x_train.shape[1] * x_train.shape[2],
            y_train.shape[1],
        )
        return FitReport(
            n_train=len(x_train),
            n_val=0 if x_val is None else len(x_val),
            best_epoch=None,
            epochs_run=1,
            best_val_loss=val_loss,
            history=[],
            n_parameters=None,
            device="cpu",
            notes=notes,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")
        return np.asarray(self._model.predict(self.flatten(x)), dtype=np.float32)

    def feature_importance(self) -> np.ndarray | None:
        if self._model is None:
            return None
        imp = getattr(self._model, "feature_importances_", None)
        return None if imp is None else np.asarray(imp, dtype=np.float64)

    def save(self, directory: str | Path) -> Path:
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            joblib.dump(self._model, directory / "model.joblib")
        return (directory / "model.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "seq_len": self._seq_len,
                    "n_channels": self._n_channels,
                    "params": self.params,
                    "channel_names": schema.channel_names(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )


#: Name aliases accepted in YAML.
ALIASES: dict[str, str] = {
    "randomforest": "RandomForest",
    "random_forest": "RandomForest",
    "rf": "RandomForest",
    "randomforestforecaster": "RandomForest",
}
