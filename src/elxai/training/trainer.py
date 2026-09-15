"""Run-directory layout and the train/evaluate orchestration.

The original code base named its output folders with a ``setting`` string built
from task/model/data/hyper-parameters. Two consequences:

* ``--loss`` was **not** part of the string, so the three runs behind the loss
  ablation table (S6.2) wrote into the same ``./results/<setting>/`` directory and
  overwrote one another;
* the SHAP output path was ``./shap_results/<model>_<setting>/``, so the
  *extreme-weather* and *normal-weather* SHAP figures for one model differed only
  if some hyper-parameter happened to differ.

Here a run directory is ``runs/<run_id>/`` where ``run_id`` encodes every field
that can change a number — including the loss — and is checked for collision.
Nothing is ever overwritten: an identical configuration gets a deterministic
suffix so re-running is idempotent to inspect but never lossy.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from elxai.config import Config
from elxai.data.schema import channel_names
from elxai.data.windows import DatasetSplits, WindowBatch
from elxai.models import FitReport, Forecaster, build_forecaster
from elxai.training import metrics as M
from elxai.training.losses import loss_metadata

log = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str) -> str:
    return _SAFE.sub("-", text).strip("-")


@dataclass
class RunPaths:
    """Everything a run writes, in one place."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def __getattr__(self, item: str) -> Path:  # pragma: no cover - convenience
        raise AttributeError(item)

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance.json"

    @property
    def dataset_report(self) -> Path:
        return self.root / "dataset_report.json"

    @property
    def splits(self) -> Path:
        return self.root / "splits.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def horizon(self) -> Path:
        return self.root / "metrics_by_horizon.csv"

    @property
    def training(self) -> Path:
        return self.root / "training.json"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions.npz"

    @property
    def shap(self) -> Path:
        return self.root / "shap"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def model(self) -> Path:
        return self.root / "model"

    def ensure(self) -> "RunPaths":
        for path in (self.root, self.shap, self.figures, self.model):
            path.mkdir(parents=True, exist_ok=True)
        return self


def new_run_paths(cfg: Config, *, root: str | Path | None = None) -> RunPaths:
    """Create (or reuse) ``runs/<run_id>`` for ``cfg``.

    ``run_id`` = ``<config.name>__<fingerprint>``. The fingerprint covers the full
    config, so changing ``training.loss`` yields a different directory.
    """
    base = Path(root or cfg.run.output_dir)
    run_id = cfg.run.run_id or f"{cfg.name}__{cfg.fingerprint()}"
    target = base / slug(run_id)
    if target.exists() and cfg.run.run_id is None:
        # Same config re-run: reuse, but never clobber an existing record silently.
        existing = target / "metrics.json"
        if existing.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            target = base / slug(f"{run_id}__{stamp}")
            log.info("an identical run already exists; writing to %s", target.name)
    return RunPaths(target).ensure()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TrainResult:
    forecaster: Forecaster
    fit_report: FitReport
    test_metrics: M.Metrics
    val_metrics: M.Metrics | None
    predictions: np.ndarray
    targets: np.ndarray
    paths: RunPaths | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit": self.fit_report.as_dict(),
            "test": self.test_metrics.as_dict(),
            "val": None if self.val_metrics is None else self.val_metrics.as_dict(),
            "extra": self.extra,
        }


def train_forecaster(
    cfg: Config, data: DatasetSplits, *, paths: RunPaths | None = None
) -> TrainResult:
    """Build, fit and evaluate one model on already-windowed splits."""
    from elxai.models.torch_adapter import TrainConfig

    train_batch = data["train"]
    val_batch = data["val"]

    if len(train_batch) == 0:
        raise ValueError("training split has no windows; check the split/purge settings")

    train_cfg = TrainConfig(
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        patience=cfg.training.patience,
        seed=cfg.training.seed,
        device=cfg.training.device,
        num_workers=cfg.training.num_workers,
        amp=cfg.training.amp,
        loss=cfg.training.loss,
        loss_kwargs=(
            {"beta": cfg.training.exp_loss_beta, "alpha": cfg.training.exp_loss_alpha}
            if cfg.training.loss == "exp"
            else None
        ),
    )

    forecaster = build_forecaster(
        cfg.model.name,
        seq_len=cfg.data.seq_len,
        pred_len=cfg.data.pred_len,
        n_channels=train_batch.n_channels,
        params=cfg.model.params,
        train=train_cfg,
    )

    log.info("training %s on %d windows (val %d)", forecaster.name, len(train_batch), len(val_batch))
    fit_report = forecaster.fit(
        train_batch.x,
        train_batch.y,
        x_val=val_batch.x if len(val_batch) else None,
        y_val=val_batch.y if len(val_batch) else None,
    )

    result = evaluate_forecaster(forecaster, fit_report, data, cfg=cfg)
    result.paths = paths
    return result


def evaluate_forecaster(
    forecaster: Forecaster,
    fit_report: FitReport,
    data: DatasetSplits,
    *,
    cfg: Config | None = None,
    include_mape: bool = False,
) -> TrainResult:
    """Score a fitted forecaster on the val and test splits."""
    test_batch = data["test"]
    val_batch = data["val"]

    pred_test = (
        forecaster.predict(test_batch.x) if len(test_batch) else np.zeros((0, data.task.pred_len))
    )
    test_metrics = (
        M.evaluate(pred_test, test_batch.y, include_mape=include_mape)
        if len(test_batch)
        else M.Metrics(mse=float("nan"), mae=float("nan"), rmse=float("nan"), n=0)
    )

    val_metrics = None
    if len(val_batch):
        pred_val = forecaster.predict(val_batch.x)
        val_metrics = M.evaluate(pred_val, val_batch.y, include_mape=include_mape)

    log.info(
        "%s -> test MSE %.4f MAE %.4f (n=%d)",
        forecaster.name,
        test_metrics.mse,
        test_metrics.mae,
        test_metrics.n,
    )

    extra: dict[str, Any] = {
        "channel_names": channel_names(),
        "target_channel": channel_names()[-1],
        "uses_internal_normalisation": forecaster.uses_internal_normalisation,
        "model_class": type(forecaster).__name__,
    }
    if fit_report.notes:
        extra["loss"] = fit_report.notes.get("loss")
        extra["loss_kwargs"] = fit_report.notes.get("loss_kwargs", {})
    if cfg is not None:
        extra["loss_metadata"] = loss_metadata(cfg.training)

    return TrainResult(
        forecaster=forecaster,
        fit_report=fit_report,
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        predictions=pred_test,
        targets=test_batch.y,
        extra=extra,
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def write_run(
    result: TrainResult,
    cfg: Config,
    data: DatasetSplits,
    dataset_report: dict[str, Any],
    *,
    paths: RunPaths | None = None,
    provenance=None,
) -> RunPaths:
    """Write the complete run record.

    The caller passes the :class:`RunPaths` produced by :func:`new_run_paths` so
    that a run which was redirected to a timestamped directory (because an
    identical configuration already existed) is not silently written back over
    the original.
    """
    if paths is None:
        paths = result.paths or new_run_paths(cfg)
    paths.ensure()

    cfg.to_yaml(paths.config)
    paths.dataset_report.write_text(
        json.dumps(dataset_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    paths.splits.write_text(
        json.dumps(data.plan.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    record = result.to_dict()
    record["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    record["scaling"] = data.scaler.as_dict()
    record["task"] = {
        "seq_len": data.task.seq_len,
        "pred_len": data.task.pred_len,
        "mode": "MS",
        "lookback_days": data.task.lookback_days,
    }
    paths.metrics.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    paths.training.write_text(
        json.dumps(
            {
                "history": result.fit_report.history or [],
                "best_epoch": result.fit_report.best_epoch,
                "epochs_run": result.fit_report.epochs_run,
                "n_parameters": result.fit_report.n_parameters,
                "device": result.fit_report.device,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    if len(result.predictions):
        np.savez_compressed(
            paths.predictions,
            pred=result.predictions,
            true=result.targets,
            start_index=data["test"].start_index,
            start_time=data["test"].start_time.astype("int64").to_numpy(),
        )
        horizon = M.horizon_table(result.predictions, result.targets)
        pd.DataFrame(horizon).to_csv(paths.horizon, index=False)

    result.forecaster.save(paths.model)
    if provenance is not None:
        provenance.write(paths.provenance)
    return paths
