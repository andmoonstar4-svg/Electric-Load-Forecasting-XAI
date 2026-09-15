"""Experiment entry points — one function per paper table or figure.

Each function returns a table (``list[dict]``) that the CLI writes to
``runs/<experiment>/<name>.csv``. Nothing here re-derives data: they all consume
the same :class:`~elxai.data.windows.DatasetSplits`.

Mapping to the paper
--------------------
============================  ==================================================
Paper artefact                :func:`...`
============================  ==================================================
table 3-1, 3-2 (normal)        :func:`run_model_comparison`
table 4-1 (extreme)            :func:`run_model_comparison` on an extreme config
table 6-1 (loss ablation)      :func:`run_loss_ablation`
fig 5-1..5-4 (SHAP)            :func:`run_xai`
S7.3 (forecast -> scheduling)  out of scope; the ``runs/`` record is the interface
============================  ==================================================
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from elxai.config import Config
from elxai.data import datasets
from elxai.data.provenance import Provenance
from elxai.training import metrics as M
from elxai.training.trainer import (
    TrainResult,
    new_run_paths,
    train_forecaster,
    write_run,
)

log = logging.getLogger(__name__)


@dataclass
class ModelRun:
    """One trained model plus where its record lives."""

    model: str
    result: TrainResult
    run_dir: Path
    report_path: Path

    def metrics_row(self, task: str, dataset: str) -> dict[str, Any]:
        m = self.result.test_metrics
        return {
            "task": task,
            "dataset": dataset,
            "model": self.model,
            "mse": m.mse,
            "mae": m.mae,
            "rmse": m.rmse,
            "n_test_windows": m.n,
            "peak_mse": m.peak_mse,
            "peak_mae": m.peak_mae,
            "bias": m.bias,
            "run_dir": str(self.run_dir),
        }


def _prepare(cfg: Config) -> tuple[Any, Provenance]:
    built = datasets.build(cfg)
    prov = built.provenance(cfg)
    prov.dataset = cfg.data.name
    return built, prov


def train_one(cfg: Config, *, output_dir: str | Path | None = None) -> ModelRun:
    """Build the dataset, train one model, write the run record."""
    built, prov = _prepare(cfg)
    if output_dir:
        cfg = copy.deepcopy(cfg)
        cfg.run.output_dir = str(output_dir)
    paths = new_run_paths(cfg)
    result = train_forecaster(cfg, built.splits, paths=paths)
    write_run(
        result,
        cfg,
        built.splits,
        built.report,
        paths=paths,
        provenance=prov,
    )
    return ModelRun(
        model=cfg.model.name,
        result=result,
        run_dir=paths.root,
        report_path=paths.metrics,
    )


def run_model_comparison(
    base_cfg: Config,
    models: Sequence[str],
    *,
    output_dir: str | Path | None = None,
    task: str | None = None,
) -> tuple[pd.DataFrame, dict[str, TrainResult]]:
    """Train every model on the *same* splits and return the paper's table.

    Reusing one built dataset (rather than rebuilding per model) is what makes
    the comparison fair: identical windows, identical scaler, identical order.
    """
    built, prov = _prepare(base_cfg)
    out_dir = Path(output_dir or base_cfg.run.output_dir) / f"{base_cfg.experiment}-{base_cfg.data.name}"

    rows: list[dict[str, Any]] = []
    results: dict[str, TrainResult] = {}
    for model in models:
        cfg = copy.deepcopy(base_cfg)
        cfg.model.name = model
        cfg.run.output_dir = str(out_dir)
        cfg.run.run_id = None
        paths = new_run_paths(cfg)
        log.info("=== %s on %s ===", model, cfg.data.name)
        result = train_forecaster(cfg, built.splits, paths=paths)
        write_run(result, cfg, built.splits, built.report, paths=paths, provenance=prov)
        results[model] = result
        rows.append(
            ModelRun(model=model, result=result, run_dir=paths.root, report_path=paths.metrics).metrics_row(
                task or base_cfg.experiment, base_cfg.data.name
            )
        )

    table = pd.DataFrame(M.ranking_table({k: v.test_metrics for k, v in results.items()}))
    table = table.merge(
        pd.DataFrame(rows)[["model", "run_dir", "peak_mse", "bias", "n_test_windows"]],
        on="model",
        how="left",
    )
    return table, results


def run_loss_ablation(
    base_cfg: Config,
    losses: Iterable[str],
    *,
    models: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], TrainResult]]:
    """Paper table 6-1.

    The original code wrote all three loss runs into the same directory because
    ``--loss`` was absent from the ``setting`` string. Here the loss is part of
    the run identity, so every cell of the table has its own record.
    """
    built, prov = _prepare(base_cfg)
    out_dir = Path(output_dir or base_cfg.run.output_dir) / f"{base_cfg.experiment}-{base_cfg.data.name}-lossablation"

    rows: list[dict[str, Any]] = []
    results: dict[tuple[str, str], TrainResult] = {}
    for loss in losses:
        for model in models or [base_cfg.model.name]:
            cfg = copy.deepcopy(base_cfg)
            cfg.training.loss = loss  # type: ignore[assignment]
            cfg.model.name = model
            cfg.run.output_dir = str(out_dir)
            cfg.run.run_id = None
            paths = new_run_paths(cfg)
            log.info("=== %s / loss=%s ===", model, loss)
            result = train_forecaster(cfg, built.splits, paths=paths)
            write_run(result, cfg, built.splits, built.report, paths=paths, provenance=prov)
            results[(model, loss)] = result
            m = result.test_metrics
            rows.append(
                {
                    "model": model,
                    "loss": loss,
                    "loss_beta": cfg.training.exp_loss_beta if loss == "exp" else None,
                    "loss_alpha": cfg.training.exp_loss_alpha if loss == "exp" else None,
                    "mse": m.mse,
                    "mae": m.mae,
                    "rmse": m.rmse,
                    "best_epoch": result.fit_report.best_epoch,
                    "epochs_run": result.fit_report.epochs_run,
                    "run_dir": str(paths.root),
                }
            )
    return pd.DataFrame(rows), results


def run_xai(cfg: Config, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Train (or reuse) a model and write its SHAP attribution into its run dir."""
    from elxai.xai import explain_from_config, plot_importance, plot_lag_profile

    if not cfg.xai.enabled:
        raise ValueError("set xai.enabled=true to run attribution")

    built, prov = _prepare(cfg)
    if output_dir:
        cfg = copy.deepcopy(cfg)
        cfg.run.output_dir = str(output_dir)
    paths = new_run_paths(cfg)

    # Always train fresh: attribution must describe the model whose metrics are
    # in this run directory, and silently pairing a stale checkpoint with new
    # config is exactly how the original SHAP figures became unattributable.
    result = train_forecaster(cfg, built.splits, paths=paths)
    write_run(result, cfg, built.splits, built.report, paths=paths, provenance=prov)

    report = explain_from_config(cfg, built.splits, result.forecaster)
    report.to_json(paths.shap / "shap_report.json")
    report.save_arrays(paths.shap)
    plot_importance(report, paths.figures / f"shap_importance_{report.model}.png")
    if cfg.xai.resolution == "variable_time":
        from elxai.data.schema import TARGET

        plot_lag_profile(report, paths.figures / f"shap_lag_profile_{report.model}.png", variable=TARGET)

    return {
        "model": report.model,
        "run_dir": str(paths.root),
        "max_abs_additivity_residual": report.max_abs_residual,
        "relative_additivity_residual": report.relative_residual,
        "importance": report.importance_table(),
    }


def summarise(run_dirs: Sequence[str | Path]) -> pd.DataFrame:
    """Collect ``metrics.json`` from several run directories into one table.

    This is how a paper table is regenerated without retraining.
    """
    rows: list[dict[str, Any]] = []
    for directory in run_dirs:
        path = Path(directory) / "metrics.json"
        if not path.exists():
            log.warning("no metrics.json in %s", directory)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = {}
        cfg_path = Path(directory) / "config.yaml"
        if cfg_path.exists():
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        rows.append(
            {
                "run_dir": str(directory),
                "experiment": (cfg.get("experiment") if cfg else None),
                "dataset": (cfg.get("data", {}) or {}).get("name") if cfg else None,
                "model": (cfg.get("model", {}) or {}).get("name") if cfg else None,
                "loss": (cfg.get("training", {}) or {}).get("loss") if cfg else None,
                "batch_size": (cfg.get("training", {}) or {}).get("batch_size") if cfg else None,
                "mse": record["test"]["mse"],
                "mae": record["test"]["mae"],
                "rmse": record["test"]["rmse"],
                "n_test_windows": record["test"]["n"],
                "best_epoch": record["fit"]["best_epoch"],
                "seed": (cfg.get("training", {}) or {}).get("seed") if cfg else None,
            }
        )
    return pd.DataFrame(rows).sort_values("mse") if rows else pd.DataFrame()
