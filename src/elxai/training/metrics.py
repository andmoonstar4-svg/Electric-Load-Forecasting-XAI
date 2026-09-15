"""Evaluation metrics.

Reproduces the paper's reported quantities and adds the decompositions that make
its claims checkable:

* ``MSE``/``MAE`` on the target channel only (MS mode) — the numbers in tables
  3-1, 3-2, 4-1 and 6-1;
* per-horizon-step errors, which the paper's qualitative claims about "日内峰谷
  变化被压缩" and "负荷峰谷变化被压缩" implicitly require but never tabulate;
* peak-band errors, which give "峰值低估" a number;
* an error-growth ratio between two tasks that is *reported but not interpreted*,
  because the paper correctly rejects that comparison (S4.3) when the two tasks
  have different training sets.

Note the horizon argument: the original ``metrics.metric`` averaged MAPE/MSPE over
the target channel *and* over all channels of the ``(B, L, C)`` tensor. Here the
target channel is selected explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class Metrics:
    """Error summary for one model on one task."""

    mse: float
    mae: float
    rmse: float
    n: int
    mape: float | None = None
    mspe: float | None = None
    #: ``(pred_len,)`` MSE per forecast step.
    mse_by_step: list[float] = field(default_factory=list)
    mae_by_step: list[float] = field(default_factory=list)
    #: MSE restricted to the top-``peak_quantile`` observed loads.
    peak_mse: float | None = None
    peak_mae: float | None = None
    peak_quantile: float | None = None
    #: Mean signed error; negative indicates systematic under-prediction.
    bias: float | None = None
    #: Decomposition of MSE into bias^2 + variance of the error.
    bias_squared: float | None = None
    error_variance: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)) ** 2))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64))))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def _safe_ratio(pred: np.ndarray, true: np.ndarray) -> tuple[float | None, float | None, int]:
    """MAPE/MSPE plus the count of usable points.

    Relative errors are only defined when the target has a meaningful scale. For
    standardised load the denominator passes through zero and MAPE inflates to
    order 1 whatever the model does; that is why the paper never quotes it. The
    guard is therefore on the magnitude of the target, not merely on the count of
    non-zero entries.

    Returns ``(None, None, n_usable)`` when the target is too small for a relative
    error to mean anything.
    """
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if true.size == 0:
        return None, None, 0

    scale = float(np.mean(np.abs(true)))
    if scale < 1.0:
        # Standardised (or otherwise dimensionless) target: MAPE is not meaningful.
        return None, None, 0

    usable = np.abs(true) > 0.01 * scale
    n_usable = int(usable.sum())
    if n_usable < 0.5 * true.size:
        return None, None, n_usable
    rel = (true[usable] - pred[usable]) / true[usable]
    return float(np.mean(np.abs(rel))), float(np.mean(rel**2)), n_usable


def evaluate(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    peak_quantile: float = 0.9,
    include_mape: bool = False,
) -> Metrics:
    """Compute the metric set for ``(n, pred_len)`` target arrays."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs true {true.shape}")
    if pred.ndim != 2:
        raise ValueError(f"expected (n, pred_len), got {pred.shape}")

    # Signed error follows the usual convention: pred - true, so a negative bias
    # means the model systematically under-predicts (the paper's "峰值低估").
    errors = pred - true
    out = Metrics(
        mse=float(np.mean(errors**2)),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        n=int(pred.size),
        mse_by_step=[float(v) for v in np.mean(errors**2, axis=0)],
        mae_by_step=[float(v) for v in np.mean(np.abs(errors), axis=0)],
        bias=float(np.mean(errors)),
        bias_squared=float(np.mean(errors) ** 2),
        error_variance=float(np.var(errors)),
        peak_quantile=peak_quantile,
    )
    if include_mape:
        m, s, n_usable = _safe_ratio(pred, true)
        out.mape, out.mspe = m, s

    if 0.0 < peak_quantile < 1.0 and true.size:
        threshold = float(np.quantile(true, peak_quantile))
        mask = true >= threshold
        if mask.any():
            out.peak_mse = float(np.mean(errors[mask] ** 2))
            out.peak_mae = float(np.mean(np.abs(errors[mask])))
    return out


def error_growth_ratio(base: Metrics, extreme: Metrics) -> dict[str, Any]:
    """Report the between-task error ratio *and* whether it is interpretable.

    The paper (S4.3) declines to read the ratio as a robustness measure because
    the two tasks have different training sets. That reading is kept: the ratio is
    returned with an explicit ``interpretable=False`` unless both tasks share the
    same test-period definition, in which case the caller can flip the flag.
    """
    return {
        "mse_ratio": extreme.mse / base.mse if base.mse else None,
        "mae_ratio": extreme.mae / base.mae if base.mae else None,
        "interpretable": False,
        "reason": (
            "the normal-weather and extreme-weather tasks are trained and tested on "
            "different data, so this ratio mixes model capability, sample distribution "
            "and task difficulty (paper S4.3)"
        ),
    }


def ranking_table(results: dict[str, Metrics]) -> list[dict[str, Any]]:
    """Models sorted by MSE, the paper's primary criterion."""
    rows = [
        {"model": name, "mse": m.mse, "mae": m.mae, "rmse": m.rmse, "n": m.n}
        for name, m in results.items()
    ]
    return sorted(rows, key=lambda r: r["mse"])


def bootstrap_ci(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    metric: str = "mse",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 2021,
) -> tuple[float, float]:
    """Percentile bootstrap CI over test windows.

    The published tables are single runs; with 24-step windows the effective
    sample size is modest, so a CI shows whether a 0.03 MSE gap is meaningful.
    """
    fn = {"mse": mse, "mae": mae, "rmse": rmse}[metric]
    pred = np.asarray(pred)
    true = np.asarray(true)
    n = len(pred)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = fn(pred[idx], true[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def sequential_significance(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    true: np.ndarray,
    *,
    seed: int = 2021,
) -> dict[str, float | bool | None]:
    """Diebold-Mariano style test on squared-error loss differences.

    Answers "is model A's MSE genuinely lower than model B's, given window-to-window
    variance?" with a two-sided normal approximation and a Newey-West-free
    variance (fine for non-overlapping 24-step horizons).
    """
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred_a.shape != pred_b.shape:
        raise ValueError("pred_a and pred_b must have the same shape")
    d = (true - pred_a) ** 2 - (true - pred_b) ** 2
    d_bar = float(d.mean())
    n = d.size
    if n < 2:
        return {"mean_diff": d_bar, "statistic": None, "p_value": None, "n": n}
    var = float(d.var(ddof=1))
    if var <= 0:
        return {
            "mean_diff": d_bar,
            "statistic": None,
            "p_value": None,
            "n": n,
            "note": "zero variance in loss differences",
        }
    stat = d_bar / np.sqrt(var / n)
    from math import erfc, sqrt

    p = erfc(abs(stat) / sqrt(2))
    return {
        "mean_diff": d_bar,
        "statistic": float(stat),
        "p_value": float(p),
        "n": int(n),
        "significant_at_5pct": bool(p < 0.05),
    }


def horizon_table(pred: np.ndarray, true: np.ndarray, step_names: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Per-step errors as rows, for the horizon plot the paper never showed."""
    m = evaluate(pred, true)
    names = list(step_names) if step_names is not None else [f"h{i + 1}" for i in range(len(m.mse_by_step))]
    return [
        {"step": names[i], "horizon": i + 1, "mse": m.mse_by_step[i], "mae": m.mae_by_step[i]}
        for i in range(len(m.mse_by_step))
    ]
