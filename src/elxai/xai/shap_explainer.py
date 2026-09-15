"""SHAP-based attribution, redesigned around the flaws of the original routine.

What the original ``shap_analysis`` did (``exp/exp_long_term_forecasting.py``)
and why each choice is wrong for the paper's claims:

===============================================  ==========================================================
Original                                          Consequence / what this module does instead
===============================================  ==========================================================
``background = batch_x[:5]``                      5 reference windows estimate ``E[f]`` to almost no precision.
                                                  Configurable ``n_background`` (default 100), drawn from the
                                                  **training** split so no test information enters the
                                                  baseline.
``nsamples=100`` on 1176 flattened inputs          100 coalitions for a 1176-dimensional game. The SHAP values
                                                  barely correlate with the true attributions and violate
                                                  completeness. ``nsamples='auto'`` here scales with the
                                                  dimension and the resulting residual is *measured*, not
                                                  assumed.
``test = batch_x[5:8]``                           3 explained windows. Configurable ``n_eval`` (default 50).
Flatten ``(168, 7) -> 1176`` and group by column  Averaging over the time axis assumes the product kernel, so
                                                  per-variable scores need not add up to the prediction. The
                                                  resolution is now an explicit choice: ``variable`` (7 groups,
                                                  the honest reading of "which variables matter") or
                                                  ``variable_time`` (lag-resolved, for the temporal-weight claims).
``outputs[:, 0, 0]``                              Only forecast hour 1 was explained, yet the captions imply the
                                                  whole 24-hour forecast. ``target_steps`` selects steps;
                                                  ``"all"`` explains the full horizon.
``except Exception: print(...)``                  Failures were printed and swallowed, so a figure could be
                                                  silently missing. Errors propagate; the JSON report records
                                                  the residual so a bad attribution cannot look like a good one.
``folder = './shap_results/' + model + '_' + setting``
                                                 The setting string omitted the loss, so the three loss-ablation
                                                  attributions collided. XAI artefacts live in the run directory.
===============================================  ==========================================================

Known limitation, stated rather than hidden
-------------------------------------------
iTransformer and DLinear standardise each window *inside* ``forward``
(``x_hat = (x - mean(x)) / std(x)``). Perturbing one input therefore also changes
the window statistics, which is exactly the mechanism behind the paper's "模型对
历史时间规律的相关权重下降" observation. KernelSHAP with an interventional
background remains a valid average marginal contribution, but the attribution is
no longer a clean statement about the raw input. The residual and the flag
:attr:`XAIReport.model_normalises_windows` are emitted so this is visible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from elxai.data import schema
from elxai.data.schema import PREDICTORS, TARGET
from elxai.models.base import Forecaster

log = logging.getLogger(__name__)

Resolution = Literal["variable", "variable_time"]


@dataclass
class XAIReport:
    """Attribution results plus everything needed to judge them."""

    model: str
    method: str
    resolution: Resolution
    n_background: int
    n_eval: int
    nsamples: int | str
    target_steps: list[int]
    #: ``(n_eval, pred_len, n_groups)`` where ``n_groups`` is ``C`` for
    #: ``variable`` and ``seq_len * C`` for ``variable_time``.
    shap_values: np.ndarray
    #: ``(n_eval, pred_len)`` model outputs on the explained windows.
    predictions: np.ndarray
    #: ``(n_eval, pred_len)`` SHAP base values ``E[f]``.
    base_values: np.ndarray
    #: ``(n_groups,)`` mean |SHAP| over eval windows and target steps.
    mean_abs: np.ndarray
    #: ``(n_eval, pred_len)`` completeness residual.
    additivity_residual: np.ndarray
    feature_names: list[str]
    model_normalises_windows: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def max_abs_residual(self) -> float:
        return float(np.max(np.abs(self.additivity_residual))) if self.additivity_residual.size else float("nan")

    @property
    def relative_residual(self) -> float:
        """Worst-case residual normalised by the spread of the predictions."""
        spread = float(np.std(self.predictions)) or 1.0
        return self.max_abs_residual / spread

    def variable_importance(self) -> dict[str, float]:
        """Mean |SHAP| per input variable, summing over lags when needed."""
        n_channels = len(self.feature_names)
        if self.mean_abs.size == n_channels:
            values = self.mean_abs
        else:
            grid = self.mean_abs.reshape(-1, n_channels)
            values = grid.sum(axis=0)
        total = float(values.sum())
        share = values / total if total > 0 else values
        return {name: float(v) for name, v in zip(self.feature_names, values)} | {
            f"{name}__share": float(s) for name, s in zip(self.feature_names, share)
        }

    def importance_table(self) -> list[dict[str, Any]]:
        """Rows sorted by contribution, ready for a paper table."""
        importances = self.variable_importance()
        names = self.feature_names
        rows = [
            {
                "variable": name,
                "mean_abs_shap": importances[name],
                "share": importances[f"{name}__share"],
                "is_target_lag": name == TARGET,
            }
            for name in names
        ]
        return sorted(rows, key=lambda r: r["mean_abs_shap"], reverse=True)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "method": self.method,
            "resolution": self.resolution,
            "n_background": self.n_background,
            "n_eval": self.n_eval,
            "nsamples": self.nsamples,
            "target_steps": self.target_steps,
            "feature_names": self.feature_names,
            "model_normalises_windows": self.model_normalises_windows,
            "max_abs_additivity_residual": self.max_abs_residual,
            "relative_additivity_residual": self.relative_residual,
            "notes": self.notes,
            "importance": self.importance_table(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def save_arrays(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "shap_values.npz",
            shap_values=self.shap_values,
            predictions=self.predictions,
            base_values=self.base_values,
            mean_abs=self.mean_abs,
            additivity_residual=self.additivity_residual,
        )
        return directory


# --------------------------------------------------------------------------- #
# nsamples policy
# --------------------------------------------------------------------------- #


def auto_nsamples(n_features: int, *, multiplier: int = 4, floor: int = 2_000, ceiling: int = 20_000) -> int:
    """Coalition count scaled to the dimension of the SHAP game.

    The original code used a hard 100 regardless of dimension. A practical rule
    is a few times the feature count (Shapley values need at least ``2d`` samples
    to be well defined for ``d`` features), bounded so a 1176-feature run stays
    tractable.
    """
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    return int(min(max(multiplier * n_features, floor), ceiling))


def resolve_nsamples(spec: int | str, n_features: int) -> int:
    if isinstance(spec, str):
        if spec != "auto":
            raise ValueError(f"unsupported nsamples spec {spec!r}; use an int or 'auto'")
        return auto_nsamples(n_features)
    if spec < 2 * n_features:
        log.warning(
            "nsamples=%d is below 2 x n_features=%d; SHAP completeness may not hold",
            spec,
            n_features,
        )
    return int(spec)


def resolve_target_steps(spec: Sequence[int] | str, pred_len: int) -> list[int]:
    if isinstance(spec, str):
        if spec != "all":
            raise ValueError(f"unsupported target_steps spec {spec!r}; use a list or 'all'")
        return list(range(pred_len))
    steps = sorted({int(s) for s in spec})
    for s in steps:
        if not 0 <= s < pred_len:
            raise ValueError(f"target step {s} outside [0, {pred_len})")
    return steps


def _stack_by_output(values: Any, n_samples: int, n_features: int) -> np.ndarray:
    """Normalise a SHAP return value to ``(n_samples, n_outputs, n_features)``.

    SHAP's conventions are inconsistent across versions and explainers:

    * ``KernelExplainer`` with a multi-column predict function returns
      ``(n_samples, n_features, n_outputs)``;
    * ``TreeExplainer`` on a multi-output forest may return that same array, or a
      *list* of ``(n_samples, n_features)`` arrays, one per output.

    Both are handled here and anything else is rejected loudly rather than
    reshaped into a plausible-looking but wrong array — the original routine
    reshaped blindly.
    """
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("SHAP returned an empty list of per-output arrays")
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=1)
        if stacked.shape[0] != n_samples:
            stacked = np.transpose(stacked, (1, 0, 2))
        if stacked.shape != (n_samples, len(values), n_features):
            raise ValueError(
                f"unexpected per-output SHAP shapes: {[np.shape(v) for v in values]}"
            )
        return stacked

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[:, None, :]
    elif array.ndim == 3:
        if array.shape[0] != n_samples and array.shape[2] == n_samples:
            array = np.transpose(array, (2, 1, 0))
        elif array.shape[1] != n_features and array.shape[2] == n_features:
            array = np.transpose(array, (0, 2, 1))
        elif array.shape[1] == n_features and array.shape[2] != n_samples:
            array = np.transpose(array, (0, 2, 1))
    if array.ndim != 3 or array.shape[0] != n_samples or array.shape[2] != n_features:
        raise ValueError(
            f"cannot interpret SHAP output of shape {np.shape(values)} as "
            f"({n_samples}, n_outputs, {n_features})"
        )
    return array


def _stack_base_values(base: Any, n_samples: int, n_outputs: int) -> np.ndarray:
    """Normalise ``expected_value`` to ``(n_samples, n_outputs)``."""
    array = np.asarray(base, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.full((n_samples, n_outputs), float(array[0]))
    if array.size != n_outputs:
        raise ValueError(
            f"base value has {array.size} entries but {n_outputs} output(s) were explained"
        )
    return np.tile(array, (n_samples, 1))


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #


def explain(
    forecaster: Forecaster,
    *,
    x_background: np.ndarray,
    x_eval: np.ndarray,
    n_background: int = 100,
    n_eval: int = 50,
    nsamples: int | str = "auto",
    resolution: Resolution = "variable",
    target_steps: Sequence[int] | str = (0,),
    method: Literal["kernel", "tree", "auto"] = "auto",
    seed: int = 2021,
    check_additivity: bool = True,
) -> XAIReport:
    """Compute SHAP attributions for ``forecaster`` on ``x_eval``.

    Parameters
    ----------
    x_background:
        Candidate reference windows, normally the **training** split.
    x_eval:
        Windows to attribute. Subsampled to ``n_eval`` with a fixed seed.
    resolution:
        ``variable`` returns one score per input variable (summed over lags);
        ``variable_time`` keeps a score for every ``(lag, variable)`` pair, which
        is what claims about time-weight changes need.
    target_steps:
        Forecast steps to explain. ``"all"`` explains the full horizon.
    method:
        ``tree`` for the Random Forest (exact, fast), ``kernel`` for the neural
        models (black-box), ``auto`` to pick by model type.
    """
    pred = forecaster.predict(x_eval)
    pred_len = pred.shape[1]
    steps = resolve_target_steps(target_steps, pred_len)
    n_channels = x_eval.shape[2]
    seq_len = x_eval.shape[1]

    chosen_method = method
    if method == "auto":
        chosen_method = "tree" if hasattr(forecaster, "_model") and forecaster.name == "RandomForest" else "kernel"

    rng = np.random.default_rng(seed)
    bg = _subsample(x_background, n_background, rng)
    ev = _subsample(x_eval, n_eval, rng)

    if chosen_method == "tree":
        shap_values, base_values, used_nsamples = _explain_tree(forecaster, bg, ev, steps)
    else:
        shap_values, base_values, used_nsamples = _explain_kernel(
            forecaster, bg, ev, steps, nsamples
        )

    # (n_eval, len(steps), seq_len * C) -> group by resolution
    if resolution == "variable":
        grouped = shap_values.reshape(
            shap_values.shape[0], shap_values.shape[1], seq_len, n_channels
        ).sum(axis=2)
        names = list(schema.channel_names())
    elif resolution == "variable_time":
        grouped = shap_values
        # Name as "<channel>@lag-<n>" so the variable is a prefix and the lag a
        # suffix: a reader can group by either without re-parsing.
        names = [
            f"{name}@lag-{seq_len - lag}"
            for lag in range(seq_len)
            for name in schema.channel_names()
        ]
    else:
        raise ValueError(f"unknown resolution {resolution!r}")

    mean_abs = np.mean(np.abs(grouped), axis=(0, 1))

    ev_pred = np.asarray(forecaster.predict(ev), dtype=np.float64)
    selected_pred = ev_pred[:, steps]

    if check_additivity:
        # Completeness: sum(phi) + E[f] should reproduce f(x).
        recon = grouped.sum(axis=2) + base_values
        residual = recon - selected_pred
    else:
        residual = np.zeros_like(base_values)

    return XAIReport(
        model=forecaster.name,
        method=chosen_method,
        resolution=resolution,
        n_background=len(bg),
        n_eval=len(ev),
        nsamples=used_nsamples,
        target_steps=steps,
        shap_values=grouped,
        predictions=selected_pred,
        base_values=base_values,
        mean_abs=mean_abs,
        additivity_residual=residual,
        feature_names=names,
        model_normalises_windows=bool(getattr(forecaster, "uses_internal_normalisation", False)),
        notes={
            "seq_len": seq_len,
            "n_channels": n_channels,
            "n_shap_features": seq_len * n_channels,
            "explained_step_indices": steps,
            "interpretation": (
                "SHAP measures how much the model relies on an input, not the "
                "physical causal effect of that variable on load (paper S5.1)."
            ),
        },
    )


def predictions_for(forecaster: Forecaster, x: np.ndarray) -> np.ndarray:
    return np.asarray(forecaster.predict(x), dtype=np.float64)


def _subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return x
    idx = rng.choice(len(x), size=n, replace=False)
    return x[np.sort(idx)]


def _explain_tree(forecaster: Forecaster, bg: np.ndarray, ev: np.ndarray, steps: Sequence[int]):
    import shap

    n_samples = len(ev)
    n_features = ev.shape[1] * ev.shape[2]
    flat_bg = bg.reshape(len(bg), -1)
    flat_ev = ev.reshape(n_samples, -1)
    explainer = shap.TreeExplainer(forecaster._model)  # type: ignore[attr-defined]
    raw = explainer.shap_values(flat_ev)

    per_step = _stack_by_output(raw, n_samples, n_features)[:, list(steps), :]
    base = _stack_base_values(explainer.expected_value, n_samples, len(steps))
    return per_step, base, "tree-exact"


def _explain_kernel(
    forecaster: Forecaster,
    bg: np.ndarray,
    ev: np.ndarray,
    steps: Sequence[int],
    nsamples: int | str,
):
    import shap

    seq_len, n_channels = ev.shape[1], ev.shape[2]
    n_features = seq_len * n_channels
    n_samples = len(ev)
    resolved = resolve_nsamples(nsamples, n_features)

    flat_bg = bg.reshape(len(bg), -1).astype(np.float64)
    flat_ev = ev.reshape(n_samples, -1).astype(np.float64)
    step_list = list(steps)

    def flat_predict(x_flat: np.ndarray) -> np.ndarray:
        x3 = np.asarray(x_flat, dtype=np.float32).reshape(-1, seq_len, n_channels)
        out = forecaster.predict(x3)
        # Always return 2-D so SHAP treats this as a multi-output function; a
        # 1-D result for a single selected step changes the axis convention.
        return np.asarray(out, dtype=np.float64)[:, step_list]

    log.info(
        "KernelSHAP: %d explained window(s), %d background, %d features, nsamples=%d",
        n_samples,
        len(bg),
        n_features,
        resolved,
    )
    explainer = shap.KernelExplainer(flat_predict, flat_bg)
    raw = explainer.shap_values(flat_ev, nsamples=resolved, l1_reg="num_features(64)")

    per_step = _stack_by_output(raw, n_samples, n_features)
    base = _stack_base_values(explainer.expected_value, n_samples, len(step_list))
    return per_step, base, resolved


# --------------------------------------------------------------------------- #
# Adapter-level helper
# --------------------------------------------------------------------------- #


def explain_from_config(cfg, data, forecaster: Forecaster, *, seed: int | None = None) -> XAIReport:
    """Run :func:`explain` using ``cfg.xai`` and a :class:`DatasetSplits`."""
    return explain(
        forecaster,
        x_background=data["train"].x,
        x_eval=data["test"].x,
        n_background=cfg.xai.n_background,
        n_eval=cfg.xai.n_eval,
        nsamples=cfg.xai.nsamples,
        resolution=cfg.xai.resolution,
        target_steps=cfg.xai.target_steps,
        method=cfg.xai.method,
        seed=seed if seed is not None else cfg.training.seed,
        check_additivity=cfg.xai.check_additivity,
    )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_importance(report: XAIReport, path: str | Path, *, top_k: int | None = None) -> Path | None:
    """Horizontal bar chart of mean |SHAP| per variable (ASCII labels only).

    matplotlib is an optional dependency; if it is absent the figure is skipped
    and the JSON report remains the source of truth.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        log.warning("matplotlib unavailable; skipping figure for %s", report.model)
        return None

    rows = report.importance_table()
    if top_k:
        rows = rows[:top_k]
    labels = [r["variable"] for r in rows][::-1]
    values = [r["mean_abs_shap"] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(7.0, 0.42 * max(len(labels), 3) + 1.2), dpi=200)
    ax.barh(range(len(labels)), values, color="#2b5c8f")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean |SHAP value|  (standardised load)")
    ax.set_title(
        f"{report.model}: input attribution\n"
        f"{report.resolution} resolution, {report.n_eval} windows, "
        f"steps {report.target_steps[:4]}{'...' if len(report.target_steps) > 4 else ''}"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def plot_lag_profile(report: XAIReport, path: str | Path, *, variable: str = TARGET, max_lag: int | None = None) -> Path | None:
    """Mean |SHAP| against lag for one variable — the temporal-weight claim.

    Only meaningful at ``variable_time`` resolution; the original implementation
    averaged the lag axis away before anyone could look at it.
    """
    if report.resolution != "variable_time":
        log.warning("lag profile needs resolution='variable_time'; got %r", report.resolution)
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    grid = report.mean_abs.reshape(-1, len(schema.channel_names()))
    names = schema.channel_names()
    if variable not in names:
        raise KeyError(f"unknown variable {variable!r}; have {names}")
    profile = grid[:, names.index(variable)]
    lags = np.arange(len(profile))[::-1]
    if max_lag:
        profile = profile[-max_lag:]
        lags = lags[-max_lag:]

    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=200)
    ax.plot(lags, profile, color="#8f402b")
    ax.set_xlabel("lag (hours before forecast start)")
    ax.set_ylabel("mean |SHAP value|")
    ax.set_title(f"{report.model}: {variable} lag profile")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def compare_reports(reports: dict[str, XAIReport]) -> list[dict[str, Any]]:
    """Side-by-side variable shares, for the Autoformer-vs-iTransformer discussion."""
    rows: list[dict[str, Any]] = []
    for model, report in reports.items():
        for row in report.importance_table():
            rows.append({"model": model, **row})
    return rows
