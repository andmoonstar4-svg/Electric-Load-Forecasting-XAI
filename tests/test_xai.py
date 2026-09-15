"""XAI: the specific defects of the original routine, encoded as tests.

Each test names the original behaviour it prevents from coming back.
"""

from __future__ import annotations

import numpy as np
import pytest

from elxai.data import schema
from elxai.models import build_forecaster
from elxai.xai import shap_explainer as S

shap = pytest.importorskip("shap")

SEQ, HOR, CH = 12, 4, 7


def linear_forecaster():
    """A deterministic stand-in with a known ground-truth attribution.

    The target is exactly ``sum over lags of x[lag, target_channel] * w[lag]``, so
    the correct attribution is that only the target channel matters and the lag
    profile follows ``|w|``.
    """

    class LinearForecaster:
        name = "Linear"
        uses_internal_normalisation = False

        def __init__(self):
            self.w = np.linspace(0.1, 1.2, SEQ, dtype=np.float64)
            self.pred_len = HOR

        def _y(self, x):
            target_lags = x[:, :, schema.TARGET_INDEX]
            base = target_lags @ self.w
            return np.stack([base * (1.0 + 0.1 * h) for h in range(HOR)], axis=1)

        def predict(self, x):
            return self._y(np.asarray(x, dtype=np.float64))

        def predict_all_channels(self, x):
            out = np.zeros((len(x), HOR, CH))
            out[:, :, schema.TARGET_INDEX] = self.predict(x)
            return out

    return LinearForecaster()


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return (
        rng.normal(size=(80, SEQ, CH)).astype(np.float32),
        rng.normal(size=(40, SEQ, CH)).astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# nsamples policy
# --------------------------------------------------------------------------- #


def test_auto_nsamples_scales_with_dimension():
    """The original used a flat 100 for a 1176-dimensional game."""
    small = S.auto_nsamples(84)
    large = S.auto_nsamples(1176)
    assert large > small
    assert S.auto_nsamples(1176) >= 4 * 1176 or S.auto_nsamples(1176) == 20_000


def test_auto_nsamples_has_a_floor_and_ceiling():
    assert S.auto_nsamples(1) == 2_000
    assert S.auto_nsamples(10**6) == 20_000


def test_resolve_nsamples_warns_when_below_the_minimum(caplog):
    with caplog.at_level("WARNING"):
        S.resolve_nsamples(10, n_features=100)
    assert any("completeness" in r.message for r in caplog.records)


def test_resolve_nsamples_rejects_unknown_strings():
    with pytest.raises(ValueError, match="unsupported nsamples spec"):
        S.resolve_nsamples("lots", 10)


# --------------------------------------------------------------------------- #
# target step selection
# --------------------------------------------------------------------------- #


def test_resolve_target_steps_all():
    assert S.resolve_target_steps("all", 24) == list(range(24))


def test_resolve_target_steps_deduplicates_and_sorts():
    assert S.resolve_target_steps([5, 0, 5, 2], 24) == [0, 2, 5]


def test_resolve_target_steps_rejects_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        S.resolve_target_steps([24], 24)


def test_original_explained_only_step_zero_this_allows_the_whole_horizon(data):
    """The original hard-coded outputs[:, 0, 0] regardless of the caption."""
    bg, ev = data
    report = S.explain(
        linear_forecaster(),
        x_background=bg,
        x_eval=ev,
        n_background=20,
        n_eval=3,
        nsamples=400,
        target_steps="all",
    )
    assert report.target_steps == list(range(HOR))
    assert report.predictions.shape == (3, HOR)
    assert report.mean_abs.shape[0] == CH


# --------------------------------------------------------------------------- #
# resolution and completeness
# --------------------------------------------------------------------------- #


def test_variable_resolution_sums_over_lags_and_matches_ground_truth(data):
    bg, ev = data
    report = S.explain(
        linear_forecaster(),
        x_background=bg,
        x_eval=ev,
        n_background=30,
        n_eval=3,
        nsamples=600,
        resolution="variable",
        target_steps=[0],
    )
    assert report.shap_values.shape == (3, 1, CH)
    shares = report.variable_importance()
    # The model depends only on the target channel's history.
    assert shares["Total Load__share"] > 0.99


def test_variable_time_resolution_keeps_the_lag_axis(data):
    bg, ev = data
    report = S.explain(
        linear_forecaster(),
        x_background=bg,
        x_eval=ev,
        n_background=30,
        n_eval=3,
        nsamples=600,
        resolution="variable_time",
        target_steps=[0],
    )
    assert report.shap_values.shape == (3, 1, SEQ * CH)
    assert len(report.feature_names) == SEQ * CH
    assert report.feature_names[0].startswith(schema.channel_names()[0])
    assert report.feature_names[0].endswith("lag-12")
    assert report.feature_names[-1].endswith("lag-1")


def test_completeness_residual_is_small_for_a_linear_model(data):
    """A correct attribution satisfies sum(phi) + E[f] == f(x)."""
    bg, ev = data
    report = S.explain(
        linear_forecaster(),
        x_background=bg,
        x_eval=ev,
        n_background=40,
        n_eval=5,
        nsamples=2000,
        resolution="variable",
        target_steps=[0],
        check_additivity=True,
    )
    assert report.relative_residual < 0.05, report.relative_residual


def test_importance_table_is_sorted_and_has_shares(data):
    bg, ev = data
    report = S.explain(
        linear_forecaster(),
        x_background=bg,
        x_eval=ev,
        n_background=20,
        n_eval=3,
        nsamples=400,
        target_steps=[0],
    )
    rows = report.importance_table()
    assert [r["mean_abs_shap"] for r in rows] == sorted(
        [r["mean_abs_shap"] for r in rows], reverse=True
    )
    assert sum(r["share"] for r in rows) == pytest.approx(1.0, rel=1e-6)
    assert all("is_target_lag" in r for r in rows)


def test_report_json_records_the_caveats(data, tmp_path):
    import json

    bg, ev = data
    report = S.explain(
        linear_forecaster(),
        x_background=bg,
        x_eval=ev,
        n_background=20,
        n_eval=3,
        nsamples=400,
        target_steps=[0],
    )
    path = report.to_json(tmp_path / "shap_report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "nsamples",
        "n_background",
        "n_eval",
        "target_steps",
        "max_abs_additivity_residual",
        "relative_additivity_residual",
        "model_normalises_windows",
    ):
        assert key in payload
    assert "not the physical causal effect" in payload["notes"]["interpretation"]


def test_arrays_round_trip(data, tmp_path):
    bg, ev = data
    report = S.explain(
        linear_forecaster(), x_background=bg, x_eval=ev, n_background=20, n_eval=3, nsamples=400
    )
    report.save_arrays(tmp_path)
    with np.load(tmp_path / "shap_values.npz") as npz:
        assert npz["shap_values"].shape == report.shap_values.shape
        assert npz["predictions"].shape == report.predictions.shape


def test_subsampling_is_deterministic(data):
    bg, ev = data
    kwargs = dict(n_background=16, n_eval=4, nsamples=300, target_steps=[0])
    a = S.explain(linear_forecaster(), x_background=bg, x_eval=ev, seed=7, **kwargs)
    b = S.explain(linear_forecaster(), x_background=bg, x_eval=ev, seed=7, **kwargs)
    np.testing.assert_allclose(a.mean_abs, b.mean_abs)


def test_background_and_eval_counts_are_respected(data):
    bg, ev = data
    report = S.explain(
        linear_forecaster(), x_background=bg, x_eval=ev, n_background=25, n_eval=6, nsamples=300
    )
    assert report.n_background == 25
    assert report.n_eval == 6


def test_more_background_than_available_is_capped(data):
    bg, ev = data
    report = S.explain(
        linear_forecaster(), x_background=bg[:5], x_eval=ev[:3], n_background=100, n_eval=100, nsamples=200
    )
    assert report.n_background == 5 and report.n_eval == 3


# --------------------------------------------------------------------------- #
# tree path
# --------------------------------------------------------------------------- #


def test_tree_method_runs_for_the_forest_baseline():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, SEQ, CH)).astype(np.float32)
    y = np.stack([x[:, :, schema.TARGET_INDEX].mean(axis=1)] * HOR, axis=1).astype(np.float32)
    forecaster = build_forecaster(
        "RandomForest",
        seq_len=SEQ,
        pred_len=HOR,
        n_channels=CH,
        params={"n_estimators": 10, "n_jobs": 1, "random_state": 0},
    )
    forecaster.fit(x[:40], y[:40])

    report = S.explain(
        forecaster,
        x_background=x[:40],
        x_eval=x[40:],
        n_background=20,
        n_eval=5,
        method="auto",
        target_steps="all",
    )
    assert report.method == "tree"
    assert report.nsamples == "tree-exact"
    assert report.shap_values.shape == (5, HOR, CH)
    assert report.relative_residual < 1e-6


def test_lag_profile_requires_variable_time_resolution(data, tmp_path):
    bg, ev = data
    report = S.explain(
        linear_forecaster(), x_background=bg, x_eval=ev, n_background=20, n_eval=3, nsamples=300
    )
    assert S.plot_lag_profile(report, tmp_path / "never.png") is None


def test_lag_profile_recovers_the_ground_truth_weighting(data, tmp_path):
    """The lag profile must track |w|, which the original averaged away."""
    bg, ev = data
    report = S.explain(
        linear_forecaster(),
        x_background=bg,
        x_eval=ev,
        n_background=40,
        n_eval=3,
        nsamples=3000,
        resolution="variable_time",
        target_steps=[0],
    )
    grid = report.mean_abs.reshape(SEQ, CH)[:, schema.TARGET_INDEX]
    forecaster = linear_forecaster()
    # Correlate the recovered profile with the true weights.
    corr = np.corrcoef(grid, np.abs(forecaster.w))[0, 1]
    assert corr > 0.7, corr


def test_plot_importance_writes_a_figure(data, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    bg, ev = data
    report = S.explain(
        linear_forecaster(), x_background=bg, x_eval=ev, n_background=20, n_eval=3, nsamples=300
    )
    path = S.plot_importance(report, tmp_path / "importance.png")
    assert path is not None and path.exists() and path.stat().st_size > 0


def test_compare_reports_stacks_models(data):
    bg, ev = data
    reports = {
        "a": S.explain(linear_forecaster(), x_background=bg, x_eval=ev, n_background=20, n_eval=3, nsamples=200),
        "b": S.explain(linear_forecaster(), x_background=bg, x_eval=ev, n_background=20, n_eval=3, nsamples=200),
    }
    rows = S.compare_reports(reports)
    assert len(rows) == 2 * CH
    assert {r["model"] for r in rows} == {"a", "b"}
