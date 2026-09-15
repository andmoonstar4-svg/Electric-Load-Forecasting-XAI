"""Metrics: agreement with reference formulas and the extra diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from elxai.training import metrics as M


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    true = rng.normal(0, 1, size=(200, 24))
    pred = true + rng.normal(0, 0.3, size=(200, 24))
    return pred, true


def test_core_metrics_match_sklearn(sample):
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    pred, true = sample
    result = M.evaluate(pred, true)
    assert result.mse == pytest.approx(sklearn_metrics.mean_squared_error(true, pred))
    assert result.mae == pytest.approx(sklearn_metrics.mean_absolute_error(true, pred))
    assert result.rmse == pytest.approx(np.sqrt(sklearn_metrics.mean_squared_error(true, pred)))
    assert result.n == true.size


def test_perfect_prediction_gives_zero():
    true = np.arange(24, dtype=float).reshape(1, 24)
    result = M.evaluate(true.copy(), true)
    assert result.mse == 0.0 and result.mae == 0.0 and result.rmse == 0.0
    assert result.bias == 0.0


def test_bias_sign_convention(sample):
    """``evaluate(pred, true)``: negative bias means the model under-predicts.

    The argument order follows the original ``utils/metrics.py`` (pred first).
    """
    pred, true = sample
    under = M.evaluate(pred - 0.5, true)
    over = M.evaluate(pred + 0.5, true)
    assert under.bias < 0 < over.bias
    # The finite sample mean of `true` is not exactly zero, hence the tolerance.
    assert under.bias == pytest.approx(-0.5, abs=0.02)


def test_mse_decomposition_is_exact(sample):
    """MSE = bias^2 + error variance, for any predictor."""
    pred, true = sample
    m = M.evaluate(pred, true)
    assert m.bias_squared + m.error_variance == pytest.approx(m.mse, rel=1e-9)


def test_horizon_decomposition_averages_to_the_total(sample):
    pred, true = sample
    m = M.evaluate(pred, true)
    assert len(m.mse_by_step) == 24 and len(m.mae_by_step) == 24
    assert np.mean(m.mse_by_step) == pytest.approx(m.mse, rel=1e-9)
    assert np.mean(m.mae_by_step) == pytest.approx(m.mae, rel=1e-9)


def test_peak_metrics_focus_on_large_loads(sample):
    pred, true = sample
    m = M.evaluate(pred, true, peak_quantile=0.9)
    assert m.peak_mse is not None and m.peak_mae is not None
    # The peak band is a subset, so its error need not equal the global error.
    assert m.peak_mse != pytest.approx(m.mse)


def test_mape_is_skipped_when_standardised_target_crosses_zero():
    """Reported as None rather than as an inflated number.

    The original code computed MAPE on the standardised target, where the
    denominator passes through zero; the value is meaningless and the paper
    correctly never quotes it.
    """
    rng = np.random.default_rng(1)
    true = rng.normal(0, 1, size=(100, 24))  # passes through zero by construction
    pred = true + rng.normal(0, 0.1, size=(100, 24))
    m = M.evaluate(pred, true, include_mape=True)
    assert m.mape is None and m.mspe is None


def test_mape_is_computed_on_a_meaningful_scale():
    """On raw load (thousands of MW) the relative error is well defined."""
    rng = np.random.default_rng(2)
    true = 10_000 + rng.normal(0, 1_000, size=(100, 24))
    pred = true * 1.01
    m = M.evaluate(pred, true, include_mape=True)
    assert m.mape is not None
    assert m.mape == pytest.approx(0.01, abs=5e-3)


def test_mape_is_computed_when_safe():
    true = np.full((50, 24), 100.0)
    pred = true * 1.01
    m = M.evaluate(pred, true, include_mape=True)
    assert m.mape == pytest.approx(0.01, rel=1e-9)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        M.evaluate(np.zeros((5, 24)), np.zeros((5, 23)))
    with pytest.raises(ValueError, match="expected"):
        M.evaluate(np.zeros(24), np.zeros(24))


def test_ranking_table_is_sorted_by_mse():
    results = {
        "a": M.evaluate(np.zeros((10, 3)), np.ones((10, 3))),
        "b": M.evaluate(np.ones((10, 3)) * 0.5, np.ones((10, 3))),
    }
    table = M.ranking_table(results)
    assert [r["model"] for r in table] == ["b", "a"]


def test_error_growth_ratio_is_reported_but_not_interpretable():
    """Paper S4.3 rejects reading the ratio as robustness; the flag says so."""
    base = M.evaluate(np.zeros((10, 3)), np.ones((10, 3)))
    extreme = M.evaluate(np.zeros((10, 3)), np.ones((10, 3)) * 2)
    ratio = M.error_growth_ratio(base, extreme)
    assert ratio["mse_ratio"] == pytest.approx(4.0)
    assert ratio["interpretable"] is False
    assert "different" in ratio["reason"]


def test_bootstrap_ci_brackets_the_point_estimate(sample):
    pred, true = sample
    lo, hi = M.bootstrap_ci(pred, true, metric="mse", n_boot=200, seed=3)
    point = M.mse(pred, true)
    assert lo <= point <= hi
    assert hi > lo


def test_bootstrap_ci_width_shrinks_with_more_data():
    rng = np.random.default_rng(0)
    small_true = rng.normal(size=(20, 24))
    small_pred = small_true + rng.normal(0, 0.3, size=(20, 24))
    big_true = rng.normal(size=(400, 24))
    big_pred = big_true + rng.normal(0, 0.3, size=(400, 24))
    lo_s, hi_s = M.bootstrap_ci(small_pred, small_true, n_boot=300, seed=1)
    lo_b, hi_b = M.bootstrap_ci(big_pred, big_true, n_boot=300, seed=1)
    assert (hi_b - lo_b) < (hi_s - lo_s)


def test_sequential_test_detects_a_clear_improvement():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(300, 24))
    good = true + rng.normal(0, 0.1, size=(300, 24))
    bad = true + rng.normal(0, 0.5, size=(300, 24))
    result = M.sequential_significance(good, bad, true)
    assert result["mean_diff"] < 0  # good has lower squared error
    assert result["p_value"] is not None and result["p_value"] < 0.05
    assert result["significant_at_5pct"] is True


def test_sequential_test_finds_no_difference_for_identical_models():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(100, 24))
    pred = true + rng.normal(0, 0.2, size=(100, 24))
    result = M.sequential_significance(pred, pred.copy(), true)
    assert result["mean_diff"] == pytest.approx(0.0)
    assert result["p_value"] is None or result["p_value"] > 0.05


def test_horizon_table_shape(sample):
    pred, true = sample
    rows = M.horizon_table(pred[:10], true[:10])
    assert len(rows) == 24
    assert rows[0]["step"] == "h1" and rows[-1]["horizon"] == 24
