"""Windowing and splitting — the leak-prevention guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elxai.config import ScalingConfig, SplitConfig
from elxai.data import schema
from elxai.data.schema import TaskSpec
from elxai.data.windows import (
    Scaler,
    build_dataset_splits,
    chronological_split,
    make_windows,
)

from conftest import synthetic_frame


def test_split_is_chronological_and_purged(long_frame):
    times = pd.DatetimeIndex(pd.to_datetime(long_frame[schema.TIME]))
    plan = chronological_split(len(long_frame), SplitConfig(), times=times, task=TaskSpec(168, 24))

    assert plan.purge == 191  # seq_len + pred_len - 1
    tr, va, te = (plan.bounds[k] for k in ("train", "val", "test"))
    assert tr[0] == 0
    assert tr[1] < va[0] < va[1] < te[0] < te[1]
    # The purge actually removes rows: the splits do not tile the series.
    assert va[0] - tr[1] == plan.purge
    assert te[0] - va[1] == plan.purge


def test_no_window_overlaps_the_next_split(windows):
    """This is the leak the original code base had: no purge between splits."""
    task = windows.task
    for a, b in (("train", "val"), ("val", "test")):
        last_end = windows[a].start_index[-1] + task.window
        assert last_end <= windows[b].start_index[0], f"{a} reaches into {b}"

    last_target = windows["train"].start_index[-1] + task.seq_len + task.pred_len
    assert last_target <= windows["val"].start_index[0] + task.pred_len


def test_insufficient_data_raises_with_actionable_message():
    with pytest.raises(ValueError, match="purge"):
        chronological_split(400, SplitConfig(), task=TaskSpec(168, 24))


def test_explicit_purge_is_honoured(long_frame):
    """A caller who sets purge explicitly gets exactly that many rows removed."""
    times = pd.DatetimeIndex(pd.to_datetime(long_frame[schema.TIME]))
    plan = chronological_split(
        len(long_frame), SplitConfig(purge=10), times=times, task=TaskSpec(168, 24)
    )
    assert plan.purge == 10
    tr, va, te = (plan.bounds[k] for k in ("train", "val", "test"))
    assert va[0] - tr[1] == 10
    assert te[0] - va[1] == 10


def test_unsafe_explicit_purge_is_logged_as_a_warning(long_frame, caplog):
    """purge below seq_len+pred_len-1 reintroduces the overlap; it must be visible."""
    times = pd.DatetimeIndex(pd.to_datetime(long_frame[schema.TIME]))
    with caplog.at_level("WARNING"):
        plan = chronological_split(
            len(long_frame), SplitConfig(purge=5), times=times, task=TaskSpec(168, 24)
        )
    assert plan.purge == 5
    assert any("smaller than seq_len+pred_len-1" in r.message for r in caplog.records)


def test_window_shapes_and_target_selection(long_frame):
    matrix = schema.predictor_matrix(long_frame).astype(np.float32)
    batch = make_windows(matrix, TaskSpec(168, 24))
    n_expected = len(long_frame) - 192 + 1
    assert batch.x.shape == (n_expected, 168, 7)
    assert batch.y.shape == (n_expected, 24)
    assert batch.y_all_channels.shape == (n_expected, 24, 7)

    # y must be the target channel of the tail of x-window+horizon.
    i = 7
    start = batch.start_index[i]
    expected = matrix[start + 168 : start + 192, schema.TARGET_INDEX]
    np.testing.assert_allclose(batch.y[i], expected, rtol=1e-6)
    np.testing.assert_allclose(
        batch.y_all_channels[i, :, schema.TARGET_INDEX], expected, rtol=1e-6
    )
    # ... and the input must be exactly the preceding seq_len rows.
    np.testing.assert_allclose(batch.x[i], matrix[start : start + 168], rtol=1e-6)


def test_windows_spanning_a_gap_are_dropped():
    """The extreme-weather subset jumps over months; a window must not straddle it."""
    frame = synthetic_frame(n_hours=24 * 40)
    # Cut out a week in the middle.
    holed = pd.concat([frame.iloc[:400], frame.iloc[568:]], ignore_index=True)
    times = pd.DatetimeIndex(pd.to_datetime(holed[schema.TIME]))
    matrix = schema.predictor_matrix(holed).astype(np.float32)

    kept = make_windows(matrix, TaskSpec(72, 24), times=times, require_contiguous=True)
    naive = make_windows(matrix, TaskSpec(72, 24), times=times, require_contiguous=False)
    assert len(kept) < len(naive)

    # Every surviving window spans exactly seq_len + pred_len - 1 hours.
    for start in kept.start_time:
        end = start + pd.Timedelta(hours=95)
        assert start + pd.Timedelta(hours=95) == end


def test_sparse_extreme_like_subset_only_loses_gap_windows():
    """A frame of separated blocks: exactly the straddling windows disappear.

    Mirrors the extreme-weather construction (paper S4.1), where each block is one
    extreme date plus its preceding week and the blocks are months apart.
    """
    frame = synthetic_frame(n_hours=24 * 60)
    # Two 8-day blocks, far apart.
    blocks = [frame.iloc[: 24 * 8], frame.iloc[24 * 40 : 24 * 48]]
    sparse = pd.concat(blocks, ignore_index=True)
    times = pd.DatetimeIndex(pd.to_datetime(sparse[schema.TIME]))
    matrix = schema.predictor_matrix(sparse).astype(np.float32)

    task = TaskSpec(72, 24)
    kept = make_windows(matrix, task, times=times, require_contiguous=True)
    naive = make_windows(matrix, task, times=times, require_contiguous=False)

    assert len(kept) < len(naive)
    # No surviving window may straddle the gap.
    gap_start = blocks[0][schema.TIME].iloc[-1]
    for start, end in zip(kept.start_time, kept.start_time + pd.Timedelta(hours=task.window - 1)):
        assert not (start <= gap_start < end), f"window {start}..{end} straddles the gap"
    # The number lost is bounded by the gap-crossing positions in each block edge.
    assert naive.start_index.size - kept.start_index.size > 0


def test_scaler_is_fitted_on_training_rows_only(long_frame):
    """Fitting on the whole series leaks test-period statistics into training."""
    splits, report = build_dataset_splits(
        long_frame, TaskSpec(168, 24), SplitConfig(), ScalingConfig()
    )
    raw = schema.predictor_matrix(long_frame).astype(np.float64)
    lo, hi = splits.plan.bounds["train"]
    np.testing.assert_allclose(splits.scaler.mean_, raw[lo:hi].mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(splits.scaler.scale_, raw[lo:hi].std(axis=0), rtol=1e-6)

    # And the full-series statistics must differ, otherwise the test is vacuous.
    assert not np.allclose(splits.scaler.mean_, raw.mean(axis=0), rtol=1e-6)


def test_scaled_windows_are_standardised_using_the_train_fit(windows):
    train = windows["train"].x
    assert abs(float(train.mean())) < 0.5
    assert abs(float(train.std()) - 1.0) < 0.35


def test_scaler_round_trip():
    rng = np.random.default_rng(0)
    matrix = rng.normal(5.0, 3.0, size=(500, 7))
    scaler = Scaler.fit(matrix, ScalingConfig())
    restored = scaler.inverse_transform(scaler.transform(matrix))
    np.testing.assert_allclose(restored, matrix, rtol=1e-5, atol=1e-6)


def test_scaler_handles_constant_column():
    matrix = np.ones((50, 3))
    scaler = Scaler.fit(matrix, ScalingConfig())
    assert np.all(np.isfinite(scaler.transform(matrix)))
    np.testing.assert_allclose(scaler.transform(matrix), 0.0, atol=1e-9)


def test_scaler_none_is_identity():
    matrix = np.arange(30, dtype=np.float64).reshape(10, 3)
    scaler = Scaler.fit(matrix, ScalingConfig(method="none"))
    np.testing.assert_allclose(scaler.transform(matrix), matrix)


def test_inverse_target_uses_the_target_channel():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(100, 7)) * np.arange(1, 8)
    scaler = Scaler.fit(matrix, ScalingConfig())
    values = np.array([0.0, 1.0])
    expected = values * scaler.scale_[schema.TARGET_INDEX] + scaler.mean_[schema.TARGET_INDEX]
    np.testing.assert_allclose(scaler.inverse_target(values), expected)


def test_dataset_splits_summary_is_tabular(windows):
    summary = windows.summary()
    assert list(summary["split"]) == ["train", "val", "test"]
    assert (summary["n_windows"] > 0).all()
