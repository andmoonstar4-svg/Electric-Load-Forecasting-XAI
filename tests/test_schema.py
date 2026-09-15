"""The data contract: feature order, validation, and the failure modes it catches."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elxai.data import schema
from elxai.data.schema import SchemaError, TaskSpec

from conftest import synthetic_frame


def test_canonical_order_matches_paper_feature_indices():
    """Paper S5.1: feature0..6 = u10, t2m, msl, ssrd, tcc, tp, Total Load."""
    assert schema.PREDICTORS == ("u10", "t2m", "msl", "ssrd", "tcc", "tp")
    assert schema.COLUMNS == ("DateTime", *schema.PREDICTORS, "Total Load")
    assert schema.TARGET_INDEX == len(schema.PREDICTORS) == 6
    assert schema.TARGET_INDEX == schema.COLUMNS.index("Total Load") - 1


def test_predictor_matrix_columns_follow_declared_order(frame):
    """A silent column reorder would invalidate every SHAP figure."""
    matrix = schema.predictor_matrix(frame)
    assert matrix.shape == (len(frame), 7)
    for i, name in enumerate((*schema.PREDICTORS, schema.TARGET)):
        assert np.allclose(matrix[:, i], frame[name].to_numpy(), equal_nan=True)


def test_validate_frame_reports_what_it_saw(frame):
    report = schema.validate_frame(frame)
    assert report.n_rows == len(frame)
    assert report.inferred_freq == "1h"
    assert report.n_gap_steps == 0
    assert report.n_duplicate_timestamps == 0


def test_validate_frame_rejects_missing_column(frame):
    with pytest.raises(SchemaError, match="missing required column"):
        schema.validate_frame(frame.drop(columns=["tcc"]))


def test_validate_frame_rejects_duplicate_timestamps(frame):
    """Duplicates let one observation enter two splits."""
    doubled = pd.concat([frame, frame.iloc[[10]]], ignore_index=True).sort_values(schema.TIME)
    with pytest.raises(SchemaError, match="duplicate timestamp"):
        schema.validate_frame(doubled.reset_index(drop=True))


def test_validate_frame_rejects_unsorted_and_nan(frame):
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    with pytest.raises(SchemaError, match="not sorted"):
        schema.validate_frame(shuffled)

    holed = frame.copy()
    holed.loc[5, "u10"] = np.nan
    with pytest.raises(SchemaError, match="NaN values present"):
        schema.validate_frame(holed)


def test_validate_frame_reports_gaps(frame):
    gapped = pd.concat([frame.iloc[:100], frame.iloc[110:]], ignore_index=True)
    report = schema.validate_frame(gapped, allow_gaps=True)
    assert report.n_gap_steps > 0
    assert report.n_duplicate_timestamps == 0


def test_validate_frame_treats_gaps_as_fatal_by_default(frame):
    """Only the extreme-weather task may opt in to gaps (its frame requires them)."""
    gapped = pd.concat([frame.iloc[:100], frame.iloc[110:]], ignore_index=True)
    with pytest.raises(SchemaError, match="regular grid"):
        schema.validate_frame(gapped)


def test_task_spec_rejects_non_day_aligned_windows():
    with pytest.raises(ValueError, match="whole number of days"):
        TaskSpec(seq_len=100, pred_len=24)


def test_task_spec_reports_lookback_days():
    assert TaskSpec(168, 24).lookback_days == 7
    assert TaskSpec(72, 24).lookback_days == 3
    assert TaskSpec(168, 24).window == 192


def test_channel_names_are_human_readable_and_ordered():
    names = schema.channel_names()
    assert len(names) == 7
    assert names[-1] == "Total Load"
    assert names[0] == "10m wind speed"


def test_feature_registry_covers_every_predictor():
    for name in schema.PREDICTORS:
        f = schema.feature(name)
        assert f.unit and f.source and f.long_name
    with pytest.raises(KeyError):
        schema.feature("not_a_predictor")


def test_synthetic_frame_is_self_consistent():
    """Guard the fixture itself: it must satisfy the contract it claims to."""
    schema.validate_frame(synthetic_frame())
