"""Extreme-weather labelling and subset selection (paper S4.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elxai.config import ExtremeConfig
from elxai.data import labels, schema

from conftest import synthetic_frame


def cfg(**kw) -> ExtremeConfig:
    base = dict(
        enabled=True,
        t2m_min_c=0.0,
        t2m_max_c=35.0,
        tp_min_m=0.005,
        tp_unit="m",
        u10_min_ms=15.0,
        mode="events_only",
        context_days=7,
    )
    base.update(kw)
    return ExtremeConfig(**base)


def test_flag_extreme_weather_marks_each_condition(extreme_frame):
    flagged = labels.flag_extreme_weather(extreme_frame, cfg())
    assert flagged.loc[100, "flag_t2m_low"] == 1
    assert flagged.loc[500, "flag_u10"] == 1
    assert flagged.loc[900, "flag_tp"] == 1
    assert flagged.loc[100:112, "is_extreme"].sum() == 13
    assert set(labels.FLAG_COLUMNS).issubset(flagged.columns)


def test_tp_threshold_unit_is_respected(extreme_frame):
    """ERA5 tp is metres; a 5 mm/h threshold is 0.005 m, not 5."""
    in_m = labels.flag_extreme_weather(extreme_frame, cfg(tp_unit="m", tp_min_m=0.005))
    in_mm = labels.flag_extreme_weather(extreme_frame, cfg(tp_unit="mm", tp_min_m=5.0))
    np.testing.assert_array_equal(in_m["flag_tp"], in_mm["flag_tp"])

    with pytest.raises(ValueError, match="unsupported tp_unit"):
        labels.flag_extreme_weather(extreme_frame, cfg(tp_unit="inches"))


def test_no_extreme_hours_is_reported_not_silent(extreme_frame):
    calm = extreme_frame.copy()
    calm["t2m"] = 15.0
    calm["u10"] = 1.0
    calm["tp"] = 0.0
    subset, report = labels.select_extreme_subset(calm, cfg())
    assert report.n_extreme_days == 0
    assert report.n_subset == len(calm)
    assert "is_extreme" in subset.columns


def test_events_only_keeps_event_day_plus_context(extreme_frame):
    subset, report = labels.select_extreme_subset(extreme_frame, cfg(context_days=7))
    days = labels.extreme_days(labels.flag_extreme_weather(extreme_frame, cfg()))

    expected_days: set[pd.Timestamp] = set()
    for day in days:
        for offset in range(8):  # context_days + the event day itself
            expected_days.add(pd.Timestamp(day) - pd.Timedelta(days=offset))
    # Context before the first observation cannot exist; the mask simply finds no
    # rows there, so compare against the days actually present.
    present = set(extreme_frame[schema.TIME].dt.normalize().unique())
    expected_days &= present

    assert set(subset[schema.TIME].dt.normalize().unique()) == expected_days
    assert report.n_subset == len(subset)
    assert report.n_extreme_days == len(days)
    # Every extreme hour survives the reduction.
    assert subset["is_extreme"].sum() == report.n_events_only


def test_ranges_mode_is_contiguous():
    """The 'ranges' mode exists so the frame can be windowed without gaps."""
    frame = synthetic_frame(n_hours=24 * 60, extreme=True)
    frame.loc[100:101, "t2m"] = -8.0
    frame.loc[24 * 45 : 24 * 45 + 2, "t2m"] = -9.0
    subset, report = labels.select_extreme_subset(frame, cfg(mode="ranges", context_days=7))
    assert report.n_subset == len(subset)
    labels.assert_contiguous(subset)


def test_flag_only_mode_keeps_every_row(frame):
    subset, report = labels.select_extreme_subset(frame, cfg(mode="flag_only"))
    assert len(subset) == len(frame)


def test_context_day_mask_is_inclusive_of_the_event_day(frame):
    flagged = labels.flag_extreme_weather(frame, cfg())
    day = pd.Timestamp("2023-03-10")
    mask = labels.context_day_mask(flagged, [day], context_days=2)
    kept_days = set(flagged.loc[mask, schema.TIME].dt.normalize().unique())
    assert kept_days == {day, day - pd.Timedelta(days=1), day - pd.Timedelta(days=2)}


def test_assert_contiguous_rejects_a_holed_frame(frame):
    holed = pd.concat([frame.iloc[:50], frame.iloc[100:]], ignore_index=True)
    with pytest.raises(ValueError, match="not contiguous"):
        labels.assert_contiguous(holed)


def test_thresholds_are_recorded_for_audit():
    thresholds = cfg().thresholds()
    assert thresholds["t2m_max_c"] == 35.0
    assert thresholds["tp_min_m"] == 0.005
    assert thresholds["u10_min_ms"] == 15.0
    assert thresholds["context_days"] == 7.0
