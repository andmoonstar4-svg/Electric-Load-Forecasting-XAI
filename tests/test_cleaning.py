"""Cleaning: forecast-column removal, the sub-hourly join bug, gap handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elxai.data import cleaning, schema

from conftest import synthetic_frame


def test_load_series_drops_model_generated_columns(tmp_path):
    """Paper S2.1: only the observed Total Load column may be used as a target."""
    n = 48
    raw = pd.DataFrame(
        {
            "Datetime": pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC"),
            "Total Load": np.arange(n, dtype=float),
            "Most recent forecast": np.arange(n, dtype=float) + 100,
            "Day-ahead 6PM forecast": np.arange(n, dtype=float) + 200,
            "Week-ahead forecast": np.arange(n, dtype=float) + 300,
        }
    )
    path = tmp_path / "elia.csv"
    raw.to_csv(path, index=False, sep=";")

    frame, report = cleaning.load_series(path)
    assert set(frame.columns) == {schema.TIME, schema.TARGET}
    assert set(report.dropped_forecast_columns) == {
        "Most recent forecast",
        "Day-ahead 6PM forecast",
        "Week-ahead forecast",
    }
    np.testing.assert_allclose(frame[schema.TARGET], raw["Total Load"])
    assert frame[schema.TIME].dt.tz is None


def test_load_series_requires_the_observed_column(tmp_path):
    raw = pd.DataFrame(
        {
            "Datetime": pd.date_range("2023-01-01", periods=5, freq="1h"),
            "Most recent forecast": np.arange(5.0),
        }
    )
    path = tmp_path / "bad.csv"
    raw.to_csv(path, index=False, sep=";")
    with pytest.raises(schema.SchemaError, match="observed-load column"):
        cleaning.load_series(path)


def test_weather_series_converts_kelvin_and_renames(tmp_path):
    n = 48
    raw = pd.DataFrame(
        {
            "valid_time": pd.date_range("2023-01-01", periods=n, freq="1h"),
            "u10": np.linspace(0, 5, n),
            "t2m": np.linspace(270.0, 300.0, n),
            "msl": np.full(n, 101_000.0),
            "ssrd": np.zeros(n),
            "tcc": np.full(n, 0.5),
            "tp": np.zeros(n),
        }
    )
    path = tmp_path / "era5.csv"
    raw.to_csv(path, index=False)

    frame, report = cleaning.weather_series(path)
    assert set(schema.PREDICTORS).issubset(frame.columns)
    assert frame["t2m"].iloc[0] == pytest.approx(270.0 - 273.15, abs=1e-6)
    assert report.weather_gap_hours == 0


def test_weather_series_counts_gaps_after_reindexing(tmp_path):
    n = 24
    times = pd.date_range("2023-01-01", periods=n, freq="1h").delete(5)
    raw = pd.DataFrame(
        {
            "valid_time": times,
            "u10": np.ones(len(times)),
            "t2m": np.full(len(times), 283.15),
            "msl": np.full(len(times), 101_000.0),
            "ssrd": np.zeros(len(times)),
            "tcc": np.zeros(len(times)),
            "tp": np.zeros(len(times)),
        }
    )
    path = tmp_path / "gapped_era5.csv"
    raw.to_csv(path, index=False)

    frame, report = cleaning.weather_series(path)
    assert report.weather_gap_hours == 1
    assert len(frame) == n
    assert frame["u10"].isna().sum() == 1


def test_merge_of_subhourly_load_keeps_every_observation():
    """Sub-hourly load is averaged to the hour, not silently filtered."""
    hours = pd.date_range("2023-01-01", periods=6, freq="1h")
    quarter = pd.date_range("2023-01-01", periods=24, freq="15min")
    load = pd.DataFrame({schema.TIME: quarter, schema.TARGET: np.ones(24) * 4.0})
    weather = pd.DataFrame(
        {
            schema.TIME: hours,
            **{c: np.linspace(1, 6, 6) for c in schema.PREDICTORS},
        }
    )

    merged, report = cleaning.merge_load_weather(load, weather, interpolate_weather=False)
    assert report.join_strategy == "resample+exact"
    assert report.load_resampled_to_hourly
    assert len(merged) == 6
    assert not merged[schema.TIME].duplicated().any()
    # The mean of four identical 15-minute readings is that value.
    np.testing.assert_allclose(merged[schema.TARGET], 4.0)


def test_readings_beyond_the_legacy_tolerance_are_reported():
    """The original 30-minute as-of join silently drops readings with no nearby hour.

    `merge_asof(direction='nearest', tolerance=...)` keeps a row whenever *any*
    weather row lies within the tolerance, so the loss appears only where the
    grid is sparse. The merge must report how many readings the legacy join would
    have discarded, and must itself keep the information by averaging to the hour.
    """
    hours = pd.date_range("2023-01-01 00:00", periods=4, freq="1h")
    load_times = pd.DatetimeIndex(
        [
            "2023-01-01 00:10",
            "2023-01-01 00:40",
            "2023-01-01 01:00",
            "2023-01-01 02:00",
            "2023-01-01 03:10",
            "2023-01-01 03:20",
        ]
    )
    load = pd.DataFrame({schema.TIME: load_times, schema.TARGET: np.arange(6.0)})
    weather = pd.DataFrame(
        {
            schema.TIME: hours,
            **{c: np.linspace(1.0, 4.0, len(hours)) for c in schema.PREDICTORS},
        }
    )

    legacy = pd.merge_asof(
        load.sort_values(schema.TIME),
        weather.sort_values(schema.TIME),
        on=schema.TIME,
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )
    # Record the legacy behaviour rather than hard-coding it: the fixture's job is
    # to prove the diagnostic equals what the old join actually discarded.
    expected_dropped = int(legacy["u10"].isna().sum())

    merged, report = cleaning.merge_load_weather(load, weather, interpolate_weather=False)
    assert report.load_rows_dropped_by_tolerance == expected_dropped
    assert report.join_strategy == "resample+exact"
    assert len(merged) == len(hours)
    # Every weather hour is represented, and no timestamp is duplicated.
    assert list(merged[schema.TIME]) == list(hours)
    assert merged[schema.TARGET].notna().all()


def test_hourly_load_uses_the_asof_join():
    hours = pd.date_range("2023-01-01", periods=6, freq="1h")
    load = pd.DataFrame({schema.TIME: hours, schema.TARGET: np.arange(6.0)})
    weather = pd.DataFrame(
        {schema.TIME: hours, **{c: np.linspace(1, 6, 6) for c in schema.PREDICTORS}}
    )
    merged, report = cleaning.merge_load_weather(load, weather, interpolate_weather=False)
    assert report.join_strategy == "asof"
    assert report.load_rows_dropped_by_tolerance == 0
    assert len(merged) == 6


def test_merge_never_produces_duplicate_timestamps():
    """Duplicates would let one instant enter two splits."""
    hours = pd.date_range("2023-01-01", periods=5, freq="1h")
    # Load rows jittered around the hour, all within tolerance.
    jittered = hours + pd.to_timedelta([0, 5, 10, -5, 0], unit="min")
    load = pd.DataFrame({schema.TIME: jittered, schema.TARGET: np.arange(5.0)})
    weather = pd.DataFrame(
        {schema.TIME: hours, **{c: np.arange(5.0) for c in schema.PREDICTORS}}
    )

    merged, report = cleaning.merge_load_weather(load, weather, interpolate_weather=False)
    assert not merged[schema.TIME].duplicated().any()
    assert merged[schema.TIME].is_monotonic_increasing
    assert report.dropped_duplicate_timestamps >= 0


def test_merge_interpolates_interior_weather_gaps_only():
    hours = pd.date_range("2023-01-01", periods=6, freq="1h")
    load = pd.DataFrame({schema.TIME: hours, schema.TARGET: np.arange(6.0)})
    weather = pd.DataFrame({schema.TIME: hours, **{c: np.arange(6.0) for c in schema.PREDICTORS}})
    weather.loc[3, "u10"] = np.nan  # type: ignore[call-overload]

    merged, _ = cleaning.merge_load_weather(load, weather, interpolate_weather=True)
    assert merged["u10"].isna().sum() == 0
    assert merged["u10"].iloc[3] == pytest.approx(3.0)


def test_clean_time_index_rejects_unparseable():
    """A garbage timestamp must abort the build, not become NaT and shift windows."""
    frame = pd.DataFrame({schema.TIME: ["99999999", "2023-01-01"], schema.TARGET: [1.0, 2.0]})
    with pytest.raises(schema.SchemaError, match="unparseable timestamp"):
        cleaning._clean_time_index(frame, label="test")


def test_clip_physical_ranges_reports_and_clips():
    frame = synthetic_frame(n_hours=10)
    frame.loc[0, "tcc"] = 5.0
    frame.loc[1, "msl"] = 10.0
    clipped, counts = cleaning.clip_physical_ranges(frame)
    assert counts["tcc"] == 1 and counts["msl"] == 1
    assert clipped["tcc"].max() <= 1.0
    assert clipped["msl"].min() >= 85_000.0


def test_restrict_period_is_half_open():
    frame = synthetic_frame(n_hours=100)
    start, end = frame[schema.TIME].iloc[10], frame[schema.TIME].iloc[20]
    sliced = cleaning.restrict_period(frame, str(start), str(end))
    assert len(sliced) == 10
    assert sliced[schema.TIME].iloc[0] == start
    assert sliced[schema.TIME].iloc[-1] < end


def test_read_csv_any_handles_both_separators(tmp_path):
    frame = synthetic_frame(n_hours=5)
    for sep, name in ((',', "comma.csv"), (";", "semi.csv")):
        path = tmp_path / name
        frame.to_csv(path, index=False, sep=sep)
        back = cleaning.read_csv_any(path)
        assert list(back.columns) == list(frame.columns)
        assert len(back) == len(frame)


def test_missing_file_message_points_at_licensing(tmp_path):
    with pytest.raises(FileNotFoundError, match="not version-controlled"):
        cleaning.read_csv_any(tmp_path / "absent.csv")
