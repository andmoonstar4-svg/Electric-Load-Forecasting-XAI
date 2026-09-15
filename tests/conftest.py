"""Shared fixtures: synthetic frames that respect the data contract.

Tests deliberately do not read any real dataset. The Belgian source is not
redistributable (see :mod:`elxai.data.provenance`), so a test suite that depended
on it could not be run by anyone else — which is exactly the failure mode this
refactor is meant to remove.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elxai.data import schema

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root, so config tests can locate the shipped YAML files."""
    return REPO_ROOT


def synthetic_frame(n_hours: int = 24 * 220, seed: int = 0, extreme: bool = False) -> pd.DataFrame:
    """A contract-valid hourly frame with realistic seasonal structure."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-01-01", periods=n_hours, freq="1h")
    hour = np.arange(n_hours)
    daily = np.sin(2 * np.pi * hour / 24)
    weekly = np.sin(2 * np.pi * hour / (24 * 7))

    frame = pd.DataFrame(
        {
            schema.TIME: t,
            "u10": np.clip(rng.gamma(2.0, 2.0, n_hours), 0, None),
            "t2m": 10.0 + 8.0 * np.sin(2 * np.pi * hour / (24 * 365)) + rng.normal(0, 2, n_hours),
            "msl": 101_000 + rng.normal(0, 500, n_hours),
            "ssrd": np.clip(5e5 + 3e5 * daily + rng.normal(0, 5e4, n_hours), 0, None),
            "tcc": np.clip(rng.random(n_hours), 0, 1),
            "tp": np.clip(rng.exponential(0.001, n_hours), 0, None),
            schema.TARGET: 10_000 + 1_500 * daily + 400 * weekly + rng.normal(0, 200, n_hours),
        }
    )
    if extreme:
        frame.loc[100:112, "t2m"] = -6.0
        frame.loc[500:505, "u10"] = 20.0
        frame.loc[900:903, "tp"] = 0.02
    return frame


@pytest.fixture
def frame() -> pd.DataFrame:
    return synthetic_frame()


@pytest.fixture
def extreme_frame() -> pd.DataFrame:
    return synthetic_frame(extreme=True)


@pytest.fixture
def long_frame() -> pd.DataFrame:
    """Long enough that purging two 191-hour boundaries still leaves a test split."""
    return synthetic_frame(n_hours=24 * 400)


@pytest.fixture
def windows(long_frame):
    from elxai.config import ScalingConfig, SplitConfig
    from elxai.data.schema import TaskSpec
    from elxai.data.windows import build_dataset_splits

    splits, _ = build_dataset_splits(
        long_frame,
        TaskSpec(seq_len=168, pred_len=24),
        SplitConfig(),
        ScalingConfig(),
    )
    return splits
