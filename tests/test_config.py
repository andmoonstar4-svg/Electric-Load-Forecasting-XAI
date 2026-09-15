"""Configuration: validation, composition, and run identity."""

from __future__ import annotations

import os

import pytest
import yaml

from elxai.config import (
    Config,
    DataConfig,
    ExtremeConfig,
    RunConfig,
    SplitConfig,
    TrainingConfig,
    deep_merge,
    load_composed,
    read_yaml,
)

REPO_CONFIGS = "configs"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_scientific_notation_is_parsed_as_float(tmp_path):
    """PyYAML's default resolver reads 1e-4 as a string; the loader must not."""
    path = tmp_path / "lr.yaml"
    path.write_text("training:\n  learning_rate: 1.0e-4\n", encoding="utf-8")
    raw = read_yaml(path)
    assert isinstance(raw["training"]["learning_rate"], float)
    cfg = Config.from_dict(raw)
    assert cfg.training.learning_rate == pytest.approx(1e-4)


def test_environment_variables_are_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("ELXAI_TEST_DIR", str(tmp_path))
    path = tmp_path / "env.yaml"
    path.write_text('data:\n  load_csv: "${ELXAI_TEST_DIR}/load.csv"\n', encoding="utf-8")
    cfg = Config.from_dict(read_yaml(path))
    from pathlib import Path as _Path

    assert _Path(cfg.data.load_csv) == _Path(tmp_path) / "load.csv"


def test_unset_variables_are_left_verbatim(tmp_path, monkeypatch):
    monkeypatch.delenv("ELXAI_DEFINITELY_UNSET", raising=False)
    path = tmp_path / "env.yaml"
    path.write_text('data:\n  load_csv: "${ELXAI_DEFINITELY_UNSET}/x.csv"\n', encoding="utf-8")
    cfg = Config.from_dict(read_yaml(path))
    assert cfg.data.load_csv.startswith("${ELXAI_DEFINITELY_UNSET}")


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("training:\n  learning_rte: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        Config.from_dict(read_yaml(path))


def test_nested_dataclass_is_built_from_mapping(tmp_path):
    path = tmp_path / "nested.yaml"
    path.write_text(
        "data:\n  name: belgium\n  extreme:\n    enabled: true\n    context_days: 3\n",
        encoding="utf-8",
    )
    cfg = Config.from_dict(read_yaml(path))
    assert isinstance(cfg.data.extreme, ExtremeConfig)
    assert cfg.data.extreme.context_days == 3


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfig(train=0.8, val=0.1, test=0.05)


def test_split_rejects_negative_purge():
    with pytest.raises(ValueError, match="purge must be >= 0"):
        SplitConfig(purge=-1)


def test_seq_len_must_be_day_aligned():
    with pytest.raises(ValueError, match="whole number of days"):
        Config.from_dict({"data": {"seq_len": 100}}).validate()


def test_exp_loss_requires_a_valid_base():
    with pytest.raises(ValueError, match="base must be > 1"):
        Config.from_dict({"training": {"loss": "exp", "exp_loss_beta": 1.0}}).validate()


def test_tree_xai_only_applies_to_the_forest():
    with pytest.raises(ValueError, match="tree"):
        Config.from_dict(
            {"xai": {"method": "tree"}, "model": {"name": "iTransformer"}}
        ).validate()


def test_repeats_must_be_positive():
    with pytest.raises(ValueError, match="repeats"):
        Config.from_dict({"training": {"repeats": 0}}).validate()


# --------------------------------------------------------------------------- #
# Composition and overrides
# --------------------------------------------------------------------------- #


def test_deep_merge_keeps_untouched_keys():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = deep_merge(base, {"a": {"y": 9}})
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3}
    assert base["a"]["y"] == 2  # the base is not mutated


def test_dotted_override_applies(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text("training:\n  loss: mse\n  epochs: 5\n", encoding="utf-8")
    cfg = Config.from_yaml(path, overrides=["training.loss=exp", "training.epochs=2"])
    assert cfg.training.loss == "exp"
    assert cfg.training.epochs == 2


def test_composition_order_matters(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("training:\n  batch_size: 48\n  loss: mse\n", encoding="utf-8")
    b.write_text("training:\n  batch_size: 8\n", encoding="utf-8")
    cfg = load_composed([a, b])
    assert cfg.training.batch_size == 8
    assert cfg.training.loss == "mse"


def test_malformed_override_is_rejected(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text("training:\n  loss: mse\n", encoding="utf-8")
    with pytest.raises(ValueError, match="section.field=value"):
        Config.from_yaml(path, overrides=["training.loss"])


# --------------------------------------------------------------------------- #
# Run identity — the loss-ablation collision
# --------------------------------------------------------------------------- #


def test_fingerprint_changes_with_the_loss():
    """The original `setting` string omitted --loss, so runs overwrote each other."""
    base = Config()
    a = Config.from_dict({**base.as_dict(), "training": {**base.as_dict()["training"], "loss": "mse"}})
    b = Config.from_dict({**base.as_dict(), "training": {**base.as_dict()["training"], "loss": "exp"}})
    assert a.fingerprint() != b.fingerprint()
    assert "mse" in a.name and "exp" in b.name


def test_fingerprint_is_stable_across_instances():
    assert Config().fingerprint() == Config().fingerprint()


def test_fingerprint_changes_with_batch_size_and_seed():
    assert Config.from_dict({"training": {"batch_size": 8}}).fingerprint() != Config.from_dict(
        {"training": {"batch_size": 48}}
    ).fingerprint()
    assert Config.from_dict({"training": {"seed": 1}}).fingerprint() != Config.from_dict(
        {"training": {"seed": 2}}
    ).fingerprint()


def test_name_is_filesystem_safe():
    cfg = Config.from_dict({"experiment": "normal weather!", "model": {"name": "iTransformer"}})
    assert " " not in cfg.name
    assert "/" not in cfg.name


def test_describe_includes_the_fingerprint():
    text = Config().describe()
    assert "fingerprint" in text and "loss" in text


def test_round_trip_through_yaml(tmp_path):
    cfg = Config.from_dict(
        {"experiment": "x", "training": {"loss": "exp", "batch_size": 8, "learning_rate": 1e-4}}
    )
    path = cfg.to_yaml(tmp_path / "cfg.yaml")
    reloaded = Config.from_yaml(path)
    assert reloaded.fingerprint() == cfg.fingerprint()


# --------------------------------------------------------------------------- #
# The shipped configs
# --------------------------------------------------------------------------- #


def test_shipped_experiment_configs_compose(repo_root):
    """Every config in configs/ must load and validate."""
    import glob
    from pathlib import Path

    root = Path(repo_root)
    experiment_dir = root / REPO_CONFIGS / "experiment"
    if not experiment_dir.exists():
        pytest.skip("configs/experiment missing")

    for path in sorted(glob.glob(str(experiment_dir / "*.yaml"))):
        cfg = load_composed([root / REPO_CONFIGS / "data" / "belgium.yaml", path])
        assert cfg.data.seq_len == 168
        assert cfg.data.pred_len == 24
        assert cfg.data.name == "belgium"


def test_shipped_model_configs_all_validate(repo_root):
    import glob
    from pathlib import Path

    root = Path(repo_root)
    for path in sorted(glob.glob(str(root / REPO_CONFIGS / "model" / "*.yaml"))):
        cfg = load_composed([root / REPO_CONFIGS / "data" / "belgium.yaml", path])
        assert cfg.model.name


def test_extreme_config_enables_the_extreme_subset(repo_root):
    from pathlib import Path

    root = Path(repo_root)
    cfg = load_composed(
        [
            root / REPO_CONFIGS / "data" / "belgium.yaml",
            root / REPO_CONFIGS / "experiment" / "extreme.yaml",
        ]
    )
    assert cfg.data.extreme.enabled is True
    assert cfg.data.extreme.context_days == 7
    assert cfg.training.batch_size == 8


def test_tetuan_config_uses_the_shorter_lookback(repo_root):
    from pathlib import Path

    root = Path(repo_root)
    cfg = load_composed([root / REPO_CONFIGS / "data" / "tetuan.yaml"])
    assert cfg.data.seq_len == 72
    assert cfg.data.pred_len == 24
