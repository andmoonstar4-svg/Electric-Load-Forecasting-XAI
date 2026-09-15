"""Adapters: upstream import hygiene, config contracts, and end-to-end learning."""

from __future__ import annotations

import numpy as np
import pytest

from elxai.data import schema
from elxai.models import NEURAL_MODELS, build_forecaster, canonical_name

torch = pytest.importorskip("torch")

from elxai.models.torch_adapter import (  # noqa: E402
    TorchForecaster,
    TrainConfig,
    required_config_attrs,
)
from elxai.third_party import UpstreamMissing, import_model, register  # noqa: E402


# --------------------------------------------------------------------------- #
# A tiny, fast configuration
# --------------------------------------------------------------------------- #

SEQ, HOR, CH = 24, 6, 7
PARAMS = {"d_model": 16, "d_ff": 16, "e_layers": 1, "n_heads": 2, "label_len": 12, "moving_avg": 5}


def make_arrays(n: int = 96, seed: int = 0, signal: bool = True):
    """Windows whose target is a deterministic function of the input.

    With a learnable signal a model that trains correctly must beat the
    predict-the-mean baseline, which is what makes the end-to-end assertions
    meaningful rather than merely shape checks.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, SEQ, CH)).astype(np.float32)
    if signal:
        w = rng.normal(size=(SEQ,))
        y = np.stack([x[:, :, schema.TARGET_INDEX] @ w for _ in range(HOR)], axis=1)
        y = y + 0.05 * rng.normal(size=(n, HOR))
    else:
        y = rng.normal(size=(n, HOR))
    return x.astype(np.float32), y.astype(np.float32)


# --------------------------------------------------------------------------- #
# Upstream plumbing
# --------------------------------------------------------------------------- #


def test_register_puts_upstream_on_sys_path():
    import sys

    path = register()
    assert str(path) in sys.path
    assert (path / "models" / "iTransformer.py").exists()


def test_import_model_returns_a_class():
    Model = import_model("iTransformer")
    assert isinstance(Model, type)
    assert Model.__name__ == "Model"


def test_upstream_package_does_not_shadow_elxai_models():
    """`import models.iTransformer` must not resolve to elxai.models."""
    import models  # noqa: F401  (the upstream package)

    import elxai.models as own

    assert own.__name__ == "elxai.models"
    assert hasattr(own, "build_forecaster")


def test_required_config_attrs_is_derived_from_upstream_source():
    """The check that turns a missing default into a named error."""
    needed = required_config_attrs("iTransformer")
    assert {"d_model", "n_heads", "seq_len", "pred_len", "enc_in"} <= needed

    dlinear = required_config_attrs("DLinear")
    assert "moving_avg" in dlinear
    assert "n_heads" not in dlinear  # DLinear has no attention


def test_missing_upstream_raises_with_remedy(monkeypatch):
    """Both failure branches must name the fix, not just fail."""
    import elxai.third_party as tp

    monkeypatch.delenv("ELXAI_THIRD_PARTY", raising=False)
    monkeypatch.setattr(tp, "upstream_path", lambda name="x": tp.REPO_ROOT / "third_party" / "nope")
    with pytest.raises(UpstreamMissing, match="fetch_third_party"):
        tp.register()

    # Branch two: the environment override points somewhere that does not exist.
    monkeypatch.setenv("ELXAI_THIRD_PARTY", str(tp.REPO_ROOT / "third_party" / "also-nope"))
    with pytest.raises(UpstreamMissing) as excinfo:
        tp.register()
    message = str(excinfo.value)
    assert "ELXAI_THIRD_PARTY" in message
    assert "also-nope" in message


def test_namespace_contains_every_attribute_the_model_reads():
    adapter = TorchForecaster(
        "iTransformer", seq_len=SEQ, pred_len=HOR, n_channels=CH, model_params=PARAMS
    )
    ns = adapter.build_namespace()
    missing = required_config_attrs("iTransformer") - set(ns.as_dict())
    assert not missing


def test_model_params_are_honoured():
    adapter = TorchForecaster(
        "iTransformer",
        seq_len=SEQ,
        pred_len=HOR,
        n_channels=CH,
        model_params={**PARAMS, "d_model": 64},
    )
    assert adapter.build_namespace().d_model == 64
    assert adapter.build_namespace().enc_in == CH


def test_unknown_model_name_is_rejected():
    with pytest.raises(KeyError, match="no adapter metadata"):
        TorchForecaster("NotAModel", seq_len=SEQ, pred_len=HOR, n_channels=CH)


def test_internal_normalisation_flag_matches_the_upstream_code():
    """iTransformer/DLinear standardise each window inside forward."""
    it = TorchForecaster("iTransformer", seq_len=SEQ, pred_len=HOR, n_channels=CH)
    dl = TorchForecaster("DLinear", seq_len=SEQ, pred_len=HOR, n_channels=CH)
    af = TorchForecaster("Autoformer", seq_len=SEQ, pred_len=HOR, n_channels=CH, model_params=PARAMS)
    assert it.uses_internal_normalisation is True
    assert dl.uses_internal_normalisation is True
    assert af.uses_internal_normalisation is False


# --------------------------------------------------------------------------- #
# End-to-end: shapes and learning
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["iTransformer", "DLinear"])
def test_neural_adapter_shapes_and_beats_mean_baseline(name):
    x, y = make_arrays(n=128)
    adapter = TorchForecaster(
        name,
        seq_len=SEQ,
        pred_len=HOR,
        n_channels=CH,
        model_params=PARAMS,
        train=TrainConfig(epochs=6, batch_size=32, patience=6, seed=0, verbose=False),
    )
    report = adapter.fit(x[:96], y[:96], x_val=x[96:], y_val=y[96:])

    pred = adapter.predict(x[96:])
    assert pred.shape == (32, HOR)
    assert np.isfinite(pred).all()
    assert report.best_epoch is not None and report.n_parameters > 0

    # A model that learned the signal beats predicting the training mean.
    mean_baseline = float(np.mean((y[96:] - y[:96].mean()) ** 2))
    assert float(np.mean((pred - y[96:]) ** 2)) < mean_baseline


def test_predict_all_channels_shape_and_target_placement():
    x, y = make_arrays(n=64)
    adapter = TorchForecaster(
        "DLinear",
        seq_len=SEQ,
        pred_len=HOR,
        n_channels=CH,
        model_params=PARAMS,
        train=TrainConfig(epochs=1, batch_size=32, verbose=False),
    )
    adapter.fit(x, y)
    all_channels = adapter.predict_all_channels(x[:5])
    assert all_channels.shape == (5, HOR, CH)
    np.testing.assert_allclose(all_channels[:, :, schema.TARGET_INDEX], adapter.predict(x[:5]), rtol=1e-5)


def test_predict_before_fit_raises():
    adapter = TorchForecaster("DLinear", seq_len=SEQ, pred_len=HOR, n_channels=CH, model_params=PARAMS)
    with pytest.raises(RuntimeError, match="call fit"):
        adapter.predict(np.zeros((2, SEQ, CH), dtype=np.float32))


def test_early_stopping_can_end_before_the_epoch_limit():
    x, y = make_arrays(n=64)
    adapter = TorchForecaster(
        "DLinear",
        seq_len=SEQ,
        pred_len=HOR,
        n_channels=CH,
        model_params=PARAMS,
        train=TrainConfig(epochs=50, batch_size=32, patience=1, seed=1, verbose=False),
    )
    report = adapter.fit(x[:48], y[:48], x_val=x[48:], y_val=y[48:])
    assert report.epochs_run <= 50
    assert len(report.history) == report.epochs_run


def test_save_writes_a_reloadable_record(tmp_path):
    x, y = make_arrays(n=32)
    adapter = TorchForecaster(
        "DLinear",
        seq_len=SEQ,
        pred_len=HOR,
        n_channels=CH,
        model_params=PARAMS,
        train=TrainConfig(epochs=1, batch_size=32, verbose=False),
    )
    adapter.fit(x, y)
    adapter.save(tmp_path)
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "model.json").exists()


def test_exponential_loss_trains_without_error():
    x, y = make_arrays(n=64)
    adapter = TorchForecaster(
        "DLinear",
        seq_len=SEQ,
        pred_len=HOR,
        n_channels=CH,
        model_params=PARAMS,
        train=TrainConfig(
            epochs=3,
            batch_size=32,
            loss="exp",
            loss_kwargs={"beta": 2.0, "alpha": 4.0},
            verbose=False,
        ),
    )
    report = adapter.fit(x[:48], y[:48], x_val=x[48:], y_val=y[48:])
    assert np.isfinite(report.best_val_loss)


# --------------------------------------------------------------------------- #
# Random Forest baseline (no torch needed, but shares the interface)
# --------------------------------------------------------------------------- #


def test_forest_alias_and_canonical_name():
    assert canonical_name("random_forest") == "RandomForest"
    assert canonical_name("rf") == "RandomForest"
    assert canonical_name("iTransformer") == "iTransformer"
    assert set(NEURAL_MODELS) == {"iTransformer", "DLinear", "Autoformer", "Transformer"}


def test_forest_shapes_and_importances():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, SEQ, CH)).astype(np.float32)
    y = np.stack([x[:, :, schema.TARGET_INDEX].mean(axis=1)] * HOR, axis=1).astype(np.float32)

    forecaster = build_forecaster(
        "RandomForest",
        seq_len=SEQ,
        pred_len=HOR,
        n_channels=CH,
        params={"n_estimators": 20, "n_jobs": 1, "random_state": 0},
    )
    report = forecaster.fit(x[:100], y[:100], x_val=x[100:], y_val=y[100:])
    assert report.n_parameters is None

    pred = forecaster.predict(x[100:])
    assert pred.shape == (20, HOR)
    assert np.isfinite(pred).all()

    flat = forecaster.feature_importance()
    assert flat is not None and flat.shape == (SEQ * CH,)
    assert flat.sum() == pytest.approx(1.0)

    per_channel = forecaster.channel_importance()
    assert per_channel is not None and per_channel.shape == (CH,)
    assert per_channel.sum() == pytest.approx(1.0)
    # The target-lag channel carries the signal, so it must dominate.
    assert int(np.argmax(per_channel)) == schema.TARGET_INDEX


def test_forest_does_not_use_internal_normalisation():
    forecaster = build_forecaster(
        "RandomForest", seq_len=SEQ, pred_len=HOR, n_channels=CH, params={"n_estimators": 5}
    )
    assert forecaster.uses_internal_normalisation is False


def test_unknown_model_name_gives_actionable_error():
    with pytest.raises(KeyError, match="unknown model"):
        build_forecaster("LSTM", seq_len=SEQ, pred_len=HOR, n_channels=CH)


# --------------------------------------------------------------------------- #
# Param filtering across composed configs
# --------------------------------------------------------------------------- #


def test_foreign_params_are_dropped_for_the_forest():
    """Composing an iTransformer config with a forest config leaves d_model behind.

    The forest would receive it as a sklearn kwarg and fail; the filter drops it.
    """
    from elxai.models import filter_model_params

    params = {"d_model": 32, "n_heads": 8, "moving_avg": 25, "n_estimators": 20}
    kept = filter_model_params("RandomForest", params)
    assert kept == {"n_estimators": 20}

    forecaster = build_forecaster(
        "RandomForest", seq_len=SEQ, pred_len=HOR, n_channels=CH, params=params
    )
    assert forecaster.params["n_estimators"] == 20


def test_foreign_params_are_dropped_for_the_neural_adapter():
    from elxai.models import filter_model_params

    params = {"d_model": 32, "n_heads": 8, "n_estimators": 300, "oob_score": True}
    kept = filter_model_params("iTransformer", params)
    assert "n_estimators" not in kept and "oob_score" not in kept
    assert kept["d_model"] == 32

    adapter = build_forecaster(
        "iTransformer", seq_len=SEQ, pred_len=HOR, n_channels=CH, params=params
    )
    assert adapter.build_namespace().d_model == 32


def test_param_keys_differ_per_model():
    from elxai.models import model_param_keys

    forest = model_param_keys("RandomForest")
    itransformer = model_param_keys("iTransformer")
    assert "n_estimators" in forest and "n_estimators" not in itransformer
    assert "d_model" in itransformer and "d_model" not in forest
    assert "moving_avg" in model_param_keys("DLinear")
    assert "n_heads" not in model_param_keys("DLinear")


def test_every_shipped_compare_config_supplies_only_valid_params(repo_root):
    """A model config must not carry params its own model cannot read."""
    import glob
    from pathlib import Path as _Path

    from elxai.config import load_composed
    from elxai.models import model_param_keys

    root = _Path(repo_root)
    for path in sorted(glob.glob(str(root / "configs" / "model" / "compare_*.yaml"))):
        cfg = load_composed([root / "configs" / "data" / "belgium.yaml", path])
        allowed = model_param_keys(cfg.model.name)
        invalid = sorted(set(cfg.model.params) - allowed)
        assert not invalid, f"{_Path(path).name}: {cfg.model.name} cannot read {invalid}"
