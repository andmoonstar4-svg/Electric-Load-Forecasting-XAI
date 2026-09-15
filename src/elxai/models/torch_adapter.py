"""Adapters that drive the vendored upstream models through the Forecaster API.

Design notes
------------
*The upstream config object.* Upstream models read a large, loosely-defined set of
attributes off ``configs`` (``d_model``, ``n_heads``, ``moving_avg``, ``factor``,
...). The original project satisfied them by passing an argparse ``Namespace``.
Here the namespace is built explicitly and then **cross-checked against the
model's own source** — :func:`required_config_attrs` parses ``configs.<name>``
occurrences out of the upstream file, so a missing default surfaces as a clear
error at build time instead of an ``AttributeError`` 40 minutes into training.

*Internal normalisation.* iTransformer and DLinear standardise each input window
inside ``forward``. That stacks on top of the dataset-level scaler. It is
recorded on the class as :attr:`uses_internal_normalisation` because it changes
what a SHAP value means: perturbing a feature also changes the window statistics
used by the instance normaliser.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from elxai.models.base import FitReport, Forecaster
from elxai.third_party import import_model, upstream_path

log = logging.getLogger(__name__)

_MODEL_METADATA: Mapping[str, dict[str, Any]] = {
    "iTransformer": {
        "uses_internal_normalisation": True,
        "defaults": {
            "d_model": 32,
            "d_ff": 32,
            "e_layers": 2,
            "n_heads": 8,
            "dropout": 0.1,
            "factor": 1,
            "activation": "gelu",
            "moving_avg": 25,
            "label_len": 48,
            "distil": True,
            "output_attention": False,
            "use_amp": False,
            "num_class": 1,
            "top_k": 5,
            "expand": 2,
            "d_conv": 4,
            "channel_independence": 0,
        },
    },
    "DLinear": {
        "uses_internal_normalisation": True,
        "defaults": {
            "moving_avg": 25,
            "label_len": 48,
            "d_model": 32,
            "d_ff": 32,
            "e_layers": 1,
            "n_heads": 8,
            "dropout": 0.1,
            "factor": 1,
            "activation": "gelu",
            "output_attention": False,
            "use_amp": False,
            "num_class": 1,
            "expand": 2,
            "d_conv": 4,
            "channel_independence": 0,
        },
    },
    "Autoformer": {
        "uses_internal_normalisation": False,
        "defaults": {
            "d_model": 32,
            "d_ff": 32,
            "e_layers": 2,
            "d_layers": 1,
            "n_heads": 8,
            "dropout": 0.1,
            "factor": 1,
            "moving_avg": 25,
            "label_len": 48,
            "activation": "gelu",
            "output_attention": False,
            "distil": True,
            "use_amp": False,
            "num_class": 1,
            "top_k": 5,
            "expand": 2,
            "d_conv": 4,
            "channel_independence": 0,
        },
    },
    "Transformer": {
        "uses_internal_normalisation": False,
        "defaults": {
            "d_model": 32,
            "d_ff": 32,
            "e_layers": 2,
            "d_layers": 1,
            "n_heads": 8,
            "dropout": 0.1,
            "factor": 1,
            "moving_avg": 25,
            "label_len": 48,
            "activation": "gelu",
            "output_attention": False,
            "distil": True,
            "use_amp": False,
            "num_class": 1,
            "expand": 2,
            "d_conv": 4,
            "channel_independence": 0,
        },
    },
}

#: Attributes every adapter supplies regardless of architecture.
_COMMON = {
    "task_name": "long_term_forecast",
    "freq": "h",
    "embed": "timeF",
    "features": "MS",
    "target": "Total Load",
    "data": "custom",
    "root_path": "./data",
    "data_path": "elxai.csv",
    "checkpoints": "./checkpoints",
    "use_gpu": False,
    "gpu": 0,
    "use_multi_gpu": False,
    "devices": "0",
    "gpu_type": "cuda",
    "use_dtw": 0,
    "augmentation_ratio": 0,
    "learning_rate": 1e-4,
    "train_epochs": 10,
    "batch_size": 32,
    "patience": 3,
    "loss": "MSE",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "jepa_loss": "mse",
    "mask_rate": 0.25,
    "top_k": 5,
}

_CONFIG_RE = re.compile(r"configs\.([A-Za-z_][A-Za-z0-9_]*)")


class _Namespace:
    """Attribute bag handed to the upstream model constructor."""

    def __init__(self, mapping: Mapping[str, Any]):
        for key, value in mapping.items():
            setattr(self, key, value)

    def __repr__(self) -> str:  # pragma: no cover
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"Namespace({items})"

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def required_config_attrs(model_name: str) -> set[str]:
    """Parse ``configs.<attr>`` accesses out of the upstream model source.

    This is the check that turns a missing default into an immediate, named error.
    """
    path = _model_source_path(model_name)
    text = path.read_text(encoding="utf-8", errors="replace")
    return set(_CONFIG_RE.findall(text))


def _model_source_path(model_name: str) -> Path:
    root = upstream_path()
    path = root / "models" / f"{model_name}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"upstream model file {path} not found; is {model_name!r} vendored?"
        )
    return path


@dataclass
class TrainConfig:
    """Training hyper-parameters, decoupled from argparse."""

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    patience: int = 3
    seed: int = 2021
    device: str = "auto"
    num_workers: int = 0
    amp: bool = False
    loss: str = "mse"
    loss_kwargs: dict[str, Any] | None = None
    grad_clip: float | None = None
    verbose: bool = True


class TorchForecaster(Forecaster):
    """Wrap one upstream ``models.<name>.Model`` as a :class:`Forecaster`."""

    def __init__(
        self,
        model_name: str,
        *,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        model_params: Mapping[str, Any] | None = None,
        train: TrainConfig | None = None,
    ):
        if model_name not in _MODEL_METADATA:
            raise KeyError(
                f"no adapter metadata for {model_name!r}; "
                f"known: {sorted(_MODEL_METADATA)}"
            )
        self.model_name = model_name
        self.name = model_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.model_params = dict(model_params or {})
        self.train_cfg = train or TrainConfig()
        self.uses_internal_normalisation = bool(
            _MODEL_METADATA[model_name]["uses_internal_normalisation"]
        )

        self._nn_module = None
        self.device = self._resolve_device(self.train_cfg.device)
        self._n_parameters: int | None = None

    # -- setup ------------------------------------------------------------- #

    def _resolve_device(self, spec: str) -> str:
        if spec and spec != "auto":
            return spec
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:  # pragma: no cover
            pass
        return "cpu"

    def build_namespace(self) -> _Namespace:
        mapping: dict[str, Any] = dict(_COMMON)
        mapping.update(_MODEL_METADATA[self.model_name]["defaults"])
        mapping.update(
            {
                "seq_len": self.seq_len,
                "pred_len": self.pred_len,
                "enc_in": self.n_channels,
                "dec_in": self.n_channels,
                "c_out": self.n_channels,
                "learning_rate": self.train_cfg.learning_rate,
                "train_epochs": self.train_cfg.epochs,
                "batch_size": self.train_cfg.batch_size,
                "patience": self.train_cfg.patience,
                "loss": self.train_cfg.loss.upper(),
            }
        )
        mapping.update(self.model_params)

        needed = required_config_attrs(self.model_name)
        missing = sorted(needed - set(mapping))
        if missing:
            raise KeyError(
                f"{self.model_name} reads configs.{missing} which the adapter does not "
                "supply. Add them to _MODEL_METADATA or to model.params in the YAML."
            )
        unused = sorted(set(self.model_params) - needed)
        if unused:
            log.debug(
                "%s: model.params keys not read from configs by the model: %s",
                self.model_name,
                unused,
            )
        return _Namespace(mapping)

    def _build(self):
        Model = import_model(self.model_name)
        module = Model(self.build_namespace())
        self._n_parameters = sum(p.numel() for p in module.parameters())
        return module.to(self.device).float()

    # -- training ---------------------------------------------------------- #

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitReport:
        import torch

        from elxai.training.losses import build_loss

        torch.manual_seed(self.train_cfg.seed)
        np.random.seed(self.train_cfg.seed)

        self._nn_module = self._build()
        optim = torch.optim.Adam(
            self._nn_module.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        criterion = build_loss(self.train_cfg.loss, **(self.train_cfg.loss_kwargs or {}))

        x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        if x_val is not None and y_val is not None and len(x_val):
            x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=self.device)
            y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
        else:
            x_val_t = y_val_t = None

        target_index = x_train.shape[-1] - 1  # schema.TARGET_INDEX
        history: list[dict[str, float]] = []
        best_val = float("inf")
        best_epoch: int | None = None
        best_state: dict[str, Any] | None = None
        stale = 0
        rng = np.random.default_rng(self.train_cfg.seed)

        for epoch in range(1, self.train_cfg.epochs + 1):
            self._nn_module.train()
            order = rng.permutation(len(x_train_t))
            epoch_losses: list[float] = []
            n_batches = 0
            for lo in range(0, len(order), self.train_cfg.batch_size):
                idx = order[lo : lo + self.train_cfg.batch_size]
                xb = x_train_t[idx]
                yb = y_train_t[idx]
                optim.zero_grad(set_to_none=True)
                out = self._forward(xb)
                pred = out[:, -self.pred_len :, target_index]
                loss = criterion(pred, yb)
                loss.backward()
                if self.train_cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self._nn_module.parameters(), self.train_cfg.grad_clip
                    )
                optim.step()
                epoch_losses.append(float(loss.detach().cpu()))
                n_batches += 1

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            val_loss = (
                self._evaluate(x_val_t, y_val_t, criterion) if x_val_t is not None else train_loss
            )
            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": float(val_loss)}
            )
            if self.train_cfg.verbose:
                log.info(
                    "epoch %d/%d | train %.6f | val %.6f",
                    epoch,
                    self.train_cfg.epochs,
                    train_loss,
                    val_loss,
                )

            if val_loss < best_val - 1e-9:
                best_val = float(val_loss)
                best_epoch = epoch
                best_state = {
                    k: v.detach().clone() for k, v in self._nn_module.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= self.train_cfg.patience:
                    if self.train_cfg.verbose:
                        log.info("early stopping at epoch %d (patience %d)", epoch, self.train_cfg.patience)
                    break

        if best_state is not None:
            self._nn_module.load_state_dict(best_state)

        return FitReport(
            n_train=len(x_train),
            n_val=0 if x_val is None else len(x_val),
            best_epoch=best_epoch,
            epochs_run=len(history),
            best_val_loss=best_val if best_epoch is not None else None,
            history=history,
            n_parameters=self._n_parameters,
            device=self.device,
            notes={
                "loss": self.train_cfg.loss,
                "loss_kwargs": self.train_cfg.loss_kwargs or {},
                "internal_normalisation": self.uses_internal_normalisation,
            },
        )

    def _forward(self, x: "Any"):
        """Call the upstream model with the decoder inputs it expects.

        The decoder receives zero-filled future values (the upstream convention
        for long-term forecasting); Autoformer additionally consumes
        ``label_len`` known steps, which is why the prefix is taken from the
        encoder window rather than being zeros.
        """
        import torch

        batch = x.shape[0]
        label_len = int(getattr(self.build_namespace(), "label_len", 48))
        label_len = min(label_len, self.seq_len)
        zeros = torch.zeros(
            batch, self.pred_len, x.shape[-1], dtype=x.dtype, device=x.device
        )
        dec_inp = torch.cat([x[:, -label_len:, :], zeros], dim=1)
        return self._nn_module(x, None, dec_inp, None)

    def _evaluate(self, x_val, y_val, criterion) -> float:
        import torch

        self._nn_module.eval()
        losses: list[float] = []
        target_index = self.n_channels - 1
        with torch.no_grad():
            for lo in range(0, len(x_val), self.train_cfg.batch_size):
                xb = x_val[lo : lo + self.train_cfg.batch_size]
                yb = y_val[lo : lo + self.train_cfg.batch_size]
                out = self._forward(xb)
                pred = out[:, -self.pred_len :, target_index]
                losses.append(float(criterion(pred, yb).detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    # -- inference --------------------------------------------------------- #

    def predict_all_channels(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self._nn_module is None:
            raise RuntimeError("call fit() before predict()")
        self._nn_module.eval()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for lo in range(0, len(x), max(self.train_cfg.batch_size, 1)):
                xb = torch.as_tensor(
                    x[lo : lo + self.train_cfg.batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                out = self._forward(xb)
                outs.append(out[:, -self.pred_len :, :].detach().cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.pred_len, self.n_channels))

    def predict(self, x: np.ndarray) -> np.ndarray:
        target_index = x.shape[-1] - 1
        return self.predict_all_channels(x)[:, :, target_index]

    # -- persistence ------------------------------------------------------- #

    def save(self, directory: str | Path) -> Path:
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._nn_module is not None:
            torch.save(self._nn_module.state_dict(), directory / "model.pt")
        return (directory / "model.json").write_text(
            __import__("json").dumps(
                {
                    "name": self.model_name,
                    "seq_len": self.seq_len,
                    "pred_len": self.pred_len,
                    "n_channels": self.n_channels,
                    "model_params": self.model_params,
                    "namespace": self.build_namespace().as_dict(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )


#: Registry used by the experiment runner.
ADAPTERS: dict[str, type[Forecaster]] = {"torch": TorchForecaster}
