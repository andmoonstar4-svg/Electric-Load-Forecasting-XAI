"""Strongly-typed experiment configuration.

Every number that the paper reports is traceable to a field here. Two rules
follow from the reproducibility gaps in the original code base:

1. **No argparse defaults scattered through the code.** Configs are YAML, are
   validated on load, and are copied verbatim into the run directory.
2. **The loss function is part of the run identity.** In the original code the
   ``setting`` string used to name output folders omitted ``--loss``, so the
   three rows of the loss ablation table (S6.2) overwrote each other.

Example
-------
>>> cfg = Config.from_yaml("configs/experiment/extreme.yaml")
>>> cfg.training.loss
'exp'
>>> cfg.name
'itransformer_...'
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, get_type_hints

import yaml

LossName = Literal["mse", "mae", "exp"]
ScalingName = Literal["standard", "none", "minmax"]
ExtremeMode = Literal["events_only", "ranges", "flag_only"]

# --------------------------------------------------------------------------- #
# YAML: accept 1e-4 for learning_rate (PyYAML's default resolver does not)
# --------------------------------------------------------------------------- #


class _Loader(yaml.SafeLoader):
    pass


_FLOAT_RE = re.compile(
    r"""^(?:
        [-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN)
    )$""",
    re.X,
)
_Loader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_RE, list("-+0123456789."))


def _expand(value: Any) -> Any:
    """Recursively expand ``${VAR}``, ``$VAR`` and ``~`` inside strings."""
    if isinstance(value, str):
        return os.path.expanduser(_VAR_RE.sub(_replace_var, value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


#: ``${NAME}`` or ``$NAME``. os.path.expandvars does not handle ``${...}`` on
#: every platform, and the configs use that form.
_VAR_RE = re.compile(r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)")


def _replace_var(match: re.Match[str]) -> str:
    name = match.group("braced") or match.group("bare")
    return os.environ.get(name, match.group(0))


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.load(fh, Loader=_Loader) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return _expand(raw)


# --------------------------------------------------------------------------- #
# Config sections
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """Dataset assembly options (paper S2.1, S4.1)."""

    name: str = "belgium"
    #: Source CSVs. Belgian data lives outside the repository on purpose.
    load_csv: str = "${ELXAI_DATA_DIR}/ods001.csv"
    weather_csv: str = "${ELXAI_DATA_DIR}/reanalysis-era5-single-levels-timeseries.csv"
    #: Optional pre-merged frame; when set it short-circuits join+clean.
    merged_csv: str | None = None
    sources: list[str] = field(default_factory=lambda: ["elia", "era5"])
    #: Restrict the observation period (ISO date strings, inclusive/exclusive).
    start: str | None = None
    end: str | None = None
    #: 168 (dataset 1) or 72 (dataset 2).
    seq_len: int = 168
    pred_len: int = 24
    #: Weather-to-load matching tolerance when load is sub-hourly.
    merge_tolerance: str = "30min"
    extreme: "ExtremeConfig" = field(default_factory=lambda: ExtremeConfig())


@dataclass
class ExtremeConfig:
    """Extreme-weather labelling (paper S4.1).

    The paper is ambiguous on precipitation units (``5??/h``) and states a wind
    threshold of 15 in unnamed units. Both are parameters here with the unit
    recorded, so the choice is explicit and revisable.
    """

    enabled: bool = True
    t2m_min_c: float = 0.0
    t2m_max_c: float = 35.0
    tp_min_m: float = 0.005
    tp_unit: str = "m"
    u10_min_ms: float = 15.0
    #: ``events_only`` keeps each extreme date plus ``context_days`` before it
    #: (the paper's 8689-row subset); ``ranges`` marks but keeps everything.
    mode: ExtremeMode = "events_only"
    context_days: int = 7

    def thresholds(self) -> dict[str, float]:
        return {
            "t2m_max_c": self.t2m_max_c,
            "t2m_min_c": self.t2m_min_c,
            "tp_min_m": self.tp_min_m,
            "u10_min_ms": self.u10_min_ms,
            "context_days": float(self.context_days),
        }


@dataclass
class SplitConfig:
    """Chronological split (paper S2.1: 70/10/20).

    ``purge`` drops this many hours between adjacent splits so that a training
    window can never overlap a validation window.
    """

    train: float = 0.7
    val: float = 0.1
    test: float = 0.2
    purge: int = 0

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total!r}")
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("every split needs a positive fraction")
        if self.purge < 0:
            raise ValueError("purge must be >= 0")


@dataclass
class ScalingConfig:
    """Standardisation bookkeeping.

    The paper reports standardised MSE/MAE, so the scaler must be fitted on the
    training split only. iTransformer/DLinear additionally normalise each window
    internally; that is a model property, recorded by the adapter, not here.
    """

    method: ScalingName = "standard"
    #: Apply scaling to every channel including the target.
    per_column: bool = True


@dataclass
class ModelConfig:
    name: str = "iTransformer"
    #: Free-form keyword arguments handed to the upstream model.
    params: dict[str, Any] = field(default_factory=dict)

    def resolved(self, seq_len: int, pred_len: int, n_channels: int) -> dict[str, Any]:
        """Merge task-derived defaults with explicit overrides."""
        base: dict[str, Any] = {
            "task_name": "long_term_forecast",
            "seq_len": seq_len,
            "pred_len": pred_len,
            "enc_in": n_channels,
            "dec_in": n_channels,
            "c_out": n_channels,
            "freq": "h",
            "embed": "timeF",
        }
        base.update(self.params)
        return base


@dataclass
class TrainingConfig:
    seed: int = 2021
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience: int = 3
    loss: LossName = "mse"
    #: Base of the exponential penalty loss (paper S6.1 uses beta = 2).
    exp_loss_beta: float = 2.0
    exp_loss_alpha: float = 4.0
    weight_decay: float = 0.0
    device: str = "auto"
    num_workers: int = 0
    amp: bool = False
    #: Repeat count; >1 turns the point estimate into mean +- std.
    repeats: int = 1


@dataclass
class XAIConfig:
    enabled: bool = False
    method: Literal["kernel", "tree", "gradient"] = "kernel"
    #: Number of background windows drawn from the *training* split.
    n_background: int = 100
    #: Number of evaluated windows.
    n_eval: int = 50
    #: KernelExplainer coalition samples. ``None`` derives a value from the
    #: effective feature count instead of silently using a small constant.
    nsamples: int | Literal["auto"] = "auto"
    #: Attribution resolution: 'variable' groups an input window into 7 scores,
    #: 'variable_time' keeps one score per (lag, variable) pair.
    resolution: Literal["variable", "variable_time"] = "variable"
    #: Target steps to explain; the original code explained only step 0.
    target_steps: list[int] | Literal["all"] = field(default_factory=lambda: [0])
    #: Record the SHAP completeness residual |sum(phi) - (f(x) - E[f])|.
    check_additivity: bool = True


@dataclass
class RunConfig:
    output_dir: str = "runs"
    #: Optional explicit run name; otherwise derived from the config content.
    run_id: str | None = None
    tags: list[str] = field(default_factory=list)
    save_predictions: bool = True


@dataclass
class Config:
    """Root configuration object."""

    experiment: str = "main"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    run: RunConfig = field(default_factory=RunConfig)
    #: Provenance of the file this was loaded from.
    source_path: str | None = None

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Config":
        raw = dict(raw)
        cfg = cls(
            experiment=str(raw.get("experiment", "main")),
            data=_build(DataConfig, raw.get("data")),
            split=_build(SplitConfig, raw.get("split")),
            scaling=_build(ScalingConfig, raw.get("scaling")),
            model=_build(ModelConfig, raw.get("model")),
            training=_build(TrainingConfig, raw.get("training")),
            xai=_build(XAIConfig, raw.get("xai")),
            run=_build(RunConfig, raw.get("run")),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Sequence[str] | None = None) -> "Config":
        raw = read_yaml(path)
        for item in overrides or ():
            _apply_override(raw, item)
        cfg = cls.from_dict(raw)
        cfg.source_path = str(Path(path).resolve())
        return cfg

    def validate(self) -> None:
        if self.data.seq_len % 24 != 0:
            raise ValueError(f"data.seq_len={self.data.seq_len} must be a whole number of days")
        if (self.data.seq_len + self.data.pred_len) > 24 * 365:
            raise ValueError("input + horizon exceeds one year of hourly history")
        if self.training.repeats < 1:
            raise ValueError("training.repeats must be >= 1")
        if self.training.loss == "exp" and self.training.exp_loss_beta <= 1:
            raise ValueError("exp loss base must be > 1 for a convex penalty")
        if self.xai.n_eval < 1 or self.xai.n_background < 1:
            raise ValueError("xai.n_eval and xai.n_background must be >= 1")
        if self.xai.method == "tree" and self.model.name.lower() not in {
            "randomforest",
            "random_forest",
            "rf",
        }:
            raise ValueError("xai.method='tree' only applies to the Random Forest baseline")

    # -- introspection ----------------------------------------------------- #

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source_path", None)
        return d

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.as_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    def fingerprint(self) -> str:
        """Stable short hash over everything that can change a number."""
        payload = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]

    @property
    def name(self) -> str:
        """Human-readable identity, safe to use as a single path component.

        Includes the loss, unlike the original ``setting`` string, which is what
        made the three loss-ablation runs collide in one directory.
        """
        parts = (
            self.experiment,
            self.data.name,
            self.model.name,
            f"L{self.data.seq_len}H{self.data.pred_len}",
            self.training.loss,
        )
        return "__".join(_slugify(p) for p in parts)

    def describe(self) -> str:
        return (
            f"{self.name}\n"
            f"  model params : {json.dumps(self.model.params, sort_keys=True, default=str)}\n"
            f"  training     : loss={self.training.loss} epochs={self.training.epochs} "
            f"bs={self.training.batch_size} lr={self.training.learning_rate:g} "
            f"patience={self.training.patience} seed={self.training.seed} "
            f"repeats={self.training.repeats}\n"
            f"  scaling      : {self.scaling.method}\n"
            f"  xai          : enabled={self.xai.enabled} method={self.xai.method} "
            f"resolution={self.xai.resolution}\n"
            f"  fingerprint  : {self.fingerprint()}"
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build(cls: type, raw: Any):
    """Instantiate a config dataclass, resolving nested dataclass fields.

    Field types are resolved through :func:`typing.get_type_hints` because the
    module uses ``from __future__ import annotations``, so ``field.type`` is a
    *string* and a naive ``is_dataclass(field.type)`` check silently never
    matches — nested sections would stay plain dicts.
    """
    if raw is None:
        return cls()
    if not isinstance(raw, Mapping):
        raise TypeError(f"{cls.__name__}: expected a mapping, got {type(raw).__name__}")

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown key(s) {sorted(unknown)}; valid: {sorted(known)}"
        )
    for key, value in raw.items():
        resolved = hints.get(key)
        if isinstance(value, Mapping) and isinstance(resolved, type) and is_dataclass(resolved):
            kwargs[key] = _build(resolved, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


_DOTTED = re.compile(r"^(?P<path>[A-Za-z_][\w.]*)=(?P<value>.*)$", re.S)
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(text: str) -> str:
    """Collapse anything that is not path-safe; used by ``Config.name``."""
    return _UNSAFE.sub("-", str(text)).strip("-") or "unnamed"


def _apply_override(raw: dict[str, Any], item: str) -> None:
    """Apply ``section.field=value`` where value is parsed as YAML."""
    if "=" not in item:
        raise ValueError(f"override {item!r} must look like section.field=value")
    path, _, text = item.partition("=")
    keys = path.strip().split(".")
    node: Any = raw
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"override {item!r}: {key!r} is not a mapping")
    node[keys[-1]] = yaml.load(text, Loader=_Loader)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_composed(
    paths: Sequence[str | Path], overrides: Sequence[str] | None = None
) -> Config:
    """Compose several YAML files left-to-right, then apply ``key=value`` overrides."""
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, read_yaml(path))
    for item in overrides or ():
        _apply_override(merged, item)
    cfg = Config.from_dict(merged)
    cfg.source_path = " + ".join(str(Path(p).resolve()) for p in paths)
    return cfg
