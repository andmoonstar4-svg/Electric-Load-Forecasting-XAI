"""Access to the vendored upstream model code.

The models come from `thuml/Time-Series-Library`. Their modules import each other
as top-level packages (``from layers.Embed import ...``, ``from models.X import
...``), so the tree must be on ``sys.path``. The original project instead relied
on the process being started from the repository root — a hidden dependency that
breaks the moment anything is imported from elsewhere (a notebook, a test, a
packaged entry point).

:func:`register` makes that dependency explicit and idempotent, and
:data:`UPSTREAM` records exactly which revision is vendored.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Repository root (…/repo), derived from this file's location.
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

#: Vendored upstream trees, in import-priority order.
UPSTREAM_DIRS: tuple[Path, ...] = (
    REPO_ROOT / "third_party" / "Time-Series-Library",
    REPO_ROOT / "third_party" / "LTSF-Linear",
)

#: Which upstream models each adapter needs. Restricting the copy keeps the
#: repository from shipping 60 unused architectures.
REQUIRED_MODEL_FILES: dict[str, tuple[str, ...]] = {
    "iTransformer": ("models/iTransformer.py", "layers/Embed.py", "layers/SelfAttention_Family.py", "layers/Transformer_EncDec.py"),
    "DLinear": ("models/DLinear.py",),
    "Autoformer": ("models/Autoformer.py", "layers/AutoCorrelation.py", "layers/Autoformer_EncDec.py", "layers/Embed.py", "layers/SelfAttention_Family.py", "layers/Transformer_EncDec.py"),
    "Transformer": ("models/Transformer.py",),
}


@dataclass(frozen=True)
class UpstreamInfo:
    name: str
    url: str
    commit: str
    licence: str
    vendored_model_files: tuple[str, ...]


UPSTREAM: dict[str, UpstreamInfo] = {
    "Time-Series-Library": UpstreamInfo(
        name="Time-Series-Library",
        url="https://github.com/thuml/Time-Series-Library",
        commit="see third_party/PROVENANCE.json",
        licence="MIT",
        vendored_model_files=tuple(
            sorted({f for files in REQUIRED_MODEL_FILES.values() for f in files})
        ),
    ),
}


class UpstreamMissing(RuntimeError):
    """Raised when the vendored code is absent, with the exact remedy."""


def upstream_path(name: str = "Time-Series-Library") -> Path:
    path = REPO_ROOT / "third_party" / name
    if not path.exists():
        raise UpstreamMissing(
            f"vendored upstream code not found at {path}.\n"
            "Run `python scripts/fetch_third_party.py` (or set ELXAI_THIRD_PARTY to a "
            "checkout of thuml/Time-Series-Library) to populate it."
        )
    return path


def register(name: str = "Time-Series-Library") -> Path:
    """Put ``third_party/<name>`` on ``sys.path``; safe to call repeatedly.

    ``ELXAI_THIRD_PARTY`` overrides the location entirely, which is how a caller
    points at their own checkout of the upstream project. Each failure mode gets
    its own message so the remedy is unambiguous.
    """
    import os

    override = os.environ.get("ELXAI_THIRD_PARTY")
    if override:
        path = Path(override)
        if not path.exists():
            raise UpstreamMissing(
                f"ELXAI_THIRD_PARTY is set to {path}, which does not exist. "
                "Unset it to use the vendored copy, or point it at a real "
                "Time-Series-Library checkout."
            )
    else:
        path = upstream_path(name)
        # Check here too rather than trusting upstream_path: this is the function
        # that is about to put the directory on sys.path, so it owns the guarantee.
        if not path.exists():
            raise UpstreamMissing(
                f"vendored upstream code not found at {path}.\n"
                "Run `python scripts/fetch_third_party.py --from <checkout>` (or "
                "--clone) to populate it, or set ELXAI_THIRD_PARTY to an existing "
                "checkout of thuml/Time-Series-Library."
            )

    for entry in (str(path), *(str(p) for p in UPSTREAM_DIRS if p.exists() and p != path)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return path


def import_model(name: str):
    """Import ``models.<name>.Model`` from the vendored tree.

    Raises
    ------
    UpstreamMissing
        If the tree is absent.
    ImportError
        With an actionable message if a torch/einops dependency is missing.
    """
    register()
    import importlib

    try:
        module = importlib.import_module(f"models.{name}")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            f"could not import upstream model {name!r} ({exc}). "
            "Install the torch extra: pip install -e '.[torch]'"
        ) from exc
    if not hasattr(module, "Model"):
        raise AttributeError(f"models.{name} has no class 'Model'")
    return module.Model


def provenance() -> dict[str, object]:
    """Contents of ``third_party/PROVENANCE.json`` if present."""
    path = REPO_ROOT / "third_party" / "PROVENANCE.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "warning": "third_party/PROVENANCE.json missing; upstream revision unknown",
        "upstream": {k: v.__dict__ for k, v in UPSTREAM.items()},
    }
