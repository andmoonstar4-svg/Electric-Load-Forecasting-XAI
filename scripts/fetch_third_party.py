#!/usr/bin/env python
"""Vendor the upstream model code into ``third_party/``.

Only the architectures the paper actually uses are copied, together with the
``layers``/``utils`` modules they import. The original repository committed two
complete forks (about 180 files, 60+ unused architectures, plus Docker and CI
configuration), which is why its own contribution was impossible to see.

Usage
-----
Copy from an existing checkout::

    python scripts/fetch_third_party.py --from /path/to/Time-Series-Library

Clone from upstream at a pinned revision::

    python scripts/fetch_third_party.py --clone --ref <commit-or-tag>

Nothing is downloaded unless ``--clone`` is given.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "third_party" / "Time-Series-Library"
UPSTREAM_URL = "https://github.com/thuml/Time-Series-Library"

MODELS = ("iTransformer.py", "DLinear.py", "Autoformer.py", "Transformer.py")
LAYERS = (
    "AutoCorrelation.py",
    "Autoformer_EncDec.py",
    "Embed.py",
    "SelfAttention_Family.py",
    "StandardNorm.py",
    "Transformer_EncDec.py",
)
UTILS = ("metrics.py", "tools.py", "timefeatures.py", "masking.py", "dtw_metric.py", "augmentation.py")


def copy_from(source: Path) -> int:
    if not (source / "models" / "iTransformer.py").exists():
        print(f"{source} does not look like a Time-Series-Library checkout", file=sys.stderr)
        return 1

    for sub, files in (("models", MODELS), ("layers", LAYERS), ("utils", UTILS)):
        (TARGET / sub).mkdir(parents=True, exist_ok=True)
        for name in files:
            src = source / sub / name
            if not src.exists():
                print(f"  missing upstream file (skipped): {sub}/{name}", file=sys.stderr)
                continue
            shutil.copy2(src, TARGET / sub / name)
            print(f"  {sub}/{name}")

    for extra in ("LICENSE", "requirements.txt"):
        src = source / extra
        if src.exists():
            name = "requirements-upstream.txt" if extra == "requirements.txt" else extra
            shutil.copy2(src, TARGET / name)

    _write_provenance(source)
    return 0


def _write_provenance(source: Path | None) -> None:
    commit = None
    if source is not None:
        try:
            commit = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None

    path = REPO_ROOT / "third_party" / "PROVENANCE.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    entry = data.setdefault("Time-Series-Library", {})
    entry.update(
        {
            "url": UPSTREAM_URL,
            "licence": "MIT",
            "vendored_files": [f"models/{m}" for m in MODELS]
            + [f"layers/{l}" for l in LAYERS]
            + [f"utils/{u}" for u in UTILS],
            "vendored_from_commit": commit,
        }
    )
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"recorded provenance in {path} (commit={commit})")


def clone_from_upstream(ref: str | None, dest: Path) -> int:
    if dest.exists():
        shutil.rmtree(dest)
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [UPSTREAM_URL, str(dest)]
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("clone failed", file=sys.stderr)
        return result.returncode
    return copy_from(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, help="existing checkout to copy from")
    parser.add_argument("--clone", action="store_true", help="clone upstream into a temp dir")
    parser.add_argument("--ref", help="commit/tag to clone (with --clone)")
    args = parser.parse_args(argv)

    if args.clone:
        return clone_from_upstream(args.ref, REPO_ROOT / ".third_party_clone")
    if args.source:
        return copy_from(args.source.resolve())

    parser.print_help()
    print(
        "\nNo source given. Either pass --from <checkout> or --clone to fetch upstream.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
