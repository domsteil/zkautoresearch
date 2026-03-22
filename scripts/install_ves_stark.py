#!/usr/bin/env python3
"""Install the local ves_stark Python bindings into the active environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def resolve_stateset_stark_dir(explicit: str | None) -> Path:
    candidates = [
        explicit,
        os.environ.get("STATESET_STARK_DIR"),
        "../stateset-stark",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "crates" / "ves-stark-python").is_dir():
            return path
    raise SystemExit(
        "Could not find stateset-stark. Pass --stateset-stark-dir or set STATESET_STARK_DIR."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the local ves_stark bindings")
    parser.add_argument(
        "--stateset-stark-dir",
        default=None,
        help="Path to the checked-out stateset-stark repository",
    )
    parser.add_argument(
        "--skip-pip-upgrade",
        action="store_true",
        help="Skip upgrading pip before installing maturin",
    )
    args = parser.parse_args()

    repo_dir = resolve_stateset_stark_dir(args.stateset_stark_dir)
    bindings_dir = repo_dir / "crates" / "ves-stark-python"

    if not args.skip_pip_upgrade:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "maturin"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "maturin", "develop", "--release"],
        cwd=bindings_dir,
        check=True,
    )

    print(f"Installed ves_stark from {bindings_dir}")


if __name__ == "__main__":
    main()
