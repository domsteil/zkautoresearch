"""Environment inspection helpers for zkautoresearch."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DependencyStatus:
    """Status information for a runtime dependency."""

    name: str
    available: bool
    location: str | None = None
    version: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentReport:
    """Summary of the current runtime environment."""

    python: str
    platform: str
    dependencies: list[DependencyStatus]

    @property
    def ready(self) -> bool:
        return all(dep.available for dep in self.dependencies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "platform": self.platform,
            "ready": self.ready,
            "dependencies": [dep.to_dict() for dep in self.dependencies],
        }


def inspect_ves_stark() -> DependencyStatus:
    """Inspect the installed `ves_stark` binding."""
    spec = importlib.util.find_spec("ves_stark")
    if spec is None:
        return DependencyStatus(
            name="ves_stark",
            available=False,
            detail=(
                "Missing. Build the bindings from stateset-stark/crates/ves-stark-python "
                "with `python scripts/install_ves_stark.py --stateset-stark-dir /path/to/stateset-stark`."
            ),
        )

    try:
        module = importlib.import_module("ves_stark")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return DependencyStatus(
            name="ves_stark",
            available=False,
            location=spec.origin,
            detail=f"Import failed: {exc}",
        )

    version = getattr(module, "__version__", None)
    return DependencyStatus(
        name="ves_stark",
        available=True,
        location=spec.origin,
        version=version,
        detail="Installed and importable.",
    )


def build_environment_report() -> EnvironmentReport:
    """Collect the environment report used by the doctor command."""
    return EnvironmentReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        dependencies=[inspect_ves_stark()],
    )


def format_environment_report(report: EnvironmentReport) -> str:
    """Render the environment report as plain text."""
    lines = [
        "zkautoresearch environment",
        f"  Python:   {report.python}",
        f"  Platform: {report.platform}",
        f"  Ready:    {'yes' if report.ready else 'no'}",
    ]
    for dep in report.dependencies:
        state = "ok" if dep.available else "missing"
        lines.append(f"  {dep.name}: {state}")
        if dep.location:
            lines.append(f"    location: {dep.location}")
        if dep.version:
            lines.append(f"    version:  {dep.version}")
        if dep.detail:
            lines.append(f"    detail:   {dep.detail}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the zkautoresearch runtime environment")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    parser.add_argument(
        "--require-ves-stark",
        action="store_true",
        help="Exit non-zero if the ves_stark binding is unavailable",
    )
    args = parser.parse_args()

    report = build_environment_report()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_environment_report(report))

    if args.require_ves_stark and not report.ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
