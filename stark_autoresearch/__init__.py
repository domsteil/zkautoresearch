"""Decentralized autoresearch network with STARK-backed proof verification."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AutoResearchAgent",
    "DependencyStatus",
    "EnvironmentReport",
    "ExperimentConfig",
    "ExperimentResult",
    "MetricImprovementProof",
    "MetricImprovementProver",
    "NetworkCoordinator",
    "build_environment_report",
    "format_environment_report",
    "inspect_ves_stark",
    "verify_improvement",
]

_LAZY_EXPORTS = {
    "AutoResearchAgent": ("stark_autoresearch.agent", "AutoResearchAgent"),
    "DependencyStatus": ("stark_autoresearch.environment", "DependencyStatus"),
    "EnvironmentReport": ("stark_autoresearch.environment", "EnvironmentReport"),
    "ExperimentConfig": ("stark_autoresearch.experiment", "ExperimentConfig"),
    "ExperimentResult": ("stark_autoresearch.experiment", "ExperimentResult"),
    "MetricImprovementProof": ("stark_autoresearch.proof", "MetricImprovementProof"),
    "MetricImprovementProver": ("stark_autoresearch.proof", "MetricImprovementProver"),
    "NetworkCoordinator": ("stark_autoresearch.coordinator", "NetworkCoordinator"),
    "build_environment_report": ("stark_autoresearch.environment", "build_environment_report"),
    "format_environment_report": ("stark_autoresearch.environment", "format_environment_report"),
    "inspect_ves_stark": ("stark_autoresearch.environment", "inspect_ves_stark"),
    "verify_improvement": ("stark_autoresearch.proof", "verify_improvement"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
