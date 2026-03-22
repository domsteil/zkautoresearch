"""
Decentralized Autoresearch Network with STARK Proofs
=====================================================

Multiple autoresearch agents run in parallel, each optimizing hyperparameters.
Every improvement is cryptographically verified via a STARK proof generated
from the stateset-stark library — no need to re-run experiments to trust results.

Modules:
    proof       — STARK proof generation/verification for metric improvement
    experiment  — Experiment data structures and config perturbation
    agent       — Individual autoresearch agent with proof generation
    coordinator — Network coordinator: proof verification, global state, broadcasting
"""

from stark_autoresearch.proof import (
    MetricImprovementProver,
    MetricImprovementProof,
    verify_improvement,
)
from stark_autoresearch.experiment import ExperimentResult, ExperimentConfig
from stark_autoresearch.agent import AutoResearchAgent
from stark_autoresearch.coordinator import NetworkCoordinator

__all__ = [
    "MetricImprovementProver",
    "MetricImprovementProof",
    "verify_improvement",
    "ExperimentResult",
    "ExperimentConfig",
    "AutoResearchAgent",
    "NetworkCoordinator",
]
