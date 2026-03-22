import asyncio
import io
import json
from contextlib import redirect_stdout

import pytest


pytest.importorskip("ves_stark")

from run_network import run_network
from stark_autoresearch.agent import AutoResearchAgent
from stark_autoresearch.coordinator import NetworkCoordinator
from stark_autoresearch.experiment import ExperimentConfig, ExperimentResult
from stark_autoresearch.proof import MetricImprovementProver


def _proposal_signature(agent: AutoResearchAgent, count: int = 5) -> list[tuple[str, str]]:
    signatures = []
    for _ in range(count):
        config = agent.propose_experiment()
        signatures.append((config.description, config.content_hash()))
    return signatures


def test_invalid_strategy_rejected() -> None:
    try:
        AutoResearchAgent(strategy="invalid")
        assert False, "Invalid strategy should raise ValueError"
    except ValueError as exc:
        assert "Unknown strategy" in str(exc)


def test_seeded_agent_proposals_are_reproducible() -> None:
    agent1 = AutoResearchAgent(agent_id="agent-seeded", strategy="adaptive", seed=7)
    agent2 = AutoResearchAgent(agent_id="agent-seeded", strategy="adaptive", seed=7)

    assert _proposal_signature(agent1) == _proposal_signature(agent2)


def test_seeded_experiment_results_are_reproducible() -> None:
    config = ExperimentConfig.from_baseline()
    agent1 = AutoResearchAgent(agent_id="agent-seeded", seed=99)
    agent2 = AutoResearchAgent(agent_id="agent-seeded", seed=99)

    result1 = asyncio.run(agent1.run_experiment(config))
    result2 = asyncio.run(agent2.run_experiment(config))

    assert result1.experiment_id == result2.experiment_id
    assert result1.avg_reward == result2.avg_reward
    assert result1.reward_std == result2.reward_std
    assert result1.success_rate == result2.success_rate
    assert result1.commit_hash == result2.commit_hash


def test_export_results_contains_bound_proof_metadata(tmp_path) -> None:
    config = ExperimentConfig.from_baseline()
    result = ExperimentResult(
        experiment_id="11111111-1111-1111-1111-111111111111",
        agent_id="agent-00",
        config=config,
        avg_reward=0.55,
    )
    proof = MetricImprovementProver().prove_improvement(
        new_reward=0.55,
        best_reward=0.52,
        experiment_id=result.experiment_id,
        agent_id=result.agent_id,
        config_hash=config.content_hash(),
    )

    async def scenario() -> NetworkCoordinator:
        coordinator = NetworkCoordinator(initial_best_reward=0.52)
        accepted = await coordinator.submit(result, proof)
        assert accepted
        return coordinator

    coordinator = asyncio.run(scenario())
    export_path = tmp_path / "results.json"
    coordinator.export_results(export_path)

    data = json.loads(export_path.read_text())
    assert data["stats"]["verified_improvements"] == 1
    exported = data["verified_experiments"][0]
    assert exported["config_hash"] == config.content_hash()
    assert exported["proof"]["config_hash"] == config.content_hash()
    assert exported["proof"]["sequence_number"] == 1
    assert exported["proof"]["new_reward_scaled"] == 550000
    assert exported["proof"]["amount_binding_hash"]


def test_seeded_network_trajectory_is_reproducible() -> None:
    async def scenario() -> tuple[list[tuple[str, float, str]], list[tuple[str, float, str]]]:
        with redirect_stdout(io.StringIO()):
            coord1, _ = await run_network(
                num_agents=2,
                experiments_per_agent=5,
                strategies=["adaptive", "random"],
                seed=42,
            )
        with redirect_stdout(io.StringIO()):
            coord2, _ = await run_network(
                num_agents=2,
                experiments_per_agent=5,
                strategies=["adaptive", "random"],
                seed=42,
            )
        sig1 = [
            (ve.result.agent_id, ve.result.avg_reward, ve.result.config.content_hash())
            for ve in coord1.verified_experiments
        ]
        sig2 = [
            (ve.result.agent_id, ve.result.avg_reward, ve.result.config.content_hash())
            for ve in coord2.verified_experiments
        ]
        return sig1, sig2

    sig1, sig2 = asyncio.run(scenario())
    assert sig1
    assert sig1 == sig2
