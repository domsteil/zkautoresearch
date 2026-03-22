"""
Network coordinator for the decentralized autoresearch network.

Responsibilities:
- Maintains global best reward and configuration.
- Receives experiment submissions + STARK proofs from agents.
- Verifies proofs cryptographically (fast, no re-running experiments).
- Broadcasts verified improvements to all agents.
- Tracks full history of verified experiments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stark_autoresearch.experiment import ExperimentConfig, ExperimentResult
from stark_autoresearch.proof import MetricImprovementProof, verify_improvement

logger = logging.getLogger(__name__)


@dataclass
class VerifiedExperiment:
    """An experiment whose improvement has been cryptographically verified."""
    result: ExperimentResult
    proof: MetricImprovementProof
    verification_time_ms: float
    verified_at: float = field(default_factory=time.time)


class NetworkCoordinator:
    """
    Central coordinator for the STARK-verified autoresearch network.

    In a fully decentralized version, this would be replaced by a consensus
    protocol (e.g., on-chain). This implementation simulates the coordinator
    role for demonstration.
    """

    def __init__(self, initial_best_reward: float = 0.0) -> None:
        self.best_reward: float = initial_best_reward
        self.best_config: ExperimentConfig = ExperimentConfig.from_baseline()
        self.best_experiment: ExperimentResult | None = None

        self.verified_experiments: list[VerifiedExperiment] = []
        self.rejected_count: int = 0
        self.total_submissions: int = 0

        self._agents: list[Any] = []
        self._lock = asyncio.Lock()

        # Stats
        self.total_proving_time_ms: int = 0
        self.total_verification_time_ms: float = 0
        self.total_proof_bytes: int = 0

    def register_agent(self, agent) -> None:
        """Register an agent to receive network broadcasts."""
        self._agents.append(agent)
        agent.best_reward = self.best_reward
        agent.best_config = ExperimentConfig(
            params=dict(self.best_config.params),
            description=self.best_config.description,
        )
        logger.info(
            "Registered %s (starting best=%.6f)",
            agent.agent_id, self.best_reward,
        )

    async def submit(
        self, result: ExperimentResult, proof: MetricImprovementProof
    ) -> bool:
        """
        Submit an experiment result with its STARK proof.

        The coordinator verifies the proof cryptographically. If valid and
        the metric improves on the global best, the improvement is accepted
        and broadcast to all agents.

        Returns True if accepted, False if rejected.
        """
        async with self._lock:
            self.total_submissions += 1

            # Verify the STARK proof.
            valid, message = verify_improvement(proof, self.best_reward)

            if not valid:
                self.rejected_count += 1
                logger.warning(
                    "REJECTED submission from %s: %s",
                    result.agent_id, message,
                )
                return False

            # Check that the claimed improvement is actually over current best.
            if result.avg_reward <= self.best_reward:
                self.rejected_count += 1
                logger.warning(
                    "REJECTED: reward %.6f not better than global best %.6f",
                    result.avg_reward, self.best_reward,
                )
                return False

            # Accept the verified improvement.
            v_time_ms = float(message.split("verified in ")[1].split("ms")[0]) if "verified in" in message else 0
            verified = VerifiedExperiment(
                result=result,
                proof=proof,
                verification_time_ms=v_time_ms,
            )
            self.verified_experiments.append(verified)

            # Update global best.
            old_best = self.best_reward
            self.best_reward = result.avg_reward
            self.best_config = result.config
            self.best_experiment = result

            # Accumulate stats.
            self.total_proving_time_ms += proof.proving_time_ms
            self.total_verification_time_ms += v_time_ms
            self.total_proof_bytes += proof.proof_size

            logger.info(
                "VERIFIED improvement: %.6f → %.6f by %s (%s)",
                old_best, result.avg_reward,
                result.agent_id, message,
            )

            # Broadcast to all agents.
            for agent in self._agents:
                if agent.agent_id != result.agent_id:
                    agent.update_from_network(self.best_reward, self.best_config)

            return True

    def get_stats(self) -> dict[str, Any]:
        """Return network statistics."""
        n_verified = len(self.verified_experiments)
        return {
            "global_best_reward": self.best_reward,
            "total_submissions": self.total_submissions,
            "verified_improvements": n_verified,
            "rejected": self.rejected_count,
            "acceptance_rate": n_verified / max(1, self.total_submissions),
            "registered_agents": len(self._agents),
            "total_proving_time_ms": self.total_proving_time_ms,
            "total_verification_time_ms": self.total_verification_time_ms,
            "total_proof_bytes": self.total_proof_bytes,
            "avg_proof_size": self.total_proof_bytes // max(1, n_verified),
            "improvement_trajectory": [
                {
                    "reward": ve.result.avg_reward,
                    "agent": ve.result.agent_id,
                    "proof_size": ve.proof.proof_size,
                    "proving_ms": ve.proof.proving_time_ms,
                    "verify_ms": ve.verification_time_ms,
                    "config": ve.result.config.description,
                }
                for ve in self.verified_experiments
            ],
        }

    def print_summary(self) -> None:
        """Print a summary of network activity."""
        stats = self.get_stats()
        print("\n" + "=" * 70)
        print("  DECENTRALIZED AUTORESEARCH NETWORK — SUMMARY")
        print("=" * 70)
        print(f"  Global best reward:      {stats['global_best_reward']:.6f}")
        print(f"  Registered agents:       {stats['registered_agents']}")
        print(f"  Total submissions:       {stats['total_submissions']}")
        print(f"  Verified improvements:   {stats['verified_improvements']}")
        print(f"  Rejected:                {stats['rejected']}")
        print(f"  Acceptance rate:         {stats['acceptance_rate']:.1%}")
        print(f"  Total proof data:        {stats['total_proof_bytes']:,} bytes")
        print(f"  Avg proof size:          {stats['avg_proof_size']:,} bytes")
        print(f"  Total proving time:      {stats['total_proving_time_ms']:,} ms")
        print(f"  Total verification time: {stats['total_verification_time_ms']:.0f} ms")

        if self.verified_experiments:
            print(f"\n  {'─' * 66}")
            print(f"  VERIFIED IMPROVEMENT TRAJECTORY (STARK-proven)")
            print(f"  {'─' * 66}")
            for i, ve in enumerate(self.verified_experiments):
                print(
                    f"  {i+1:3d}. reward={ve.result.avg_reward:.6f}  "
                    f"agent={ve.result.agent_id}  "
                    f"proof={ve.proof.proof_size:,}B  "
                    f"prove={ve.proof.proving_time_ms}ms  "
                    f"verify={ve.verification_time_ms:.0f}ms"
                )
                print(f"       config: {ve.result.config.description}")

        print("=" * 70)

    def export_results(self, path: str | Path) -> None:
        """Export network results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stats": self.get_stats(),
            "verified_experiments": [
                {
                    **ve.result.to_dict(),
                    "proof": ve.proof.to_dict(),
                    "verification_time_ms": ve.verification_time_ms,
                }
                for ve in self.verified_experiments
            ],
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Results exported to %s", path)
