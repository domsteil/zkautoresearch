"""
Individual autoresearch agent that runs experiments, generates STARK proofs
for improvements, and learns from the network's verified results.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from stark_autoresearch.experiment import (
    BASELINE_PARAMS,
    ExperimentConfig,
    ExperimentResult,
    perturb_config,
)
from stark_autoresearch.proof import MetricImprovementProof, MetricImprovementProver

logger = logging.getLogger(__name__)


# ─── Simulated reward surface ────────────────────────────────────────────────
# A synthetic function that maps hyperparameters to a reward signal.
# This lets us demo the network without running real training (which takes
# minutes per experiment). The surface has a global optimum that agents
# must discover collaboratively.

def _simulated_reward(params: dict[str, Any], noise_std: float = 0.015) -> float:
    """
    Simulated reward surface over the hyperparameter space.

    The surface is designed so that:
    - Baseline params (current best) give ~0.52
    - There's a reachable optimum around ~0.62
    - The surface is non-trivial (interactions between params)
    - Random noise simulates eval variance
    """
    lr = params.get("learning_rate", 5e-6)
    temp = params.get("temperature", 0.5)
    lora_r = params.get("lora_r", 4)
    lora_alpha = params.get("lora_alpha", 8)
    gens = params.get("num_generations", 2)
    warmup = params.get("warmup_ratio", 0.1)
    entropy = params.get("entropy_coef", 0.01)
    beta = params.get("beta", 0.0)
    outer = params.get("num_outer_iterations", 1)

    # Base reward from LR (optimal around 8e-6)
    lr_score = 0.5 - 0.3 * (math.log10(lr) - math.log10(8e-6)) ** 2

    # Temperature sweet spot around 0.6
    temp_score = 0.15 * math.exp(-((temp - 0.6) ** 2) / 0.08)

    # LoRA: r=4..8, alpha=2*r is optimal
    lora_ratio = lora_alpha / max(lora_r, 1)
    lora_score = 0.1 * math.exp(-((lora_ratio - 2.0) ** 2) / 0.5)

    # Generations: 2-4 is sweet spot
    gen_score = 0.05 * math.exp(-((gens - 3) ** 2) / 4)

    # Warmup: 0.05-0.15 is best
    warmup_score = 0.03 * math.exp(-((warmup - 0.1) ** 2) / 0.01)

    # Entropy: small positive helps
    entropy_score = 0.02 * math.exp(-((math.log10(max(entropy, 1e-6)) + 2) ** 2) / 2)

    # Beta: small positive can help
    beta_score = 0.02 * max(0, 1 - 4 * beta) if beta > 0 else 0.01

    # Outer iterations: 1-2 best
    outer_score = 0.02 * math.exp(-((outer - 1.5) ** 2) / 2)

    # Interaction: lr × temperature synergy
    interaction = 0.05 * math.exp(-((math.log10(lr) + 5.1) ** 2 + (temp - 0.6) ** 2) / 0.2)

    base_reward = (
        lr_score + temp_score + lora_score + gen_score +
        warmup_score + entropy_score + beta_score + outer_score +
        interaction
    )

    # Clamp to [0, 1] and add noise
    reward = max(0.0, min(1.0, base_reward))
    reward += random.gauss(0, noise_std)
    return max(0.0, min(1.0, reward))


@dataclass
class AgentMemory:
    """
    What the agent has learned from its own experiments and the network.
    Used to guide future proposals (self-learning).
    """
    experiments: list[ExperimentResult] = field(default_factory=list)
    improvements: list[ExperimentResult] = field(default_factory=list)
    param_scores: dict[str, list[tuple[Any, float]]] = field(default_factory=dict)

    def record(self, result: ExperimentResult, improved: bool) -> None:
        self.experiments.append(result)
        if improved:
            self.improvements.append(result)
        # Track which parameter values led to what rewards.
        for key, val in result.config.params.items():
            if key not in self.param_scores:
                self.param_scores[key] = []
            self.param_scores[key].append((val, result.avg_reward))

    def best_value_for(self, key: str) -> Any | None:
        """Return the parameter value that correlated with the highest reward."""
        entries = self.param_scores.get(key, [])
        if not entries:
            return None
        return max(entries, key=lambda x: x[1])[0]


class AutoResearchAgent:
    """
    A single autoresearch agent on the decentralized network.

    Each agent:
    1. Proposes hyperparameter experiments (learning from past results).
    2. Runs experiments (simulated or real).
    3. Generates STARK proofs for improvements.
    4. Submits proofs to the network coordinator.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        strategy: str = "adaptive",
        reward_fn: Callable[[dict[str, Any]], float] | None = None,
    ) -> None:
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.strategy = strategy  # "random", "adaptive", "exploit"
        self.reward_fn = reward_fn or _simulated_reward
        self.prover = MetricImprovementProver()
        self.memory = AgentMemory()
        self.best_reward: float = 0.0
        self.best_config: ExperimentConfig = ExperimentConfig.from_baseline()
        self._rng = random.Random(hash(self.agent_id))

    def propose_experiment(self) -> ExperimentConfig:
        """Propose the next experiment based on strategy and learned memory."""
        if self.strategy == "random":
            return perturb_config(self.best_config, num_params=3, magnitude=0.4)

        if self.strategy == "exploit":
            return perturb_config(self.best_config, num_params=1, magnitude=0.1)

        # Adaptive: blend exploration and exploitation based on history.
        n = len(self.memory.experiments)
        if n < 3:
            # Early: explore broadly.
            return perturb_config(self.best_config, num_params=3, magnitude=0.4)

        # Compute improvement rate.
        n_improved = len(self.memory.improvements)
        improvement_rate = n_improved / n if n > 0 else 0

        if improvement_rate > 0.3:
            # High hit rate: exploit harder.
            magnitude = 0.1
            num_params = 1
        elif improvement_rate > 0.1:
            # Medium: balanced.
            magnitude = 0.25
            num_params = 2
        else:
            # Low: explore more aggressively.
            magnitude = 0.5
            num_params = 3

        # Bias toward parameter values that worked well.
        config = perturb_config(self.best_config, num_params=num_params, magnitude=magnitude)
        for key in list(config.params.keys()):
            best_val = self.memory.best_value_for(key)
            if best_val is not None and self._rng.random() < 0.3:
                # 30% chance to use the historically best value.
                config.params[key] = best_val

        return config

    async def run_experiment(
        self, config: ExperimentConfig
    ) -> ExperimentResult:
        """Run a single experiment and return the result."""
        experiment_id = str(uuid.uuid4())
        start = time.time()

        # Simulate training time (50-200ms for demo).
        await asyncio.sleep(self._rng.uniform(0.05, 0.2))

        # Compute reward.
        reward = self.reward_fn(config.params)
        elapsed = time.time() - start

        commit_hash = hashlib.sha256(
            config.content_hash().encode() + self.agent_id.encode()
        ).hexdigest()[:7]

        return ExperimentResult(
            experiment_id=experiment_id,
            agent_id=self.agent_id,
            config=config,
            avg_reward=reward,
            reward_std=self._rng.uniform(0.01, 0.05),
            success_rate=min(1.0, reward + self._rng.uniform(0, 0.2)),
            training_seconds=elapsed,
            commit_hash=commit_hash,
        )

    def generate_proof(
        self, result: ExperimentResult
    ) -> MetricImprovementProof | None:
        """
        Generate a STARK proof if the result improves on the current best.
        Returns None if no improvement.
        """
        if result.avg_reward <= self.best_reward:
            return None

        try:
            proof = self.prover.prove_improvement(
                new_reward=result.avg_reward,
                best_reward=self.best_reward,
                experiment_id=result.experiment_id,
                agent_id=self.agent_id,
            )
            return proof
        except ValueError:
            return None

    def update_from_network(
        self, new_best_reward: float, new_best_config: ExperimentConfig
    ) -> None:
        """Update agent state when the network broadcasts a verified improvement."""
        if new_best_reward > self.best_reward:
            self.best_reward = new_best_reward
            self.best_config = new_best_config
            logger.debug(
                "[%s] Updated from network: best=%.6f",
                self.agent_id, new_best_reward,
            )

    async def run_loop(
        self,
        submit_fn,
        max_experiments: int = 20,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """
        Main agent loop: propose → run → prove → submit.

        Args:
            submit_fn: async callable(result, proof) → bool (accepted by coordinator)
            max_experiments: max experiments before stopping
            stop_event: external stop signal
        """
        for i in range(max_experiments):
            if stop_event and stop_event.is_set():
                break

            config = self.propose_experiment()
            result = await self.run_experiment(config)
            proof = self.generate_proof(result)

            improved = proof is not None
            self.memory.record(result, improved)

            if proof:
                accepted = await submit_fn(result, proof)
                if accepted:
                    self.best_reward = result.avg_reward
                    self.best_config = result.config
                    logger.info(
                        "[%s] Exp %d/%d: %.6f → PROVED & ACCEPTED (proof %d bytes, %dms)",
                        self.agent_id, i + 1, max_experiments,
                        result.avg_reward, proof.proof_size, proof.proving_time_ms,
                    )
                else:
                    logger.info(
                        "[%s] Exp %d/%d: %.6f → proof rejected by coordinator",
                        self.agent_id, i + 1, max_experiments, result.avg_reward,
                    )
            else:
                logger.debug(
                    "[%s] Exp %d/%d: %.6f (no improvement over %.6f)",
                    self.agent_id, i + 1, max_experiments,
                    result.avg_reward, self.best_reward,
                )

            # Small delay to simulate realistic pacing.
            await asyncio.sleep(0.01)
