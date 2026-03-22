"""
Experiment data structures and hyperparameter perturbation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

RandomSource = random.Random


# Default baseline hyperparameters (matches current best: commit dbe3c69)
BASELINE_PARAMS: dict[str, Any] = {
    "learning_rate": 5e-6,
    "num_generations": 2,
    "num_outer_iterations": 1,
    "generations_per_iteration": 3,
    "clip_range_left": 3e-4,
    "clip_range_right": 4e-4,
    "lora_r": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "temperature": 0.5,
    "top_p": 0.9,
    "warmup_ratio": 0.1,
    "entropy_coef": 0.01,
    "beta": 0.0,
}

# Search ranges for each parameter
SEARCH_RANGES: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "log_float", "low": 1e-7, "high": 1e-3},
    "num_generations": {"type": "int", "low": 1, "high": 16},
    "num_outer_iterations": {"type": "int", "low": 1, "high": 5},
    "generations_per_iteration": {"type": "int", "low": 1, "high": 10},
    "clip_range_left": {"type": "log_float", "low": 1e-5, "high": 1e-2},
    "clip_range_right": {"type": "log_float", "low": 1e-5, "high": 1e-2},
    "lora_r": {"type": "choice", "values": [2, 4, 8, 16, 32]},
    "lora_alpha": {"type": "choice", "values": [4, 8, 16, 32, 64]},
    "lora_dropout": {"type": "float", "low": 0.0, "high": 0.3},
    "temperature": {"type": "float", "low": 0.1, "high": 1.5},
    "top_p": {"type": "float", "low": 0.5, "high": 1.0},
    "warmup_ratio": {"type": "float", "low": 0.0, "high": 0.3},
    "entropy_coef": {"type": "log_float", "low": 1e-4, "high": 0.1},
    "beta": {"type": "float", "low": 0.0, "high": 0.5},
}


@dataclass
class ExperimentConfig:
    """Hyperparameter configuration for one experiment."""
    params: dict[str, Any]
    description: str = ""

    def to_json(self) -> str:
        return json.dumps({"params": self.params, "description": self.description}, sort_keys=True)

    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @staticmethod
    def from_baseline() -> ExperimentConfig:
        return ExperimentConfig(params=dict(BASELINE_PARAMS), description="baseline")


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    config: ExperimentConfig = field(default_factory=ExperimentConfig.from_baseline)
    avg_reward: float = 0.0
    reward_std: float = 0.0
    success_rate: float = 0.0
    training_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)
    commit_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "agent_id": self.agent_id,
            "config": self.config.params,
            "config_hash": self.config.content_hash(),
            "description": self.config.description,
            "avg_reward": self.avg_reward,
            "reward_std": self.reward_std,
            "success_rate": self.success_rate,
            "training_seconds": self.training_seconds,
            "timestamp": self.timestamp,
            "commit_hash": self.commit_hash,
        }


def perturb_config(
    base: ExperimentConfig,
    num_params: int = 2,
    magnitude: float = 0.3,
    rng: RandomSource | None = None,
) -> ExperimentConfig:
    """Create a new config by randomly perturbing `num_params` parameters."""
    rng = rng or random
    params = copy.deepcopy(base.params)
    keys = rng.sample(list(SEARCH_RANGES.keys()), min(num_params, len(SEARCH_RANGES)))
    changes = []

    for key in keys:
        spec = SEARCH_RANGES[key]
        old_val = params.get(key, BASELINE_PARAMS.get(key))

        if spec["type"] == "log_float":
            import math
            log_val = math.log(old_val) if old_val > 0 else math.log(spec["low"])
            log_range = math.log(spec["high"]) - math.log(spec["low"])
            new_log = log_val + rng.gauss(0, magnitude * log_range)
            new_val = max(spec["low"], min(spec["high"], math.exp(new_log)))
            params[key] = new_val

        elif spec["type"] == "float":
            span = spec["high"] - spec["low"]
            new_val = old_val + rng.gauss(0, magnitude * span)
            new_val = max(spec["low"], min(spec["high"], new_val))
            params[key] = new_val

        elif spec["type"] == "int":
            delta = max(1, int(magnitude * (spec["high"] - spec["low"])))
            new_val = old_val + rng.randint(-delta, delta)
            new_val = max(spec["low"], min(spec["high"], new_val))
            params[key] = new_val

        elif spec["type"] == "choice":
            params[key] = rng.choice(spec["values"])

        changes.append(f"{key}={params[key]}")

    return ExperimentConfig(params=params, description=", ".join(changes))
