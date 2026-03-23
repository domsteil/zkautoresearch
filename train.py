"""
Stateset Agents — Autonomous RL Research Training Script
========================================================

This is the single-experiment runner used by the autoresearch loop.
It supports per-run JSON overrides so experiments can be launched in
isolated subprocesses with hard wall-clock limits.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from experiment_runtime import (
    DeviceInfo,
    RunSummary,
    detect_device_info,
    getenv_bool,
    write_run_summary,
)
from prepare import (
    EVAL_EPISODES,
    EVAL_SCENARIOS,
    LOCKED_SELECTION_METRIC_NAME,
    TIME_BUDGET,
    TRAIN_SCENARIOS,
    evaluate_selection_metric,
    evaluate_trained_agent,
    get_peak_vram_mb,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _coerce_optional_float_map(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("custom_reward_weights must be a JSON object")

    weights: dict[str, float] = {}
    for key, raw in value.items():
        weight = float(raw)
        if weight < 0:
            raise ValueError(f"Reward weight for {key!r} must be >= 0")
        weights[str(key)] = weight
    return weights or None


def _coerce_scenarios(value: Any, default: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError("train_scenarios must be a list")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each training scenario must be a JSON object")
        normalized.append(item)
    return normalized


def _load_run_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    config_path = os.getenv("AUTORESEARCH_RUN_CONFIG")
    if config_path:
        payload = json.loads(Path(config_path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("AUTORESEARCH_RUN_CONFIG must point to a JSON object")
        overrides.update(payload)

    raw_json = os.getenv("AUTORESEARCH_RUN_CONFIG_JSON")
    if raw_json:
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("AUTORESEARCH_RUN_CONFIG_JSON must decode to a JSON object")
        overrides.update(payload)

    return overrides


# ============================================================================
# MODEL CONFIGURATION — defaults, overridden per run by AUTORESEARCH_RUN_CONFIG
# ============================================================================

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
SYSTEM_PROMPT = (
    "You are an expert customer service agent. Follow these rules strictly: "
    "1) Greet the customer warmly. 2) Acknowledge their issue with empathy. "
    "3) Ask clarifying questions. 4) Provide a clear solution or next steps. "
    "5) Confirm resolution. Always be polite, professional, and concise."
)
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.5
TOP_P = 0.9

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

LEARNING_RATE = 5e-6
NUM_EPISODES = 1
NUM_GENERATIONS = 2
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 1
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# ============================================================================
# RL ALGORITHM PARAMETERS
# ============================================================================

CLIP_RATIO = 0.2
ENTROPY_COEF = 0.01
BETA = 0.0
GAMMA = 0.99
GAE_LAMBDA = 0.95
ADVANTAGE_NORMALIZATION = True
BASELINE_TYPE = "group_mean"

# ============================================================================
# GSPO-SPECIFIC PARAMETERS
# ============================================================================

NUM_OUTER_ITERATIONS = 1
GENERATIONS_PER_ITERATION = 3
CLIP_RANGE_LEFT = 3e-4
CLIP_RANGE_RIGHT = 4e-4

# ============================================================================
# MODEL OPTIMIZATION
# ============================================================================

USE_LORA = True
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
USE_4BIT = False
USE_8BIT = False
GRADIENT_CHECKPOINTING = False
BF16 = False
RUN_POST_EVAL = True
POST_EVAL_SAMPLES = 5
POST_EVAL_DETAILED = True

# ============================================================================
# REWARD CONFIGURATION
# ============================================================================

REWARD_DOMAIN = "customer_service"
CUSTOM_REWARD_WEIGHTS: dict[str, float] | None = None

# ============================================================================
# TRAINING SCENARIOS
# ============================================================================

SCENARIOS = TRAIN_SCENARIOS
EVAL_SCENARIO_SET = EVAL_SCENARIOS
OUTPUT_DIR = "./outputs"
ALGORITHM = "gspo"
EVAL_EPISODES_VALUE = EVAL_EPISODES
SKIP_TRAIN = False
EXPERIMENT_ID: str | None = None
RUNTIME_PROFILE = "standard"
TRAIN_SCENARIO_LIMIT: int | None = None
EVAL_SCENARIO_LIMIT: int | None = None

RUN_OVERRIDES = _load_run_overrides()

MODEL_NAME = str(RUN_OVERRIDES.get("model_name", MODEL_NAME))
SYSTEM_PROMPT = str(RUN_OVERRIDES.get("system_prompt", SYSTEM_PROMPT))
MAX_NEW_TOKENS = _coerce_int(
    RUN_OVERRIDES.get("max_new_tokens", RUN_OVERRIDES.get("max_completion_length")),
    MAX_NEW_TOKENS,
)
TEMPERATURE = _coerce_float(RUN_OVERRIDES.get("temperature"), TEMPERATURE)
TOP_P = _coerce_float(RUN_OVERRIDES.get("top_p"), TOP_P)
LEARNING_RATE = _coerce_float(RUN_OVERRIDES.get("learning_rate"), LEARNING_RATE)
NUM_EPISODES = _coerce_int(RUN_OVERRIDES.get("num_episodes"), NUM_EPISODES)
NUM_GENERATIONS = _coerce_int(RUN_OVERRIDES.get("num_generations"), NUM_GENERATIONS)
BATCH_SIZE = _coerce_int(
    RUN_OVERRIDES.get("batch_size", RUN_OVERRIDES.get("per_device_train_batch_size")),
    BATCH_SIZE,
)
GRADIENT_ACCUMULATION = _coerce_int(
    RUN_OVERRIDES.get("gradient_accumulation", RUN_OVERRIDES.get("gradient_accumulation_steps")),
    GRADIENT_ACCUMULATION,
)
WARMUP_RATIO = _coerce_float(RUN_OVERRIDES.get("warmup_ratio"), WARMUP_RATIO)
LR_SCHEDULER = str(RUN_OVERRIDES.get("lr_scheduler", LR_SCHEDULER))
WEIGHT_DECAY = _coerce_float(RUN_OVERRIDES.get("weight_decay"), WEIGHT_DECAY)
MAX_GRAD_NORM = _coerce_float(RUN_OVERRIDES.get("max_grad_norm"), MAX_GRAD_NORM)
CLIP_RATIO = _coerce_float(RUN_OVERRIDES.get("clip_ratio"), CLIP_RATIO)
ENTROPY_COEF = _coerce_float(RUN_OVERRIDES.get("entropy_coef"), ENTROPY_COEF)
BETA = _coerce_float(RUN_OVERRIDES.get("beta"), BETA)
GAMMA = _coerce_float(RUN_OVERRIDES.get("gamma"), GAMMA)
GAE_LAMBDA = _coerce_float(RUN_OVERRIDES.get("gae_lambda"), GAE_LAMBDA)
ADVANTAGE_NORMALIZATION = _coerce_bool(
    RUN_OVERRIDES.get("advantage_normalization"),
    ADVANTAGE_NORMALIZATION,
)
BASELINE_TYPE = str(RUN_OVERRIDES.get("baseline_type", BASELINE_TYPE))
NUM_OUTER_ITERATIONS = _coerce_int(
    RUN_OVERRIDES.get("num_outer_iterations"), NUM_OUTER_ITERATIONS
)
GENERATIONS_PER_ITERATION = _coerce_int(
    RUN_OVERRIDES.get("generations_per_iteration"), GENERATIONS_PER_ITERATION
)
CLIP_RANGE_LEFT = _coerce_float(RUN_OVERRIDES.get("clip_range_left"), CLIP_RANGE_LEFT)
CLIP_RANGE_RIGHT = _coerce_float(RUN_OVERRIDES.get("clip_range_right"), CLIP_RANGE_RIGHT)
USE_LORA = _coerce_bool(RUN_OVERRIDES.get("use_lora"), USE_LORA)
LORA_R = _coerce_int(RUN_OVERRIDES.get("lora_r"), LORA_R)
LORA_ALPHA = _coerce_int(RUN_OVERRIDES.get("lora_alpha"), LORA_ALPHA)
LORA_DROPOUT = _coerce_float(RUN_OVERRIDES.get("lora_dropout"), LORA_DROPOUT)
USE_4BIT = _coerce_bool(RUN_OVERRIDES.get("use_4bit"), USE_4BIT)
USE_8BIT = _coerce_bool(RUN_OVERRIDES.get("use_8bit"), USE_8BIT)
GRADIENT_CHECKPOINTING = _coerce_bool(
    RUN_OVERRIDES.get("gradient_checkpointing"),
    GRADIENT_CHECKPOINTING,
)
BF16 = _coerce_bool(RUN_OVERRIDES.get("bf16"), BF16)
RUN_POST_EVAL = _coerce_bool(RUN_OVERRIDES.get("run_post_eval"), RUN_POST_EVAL)
POST_EVAL_SAMPLES = _coerce_int(RUN_OVERRIDES.get("post_eval_samples"), POST_EVAL_SAMPLES)
POST_EVAL_DETAILED = _coerce_bool(
    RUN_OVERRIDES.get("post_eval_detailed"),
    POST_EVAL_DETAILED,
)
REWARD_DOMAIN = str(RUN_OVERRIDES.get("reward_domain", REWARD_DOMAIN))
CUSTOM_REWARD_WEIGHTS = _coerce_optional_float_map(
    RUN_OVERRIDES.get("custom_reward_weights", CUSTOM_REWARD_WEIGHTS)
)
SCENARIOS = _coerce_scenarios(
    RUN_OVERRIDES.get("train_scenarios", RUN_OVERRIDES.get("scenarios")),
    SCENARIOS,
)
EVAL_SCENARIO_SET = _coerce_scenarios(
    RUN_OVERRIDES.get("eval_scenarios"),
    EVAL_SCENARIO_SET,
)
TRAIN_SCENARIO_LIMIT = _coerce_optional_int(RUN_OVERRIDES.get("train_scenario_limit"))
EVAL_SCENARIO_LIMIT = _coerce_optional_int(RUN_OVERRIDES.get("eval_scenario_limit"))
if TRAIN_SCENARIO_LIMIT is not None:
    if TRAIN_SCENARIO_LIMIT < 1:
        raise ValueError(f"train_scenario_limit must be >= 1, got {TRAIN_SCENARIO_LIMIT}")
    SCENARIOS = SCENARIOS[:TRAIN_SCENARIO_LIMIT]
if EVAL_SCENARIO_LIMIT is not None:
    if EVAL_SCENARIO_LIMIT < 1:
        raise ValueError(f"eval_scenario_limit must be >= 1, got {EVAL_SCENARIO_LIMIT}")
    EVAL_SCENARIO_SET = EVAL_SCENARIO_SET[:EVAL_SCENARIO_LIMIT]
OUTPUT_DIR = str(RUN_OVERRIDES.get("output_dir", OUTPUT_DIR))
ALGORITHM = str(RUN_OVERRIDES.get("algorithm", ALGORITHM)).lower()
EVAL_EPISODES_VALUE = _coerce_int(RUN_OVERRIDES.get("eval_episodes"), EVAL_EPISODES_VALUE)
SKIP_TRAIN = _coerce_bool(RUN_OVERRIDES.get("skip_train"), SKIP_TRAIN)
RUNTIME_PROFILE = str(RUN_OVERRIDES.get("runtime_profile", RUNTIME_PROFILE))
if RUN_OVERRIDES.get("experiment_id"):
    EXPERIMENT_ID = str(RUN_OVERRIDES["experiment_id"])

if ALGORITHM != "gspo":
    raise ValueError(
        f"train.py currently supports only algorithm='gspo', got {ALGORITHM!r}"
    )

RUN_SUMMARY_PATH = Path(
    os.getenv("AUTORESEARCH_RUN_SUMMARY", f"{OUTPUT_DIR}/latest_run.json")
)
PREFER_GPU = getenv_bool("AUTORESEARCH_USE_GPU", True)
REQUIRE_GPU = getenv_bool("AUTORESEARCH_REQUIRE_GPU", False)


def _device_label(device_info: DeviceInfo) -> str:
    if device_info.device_name:
        return f"{device_info.accelerator} ({device_info.device_name})"
    return device_info.accelerator


def _config_snapshot(device_info: DeviceInfo) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "algorithm": ALGORITHM,
        "skip_train": SKIP_TRAIN,
        "output_dir": OUTPUT_DIR,
        "model_name": MODEL_NAME,
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "learning_rate": LEARNING_RATE,
        "num_episodes": NUM_EPISODES,
        "num_generations": NUM_GENERATIONS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler": LR_SCHEDULER,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "clip_ratio": CLIP_RATIO,
        "entropy_coef": ENTROPY_COEF,
        "beta": BETA,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "advantage_normalization": ADVANTAGE_NORMALIZATION,
        "baseline_type": BASELINE_TYPE,
        "num_outer_iterations": NUM_OUTER_ITERATIONS,
        "generations_per_iteration": GENERATIONS_PER_ITERATION,
        "clip_range_left": CLIP_RANGE_LEFT,
        "clip_range_right": CLIP_RANGE_RIGHT,
        "use_lora": USE_LORA,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "use_4bit": USE_4BIT,
        "use_8bit": USE_8BIT,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "bf16_requested": BF16,
        "run_post_eval": RUN_POST_EVAL,
        "post_eval_samples": POST_EVAL_SAMPLES,
        "post_eval_detailed": POST_EVAL_DETAILED,
        "reward_domain": REWARD_DOMAIN,
        "custom_reward_weights": CUSTOM_REWARD_WEIGHTS,
        "runtime_profile": RUNTIME_PROFILE,
        "train_scenario_limit": TRAIN_SCENARIO_LIMIT,
        "eval_scenario_limit": EVAL_SCENARIO_LIMIT,
        "train_scenario_count": len(SCENARIOS),
        "eval_scenario_count": len(EVAL_SCENARIO_SET),
        "train_scenarios": SCENARIOS,
        "eval_scenarios": EVAL_SCENARIO_SET,
        "eval_episodes": EVAL_EPISODES_VALUE,
        "selection_metric": LOCKED_SELECTION_METRIC_NAME,
        "selection_eval_deterministic": True,
        "selection_bootstrap_samples": int(os.getenv("AUTORESEARCH_SELECTION_BOOTSTRAP_SAMPLES", "128")),
        "time_budget_seconds": TIME_BUDGET,
        "prefer_gpu": PREFER_GPU,
        "require_gpu": REQUIRE_GPU,
        "resolved_device": device_info.to_dict(),
    }


def _emit_run_summary(summary: RunSummary, summary_path: Path) -> None:
    metrics = summary.metrics
    print("---")
    if summary.status == "ok":
        print(f"selection_score:  {metrics.get(LOCKED_SELECTION_METRIC_NAME, 0.0):.6f}")
        print(f"selection_std:    {metrics.get('selection_score_std', 0.0):.6f}")
        print(f"selection_boot:   {metrics.get('selection_score_bootstrap_std', 0.0):.6f}")
        print(f"selection_low:    {metrics.get('selection_score_lower_bound', 0.0):.6f}")
        print(f"selection_high:   {metrics.get('selection_score_upper_bound', 0.0):.6f}")
        print(f"selection_hit:    {metrics.get('selection_success_rate', 0.0):.4f}")
        print(f"avg_reward:       {metrics.get('eval_reward', 0.0):.6f}")
        print(f"reward_std:       {metrics.get('eval_reward_std', 0.0):.6f}")
        print(f"success_rate:     {metrics.get('eval_success_rate', 0.0):.4f}")
        print(f"avg_ep_length:    {metrics.get('eval_episode_length', 0.0):.1f}")
    print(f"training_seconds: {summary.training_seconds or 0.0:.1f}")
    print(f"total_seconds:    {summary.total_seconds:.1f}")
    print(f"peak_vram_mb:     {summary.peak_vram_mb:.1f}")
    print(f"status:           {summary.status}")
    print(
        "device:           "
        + _device_label(
            DeviceInfo(
                summary.accelerator,
                summary.accelerator == "cpu",
                summary.cuda_available,
                summary.device_name,
                summary.torch_version,
                summary.bf16_enabled,
            )
        )
    )
    print(f"time_budget:      {summary.time_budget_seconds}")
    print(f"model:            {MODEL_NAME}")
    print(f"reward_domain:    {REWARD_DOMAIN}")
    print(f"runtime_profile:  {summary.config.get('runtime_profile', 'standard')}")
    print(f"train_scenarios:  {summary.config.get('train_scenario_count', 0)}")
    print(f"eval_scenarios:   {summary.config.get('eval_scenario_count', 0)}")
    print(f"summary_path:     {summary_path}")
    if summary.error:
        print(f"error:            {summary.error}")



def _rebuild_generation_config(agent: Any, fallback: Any) -> Any:
    if not hasattr(agent, "_build_generation_config"):
        return fallback
    if getattr(agent, "tokenizer", None) is None:
        return fallback
    try:
        return agent._build_generation_config()
    except Exception as exc:
        logger.warning("Failed to rebuild generation config for locked evaluation: %s", exc)
        return fallback


@contextmanager
def _temporary_deterministic_generation(agent: Any):
    config = getattr(agent, "config", None)
    if config is None:
        yield
        return

    previous = {
        "temperature": getattr(config, "temperature", None),
        "top_p": getattr(config, "top_p", None),
        "top_k": getattr(config, "top_k", None),
        "do_sample": getattr(config, "do_sample", None),
    }
    previous_generation_config = getattr(agent, "generation_config", None)

    config.temperature = 0.0
    if hasattr(config, "top_p"):
        config.top_p = 1.0
    if hasattr(config, "top_k"):
        config.top_k = 1
    if hasattr(config, "do_sample"):
        config.do_sample = False
    agent.generation_config = _rebuild_generation_config(agent, previous_generation_config)

    try:
        yield
    finally:
        for field, value in previous.items():
            if value is not None or hasattr(config, field):
                setattr(config, field, value)
        agent.generation_config = _rebuild_generation_config(agent, previous_generation_config)


def _build_custom_reward_model() -> Any | None:
    if not CUSTOM_REWARD_WEIGHTS:
        return None

    from stateset_agents.rewards.multi_objective_components import (
        ActionOrientedRewardComponent,
        EmpathyRewardComponent,
        LengthRewardComponent,
        ProfessionalismRewardComponent,
        ReasoningRewardComponent,
    )
    from stateset_agents.rewards.multi_objective_reward import MultiObjectiveRewardFunction

    component_specs = [
        ("empathy", EmpathyRewardComponent, {}),
        ("professionalism", ProfessionalismRewardComponent, {}),
        ("action_oriented", ActionOrientedRewardComponent, {}),
        ("reasoning", ReasoningRewardComponent, {}),
        (
            "length",
            LengthRewardComponent,
            {"min_length": 20, "optimal_range": (50, 200)},
        ),
    ]

    components = []
    unsupported = []
    for key in CUSTOM_REWARD_WEIGHTS:
        if key not in {spec[0] for spec in component_specs}:
            unsupported.append(key)
    if unsupported:
        logger.warning("Ignoring unsupported reward weights: %s", ", ".join(sorted(unsupported)))

    for key, component_cls, kwargs in component_specs:
        weight = float(CUSTOM_REWARD_WEIGHTS.get(key, 0.0)) if CUSTOM_REWARD_WEIGHTS else 0.0
        if weight > 0.0:
            components.append(component_cls(weight=weight, **kwargs))

    if not components:
        return None

    logger.info("Using custom multi-objective reward weights: %s", CUSTOM_REWARD_WEIGHTS)
    return MultiObjectiveRewardFunction(components=components)


def _create_reward_model() -> Any:
    custom_reward_model = _build_custom_reward_model()
    if custom_reward_model is not None:
        return custom_reward_model

    from stateset_agents.rewards import create_domain_reward

    return create_domain_reward(REWARD_DOMAIN)


async def main() -> None:
    start_time = time.time()
    device_info = detect_device_info(
        prefer_gpu=PREFER_GPU,
        require_gpu=REQUIRE_GPU,
        requested_bf16=BF16,
    )
    config_snapshot = _config_snapshot(device_info)

    training_seconds: float | None = None
    eval_results: dict[str, float] = {}
    phase_timings: dict[str, float] = {}
    error: Exception | None = None
    status = "error"

    try:
        from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
        from stateset_agents.core.environment import ConversationEnvironment
        from stateset_agents.training.config import TrainingConfig
        from stateset_agents.training.gspo_config import GSPOConfig
        from stateset_agents.training.gspo_entrypoints import train_with_gspo

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        logger.info("Resolved device: %s", _device_label(device_info))
        logger.info("Training budget: %ss", TIME_BUDGET)
        if EXPERIMENT_ID:
            logger.info("Experiment id: %s", EXPERIMENT_ID)
        if SKIP_TRAIN:
            logger.info("Skip-train mode enabled: running evaluation only")

        phase_started = time.time()
        agent_config = AgentConfig(
            model_name=MODEL_NAME,
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            attn_implementation="sdpa",
        )
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()
        phase_timings["agent_initialize_seconds"] = time.time() - phase_started

        phase_started = time.time()
        environment = ConversationEnvironment(
            scenarios=SCENARIOS,
            max_turns=8,
        )
        phase_timings["environment_setup_seconds"] = time.time() - phase_started

        phase_started = time.time()
        reward_model = _create_reward_model()
        phase_timings["reward_setup_seconds"] = time.time() - phase_started

        if SKIP_TRAIN:
            trained_agent = agent
            training_seconds = 0.0
        else:
            base_config = TrainingConfig(
                model_name=MODEL_NAME,
                num_episodes=NUM_EPISODES,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                num_generations=NUM_GENERATIONS,
                learning_rate=LEARNING_RATE,
                warmup_ratio=WARMUP_RATIO,
                lr_scheduler_type=LR_SCHEDULER,
                weight_decay=WEIGHT_DECAY,
                max_grad_norm=MAX_GRAD_NORM,
                clip_ratio=CLIP_RATIO,
                entropy_coef=ENTROPY_COEF,
                beta=BETA,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                advantage_normalization=ADVANTAGE_NORMALIZATION,
                baseline_type=BASELINE_TYPE,
                bf16=device_info.bf16_enabled,
                use_cpu=device_info.use_cpu,
                gradient_checkpointing=GRADIENT_CHECKPOINTING,
                output_dir=OUTPUT_DIR,
                report_to="none",
                run_post_eval=RUN_POST_EVAL,
                post_eval_samples=POST_EVAL_SAMPLES,
                post_eval_detailed=POST_EVAL_DETAILED,
            )
            Path(base_config.output_dir).mkdir(parents=True, exist_ok=True)

            gspo_config = GSPOConfig.from_training_config(
                base_config,
                num_outer_iterations=NUM_OUTER_ITERATIONS,
                generations_per_iteration=GENERATIONS_PER_ITERATION,
                clip_range_left=CLIP_RANGE_LEFT,
                clip_range_right=CLIP_RANGE_RIGHT,
                use_lora=USE_LORA,
                lora_r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                use_4bit=USE_4BIT,
                use_8bit=USE_8BIT,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_completion_length=MAX_NEW_TOKENS,
            )

            logger.info("Starting GSPO training...")
            train_started = time.time()
            trained_agent = await asyncio.wait_for(
                train_with_gspo(
                    config=gspo_config,
                    agent=agent,
                    environment=environment,
                    reward_model=reward_model,
                ),
                timeout=TIME_BUDGET,
            )
            training_seconds = time.time() - train_started
            phase_timings["training_seconds"] = training_seconds

        logger.info("Evaluating training reward on held-out scenarios...")
        reward_eval_started = time.time()
        eval_results = await evaluate_trained_agent(
            trained_agent,
            reward_model,
            scenarios=EVAL_SCENARIO_SET,
            num_episodes=EVAL_EPISODES_VALUE,
        )
        reward_eval_seconds = time.time() - reward_eval_started
        phase_timings["reward_evaluation_seconds"] = reward_eval_seconds

        logger.info("Running locked deterministic selection evaluation...")
        selection_eval_started = time.time()
        with _temporary_deterministic_generation(trained_agent):
            selection_results = await evaluate_selection_metric(
                trained_agent,
                scenarios=EVAL_SCENARIO_SET,
                num_episodes=EVAL_EPISODES_VALUE,
            )
        selection_eval_seconds = time.time() - selection_eval_started
        phase_timings["selection_evaluation_seconds"] = selection_eval_seconds
        phase_timings["evaluation_seconds"] = reward_eval_seconds + selection_eval_seconds
        eval_results.update(selection_results)
        status = "ok"
    except asyncio.TimeoutError as exc:
        status = "timeout"
        training_seconds = time.time() - start_time
        error = RuntimeError(f"Training exceeded TIME_BUDGET={TIME_BUDGET}s")
        error.__cause__ = exc
    except Exception as exc:
        status = "error"
        error = exc
    finally:
        total_seconds = time.time() - start_time
        phase_timings.setdefault(
            "startup_seconds",
            max(0.0, total_seconds - sum(phase_timings.values())),
        )
        peak_vram = get_peak_vram_mb()
        summary = RunSummary(
            status=status,
            started_at=start_time,
            completed_at=time.time(),
            time_budget_seconds=TIME_BUDGET,
            accelerator=device_info.accelerator,
            cuda_available=device_info.cuda_available,
            device_name=device_info.device_name,
            torch_version=device_info.torch_version,
            bf16_enabled=device_info.bf16_enabled,
            training_seconds=training_seconds,
            total_seconds=total_seconds,
            peak_vram_mb=peak_vram,
            metrics=eval_results,
            config=config_snapshot,
            phase_timings=phase_timings,
            error=str(error) if error else None,
        )
        summary_path = write_run_summary(RUN_SUMMARY_PATH, summary)
        _emit_run_summary(summary, summary_path)

    if error is not None:
        raise error


if __name__ == "__main__":
    asyncio.run(main())
