"""
Fixed constants, evaluation harness, and data definitions for stateset-agents
autonomous research. DO NOT MODIFY this file — the agent only edits train.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from statistics import fmean, pstdev
from typing import Any

from experiment_runtime import detect_device_info

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare")

# ─── Fixed constants ───────────────────────────────────────────────────────────

TIME_BUDGET = int(os.getenv('AUTORESEARCH_TIME_BUDGET', '300'))
EVAL_EPISODES = int(os.getenv('AUTORESEARCH_EVAL_EPISODES', '10'))
EVAL_SEED = int(os.getenv('AUTORESEARCH_EVAL_SEED', '42'))
LOCKED_SELECTION_METRIC_NAME = 'selection_score'
SELECTION_BOOTSTRAP_SAMPLES = int(os.getenv('AUTORESEARCH_SELECTION_BOOTSTRAP_SAMPLES', '128'))

# ─── Evaluation scenarios (the "validation set") ──────────────────────────────
# These are FIXED and NOT modified by the agent. They cover the core customer
# service domain so experiments are directly comparable.

EVAL_SCENARIOS: list[dict[str, Any]] = [
    {
        "topic": "order_status",
        "user_goal": "Check the status of an order",
        "context": "Customer placed an order 3 days ago and wants an update.",
        "user_responses": [
            "Hi, I placed an order on Monday and haven't received any updates.",
            "The order number is #12345.",
            "Can you tell me when it will arrive?",
        ],
        "reference_response": (
            "I can help check your order status. Please confirm the order number so I can "
            "review the tracking details and estimated delivery date."
        ),
        "required_concepts": [
            ["order number", "order #", "order"],
            ["tracking", "status", "shipment"],
            ["delivery", "arrive", "estimated"],
        ],
    },
    {
        "topic": "product_return",
        "user_goal": "Return a defective product",
        "context": "Customer received a damaged item and wants to return it.",
        "user_responses": [
            "I received my order but the item is broken.",
            "It's a wireless keyboard, the spacebar doesn't work.",
            "I'd like a refund please.",
        ],
        "reference_response": (
            "I'm sorry the keyboard arrived damaged. I can help start a return or refund "
            "and share the next steps for sending it back."
        ),
        "required_concepts": [
            ["sorry", "apologize"],
            ["return", "refund"],
            ["next steps", "instructions", "label"],
        ],
    },
    {
        "topic": "billing_dispute",
        "user_goal": "Resolve a billing discrepancy",
        "context": "Customer was charged twice for the same order.",
        "user_responses": [
            "I was charged twice for order #67890.",
            "Yes I can see both charges on my credit card statement.",
            "When will the refund appear?",
        ],
        "reference_response": (
            "I can help review the duplicate charge for your order and start the refund "
            "process, including what timeline to expect."
        ),
        "required_concepts": [
            ["charged twice", "duplicate charge", "double charge"],
            ["refund", "credit"],
            ["timeline", "business days", "when"],
        ],
    },
    {
        "topic": "account_help",
        "user_goal": "Reset account password",
        "context": "Customer is locked out of their account.",
        "user_responses": [
            "I can't log into my account.",
            "I've tried the forgot password link but I'm not getting the email.",
            "My email is user@example.com.",
        ],
        "reference_response": (
            "I can help with the password reset. Let's verify the email address and try the "
            "reset flow again or use a support fallback if the email is not arriving."
        ),
        "required_concepts": [
            ["password", "reset", "login"],
            ["email", "inbox"],
            ["support", "fallback", "next step"],
        ],
    },
    {
        "topic": "product_inquiry",
        "user_goal": "Learn about product features before purchasing",
        "context": "Customer is comparing products and needs guidance.",
        "user_responses": [
            "What's the difference between the Pro and Standard plans?",
            "Does the Pro plan include priority support?",
            "I think I'll go with Pro. How do I sign up?",
        ],
        "reference_response": (
            "I can explain the difference between the Pro and Standard plans, confirm whether "
            "priority support is included, and show how to sign up for Pro."
        ),
        "required_concepts": [
            ["pro", "standard"],
            ["priority support", "support"],
            ["sign up", "upgrade", "purchase"],
        ],
    },
    {
        "topic": "shipping_issue",
        "user_goal": "Resolve a shipping problem",
        "context": "Package was marked delivered but customer didn't receive it.",
        "user_responses": [
            "My package says delivered but I never got it.",
            "I've checked with my neighbors and the front desk.",
            "Can you send a replacement?",
        ],
        "reference_response": (
            "I'm sorry the package shows delivered but has not arrived. I can help open an "
            "investigation with the carrier and review replacement or refund options."
        ),
        "required_concepts": [
            ["sorry", "apologize"],
            ["delivered", "carrier", "investigation", "trace"],
            ["replacement", "refund", "reship"],
        ],
    },
    {
        "topic": "subscription_cancel",
        "user_goal": "Cancel a subscription",
        "context": "Customer wants to cancel their monthly subscription.",
        "user_responses": [
            "I'd like to cancel my subscription.",
            "I'm just not using the service enough to justify the cost.",
            "No thanks, I'd just like to cancel.",
        ],
        "reference_response": (
            "I can help cancel the subscription, confirm when the cancellation takes effect, "
            "and explain any final billing details."
        ),
        "required_concepts": [
            ["cancel", "cancellation"],
            ["billing", "charged", "final bill"],
            ["effective", "takes effect", "confirm"],
        ],
    },
    {
        "topic": "technical_support",
        "user_goal": "Fix an app that keeps crashing",
        "context": "Customer's mobile app crashes on startup.",
        "user_responses": [
            "The app keeps crashing when I open it.",
            "I'm on Android 14, Samsung Galaxy S24.",
            "I already tried reinstalling it.",
        ],
        "reference_response": (
            "I can help troubleshoot the app crash on Android, confirm the device details, "
            "and walk through the next diagnostic or escalation steps."
        ),
        "required_concepts": [
            ["crash", "crashing"],
            ["android", "device", "samsung"],
            ["troubleshoot", "diagnostic", "update", "escalate"],
        ],
    },
]

# Separate training scenarios for the agent to train on (distinct from eval).
TRAIN_SCENARIOS: list[dict[str, Any]] = [
    {
        "topic": "general_inquiry",
        "user_goal": "Get help with a general question",
        "context": "Customer has a general question about services.",
        "user_responses": [
            "Hi, I have a question about your services.",
            "What are your business hours?",
            "Thanks for the help!",
        ],
    },
    {
        "topic": "feedback",
        "user_goal": "Provide feedback on a recent experience",
        "context": "Customer wants to share feedback about a recent interaction.",
        "user_responses": [
            "I wanted to give feedback about my recent order.",
            "The delivery was faster than expected, great job!",
            "Keep up the good work.",
        ],
    },
    {
        "topic": "upgrade_request",
        "user_goal": "Upgrade their plan",
        "context": "Customer wants to upgrade from Basic to Premium.",
        "user_responses": [
            "I'd like to upgrade my plan.",
            "I'm currently on Basic and want Premium.",
            "Will my existing data carry over?",
        ],
    },
    {
        "topic": "complaint",
        "user_goal": "File a complaint about poor service",
        "context": "Customer had a bad experience with a previous agent.",
        "user_responses": [
            "I need to file a complaint.",
            "The agent I spoke with yesterday was very unhelpful.",
            "I want this escalated to a manager.",
        ],
    },
    {
        "topic": "warranty_claim",
        "user_goal": "Make a warranty claim",
        "context": "Customer's product broke within the warranty period.",
        "user_responses": [
            "My product stopped working and it's still under warranty.",
            "I bought it 6 months ago.",
            "I have the receipt and warranty card.",
        ],
    },
    {
        "topic": "new_feature",
        "user_goal": "Ask about a new feature",
        "context": "Customer heard about a new feature and wants details.",
        "user_responses": [
            "I heard you launched a new feature recently.",
            "How does the AI assistant integration work?",
            "Is it available on my current plan?",
        ],
    },
]


def _assistant_transcript(turns: list[Any]) -> str:
    parts: list[str] = []
    for turn in turns:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content", "")
        else:
            role = getattr(turn, "role", None)
            content = getattr(turn, "content", "")
        if role == "assistant" and content:
            parts.append(str(content))
    return " ".join(parts).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = set(_tokenize(candidate))
    reference_tokens = set(_tokenize(reference))
    if not candidate_tokens or not reference_tokens:
        return 0.0

    overlap = len(candidate_tokens & reference_tokens)
    if overlap == 0:
        return 0.0

    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)


def _concept_coverage(candidate: str, required_concepts: list[list[str]] | None) -> float:
    if not required_concepts:
        return 1.0

    lowered = candidate.lower()
    satisfied = 0
    for group in required_concepts:
        terms = [str(term).lower() for term in group]
        if any(term in lowered for term in terms):
            satisfied += 1
    return satisfied / len(required_concepts)


def _selection_topic_metric_name(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return f"selection_topic_{slug or 'unknown'}"


def _bootstrap_selection_means(scores: list[float]) -> list[float]:
    if not scores:
        return [0.0]

    rng = random.Random(EVAL_SEED)
    count = len(scores)
    means: list[float] = []
    for _ in range(max(1, SELECTION_BOOTSTRAP_SAMPLES)):
        resampled = [scores[rng.randrange(count)] for _ in range(count)]
        means.append(float(fmean(resampled)))
    return means


class FixedReferenceReward:
    """Locked evaluation reward used for keep/discard decisions."""

    async def compute_reward(
        self,
        turns: list[Any],
        context: dict[str, Any] | None = None,
    ) -> float:
        scenario = context or {}
        transcript = _assistant_transcript(turns)
        if not transcript:
            return 0.0

        coverage = _concept_coverage(
            transcript,
            scenario.get("required_concepts") or [],
        )
        similarity = _token_f1(
            transcript,
            str(scenario.get("reference_response") or ""),
        )
        score = 0.7 * coverage + 0.3 * similarity
        return max(0.0, min(1.0, score))


# ─── Evaluation harness ───────────────────────────────────────────────────────

async def evaluate_trained_agent(
    agent: Any,
    reward_model: Any,
    *,
    scenarios: list[dict[str, Any]] | None = None,
    num_episodes: int = EVAL_EPISODES,
) -> dict[str, float]:
    """
    Evaluate an agent on fixed scenarios with the training reward model.
    This is diagnostic only; model promotion uses evaluate_selection_metric().
    """
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.training.evaluation import EvaluationConfig, evaluate_agent

    eval_scenarios = scenarios or EVAL_SCENARIOS

    environment = ConversationEnvironment(
        scenarios=eval_scenarios,
        max_turns=8,
    )

    eval_config = EvaluationConfig(
        num_episodes=num_episodes,
        num_generations=1,
        seed=EVAL_SEED,
        concurrency=1,
    )

    results = await evaluate_agent(
        agent=agent,
        environment=environment,
        reward_fn=reward_model,
        config=eval_config,
    )

    return results


async def evaluate_selection_metric(
    agent: Any,
    *,
    scenarios: list[dict[str, Any]] | None = None,
    num_episodes: int = EVAL_EPISODES,
) -> dict[str, float]:
    """Run the locked held-out evaluation used for keep/discard decisions."""
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.training.evaluation import EvaluationConfig, evaluate_agent

    eval_scenarios = list(scenarios or EVAL_SCENARIOS)
    reward_fn = FixedReferenceReward()
    per_topic_scores: dict[str, float] = {}
    per_topic_lengths: dict[str, float] = {}
    topic_scores: list[float] = []
    topic_lengths: list[float] = []

    for index, scenario in enumerate(eval_scenarios):
        environment = ConversationEnvironment(
            scenarios=[scenario],
            max_turns=8,
        )
        eval_config = EvaluationConfig(
            num_episodes=1,
            num_generations=1,
            seed=EVAL_SEED + index,
            concurrency=1,
        )
        results = await evaluate_agent(
            agent=agent,
            environment=environment,
            reward_fn=reward_fn,
            config=eval_config,
        )
        topic_name = _selection_topic_metric_name(str(scenario.get("topic") or index))
        topic_score = float(results["eval_reward"])
        topic_length = float(results["eval_episode_length"])
        per_topic_scores[topic_name] = topic_score
        per_topic_lengths[f"{topic_name}_length"] = topic_length
        topic_scores.append(topic_score)
        topic_lengths.append(topic_length)

    selection_score = float(fmean(topic_scores)) if topic_scores else 0.0
    selection_std = float(pstdev(topic_scores)) if len(topic_scores) > 1 else 0.0
    bootstrap_means = _bootstrap_selection_means(topic_scores)
    bootstrap_std = float(pstdev(bootstrap_means)) if len(bootstrap_means) > 1 else 0.0
    selection_lower_bound = max(0.0, selection_score - bootstrap_std)
    selection_upper_bound = min(1.0, selection_score + bootstrap_std)

    metrics = {
        LOCKED_SELECTION_METRIC_NAME: selection_score,
        "selection_score_std": selection_std,
        "selection_score_bootstrap_std": bootstrap_std,
        "selection_score_lower_bound": selection_lower_bound,
        "selection_score_upper_bound": selection_upper_bound,
        "selection_score_worst_topic": min(topic_scores) if topic_scores else 0.0,
        "selection_episode_length": float(fmean(topic_lengths)) if topic_lengths else 0.0,
        "selection_success_rate": float(sum(score > 0.5 for score in topic_scores) / len(topic_scores)) if topic_scores else 0.0,
        "selection_num_episodes": float(len(topic_scores)),
        "selection_topic_count": float(len(topic_scores)),
    }
    metrics.update(per_topic_scores)
    metrics.update(per_topic_lengths)
    return metrics


def get_peak_vram_mb() -> float:
    """Return peak GPU memory usage in MB, or 0 if not available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ─── Setup verification ──────────────────────────────────────────────────────

def verify_setup() -> bool:
    """Verify that the environment is correctly set up."""
    ok = True
    device_info = detect_device_info(prefer_gpu=True, require_gpu=False, requested_bf16=False)
    logger.info("Preferred accelerator: %s", device_info.device_name or device_info.accelerator)

    # Check stateset-agents is importable
    try:
        import stateset_agents
        logger.info(f"stateset-agents {stateset_agents.__version__} found")
    except ImportError:
        logger.error("stateset-agents not installed. Run: uv sync")
        ok = False

    # Check torch
    try:
        import torch
        logger.info(f"torch {torch.__version__} found, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available — experiments will run on CPU unless a GPU is attached.")
    except ImportError:
        logger.error("torch not installed. Run: uv sync")
        ok = False

    # Check transformers
    try:
        import transformers
        logger.info(f"transformers {transformers.__version__} found")
    except ImportError:
        logger.error("transformers not installed. Run: uv sync")
        ok = False

    # Check peft
    try:
        import peft
        logger.info(f"peft {peft.__version__} found")
    except ImportError:
        logger.warning("peft not installed — LoRA will not be available")

    if ok:
        logger.info("Setup OK — ready to run experiments")
    return ok


if __name__ == "__main__":
    verify_setup()
