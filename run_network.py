#!/usr/bin/env python3
"""
Decentralized Autoresearch Network — Demo
==========================================

Runs multiple autoresearch agents in parallel, each generating STARK proofs
for every metric improvement. A network coordinator verifies proofs
cryptographically (no re-running experiments) and broadcasts improvements.

Usage:
    python run_network.py                          # 4 agents, 20 experiments each
    python run_network.py --agents 8 --experiments 50
    python run_network.py --initial-best 0.523     # Start from current best
    python run_network.py --strategy adaptive      # All agents use adaptive strategy
    python run_network.py --seed 42                # Reproducible search trajectory
    python run_network.py --export results.json    # Export results
"""

import argparse
import asyncio
import logging
import sys
import time

from stark_autoresearch.agent import AutoResearchAgent
from stark_autoresearch.coordinator import NetworkCoordinator

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
# Quiet down debug noise from agents during the run.
logging.getLogger("stark_autoresearch.agent").setLevel(logging.INFO)


async def run_network(
    num_agents: int = 4,
    experiments_per_agent: int = 20,
    initial_best: float = 0.0,
    strategies: list[str] | None = None,
    export_path: str | None = None,
    seed: int | None = None,
) -> tuple[NetworkCoordinator, list[AutoResearchAgent]]:
    """Run the decentralized STARK-verified autoresearch network."""

    print("=" * 70)
    print("  DECENTRALIZED AUTORESEARCH NETWORK")
    print("  with STARK Cryptographic Verification")
    print("=" * 70)
    print(f"  Agents:              {num_agents}")
    print(f"  Experiments/agent:   {experiments_per_agent}")
    print(f"  Initial best reward: {initial_best:.6f}")
    print(f"  Total experiments:   {num_agents * experiments_per_agent}")
    if seed is not None:
        print(f"  Seed:                {seed}")
    print()

    # ── Create coordinator ──
    coordinator = NetworkCoordinator(initial_best_reward=initial_best)

    # ── Create agents with diverse strategies ──
    if strategies is None:
        # Mix of strategies for diversity.
        strategy_pool = ["adaptive", "random", "exploit", "adaptive"]
        strategies = [strategy_pool[i % len(strategy_pool)] for i in range(num_agents)]

    agents = []
    for i in range(num_agents):
        agent = AutoResearchAgent(
            agent_id=f"agent-{i:02d}",
            strategy=strategies[i],
            seed=seed,
        )
        coordinator.register_agent(agent)
        agents.append(agent)
        print(f"  Registered {agent.agent_id} (strategy={strategies[i]})")

    print()
    print("─" * 70)
    print("  Starting parallel experiment loop...")
    print("  Each improvement generates a STARK proof (proving metric > best)")
    print("  Coordinator verifies proofs cryptographically — no re-running!")
    print("─" * 70)
    print()

    start_time = time.time()
    stop_event = asyncio.Event()

    # ── Run all agents in parallel ──
    async def agent_loop(agent: AutoResearchAgent) -> None:
        await agent.run_loop(
            submit_fn=coordinator.submit,
            max_experiments=experiments_per_agent,
            stop_event=stop_event,
        )

    await asyncio.gather(*[agent_loop(a) for a in agents])

    elapsed = time.time() - start_time

    # ── Results ──
    coordinator.print_summary()

    print(f"\n  Wall-clock time: {elapsed:.1f}s")
    print(f"  Throughput: {num_agents * experiments_per_agent / elapsed:.1f} experiments/sec")

    # Per-agent stats.
    print(f"\n  {'─' * 66}")
    print("  PER-AGENT STATISTICS")
    print(f"  {'─' * 66}")
    for agent in agents:
        n_total = len(agent.memory.experiments)
        n_improved = len(agent.memory.improvements)
        print(
            f"  {agent.agent_id} ({agent.strategy:>8s}): "
            f"{n_total} experiments, {n_improved} improvements "
            f"({n_improved/max(1,n_total):.0%} hit rate), "
            f"best={agent.best_reward:.6f}"
        )

    # ── Show STARK proof details for the last verified improvement ──
    if coordinator.verified_experiments:
        last = coordinator.verified_experiments[-1]
        print(f"\n  {'─' * 66}")
        print("  EXAMPLE STARK PROOF (last verified improvement)")
        print(f"  {'─' * 66}")
        print(f"  Experiment ID:       {last.result.experiment_id}")
        print(f"  Agent:               {last.result.agent_id}")
        print(f"  Reward:              {last.result.avg_reward:.6f}")
        print(f"  Proof size:          {last.proof.proof_size:,} bytes")
        print(f"  Proving time:        {last.proof.proving_time_ms} ms")
        print(f"  Verification time:   {last.verification_time_ms:.1f} ms")
        print(f"  Witness commitment:  {last.proof.witness_commitment_hex[:32]}...")
        print(f"  Policy hash:         {last.proof.policy_hash[:32]}...")
        print(f"  Proof hash:          {last.proof.proof_hash[:32]}...")

    if export_path:
        coordinator.export_results(export_path)
        print(f"\n  Results exported to {export_path}")

    print()
    return coordinator, agents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decentralized Autoresearch Network with STARK Proofs",
    )
    parser.add_argument(
        "--agents", type=int, default=4,
        help="Number of parallel agents (default: 4)",
    )
    parser.add_argument(
        "--experiments", type=int, default=20,
        help="Experiments per agent (default: 20)",
    )
    parser.add_argument(
        "--initial-best", type=float, default=0.0,
        help="Starting best reward (default: 0.0, use 0.523 for current best)",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=["adaptive", "random", "exploit"],
        help="Force all agents to use this strategy (default: mixed)",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for reproducible experiment proposals and rewards",
    )

    args = parser.parse_args()

    strategies = None
    if args.strategy:
        strategies = [args.strategy] * args.agents

    asyncio.run(run_network(
        num_agents=args.agents,
        experiments_per_agent=args.experiments,
        initial_best=args.initial_best,
        strategies=strategies,
        export_path=args.export,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
