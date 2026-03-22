#!/usr/bin/env python3
"""
STARK integrity tests for metric-improvement proofs.

Covers:
1. Valid proofs verify correctly.
2. Tampered proofs are rejected.
3. Non-improvements cannot generate proofs.
4. Wrong baselines or claimed rewards are rejected.
5. Multiple proofs from the same prover verify correctly.
6. The coordinator rejects mismatched rewards and configs.
"""

import asyncio
import logging
import uuid

import pytest

pytest.importorskip("ves_stark")

from stark_autoresearch.coordinator import NetworkCoordinator
from stark_autoresearch.experiment import ExperimentConfig, ExperimentResult
from stark_autoresearch.proof import (
    MetricImprovementProof,
    MetricImprovementProver,
    verify_improvement,
)


def _baseline_config() -> ExperimentConfig:
    return ExperimentConfig.from_baseline()


def _make_proof(
    new_reward: float = 0.55,
    best_reward: float = 0.52,
    config: ExperimentConfig | None = None,
    prover: MetricImprovementProver | None = None,
    agent_id: str = "test-agent",
) -> MetricImprovementProof:
    config = config or _baseline_config()
    prover = prover or MetricImprovementProver()
    return prover.prove_improvement(
        new_reward=new_reward,
        best_reward=best_reward,
        experiment_id=str(uuid.uuid4()),
        agent_id=agent_id,
        config_hash=config.content_hash(),
    )


def test_valid_proof() -> None:
    """A genuine improvement produces a valid STARK proof."""
    config = _baseline_config()
    proof = _make_proof(config=config)
    valid, msg = verify_improvement(
        proof,
        claimed_best_reward=0.52,
        claimed_new_reward=0.55,
        config_hash=config.content_hash(),
    )
    assert valid, f"Valid proof should verify: {msg}"



def test_tampered_proof() -> None:
    """A proof with tampered bytes must be rejected."""
    config = _baseline_config()
    original_proof = _make_proof(config=config)
    tampered = MetricImprovementProof(
        experiment_id=original_proof.experiment_id,
        agent_id=original_proof.agent_id,
        sequence_number=original_proof.sequence_number,
        config_hash=original_proof.config_hash,
        best_reward_scaled=original_proof.best_reward_scaled,
        new_reward_scaled=original_proof.new_reward_scaled,
        improvement_delta_scaled=original_proof.improvement_delta_scaled,
        proof_bytes=bytearray(original_proof.proof_bytes),
        proof_hash=original_proof.proof_hash,
        witness_commitment=list(original_proof.witness_commitment),
        witness_commitment_hex=original_proof.witness_commitment_hex,
        policy_hash=original_proof.policy_hash,
        amount_binding_hash=original_proof.amount_binding_hash,
        proving_time_ms=original_proof.proving_time_ms,
        proof_size=original_proof.proof_size,
        timestamp=original_proof.timestamp,
    )
    proof_bytes = bytearray(tampered.proof_bytes)
    proof_bytes[100] ^= 0xFF
    tampered.proof_bytes = bytes(proof_bytes)

    valid, msg = verify_improvement(
        tampered,
        claimed_best_reward=0.52,
        claimed_new_reward=0.55,
        config_hash=config.content_hash(),
    )
    assert not valid, "Tampered proof should NOT verify"
    assert "mismatch" in msg.lower() or "failed" in msg.lower() or "error" in msg.lower()



def test_no_improvement() -> None:
    """Cannot generate a proof when the metric did not improve."""
    config = _baseline_config()
    prover = MetricImprovementProver()
    try:
        prover.prove_improvement(
            new_reward=0.50,
            best_reward=0.52,
            experiment_id=str(uuid.uuid4()),
            agent_id="test-agent",
            config_hash=config.content_hash(),
        )
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "Cannot prove improvement" in str(exc)



def test_wrong_baseline() -> None:
    """Proofs verified against a different baseline must fail."""
    config = _baseline_config()
    proof = _make_proof(new_reward=0.60, best_reward=0.52, config=config)
    valid, msg = verify_improvement(
        proof,
        claimed_best_reward=0.40,
        claimed_new_reward=0.60,
        config_hash=config.content_hash(),
    )
    assert not valid, "Proof against wrong baseline should NOT verify"
    assert "Best-reward mismatch" in msg



def test_wrong_claimed_reward() -> None:
    """Proofs cannot be rebound to a larger claimed reward."""
    config = _baseline_config()
    proof = _make_proof(new_reward=0.520001, best_reward=0.52, config=config)
    valid, msg = verify_improvement(
        proof,
        claimed_best_reward=0.52,
        claimed_new_reward=0.99,
        config_hash=config.content_hash(),
    )
    assert not valid, "Proof against a different claimed reward should NOT verify"
    assert "New-reward mismatch" in msg



def test_multiple_proofs_from_same_prover_verify() -> None:
    """Later proofs from the same prover must verify as well."""
    config = _baseline_config()
    prover = MetricImprovementProver()
    proof1 = _make_proof(new_reward=0.55, best_reward=0.52, config=config, prover=prover)
    proof2 = _make_proof(new_reward=0.60, best_reward=0.55, config=config, prover=prover)

    valid1, msg1 = verify_improvement(
        proof1,
        claimed_best_reward=0.52,
        claimed_new_reward=0.55,
        config_hash=config.content_hash(),
    )
    valid2, msg2 = verify_improvement(
        proof2,
        claimed_best_reward=0.55,
        claimed_new_reward=0.60,
        config_hash=config.content_hash(),
    )

    assert valid1, f"First proof should verify: {msg1}"
    assert valid2, f"Second proof should verify: {msg2}"
    assert proof2.sequence_number == proof1.sequence_number + 1



def test_coordinator_rejects_mismatched_reward() -> None:
    """The coordinator must reject submissions whose reward does not match the proof."""
    config = _baseline_config()
    proof = _make_proof(new_reward=0.520001, best_reward=0.52, config=config, agent_id="attacker")

    async def scenario() -> tuple[bool, float]:
        coordinator = NetworkCoordinator(initial_best_reward=0.52)
        fake_result = ExperimentResult(
            experiment_id=proof.experiment_id,
            agent_id="attacker",
            config=config,
            avg_reward=0.99,
        )
        accepted = await coordinator.submit(fake_result, proof)
        return accepted, coordinator.best_reward

    accepted, best_reward = asyncio.run(scenario())
    assert not accepted
    assert best_reward == 0.52



def test_coordinator_rejects_config_mismatch() -> None:
    """The coordinator must reject a proof paired with a different config."""
    good_config = _baseline_config()
    bad_params = dict(good_config.params)
    bad_params["temperature"] = 0.9
    bad_config = ExperimentConfig(params=bad_params, description="tampered")
    proof = _make_proof(new_reward=0.55, best_reward=0.52, config=good_config)

    async def scenario() -> bool:
        coordinator = NetworkCoordinator(initial_best_reward=0.52)
        fake_result = ExperimentResult(
            experiment_id=proof.experiment_id,
            agent_id=proof.agent_id,
            config=bad_config,
            avg_reward=0.55,
        )
        return await coordinator.submit(fake_result, proof)

    accepted = asyncio.run(scenario())
    assert not accepted



def test_proof_sizes() -> None:
    """Show proof sizes for different improvement magnitudes."""
    config = _baseline_config()
    prover = MetricImprovementProver()
    for delta in [0.001, 0.01, 0.05, 0.1, 0.3]:
        proof = prover.prove_improvement(
            new_reward=0.52 + delta,
            best_reward=0.52,
            experiment_id=str(uuid.uuid4()),
            agent_id="test-agent",
            config_hash=config.content_hash(),
        )
        assert proof.proof_size > 0
        assert proof.improvement_delta_scaled > 0



def main() -> None:
    logging.getLogger("stark_autoresearch.coordinator").setLevel(logging.ERROR)
    config = _baseline_config()
    print("=" * 60)
    print("  STARK PROOF INTEGRITY TEST")
    print("=" * 60)
    print()

    print("1. Valid improvement proof:")
    proof = _make_proof(config=config)
    valid, msg = verify_improvement(
        proof,
        claimed_best_reward=0.52,
        claimed_new_reward=0.55,
        config_hash=config.content_hash(),
    )
    assert valid, msg
    print(f"  [PASS] Valid proof verified ({proof.proof_size:,} bytes, {proof.proving_time_ms}ms prove)")

    print("\n2. Tampered proof detection:")
    test_tampered_proof()
    print("  [PASS] Tampered proof rejected")

    print("\n3. Non-improvement rejection:")
    test_no_improvement()
    print("  [PASS] Non-improvement correctly rejected")

    print("\n4. Wrong baseline detection:")
    test_wrong_baseline()
    print("  [PASS] Wrong baseline rejected")

    print("\n5. Wrong claimed reward detection:")
    test_wrong_claimed_reward()
    print("  [PASS] Wrong claimed reward rejected")

    print("\n6. Multiple proofs from one prover:")
    test_multiple_proofs_from_same_prover_verify()
    print("  [PASS] Later proofs verify correctly")

    print("\n7. Coordinator-side submission checks:")
    test_coordinator_rejects_mismatched_reward()
    test_coordinator_rejects_config_mismatch()
    print("  [PASS] Coordinator rejected mismatched reward/config")

    print("\n8. Proof sizes by improvement magnitude:")
    print("  Improvement magnitudes and proof sizes:")
    prover = MetricImprovementProver()
    for delta in [0.001, 0.01, 0.05, 0.1, 0.3]:
        proof = prover.prove_improvement(
            new_reward=0.52 + delta,
            best_reward=0.52,
            experiment_id=str(uuid.uuid4()),
            agent_id="test-agent",
            config_hash=config.content_hash(),
        )
        print(
            f"    delta={delta:.3f}  ->  proof={proof.proof_size:,} bytes, "
            f"prove={proof.proving_time_ms}ms"
        )

    print()
    print("=" * 60)
    print("  ALL INTEGRITY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
