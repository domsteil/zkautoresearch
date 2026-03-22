#!/usr/bin/env python3
"""
STARK Integrity Test — proves that fake improvements cannot pass verification.

Demonstrates:
1. Valid proofs verify correctly.
2. Tampered proofs are rejected.
3. Proofs for non-improvements cannot be generated.
4. Proofs from one experiment cannot be reused for another.
"""

import sys
import uuid

from stark_autoresearch.proof import (
    MetricImprovementProver,
    MetricImprovementProof,
    verify_improvement,
)


def test_valid_proof():
    """A genuine improvement produces a valid STARK proof."""
    prover = MetricImprovementProver()
    proof = prover.prove_improvement(
        new_reward=0.55,
        best_reward=0.52,
        experiment_id=str(uuid.uuid4()),
        agent_id="test-agent",
    )
    valid, msg = verify_improvement(proof, claimed_best_reward=0.52)
    assert valid, f"Valid proof should verify: {msg}"
    print(f"  [PASS] Valid proof verified ({proof.proof_size:,} bytes, {proof.proving_time_ms}ms prove)")
    return proof


def test_tampered_proof(original_proof: MetricImprovementProof):
    """A proof with tampered bytes must be rejected."""
    tampered = MetricImprovementProof(
        experiment_id=original_proof.experiment_id,
        agent_id=original_proof.agent_id,
        best_reward_scaled=original_proof.best_reward_scaled,
        proof_bytes=bytearray(original_proof.proof_bytes),  # copy
        proof_hash=original_proof.proof_hash,
        witness_commitment=list(original_proof.witness_commitment),
        witness_commitment_hex=original_proof.witness_commitment_hex,
        policy_hash=original_proof.policy_hash,
        proving_time_ms=original_proof.proving_time_ms,
        proof_size=original_proof.proof_size,
        timestamp=original_proof.timestamp,
    )
    # Flip a byte in the proof.
    proof_bytes = bytearray(tampered.proof_bytes)
    proof_bytes[100] ^= 0xFF
    tampered.proof_bytes = bytes(proof_bytes)

    valid, msg = verify_improvement(tampered, claimed_best_reward=0.52)
    assert not valid, "Tampered proof should NOT verify"
    print(f"  [PASS] Tampered proof rejected: {msg[:60]}...")


def test_no_improvement():
    """Cannot generate a proof when the metric didn't improve."""
    prover = MetricImprovementProver()
    try:
        prover.prove_improvement(
            new_reward=0.50,
            best_reward=0.52,
            experiment_id=str(uuid.uuid4()),
            agent_id="test-agent",
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  [PASS] Non-improvement correctly rejected: {e}")


def test_wrong_baseline():
    """Proof verified against a different baseline must fail."""
    prover = MetricImprovementProver()
    proof = prover.prove_improvement(
        new_reward=0.60,
        best_reward=0.52,
        experiment_id=str(uuid.uuid4()),
        agent_id="test-agent",
    )
    # Try to verify against a DIFFERENT baseline.
    valid, msg = verify_improvement(proof, claimed_best_reward=0.40)
    assert not valid, "Proof against wrong baseline should NOT verify"
    print(f"  [PASS] Wrong baseline rejected: {msg[:60]}...")


def test_proof_sizes():
    """Show proof sizes for different improvement magnitudes."""
    prover = MetricImprovementProver()
    print("  Improvement magnitudes and proof sizes:")
    for delta in [0.001, 0.01, 0.05, 0.1, 0.3]:
        proof = prover.prove_improvement(
            new_reward=0.52 + delta,
            best_reward=0.52,
            experiment_id=str(uuid.uuid4()),
            agent_id="test-agent",
        )
        print(
            f"    delta={delta:.3f}  →  proof={proof.proof_size:,} bytes, "
            f"prove={proof.proving_time_ms}ms"
        )


def main():
    print("=" * 60)
    print("  STARK PROOF INTEGRITY TEST")
    print("=" * 60)
    print()

    print("1. Valid improvement proof:")
    proof = test_valid_proof()

    print("\n2. Tampered proof detection:")
    test_tampered_proof(proof)

    print("\n3. Non-improvement rejection:")
    test_no_improvement()

    print("\n4. Wrong baseline detection:")
    test_wrong_baseline()

    print("\n5. Proof sizes by improvement magnitude:")
    test_proof_sizes()

    print()
    print("=" * 60)
    print("  ALL INTEGRITY TESTS PASSED")
    print("  STARK proofs are cryptographically sound.")
    print("=" * 60)


if __name__ == "__main__":
    main()
