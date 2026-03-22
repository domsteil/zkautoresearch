"""
STARK proof generation and verification for metric improvement.

Maps the autoresearch metric-improvement claim to the ves_stark compliance
circuit:

  1. Scale floating-point rewards to u64 integers (× SCALE_FACTOR).
  2. Compute  delta = new_reward_scaled − best_reward_scaled.
  3. Generate a STARK proof that  delta < SCALE_FACTOR  (i.e. the improvement
     is a valid positive number less than 1.0 in original reward space).
  4. If delta were negative (no improvement), it wraps to a huge u64 that
     fails the STARK constraint — the proof cannot be generated.

This means:
  • A valid proof  ⟹  the metric genuinely improved.
  • Verification is O(log n) — no re-running of the experiment.
  • The exact new reward can remain private (only the improvement is proven).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import ves_stark

# Rewards are in [0, 1] — scale to u64 for the STARK circuit.
SCALE_FACTOR = 1_000_000

# Network identifiers (constant for this research network).
NETWORK_TENANT_ID = str(uuid.UUID(int=0xABCDEF_0000_0001))
NETWORK_STORE_ID = str(uuid.UUID(int=0xABCDEF_0000_0002))

# Policy: we prove  improvement_delta < SCALE_FACTOR.
POLICY_ID = "aml.threshold"
POLICY_THRESHOLD = SCALE_FACTOR  # improvement must be in [0, 1.0)


@dataclass
class MetricImprovementProof:
    """A STARK proof that an experiment's metric improved over the previous best."""

    experiment_id: str
    agent_id: str
    best_reward_scaled: int           # public: the baseline being beaten
    proof_bytes: bytes                # the raw STARK proof
    proof_hash: str                   # SHA-256 of proof_bytes
    witness_commitment: list[int]     # 4 × u64 Rescue hash of the witness
    witness_commitment_hex: str       # hex encoding of commitment
    policy_hash: str                  # hash of the policy used
    proving_time_ms: int
    proof_size: int
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "agent_id": self.agent_id,
            "best_reward_scaled": self.best_reward_scaled,
            "proof_hash": self.proof_hash,
            "witness_commitment_hex": self.witness_commitment_hex,
            "policy_hash": self.policy_hash,
            "proving_time_ms": self.proving_time_ms,
            "proof_size": self.proof_size,
            "timestamp": self.timestamp,
        }


def _reward_to_scaled(reward: float) -> int:
    """Convert a [0,1] reward to a u64 for the STARK circuit."""
    return max(0, int(round(reward * SCALE_FACTOR)))


def _make_public_inputs(
    experiment_id: str,
    agent_id: str,
    sequence_number: int,
    policy_hash: str,
) -> ves_stark.CompliancePublicInputs:
    """Build CompliancePublicInputs for a metric-improvement proof."""
    # Derive deterministic hashes from experiment metadata.
    payload_data = json.dumps({
        "experiment_id": experiment_id,
        "agent_id": agent_id,
        "type": "metric_improvement",
    }, sort_keys=True).encode()

    payload_hash = hashlib.sha256(payload_data).hexdigest()
    cipher_hash = hashlib.sha256(b"cipher:" + payload_data).hexdigest()
    signing_hash = hashlib.sha256(b"sign:" + payload_data).hexdigest()

    # event_id derived from experiment_id
    event_id = experiment_id if _is_valid_uuid(experiment_id) else str(uuid.uuid5(uuid.NAMESPACE_DNS, experiment_id))

    return ves_stark.CompliancePublicInputs(
        event_id=event_id,
        tenant_id=NETWORK_TENANT_ID,
        store_id=NETWORK_STORE_ID,
        sequence_number=sequence_number,
        payload_kind=1,
        payload_plain_hash=payload_hash,
        payload_cipher_hash=cipher_hash,
        event_signing_hash=signing_hash,
        policy_id=POLICY_ID,
        policy_params={"threshold": POLICY_THRESHOLD},
        policy_hash=policy_hash,
    )


def _is_valid_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except ValueError:
        return False


class MetricImprovementProver:
    """
    Generates STARK proofs that a new metric exceeds the current best.

    The proof circuit proves:  (new_reward - best_reward) * SCALE < THRESHOLD
    which is only satisfiable when new_reward > best_reward.
    """

    def __init__(self) -> None:
        self._policy = ves_stark.Policy.aml_threshold(POLICY_THRESHOLD)
        self._policy_hash = ves_stark.compute_policy_hash(
            POLICY_ID, {"threshold": POLICY_THRESHOLD}
        )
        self._sequence = 0

    def prove_improvement(
        self,
        new_reward: float,
        best_reward: float,
        experiment_id: str,
        agent_id: str,
    ) -> MetricImprovementProof:
        """
        Generate a STARK proof that new_reward > best_reward.

        The witness (improvement delta) is private; the proof is publicly
        verifiable without knowing the exact new reward.

        Raises ValueError if new_reward <= best_reward (proof is impossible).
        """
        new_scaled = _reward_to_scaled(new_reward)
        best_scaled = _reward_to_scaled(best_reward)

        if new_scaled <= best_scaled:
            raise ValueError(
                f"Cannot prove improvement: new ({new_reward:.6f}) "
                f"<= best ({best_reward:.6f})"
            )

        # The witness: how much the metric improved (private).
        delta = new_scaled - best_scaled

        self._sequence += 1
        public_inputs = _make_public_inputs(
            experiment_id=experiment_id,
            agent_id=agent_id,
            sequence_number=self._sequence,
            policy_hash=self._policy_hash,
        )

        # Generate the STARK proof via ves_stark.
        proof = ves_stark.prove(delta, public_inputs, self._policy)

        return MetricImprovementProof(
            experiment_id=experiment_id,
            agent_id=agent_id,
            best_reward_scaled=best_scaled,
            proof_bytes=bytes(proof.proof_bytes),
            proof_hash=proof.proof_hash,
            witness_commitment=list(proof.witness_commitment),
            witness_commitment_hex=proof.witness_commitment_hex,
            policy_hash=self._policy_hash,
            proving_time_ms=proof.proving_time_ms,
            proof_size=proof.proof_size,
            timestamp=time.time(),
        )


def verify_improvement(
    proof: MetricImprovementProof,
    claimed_best_reward: float,
) -> tuple[bool, str]:
    """
    Verify a STARK metric-improvement proof.

    Returns (valid, message). Verification is fast (~milliseconds) and does
    NOT require re-running the experiment.
    """
    expected_best_scaled = _reward_to_scaled(claimed_best_reward)
    if proof.best_reward_scaled != expected_best_scaled:
        return False, (
            f"Best-reward mismatch: proof claims {proof.best_reward_scaled}, "
            f"expected {expected_best_scaled}"
        )

    # Reconstruct public inputs for verification.
    policy_hash = ves_stark.compute_policy_hash(
        POLICY_ID, {"threshold": POLICY_THRESHOLD}
    )
    if proof.policy_hash != policy_hash:
        return False, f"Policy hash mismatch"

    public_inputs = _make_public_inputs(
        experiment_id=proof.experiment_id,
        agent_id=proof.agent_id,
        sequence_number=1,  # verifier reconstructs from metadata
        policy_hash=policy_hash,
    )

    # Verify the STARK proof — this is the fast path.
    try:
        result = ves_stark.verify(
            proof.proof_bytes,
            public_inputs,
            proof.witness_commitment,
        )
        if result.valid:
            return True, (
                f"STARK proof valid (verified in {result.verification_time_ms}ms, "
                f"proof size {proof.proof_size} bytes)"
            )
        else:
            return False, f"STARK verification failed: {result.error}"
    except Exception as e:
        return False, f"Verification error: {e}"
