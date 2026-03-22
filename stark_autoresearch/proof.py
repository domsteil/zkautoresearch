"""
STARK proof generation and verification for metric improvement.

This module binds each proof to concrete experiment metadata, config hash,
baseline reward, claimed new reward, and improvement delta. The STARK proves
that the private improvement delta satisfies the threshold policy, while the
payload amount binding ensures the proven witness matches the public claim.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

try:
    import ves_stark
except ImportError as exc:
    raise ImportError(
        "ves_stark is required. Run `python -m stark_autoresearch.environment` "
        "for setup guidance, then install the bindings with `python scripts/install_ves_stark.py --stateset-stark-dir /path/to/stateset-stark`."
    ) from exc

# Rewards are in [0, 1] for this demo. Scale them into u64 integers.
SCALE_FACTOR = 1_000_000

# Network identifiers (constant for this research network).
NETWORK_TENANT_ID = str(uuid.UUID(int=0xABCDEF_0000_0001))
NETWORK_STORE_ID = str(uuid.UUID(int=0xABCDEF_0000_0002))

# Policy: we prove improvement_delta < SCALE_FACTOR.
POLICY_ID = "aml.threshold"
POLICY_THRESHOLD = SCALE_FACTOR


@dataclass
class MetricImprovementProof:
    """A STARK proof that an experiment's metric improved over the previous best."""

    experiment_id: str
    agent_id: str
    sequence_number: int
    config_hash: str
    best_reward_scaled: int
    new_reward_scaled: int
    improvement_delta_scaled: int
    proof_bytes: bytes
    proof_hash: str
    witness_commitment: list[int]
    witness_commitment_hex: str
    policy_hash: str
    amount_binding_hash: str
    proving_time_ms: int
    proof_size: int
    timestamp: float

    @property
    def new_reward(self) -> float:
        """Canonical float reward implied by the proof metadata."""
        return _scaled_to_reward(self.new_reward_scaled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "agent_id": self.agent_id,
            "sequence_number": self.sequence_number,
            "config_hash": self.config_hash,
            "best_reward_scaled": self.best_reward_scaled,
            "new_reward_scaled": self.new_reward_scaled,
            "improvement_delta_scaled": self.improvement_delta_scaled,
            "proof_hash": self.proof_hash,
            "witness_commitment_hex": self.witness_commitment_hex,
            "policy_hash": self.policy_hash,
            "amount_binding_hash": self.amount_binding_hash,
            "proving_time_ms": self.proving_time_ms,
            "proof_size": self.proof_size,
            "timestamp": self.timestamp,
        }


def _reward_to_scaled(reward: float) -> int:
    """Convert a reward to the circuit's scaled integer representation."""
    return max(0, int(round(reward * SCALE_FACTOR)))


def _scaled_to_reward(scaled_reward: int) -> float:
    """Convert a scaled reward back into the canonical float representation."""
    return scaled_reward / SCALE_FACTOR


def canonicalize_reward(reward: float) -> float:
    """Round-trip a reward through the circuit scale factor."""
    return _scaled_to_reward(_reward_to_scaled(reward))


def _payload_hashes(
    experiment_id: str,
    agent_id: str,
    config_hash: str,
    best_reward_scaled: int,
    new_reward_scaled: int,
    improvement_delta_scaled: int,
) -> tuple[str, str, str]:
    """Derive deterministic hashes from the proof-bound experiment metadata."""
    payload_data = json.dumps(
        {
            "agent_id": agent_id,
            "best_reward_scaled": best_reward_scaled,
            "config_hash": config_hash,
            "experiment_id": experiment_id,
            "improvement_delta_scaled": improvement_delta_scaled,
            "new_reward_scaled": new_reward_scaled,
            "type": "metric_improvement",
        },
        sort_keys=True,
    ).encode()
    payload_hash = hashlib.sha256(payload_data).hexdigest()
    cipher_hash = hashlib.sha256(b"cipher:" + payload_data).hexdigest()
    signing_hash = hashlib.sha256(b"sign:" + payload_data).hexdigest()
    return payload_hash, cipher_hash, signing_hash


def _make_public_inputs(
    experiment_id: str,
    agent_id: str,
    sequence_number: int,
    policy_hash: str,
    config_hash: str,
    best_reward_scaled: int,
    new_reward_scaled: int,
    improvement_delta_scaled: int,
    amount_binding_hash: str | None = None,
) -> ves_stark.CompliancePublicInputs:
    """Build CompliancePublicInputs for a metric-improvement proof."""
    payload_hash, cipher_hash, signing_hash = _payload_hashes(
        experiment_id=experiment_id,
        agent_id=agent_id,
        config_hash=config_hash,
        best_reward_scaled=best_reward_scaled,
        new_reward_scaled=new_reward_scaled,
        improvement_delta_scaled=improvement_delta_scaled,
    )

    event_id = (
        experiment_id
        if _is_valid_uuid(experiment_id)
        else str(uuid.uuid5(uuid.NAMESPACE_DNS, experiment_id))
    )

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
        amount_binding_hash=amount_binding_hash,
    )


def _make_amount_bound_public_inputs(
    experiment_id: str,
    agent_id: str,
    sequence_number: int,
    policy_hash: str,
    config_hash: str,
    best_reward_scaled: int,
    new_reward_scaled: int,
    improvement_delta_scaled: int,
) -> tuple[ves_stark.CompliancePublicInputs, dict[str, Any]]:
    """Create public inputs and the canonical amount binding for the proof."""
    base_inputs = _make_public_inputs(
        experiment_id=experiment_id,
        agent_id=agent_id,
        sequence_number=sequence_number,
        policy_hash=policy_hash,
        config_hash=config_hash,
        best_reward_scaled=best_reward_scaled,
        new_reward_scaled=new_reward_scaled,
        improvement_delta_scaled=improvement_delta_scaled,
    )
    amount_binding = ves_stark.create_payload_amount_binding(
        base_inputs, improvement_delta_scaled
    )
    bound_inputs = _make_public_inputs(
        experiment_id=experiment_id,
        agent_id=agent_id,
        sequence_number=sequence_number,
        policy_hash=policy_hash,
        config_hash=config_hash,
        best_reward_scaled=best_reward_scaled,
        new_reward_scaled=new_reward_scaled,
        improvement_delta_scaled=improvement_delta_scaled,
        amount_binding_hash=amount_binding["bindingHash"],
    )
    return bound_inputs, amount_binding


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


class MetricImprovementProver:
    """Generate STARK proofs that a new metric exceeds the current best."""

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
        config_hash: str,
    ) -> MetricImprovementProof:
        """
        Generate a STARK proof that new_reward > best_reward.

        Raises ValueError if new_reward <= best_reward at the circuit scale.
        """
        new_scaled = _reward_to_scaled(new_reward)
        best_scaled = _reward_to_scaled(best_reward)

        if new_scaled <= best_scaled:
            raise ValueError(
                f"Cannot prove improvement: new ({new_reward:.6f}) "
                f"<= best ({best_reward:.6f})"
            )

        delta = new_scaled - best_scaled

        self._sequence += 1
        public_inputs, amount_binding = _make_amount_bound_public_inputs(
            experiment_id=experiment_id,
            agent_id=agent_id,
            sequence_number=self._sequence,
            policy_hash=self._policy_hash,
            config_hash=config_hash,
            best_reward_scaled=best_scaled,
            new_reward_scaled=new_scaled,
            improvement_delta_scaled=delta,
        )

        proof = ves_stark.prove(delta, public_inputs, self._policy)

        return MetricImprovementProof(
            experiment_id=experiment_id,
            agent_id=agent_id,
            sequence_number=self._sequence,
            config_hash=config_hash,
            best_reward_scaled=best_scaled,
            new_reward_scaled=new_scaled,
            improvement_delta_scaled=delta,
            proof_bytes=bytes(proof.proof_bytes),
            proof_hash=proof.proof_hash,
            witness_commitment=list(proof.witness_commitment),
            witness_commitment_hex=proof.witness_commitment_hex,
            policy_hash=self._policy_hash,
            amount_binding_hash=amount_binding["bindingHash"],
            proving_time_ms=proof.proving_time_ms,
            proof_size=proof.proof_size,
            timestamp=time.time(),
        )


def verify_improvement(
    proof: MetricImprovementProof,
    claimed_best_reward: float,
    claimed_new_reward: float | None = None,
    config_hash: str | None = None,
) -> tuple[bool, str]:
    """
    Verify a STARK metric-improvement proof.

    If claimed_new_reward and/or config_hash are supplied, the verifier also
    checks that the proof matches the caller's specific submission metadata.
    """
    expected_best_scaled = _reward_to_scaled(claimed_best_reward)
    if proof.best_reward_scaled != expected_best_scaled:
        return False, (
            f"Best-reward mismatch: proof claims {proof.best_reward_scaled}, "
            f"expected {expected_best_scaled}"
        )

    if claimed_new_reward is not None:
        expected_new_scaled = _reward_to_scaled(claimed_new_reward)
        if proof.new_reward_scaled != expected_new_scaled:
            return False, (
                f"New-reward mismatch: proof claims {proof.new_reward_scaled}, "
                f"expected {expected_new_scaled}"
            )

    if config_hash is not None and proof.config_hash != config_hash:
        return False, "Config hash mismatch"

    if proof.sequence_number < 1:
        return False, "Invalid sequence number"

    if proof.new_reward_scaled <= proof.best_reward_scaled:
        return False, "Proof does not encode a positive improvement"

    expected_delta = proof.new_reward_scaled - proof.best_reward_scaled
    if proof.improvement_delta_scaled != expected_delta:
        return False, (
            f"Improvement mismatch: proof claims {proof.improvement_delta_scaled}, "
            f"expected {expected_delta}"
        )

    policy_hash = ves_stark.compute_policy_hash(
        POLICY_ID, {"threshold": POLICY_THRESHOLD}
    )
    if proof.policy_hash != policy_hash:
        return False, "Policy hash mismatch"

    public_inputs, amount_binding = _make_amount_bound_public_inputs(
        experiment_id=proof.experiment_id,
        agent_id=proof.agent_id,
        sequence_number=proof.sequence_number,
        policy_hash=policy_hash,
        config_hash=proof.config_hash,
        best_reward_scaled=proof.best_reward_scaled,
        new_reward_scaled=proof.new_reward_scaled,
        improvement_delta_scaled=proof.improvement_delta_scaled,
    )
    if proof.amount_binding_hash != amount_binding["bindingHash"]:
        return False, "Amount binding hash mismatch"

    try:
        result = ves_stark.verify_with_amount_binding(
            proof.proof_bytes,
            public_inputs,
            amount_binding,
        )
        if result.valid:
            return True, (
                f"STARK proof valid (verified in {result.verification_time_ms}ms, "
                f"proof size {proof.proof_size} bytes)"
            )
        return False, f"STARK verification failed: {result.error}"
    except Exception as exc:
        return False, f"Verification error: {exc}"
