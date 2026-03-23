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
from pathlib import Path
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
class VerificationResult:
    """Structured proof verification result."""

    valid: bool
    message: str
    verification_time_ms: float = 0.0

    def __iter__(self):
        yield self.valid
        yield self.message

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class MetricImprovementProof:
    """A STARK proof that an experiment's metric improved over the previous best."""

    experiment_id: str
    agent_id: str
    best_reward_scaled: int
    sequence_number: int
    proof_bytes: bytes
    proof_hash: str
    witness_commitment: list[int]
    witness_commitment_hex: str
    policy_hash: str
    proving_time_ms: int
    proof_size: int
    timestamp: float
    config_hash: str | None = None
    new_reward_scaled: int | None = None
    improvement_delta_scaled: int | None = None
    envelope_sha256: str | None = None
    amount_binding_hash: str | None = None

    @property
    def new_reward(self) -> float:
        """Canonical float reward implied by the proof metadata."""
        if self.new_reward_scaled is not None:
            return _scaled_to_reward(self.new_reward_scaled)
        if self.improvement_delta_scaled is not None:
            return _scaled_to_reward(self.best_reward_scaled + self.improvement_delta_scaled)
        return _scaled_to_reward(self.best_reward_scaled)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "experiment_id": self.experiment_id,
            "agent_id": self.agent_id,
            "sequence_number": self.sequence_number,
            "best_reward_scaled": self.best_reward_scaled,
            "proof_hash": self.proof_hash,
            "witness_commitment_hex": self.witness_commitment_hex,
            "policy_hash": self.policy_hash,
            "proving_time_ms": self.proving_time_ms,
            "proof_size": self.proof_size,
            "timestamp": self.timestamp,
        }
        if self.config_hash is not None:
            payload["config_hash"] = self.config_hash
        if self.new_reward_scaled is not None:
            payload["new_reward_scaled"] = self.new_reward_scaled
        if self.improvement_delta_scaled is not None:
            payload["improvement_delta_scaled"] = self.improvement_delta_scaled
        if self.envelope_sha256 is not None:
            payload["envelope_sha256"] = self.envelope_sha256
        if self.amount_binding_hash is not None:
            payload["amount_binding_hash"] = self.amount_binding_hash
        return payload


def _reward_to_scaled(reward: float) -> int:
    """Convert a reward to the circuit's scaled integer representation."""
    return max(0, int(round(reward * SCALE_FACTOR)))


def _scaled_to_reward(scaled_reward: int) -> float:
    """Convert a scaled reward back into the canonical float representation."""
    return scaled_reward / SCALE_FACTOR


def canonicalize_reward(reward: float) -> float:
    """Round-trip a reward through the circuit scale factor."""
    return _scaled_to_reward(_reward_to_scaled(reward))


def _normalized_config_hash(config_hash: str | None) -> str:
    return "" if config_hash is None else str(config_hash)


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


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


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
        config_hash: str | None = None,
        envelope_sha256: str | None = None,
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
        normalized_config_hash = _normalized_config_hash(config_hash)
        public_inputs, amount_binding = _make_amount_bound_public_inputs(
            experiment_id=experiment_id,
            agent_id=agent_id,
            sequence_number=self._sequence,
            policy_hash=self._policy_hash,
            config_hash=normalized_config_hash,
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
            envelope_sha256=envelope_sha256,
        )

    def prove_improvement_from_provenance(
        self,
        *,
        provenance_path: str | Path,
        best_reward: float,
        agent_id: str,
        require_signature: bool,
    ) -> MetricImprovementProof:
        from experiment_runtime import verify_provenance_envelope

        verification = verify_provenance_envelope(
            provenance_path,
            require_signature=require_signature,
        )
        if not verification.get("ok"):
            reason = verification.get("signature_reason") or "provenance verification failed"
            raise ValueError(f"Invalid provenance envelope: {reason}")

        payload = _load_json_object(provenance_path)
        try:
            objective_value = float(payload["objective_value"])
        except KeyError as exc:
            raise ValueError(f"Provenance envelope missing objective_value: {provenance_path}") from exc

        config_hash = payload.get("summary_config_sha256")
        if config_hash is not None:
            config_hash = str(config_hash)

        return self.prove_improvement(
            new_reward=objective_value,
            best_reward=best_reward,
            experiment_id=str(payload["experiment_id"]),
            agent_id=agent_id,
            config_hash=config_hash,
            envelope_sha256=payload.get("envelope_sha256"),
        )


def verify_improvement(
    proof: MetricImprovementProof,
    claimed_best_reward: float,
    claimed_new_reward: float | None = None,
    config_hash: str | None = None,
    provenance_path: str | Path | None = None,
    require_provenance_signature: bool = False,
) -> VerificationResult:
    """
    Verify a STARK metric-improvement proof.

    If claimed_new_reward and/or config_hash are supplied, the verifier also
    checks that the proof matches the caller's specific submission metadata.
    When provenance_path is supplied, the proof can be bound to a signed
    provenance envelope instead of an explicit claimed_new_reward.
    """
    expected_best_scaled = _reward_to_scaled(claimed_best_reward)
    if proof.best_reward_scaled != expected_best_scaled:
        return VerificationResult(
            False,
            f"Best-reward mismatch: proof claims {proof.best_reward_scaled}, expected {expected_best_scaled}",
        )

    expected_new_scaled: int | None = None
    inferred_config_hash: str | None = None
    if provenance_path is not None:
        from experiment_runtime import verify_provenance_envelope

        provenance_verification = verify_provenance_envelope(
            provenance_path,
            require_signature=require_provenance_signature,
        )
        if not provenance_verification.get("ok"):
            reason = provenance_verification.get("signature_reason") or "provenance verification failed"
            return VerificationResult(False, f"Provenance verification failed: {reason}")

        provenance_payload = _load_json_object(provenance_path)
        if str(provenance_payload.get("experiment_id")) != proof.experiment_id:
            return VerificationResult(False, "Experiment/provenance mismatch")
        if proof.envelope_sha256 is not None and provenance_payload.get("envelope_sha256") != proof.envelope_sha256:
            return VerificationResult(False, "Envelope SHA mismatch")
        if "objective_value" not in provenance_payload:
            return VerificationResult(False, "Provenance missing objective_value")
        inferred_config_hash = provenance_payload.get("summary_config_sha256")
        if inferred_config_hash is not None:
            inferred_config_hash = str(inferred_config_hash)
        expected_new_scaled = _reward_to_scaled(float(provenance_payload["objective_value"]))

    if claimed_new_reward is not None:
        claimed_new_scaled = _reward_to_scaled(claimed_new_reward)
        if expected_new_scaled is None:
            expected_new_scaled = claimed_new_scaled
        elif claimed_new_scaled != expected_new_scaled:
            return VerificationResult(False, "Claimed reward/provenance mismatch")

    if expected_new_scaled is None:
        if proof.new_reward_scaled is None:
            return VerificationResult(False, "New reward unavailable for verification")
        expected_new_scaled = int(proof.new_reward_scaled)

    if proof.new_reward_scaled is not None and int(proof.new_reward_scaled) != expected_new_scaled:
        return VerificationResult(
            False,
            f"New-reward mismatch: proof claims {proof.new_reward_scaled}, expected {expected_new_scaled}",
        )

    if expected_new_scaled <= proof.best_reward_scaled:
        return VerificationResult(False, "Proof does not encode a positive improvement")

    expected_delta = expected_new_scaled - proof.best_reward_scaled
    if proof.improvement_delta_scaled is not None and int(proof.improvement_delta_scaled) != expected_delta:
        return VerificationResult(
            False,
            f"Improvement mismatch: proof claims {proof.improvement_delta_scaled}, expected {expected_delta}",
        )

    effective_config_hash = proof.config_hash if proof.config_hash is not None else inferred_config_hash
    normalized_proof_config_hash = _normalized_config_hash(effective_config_hash)
    normalized_expected_config_hash = _normalized_config_hash(config_hash)
    if config_hash is not None and normalized_proof_config_hash != normalized_expected_config_hash:
        return VerificationResult(False, "Config hash mismatch")

    if proof.sequence_number < 1:
        return VerificationResult(False, "Invalid sequence number")

    policy_hash = ves_stark.compute_policy_hash(
        POLICY_ID, {"threshold": POLICY_THRESHOLD}
    )
    if proof.policy_hash != policy_hash:
        return VerificationResult(False, "Policy hash mismatch")

    public_inputs, amount_binding = _make_amount_bound_public_inputs(
        experiment_id=proof.experiment_id,
        agent_id=proof.agent_id,
        sequence_number=proof.sequence_number,
        policy_hash=policy_hash,
        config_hash=normalized_proof_config_hash,
        best_reward_scaled=proof.best_reward_scaled,
        new_reward_scaled=expected_new_scaled,
        improvement_delta_scaled=expected_delta,
    )
    if proof.amount_binding_hash is not None and proof.amount_binding_hash != amount_binding["bindingHash"]:
        return VerificationResult(False, "Amount binding hash mismatch")

    try:
        result = ves_stark.verify_with_amount_binding(
            proof.proof_bytes,
            public_inputs,
            amount_binding,
        )
        if result.valid:
            return VerificationResult(
                True,
                f"STARK proof valid (verified in {result.verification_time_ms}ms, proof size {proof.proof_size} bytes)",
                verification_time_ms=float(result.verification_time_ms),
            )
        return VerificationResult(False, f"STARK verification failed: {result.error}")
    except Exception as exc:
        return VerificationResult(False, f"Verification error: {exc}")
