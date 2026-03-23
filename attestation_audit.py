
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from experiment_runtime import (
    sha256_file,
    sha256_json,
    sign_json_payload_if_configured,
    verify_json_signature,
    verify_provenance_envelope,
)
from stateset_agents.training.auto_research.experiment_tracker import ExperimentTracker


def objective_is_improvement(
    value: float,
    current_best: float | None,
    *,
    direction: str,
) -> bool:
    if current_best is None:
        return True
    if direction == "maximize":
        return value > current_best
    return value < current_best


def _missing_provenance_result() -> dict[str, Any]:
    return {
        "ok": False,
        "path": None,
        "signature_present": False,
        "signature_reason": "missing",
        "checks": {},
    }


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_result(
    *,
    path: Path,
    status: str,
    reason: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    result = {
        "path": str(path),
        "present": path.exists(),
        "status": status,
        "ok": status in {"ok", "not_required"},
        "reason": reason,
    }
    result.update(details)
    return result


def _best_artifact_paths(output_path: Path) -> dict[str, Path]:
    return {
        "best_config": output_path / "best_config.json",
        "best_summary": output_path / "best_summary.json",
        "best_provenance": output_path / "best_provenance.json",
        "best_runtime_environment": output_path / "best_runtime_environment.json",
        "best_proof": output_path / "best_proof.json",
        "best_proof_bin": output_path / "best_proof.bin",
    }


def _best_bundle_paths(output_path: Path) -> dict[str, Path]:
    root = output_path / "best_bundle"
    return {
        "root": root,
        "config": root / "config.json",
        "summary": root / "summary.json",
        "log": root / "train.log",
        "runtime_environment": root / "runtime_environment.json",
        "provenance": root / "provenance.json",
        "proof": root / "proof.json",
        "proof_bin": root / "proof.bin",
        "model_dir": root / "outputs" / "final_model",
    }


def _relativize(output_path: Path, target: str | Path) -> str:
    target_path = Path(target)
    try:
        return str(target_path.relative_to(output_path))
    except ValueError:
        return str(target_path)


def _artifact_digest(path: Path) -> dict[str, Any]:
    return {
        "path": path.name if path.is_absolute() else str(path),
        "sha256": sha256_file(path),
    }


def _provenance_runtime_artifact_path(
    provenance_path: str | Path | None,
    *,
    fallback_run_dir: Path | None = None,
) -> Path | None:
    if provenance_path is not None:
        payload = _load_json_object(Path(provenance_path))
        if payload is not None:
            runtime_rel = payload.get("runtime_artifact_path")
            if runtime_rel:
                return Path(provenance_path).parent / str(runtime_rel)
    if fallback_run_dir is not None:
        fallback = fallback_run_dir / "runtime_environment.json"
        if fallback.exists():
            return fallback
    return None


def _best_artifact_issues(best_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    checks = best_artifacts.get("checks") or {}
    issues: list[dict[str, Any]] = []
    for name, check in checks.items():
        if not isinstance(check, dict):
            continue
        status = str(check.get("status") or "")
        if status not in {"fail", "missing"}:
            continue
        issue = {"name": name, "status": status}
        reason = check.get("reason")
        if reason is not None:
            issue["reason"] = str(reason)
        issues.append(issue)
    return issues


def verify_proof_artifact(path: str | Path, *, require_signature: bool) -> dict[str, Any]:
    try:
        from stark_autoresearch.proof import MetricImprovementProof, verify_improvement

        proof_path = Path(path)
        payload = json.loads(proof_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Proof artifact at {proof_path} must be a JSON object")

        proof_rel = str(payload.get("proof_path") or "proof.bin")
        proof_bin_path = proof_path.parent / proof_rel
        if not proof_bin_path.exists():
            raise FileNotFoundError(f"Proof bytes not found at {proof_bin_path}")

        expected_proof_sha256 = payload.get("proof_bytes_sha256")
        actual_proof_sha256 = sha256_file(proof_bin_path)
        proof_hash_ok = (
            expected_proof_sha256 is None or actual_proof_sha256 == expected_proof_sha256
        )

        provenance_rel = payload.get("provenance_path")
        if provenance_rel:
            provenance_path = proof_path.parent / str(provenance_rel)
        elif proof_path.name == "best_proof.json":
            provenance_path = proof_path.parent / "best_provenance.json"
        else:
            provenance_path = proof_path.parent / "provenance.json"
        if not provenance_path.exists():
            provenance_path = None

        proof = MetricImprovementProof(
            experiment_id=str(payload["experiment_id"]),
            agent_id=str(payload["agent_id"]),
            best_reward_scaled=int(payload["best_reward_scaled"]),
            sequence_number=int(payload["sequence_number"]),
            proof_bytes=proof_bin_path.read_bytes(),
            proof_hash=str(payload["proof_hash"]),
            witness_commitment=list(payload["witness_commitment"]),
            witness_commitment_hex=str(payload["witness_commitment_hex"]),
            policy_hash=str(payload["policy_hash"]),
            proving_time_ms=int(payload["proving_time_ms"]),
            proof_size=int(payload["proof_size"]),
            timestamp=float(payload["timestamp"]),
            config_hash=None if payload.get("config_hash") is None else str(payload.get("config_hash")),
            new_reward_scaled=(
                None if payload.get("new_reward_scaled") is None else int(payload.get("new_reward_scaled"))
            ),
            improvement_delta_scaled=(
                None
                if payload.get("improvement_delta_scaled") is None
                else int(payload.get("improvement_delta_scaled"))
            ),
            envelope_sha256=payload.get("envelope_sha256"),
            amount_binding_hash=payload.get("amount_binding_hash"),
        )
        verification = verify_improvement(
            proof,
            claimed_best_reward=float(payload["claimed_best_reward"]),
            provenance_path=provenance_path,
            require_provenance_signature=require_signature,
        )
        ok = proof_hash_ok and verification.valid
        return {
            "ok": ok,
            "path": str(proof_path),
            "proof_bin_path": str(proof_bin_path),
            "proof_bytes_sha256_ok": proof_hash_ok,
            "provenance_path": str(provenance_path) if provenance_path is not None else None,
            "verification_message": verification.message,
            "verification_time_ms": verification.verification_time_ms,
        }
    except Exception as exc:
        return {
            "ok": False,
            "path": str(path),
            "proof_bin_path": None,
            "proof_bytes_sha256_ok": False,
            "provenance_path": None,
            "verification_message": str(exc),
            "verification_time_ms": 0.0,
        }


def evaluate_record_attestation(
    *,
    record_status: str,
    objective_value: float,
    current_best: float | None,
    direction: str,
    provenance_path: str | Path | None,
    proof_path: str | Path | None,
    require_signature: bool,
) -> dict[str, Any]:
    improved = record_status == "keep" and objective_is_improvement(
        objective_value,
        current_best,
        direction=direction,
    )
    proof_required = record_status == "keep" and current_best is not None and improved

    if provenance_path:
        provenance_result = verify_provenance_envelope(
            provenance_path,
            require_signature=require_signature,
        )
        provenance_status = "ok" if provenance_result["ok"] else "fail"
    else:
        provenance_result = _missing_provenance_result()
        provenance_status = "missing"

    if proof_path:
        proof_result = verify_proof_artifact(proof_path, require_signature=require_signature)
        proof_status = "ok" if proof_result["ok"] else "fail"
    elif proof_required:
        proof_result = {
            "ok": False,
            "path": None,
            "proof_bin_path": None,
            "proof_bytes_sha256_ok": False,
            "provenance_path": str(provenance_path) if provenance_path else None,
            "verification_message": "Required proof missing",
            "verification_time_ms": 0.0,
        }
        proof_status = "missing"
    else:
        proof_result = None
        proof_status = "not_required"

    ok = provenance_status == "ok" and proof_status in {"ok", "not_required"}
    return {
        "ok": ok,
        "improved": improved,
        "proof_required": proof_required,
        "provenance_status": provenance_status,
        "provenance_result": provenance_result,
        "provenance_message": provenance_result.get("signature_reason")
        if provenance_status != "ok"
        else None,
        "proof_status": proof_status,
        "proof_result": proof_result,
        "proof_message": None if proof_result is None else proof_result.get("verification_message"),
    }


def verify_best_artifacts(
    output_dir: str | Path,
    *,
    tracker: ExperimentTracker,
    require_signature: bool,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    artifact_paths = _best_artifact_paths(output_path)
    bundle_paths = _best_bundle_paths(output_path)

    best_record = tracker.best_record
    checks: dict[str, dict[str, Any]] = {}
    overall_ok = True

    if best_record is None:
        for name, path in artifact_paths.items():
            if path.exists():
                checks[name] = _artifact_result(
                    path=path,
                    status="fail",
                    reason="Unexpected best artifact without a best record",
                )
                overall_ok = False
            else:
                checks[name] = _artifact_result(path=path, status="not_required")
        if bundle_paths["root"].exists():
            checks["best_bundle"] = _artifact_result(
                path=bundle_paths["root"],
                status="fail",
                reason="Unexpected best bundle without a best record",
            )
            overall_ok = False
        else:
            checks["best_bundle"] = _artifact_result(path=bundle_paths["root"], status="not_required")
        return {"ok": overall_ok, "checks": checks}

    source_run_dir = Path(best_record.checkpoint_path) if best_record.checkpoint_path else None
    source_config_path = source_run_dir / "config.json" if source_run_dir is not None else None
    source_summary_path = source_run_dir / "summary.json" if source_run_dir is not None else None
    source_log_path = source_run_dir / "train.log" if source_run_dir is not None else None
    source_model_dir = source_run_dir / "outputs" / "final_model" if source_run_dir is not None else None
    source_provenance_path = Path(best_record.provenance_path) if best_record.provenance_path else None
    source_runtime_path = _provenance_runtime_artifact_path(
        source_provenance_path,
        fallback_run_dir=source_run_dir,
    )
    source_proof_path = Path(best_record.proof_path) if best_record.proof_path else None
    source_proof_payload = _load_json_object(source_proof_path) if source_proof_path is not None else None
    source_proof_bin_name = "proof.bin"
    if source_proof_payload is not None:
        source_proof_bin_name = str(source_proof_payload.get("proof_path") or "proof.bin")
    source_proof_bin_path = (
        source_proof_path.parent / source_proof_bin_name
        if source_proof_path is not None
        else None
    )

    best_config_path = artifact_paths["best_config"]
    config_payload = _load_json_object(best_config_path)
    if not best_config_path.exists():
        checks["best_config"] = _artifact_result(
            path=best_config_path,
            status="missing",
            reason="Best config artifact missing",
        )
    elif source_config_path is not None and source_config_path.exists():
        if sha256_file(best_config_path) != sha256_file(source_config_path):
            checks["best_config"] = _artifact_result(
                path=best_config_path,
                status="fail",
                reason="Best config does not match source config",
                source_path=str(source_config_path),
            )
            overall_ok = False
        else:
            checks["best_config"] = _artifact_result(
                path=best_config_path,
                status="ok",
                source_path=str(source_config_path),
            )
    elif config_payload != dict(best_record.params):
        checks["best_config"] = _artifact_result(
            path=best_config_path,
            status="fail",
            reason="Best config does not match tracker.best_record.params",
        )
        overall_ok = False
    else:
        checks["best_config"] = _artifact_result(path=best_config_path, status="ok")

    best_summary_path = artifact_paths["best_summary"]
    if not best_summary_path.exists():
        checks["best_summary"] = _artifact_result(
            path=best_summary_path,
            status="missing",
            reason="Best summary artifact missing",
        )
    elif source_summary_path is None or not source_summary_path.exists():
        checks["best_summary"] = _artifact_result(
            path=best_summary_path,
            status="fail",
            reason="Source summary for best record is missing",
        )
        overall_ok = False
    elif sha256_file(best_summary_path) != sha256_file(source_summary_path):
        checks["best_summary"] = _artifact_result(
            path=best_summary_path,
            status="fail",
            reason="Best summary does not match source summary",
            source_path=str(source_summary_path),
        )
        overall_ok = False
    else:
        checks["best_summary"] = _artifact_result(
            path=best_summary_path,
            status="ok",
            source_path=str(source_summary_path),
        )

    best_provenance_path = artifact_paths["best_provenance"]
    if not best_provenance_path.exists():
        checks["best_provenance"] = _artifact_result(
            path=best_provenance_path,
            status="missing",
            reason="Best provenance artifact missing",
        )
    elif source_provenance_path is None or not source_provenance_path.exists():
        checks["best_provenance"] = _artifact_result(
            path=best_provenance_path,
            status="fail",
            reason="Source provenance for best record is missing",
        )
        overall_ok = False
    else:
        provenance_match = sha256_file(best_provenance_path) == sha256_file(source_provenance_path)
        source_provenance_result = verify_provenance_envelope(
            source_provenance_path,
            require_signature=require_signature,
        )
        if not provenance_match or not source_provenance_result["ok"]:
            reason = "Best provenance does not match source provenance"
            if provenance_match and not source_provenance_result["ok"]:
                reason = source_provenance_result.get("signature_reason") or "Source provenance verification failed"
            checks["best_provenance"] = _artifact_result(
                path=best_provenance_path,
                status="fail",
                reason=reason,
                source_path=str(source_provenance_path),
                verification=source_provenance_result,
            )
            overall_ok = False
        else:
            checks["best_provenance"] = _artifact_result(
                path=best_provenance_path,
                status="ok",
                source_path=str(source_provenance_path),
                verification=source_provenance_result,
            )

    best_runtime_path = artifact_paths["best_runtime_environment"]
    if source_runtime_path is None:
        if best_runtime_path.exists():
            checks["best_runtime_environment"] = _artifact_result(
                path=best_runtime_path,
                status="fail",
                reason="Unexpected best runtime artifact without runtime attestation in source provenance",
            )
            overall_ok = False
        else:
            checks["best_runtime_environment"] = _artifact_result(path=best_runtime_path, status="not_required")
    elif not best_runtime_path.exists():
        checks["best_runtime_environment"] = _artifact_result(
            path=best_runtime_path,
            status="missing",
            reason="Best runtime environment artifact missing",
            source_path=str(source_runtime_path),
        )
        overall_ok = False
    elif not source_runtime_path.exists():
        checks["best_runtime_environment"] = _artifact_result(
            path=best_runtime_path,
            status="fail",
            reason="Source runtime environment artifact for best record is missing",
            source_path=str(source_runtime_path),
        )
        overall_ok = False
    elif sha256_file(best_runtime_path) != sha256_file(source_runtime_path):
        checks["best_runtime_environment"] = _artifact_result(
            path=best_runtime_path,
            status="fail",
            reason="Best runtime environment does not match source runtime environment",
            source_path=str(source_runtime_path),
        )
        overall_ok = False
    else:
        checks["best_runtime_environment"] = _artifact_result(
            path=best_runtime_path,
            status="ok",
            source_path=str(source_runtime_path),
        )

    best_proof_path = artifact_paths["best_proof"]
    best_proof_bin_path = artifact_paths["best_proof_bin"]
    if source_proof_path is None:
        if best_proof_path.exists():
            checks["best_proof"] = _artifact_result(
                path=best_proof_path,
                status="fail",
                reason="Unexpected best proof artifact for a best record without proof_path",
            )
            overall_ok = False
        else:
            checks["best_proof"] = _artifact_result(path=best_proof_path, status="not_required")
        if best_proof_bin_path.exists():
            checks["best_proof_bin"] = _artifact_result(
                path=best_proof_bin_path,
                status="fail",
                reason="Unexpected best proof binary for a best record without proof_path",
            )
            overall_ok = False
        else:
            checks["best_proof_bin"] = _artifact_result(path=best_proof_bin_path, status="not_required")
    else:
        if not best_proof_path.exists():
            checks["best_proof"] = _artifact_result(
                path=best_proof_path,
                status="missing",
                reason="Best proof artifact missing",
            )
        else:
            proof_payload = _load_json_object(best_proof_path)
            source_proof_result = verify_proof_artifact(source_proof_path, require_signature=require_signature)
            payload_ok = isinstance(proof_payload, dict)
            experiment_ok = payload_ok and proof_payload.get("experiment_id") == best_record.experiment_id
            provenance_ref_ok = payload_ok and proof_payload.get("provenance_path") == "best_provenance.json"
            proof_path_ok = payload_ok and proof_payload.get("proof_path") == "best_proof.bin"
            payload_hash_ok = (
                payload_ok
                and best_proof_bin_path.exists()
                and proof_payload.get("proof_bytes_sha256") == sha256_file(best_proof_bin_path)
            )
            if not source_proof_result["ok"] or not experiment_ok or not provenance_ref_ok or not proof_path_ok or not payload_hash_ok:
                reason = source_proof_result.get("verification_message") or "Source proof verification failed"
                if source_proof_result["ok"] and not experiment_ok:
                    reason = "Best proof experiment_id does not match tracker.best_record"
                elif source_proof_result["ok"] and experiment_ok and not provenance_ref_ok:
                    reason = "Best proof does not point at best_provenance.json"
                elif source_proof_result["ok"] and experiment_ok and provenance_ref_ok and not proof_path_ok:
                    reason = "Best proof does not point at best_proof.bin"
                elif source_proof_result["ok"] and experiment_ok and provenance_ref_ok and proof_path_ok and not payload_hash_ok:
                    reason = "Best proof hash does not match best_proof.bin"
                checks["best_proof"] = _artifact_result(
                    path=best_proof_path,
                    status="fail",
                    reason=reason,
                    verification=source_proof_result,
                )
                overall_ok = False
            else:
                checks["best_proof"] = _artifact_result(
                    path=best_proof_path,
                    status="ok",
                    verification=source_proof_result,
                )

        if not best_proof_bin_path.exists():
            checks["best_proof_bin"] = _artifact_result(
                path=best_proof_bin_path,
                status="missing",
                reason="Best proof binary missing",
            )
        elif source_proof_bin_path is None or not source_proof_bin_path.exists():
            checks["best_proof_bin"] = _artifact_result(
                path=best_proof_bin_path,
                status="fail",
                reason="Source proof binary is missing",
                source_path=None if source_proof_bin_path is None else str(source_proof_bin_path),
            )
            overall_ok = False
        elif sha256_file(best_proof_bin_path) != sha256_file(source_proof_bin_path):
            checks["best_proof_bin"] = _artifact_result(
                path=best_proof_bin_path,
                status="fail",
                reason="Best proof binary does not match source proof binary",
                source_path=str(source_proof_bin_path),
            )
            overall_ok = False
        else:
            checks["best_proof_bin"] = _artifact_result(
                path=best_proof_bin_path,
                status="ok",
                source_path=str(source_proof_bin_path),
            )

    bundle_root = bundle_paths["root"]
    bundle_provenance_path = bundle_paths["provenance"]
    if source_provenance_path is None or not source_provenance_path.exists():
        if bundle_root.exists():
            checks["best_bundle"] = _artifact_result(
                path=bundle_root,
                status="fail",
                reason="Unexpected portable best bundle without source provenance",
            )
            overall_ok = False
        else:
            checks["best_bundle"] = _artifact_result(path=bundle_root, status="not_required")
    elif not bundle_provenance_path.exists():
        checks["best_bundle"] = _artifact_result(
            path=bundle_root,
            status="missing",
            reason="Portable best bundle provenance missing",
            source_path=str(source_run_dir) if source_run_dir is not None else None,
        )
        overall_ok = False
    else:
        bundle_provenance_payload = _load_json_object(bundle_provenance_path)
        bundle_provenance_result = verify_provenance_envelope(
            bundle_provenance_path,
            require_signature=require_signature,
        )
        bundle_experiment_ok = (
            isinstance(bundle_provenance_payload, dict)
            and bundle_provenance_payload.get("experiment_id") == best_record.experiment_id
        )
        bundle_objective_ok = (
            isinstance(bundle_provenance_payload, dict)
            and bundle_provenance_payload.get("objective_value") == best_record.objective_value
        )
        bundle_proof_result = None
        if source_proof_path is None:
            bundle_proof_ok = not bundle_paths["proof"].exists() and not bundle_paths["proof_bin"].exists()
            bundle_reason = None if bundle_proof_ok else "Portable best bundle unexpectedly contains proof artifacts"
        elif not bundle_paths["proof"].exists():
            bundle_proof_ok = False
            bundle_reason = "Portable best bundle proof missing"
        else:
            bundle_proof_result = verify_proof_artifact(
                bundle_paths["proof"],
                require_signature=require_signature,
            )
            bundle_proof_ok = bundle_proof_result["ok"]
            bundle_reason = None if bundle_proof_ok else bundle_proof_result.get("verification_message")

        if not bundle_provenance_result["ok"]:
            checks["best_bundle"] = _artifact_result(
                path=bundle_root,
                status="fail",
                reason=bundle_provenance_result.get("signature_reason") or "Portable best bundle provenance verification failed",
                verification=bundle_provenance_result,
                proof_verification=bundle_proof_result,
            )
            overall_ok = False
        elif not bundle_experiment_ok:
            checks["best_bundle"] = _artifact_result(
                path=bundle_root,
                status="fail",
                reason="Portable best bundle provenance experiment_id does not match tracker.best_record",
                verification=bundle_provenance_result,
                proof_verification=bundle_proof_result,
            )
            overall_ok = False
        elif not bundle_objective_ok:
            checks["best_bundle"] = _artifact_result(
                path=bundle_root,
                status="fail",
                reason="Portable best bundle provenance objective_value does not match tracker.best_record",
                verification=bundle_provenance_result,
                proof_verification=bundle_proof_result,
            )
            overall_ok = False
        elif not bundle_proof_ok:
            checks["best_bundle"] = _artifact_result(
                path=bundle_root,
                status="fail",
                reason=bundle_reason or "Portable best bundle proof verification failed",
                verification=bundle_provenance_result,
                proof_verification=bundle_proof_result,
            )
            overall_ok = False
        else:
            checks["best_bundle"] = _artifact_result(
                path=bundle_root,
                status="ok",
                verification=bundle_provenance_result,
                proof_verification=bundle_proof_result,
            )

    return {"ok": overall_ok, "checks": checks}


def build_audit_report(output_dir: str | Path, *, require_signature: bool) -> dict[str, Any]:
    output_path = Path(output_dir)
    if not (output_path / "experiments.jsonl").exists():
        raise FileNotFoundError(f"No platform results found in {output_path}")

    tracker = ExperimentTracker.load(output_path)
    current_best: float | None = None
    experiments: list[dict[str, Any]] = []
    summary = {
        "total_experiments": tracker.num_experiments,
        "provenance_ok": 0,
        "provenance_fail": 0,
        "provenance_missing": 0,
        "proofs_present": 0,
        "proofs_ok": 0,
        "proofs_fail": 0,
        "proofs_missing_required": 0,
        "proofs_not_required": 0,
        "best_artifacts_ok": 0,
        "best_artifacts_fail": 0,
        "best_artifacts_missing": 0,
        "best_artifacts_not_required": 0,
    }

    overall_ok = True
    for record in tracker.records:
        attestation = evaluate_record_attestation(
            record_status=record.status,
            objective_value=record.objective_value,
            current_best=current_best,
            direction=tracker.direction,
            provenance_path=record.provenance_path,
            proof_path=record.proof_path,
            require_signature=require_signature,
        )
        provenance_status = attestation["provenance_status"]
        proof_status = attestation["proof_status"]
        proof_result = attestation["proof_result"]

        if provenance_status == "ok":
            summary["provenance_ok"] += 1
        elif provenance_status == "missing":
            summary["provenance_missing"] += 1
            overall_ok = False
        else:
            summary["provenance_fail"] += 1
            overall_ok = False

        if record.proof_path:
            summary["proofs_present"] += 1
        if proof_status == "ok":
            summary["proofs_ok"] += 1
        elif proof_status == "fail":
            summary["proofs_fail"] += 1
            overall_ok = False
        elif proof_status == "missing":
            summary["proofs_missing_required"] += 1
            overall_ok = False
        else:
            summary["proofs_not_required"] += 1

        experiments.append(
            {
                "experiment_id": record.experiment_id,
                "status": record.status,
                "objective_value": record.objective_value,
                "provenance_status": provenance_status,
                "provenance_path": record.provenance_path,
                "proof_required": attestation["proof_required"],
                "proof_status": proof_status,
                "proof_path": record.proof_path,
                "proof_message": None if proof_result is None else proof_result["verification_message"],
            }
        )

        if attestation["improved"]:
            current_best = record.objective_value

    best_artifacts = verify_best_artifacts(
        output_path,
        tracker=tracker,
        require_signature=require_signature,
    )
    for check in best_artifacts["checks"].values():
        summary[f"best_artifacts_{check['status']}"] += 1
        if check["status"] in {"fail", "missing"}:
            overall_ok = False

    repair_history = build_repair_history_summary(
        output_path,
        require_signature=require_signature,
    )
    summary.update(_repair_history_headline_summary(repair_history))
    if repair_history.get("present") and not repair_history.get("ok"):
        overall_ok = False

    return {
        "ok": overall_ok,
        "output_dir": str(output_path),
        "objective_metric": tracker.objective_metric,
        "direction": tracker.direction,
        "require_signature": require_signature,
        "summary": summary,
        "experiments": experiments,
        "best_artifacts": best_artifacts,
        "repair_history": repair_history,
    }


def rebuild_best_artifacts(
    output_dir: str | Path,
    *,
    tracker: ExperimentTracker | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    resolved_tracker = tracker
    if resolved_tracker is None:
        if not (output_path / "experiments.jsonl").exists():
            raise FileNotFoundError(f"No platform results found in {output_path}")
        resolved_tracker = ExperimentTracker.load(output_path)

    artifact_paths = _best_artifact_paths(output_path)
    bundle_paths = _best_bundle_paths(output_path)
    written: list[str] = []
    removed: list[str] = []
    missing_sources: list[str] = []
    notes: list[str] = []

    def _write_bytes(target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        written.append(str(target))

    def _write_json(target: Path, payload: dict[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))
        written.append(str(target))

    def _remove_path_if_exists(target: Path, *, record: bool = True) -> None:
        if not target.exists():
            return
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        if record:
            removed.append(str(target))

    def _copy_file(source: Path, target: Path) -> None:
        _write_bytes(target, source.read_bytes())

    def _copy_tree(source: Path, target: Path) -> None:
        _remove_path_if_exists(target)
        for candidate in sorted(source.rglob("*")):
            if candidate.is_file():
                _copy_file(candidate, target / candidate.relative_to(source))

    best_record = resolved_tracker.best_record
    if best_record is None:
        for path in artifact_paths.values():
            _remove_path_if_exists(path)
        _remove_path_if_exists(bundle_paths["root"], record=False)
        return {
            "ok": True,
            "output_dir": str(output_path),
            "best_record": None,
            "written": written,
            "removed": removed,
            "missing_sources": missing_sources,
            "notes": notes,
        }

    run_dir = Path(best_record.checkpoint_path) if best_record.checkpoint_path else None
    source_config_path = run_dir / "config.json" if run_dir is not None else None
    source_summary_path = run_dir / "summary.json" if run_dir is not None else None
    source_log_path = run_dir / "train.log" if run_dir is not None else None
    source_model_dir = run_dir / "outputs" / "final_model" if run_dir is not None else None
    source_provenance_path = Path(best_record.provenance_path) if best_record.provenance_path else None
    source_runtime_path = _provenance_runtime_artifact_path(
        source_provenance_path,
        fallback_run_dir=run_dir,
    )

    if source_config_path is not None and source_config_path.exists():
        _copy_file(source_config_path, artifact_paths["best_config"])
    else:
        if source_config_path is not None:
            missing_sources.append(str(source_config_path))
            notes.append("Source config missing; rebuilt best_config.json from tracker.best_record.params")
        _write_json(artifact_paths["best_config"], dict(best_record.params))

    if source_summary_path is not None and source_summary_path.exists():
        _copy_file(source_summary_path, artifact_paths["best_summary"])
    else:
        if source_summary_path is not None:
            missing_sources.append(str(source_summary_path))
        _remove_path_if_exists(artifact_paths["best_summary"])

    if source_provenance_path is not None and source_provenance_path.exists():
        _copy_file(source_provenance_path, artifact_paths["best_provenance"])
    else:
        if source_provenance_path is not None:
            missing_sources.append(str(source_provenance_path))
        _remove_path_if_exists(artifact_paths["best_provenance"])

    if source_runtime_path is not None and source_runtime_path.exists():
        _copy_file(source_runtime_path, artifact_paths["best_runtime_environment"])
    else:
        if source_runtime_path is not None:
            missing_sources.append(str(source_runtime_path))
        _remove_path_if_exists(artifact_paths["best_runtime_environment"])

    source_proof_path = Path(best_record.proof_path) if best_record.proof_path else None
    source_proof_payload = _load_json_object(source_proof_path) if source_proof_path is not None else None
    source_proof_bin_name = "proof.bin"
    if source_proof_payload is not None:
        source_proof_bin_name = str(source_proof_payload.get("proof_path") or "proof.bin")
    source_proof_bin_path = (
        source_proof_path.parent / source_proof_bin_name
        if source_proof_path is not None
        else None
    )

    if source_proof_path is not None:
        if source_proof_path.exists():
            if source_proof_payload is None:
                missing_sources.append(str(source_proof_path))
                _remove_path_if_exists(artifact_paths["best_proof"])
                _remove_path_if_exists(artifact_paths["best_proof_bin"])
            elif source_proof_bin_path is None or not source_proof_bin_path.exists():
                if source_proof_bin_path is not None:
                    missing_sources.append(str(source_proof_bin_path))
                _remove_path_if_exists(artifact_paths["best_proof"])
                _remove_path_if_exists(artifact_paths["best_proof_bin"])
            else:
                _copy_file(source_proof_bin_path, artifact_paths["best_proof_bin"])
                proof_payload = dict(source_proof_payload)
                proof_payload["proof_path"] = "best_proof.bin"
                proof_payload["proof_bytes_sha256"] = sha256_file(artifact_paths["best_proof_bin"])
                if artifact_paths["best_provenance"].exists():
                    proof_payload["provenance_path"] = "best_provenance.json"
                _write_json(artifact_paths["best_proof"], proof_payload)
        else:
            missing_sources.append(str(source_proof_path))
            _remove_path_if_exists(artifact_paths["best_proof"])
            _remove_path_if_exists(artifact_paths["best_proof_bin"])
    else:
        _remove_path_if_exists(artifact_paths["best_proof"])
        _remove_path_if_exists(artifact_paths["best_proof_bin"])

    _remove_path_if_exists(bundle_paths["root"], record=False)
    bundle_missing: list[str] = []
    bundle_ready = source_provenance_path is not None and source_provenance_path.exists()
    if source_config_path is None or not source_config_path.exists():
        bundle_ready = False
        if source_config_path is not None:
            bundle_missing.append(str(source_config_path))
    if source_summary_path is None or not source_summary_path.exists():
        bundle_ready = False
        if source_summary_path is not None:
            bundle_missing.append(str(source_summary_path))
    if source_log_path is None or not source_log_path.exists():
        bundle_ready = False
        if source_log_path is not None:
            bundle_missing.append(str(source_log_path))
    if source_model_dir is None or not source_model_dir.exists():
        bundle_ready = False
        if source_model_dir is not None:
            bundle_missing.append(str(source_model_dir))
    if source_runtime_path is not None and not source_runtime_path.exists():
        bundle_ready = False
        bundle_missing.append(str(source_runtime_path))
    if source_proof_path is not None and (
        not source_proof_path.exists()
        or source_proof_payload is None
        or source_proof_bin_path is None
        or not source_proof_bin_path.exists()
    ):
        bundle_ready = False
        bundle_missing.append(str(source_proof_path))
        if source_proof_bin_path is not None and not source_proof_bin_path.exists():
            bundle_missing.append(str(source_proof_bin_path))

    if bundle_ready:
        _copy_file(source_config_path, bundle_paths["config"])
        _copy_file(source_summary_path, bundle_paths["summary"])
        _copy_file(source_log_path, bundle_paths["log"])
        _copy_file(source_provenance_path, bundle_paths["provenance"])
        if source_runtime_path is not None:
            _copy_file(source_runtime_path, bundle_paths["runtime_environment"])
        if source_model_dir is not None:
            _copy_tree(source_model_dir, bundle_paths["model_dir"])
        if source_proof_path is not None and source_proof_payload is not None and source_proof_bin_path is not None:
            _copy_file(source_proof_bin_path, bundle_paths["proof_bin"])
            bundle_proof_payload = dict(source_proof_payload)
            bundle_proof_payload["proof_path"] = "proof.bin"
            bundle_proof_payload["provenance_path"] = "provenance.json"
            bundle_proof_payload["proof_bytes_sha256"] = sha256_file(bundle_paths["proof_bin"])
            _write_json(bundle_paths["proof"], bundle_proof_payload)
    else:
        for item in bundle_missing:
            if item not in missing_sources:
                missing_sources.append(item)

    ok = not missing_sources
    return {
        "ok": ok,
        "output_dir": str(output_path),
        "best_record": {
            "experiment_id": best_record.experiment_id,
            "status": best_record.status,
            "objective_value": best_record.objective_value,
            "checkpoint_path": best_record.checkpoint_path,
            "provenance_path": best_record.provenance_path,
            "proof_path": best_record.proof_path,
        },
        "written": written,
        "removed": removed,
        "missing_sources": missing_sources,
        "notes": notes,
    }


def _attestation_summary_repair_core(payload: dict[str, Any]) -> dict[str, Any]:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"repair_history", "repair_report_verification"}
    }
    summary = dict(core.get("summary") or {})
    for key in (
        "repair_history_present",
        "repair_history_ok",
        "repair_history_entry_count",
        "repair_history_latest_issue_count",
        "repair_history_latest_issues",
        "repair_history_latest_repair_report_snapshot_ok",
        "repair_history_latest_signed_repair_report_snapshot_ok",
        "repair_history_latest_repair_report_snapshot_path",
        "repair_history_latest_repair_report_snapshot_expected_sha256",
        "repair_history_latest_repair_report_snapshot_actual_sha256",
        "repair_history_latest_repair_report_snapshot_expected_best_record",
        "repair_history_latest_repair_report_snapshot_actual_best_record",
        "repair_history_latest_repair_report_snapshot_actual_parse_error",
        "repair_history_latest_best_artifact_issue_count",
        "repair_history_latest_best_artifact_issues",
    ):
        summary.pop(key, None)
    core["summary"] = summary
    return core


def _audit_report_repair_core(payload: dict[str, Any]) -> dict[str, Any]:
    core = {
        key: value
        for key, value in payload.items()
        if key != "repair_history"
    }
    summary = dict(core.get("summary") or {})
    for key in (
        "repair_history_present",
        "repair_history_ok",
        "repair_history_entry_count",
        "repair_history_latest_issue_count",
        "repair_history_latest_issues",
        "repair_history_latest_repair_report_snapshot_ok",
        "repair_history_latest_signed_repair_report_snapshot_ok",
        "repair_history_latest_best_artifact_issue_count",
        "repair_history_latest_best_artifact_issues",
    ):
        summary.pop(key, None)
    core["summary"] = summary
    return core


def _repair_report_history_core(payload: dict[str, Any]) -> dict[str, Any]:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"repair_timestamp", "post_repair_repair_history_summary"}
    }

    for field in ("post_repair_audit", "post_repair_attestation"):
        value = core.get(field)
        if not isinstance(value, dict):
            continue
        normalized = {
            "path": value.get("path"),
            "ok": value.get("ok"),
        }
        digest = value.get("core_sha256") or value.get("sha256")
        if digest is not None:
            normalized["core_sha256"] = digest
        core[field] = normalized

    return core


def build_repair_report(
    output_dir: str | Path,
    *,
    repair_result: dict[str, Any],
    audit_report: dict[str, Any],
    attestation_summary: dict[str, Any],
    repair_timestamp: float | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    audit_path = output_path / "audit_report.json"
    attestation_path = output_path / "attestation_summary.json"
    attestation_core = _attestation_summary_repair_core(attestation_summary)

    written_entries = []
    for item in repair_result.get("written", []):
        item_path = Path(item)
        entry = {"path": _relativize(output_path, item_path), "present": item_path.exists()}
        if item_path.exists():
            entry["sha256"] = sha256_file(item_path)
        written_entries.append(entry)

    removed_entries = []
    for item in repair_result.get("removed", []):
        item_path = Path(item)
        removed_entries.append({
            "path": _relativize(output_path, item_path),
            "present": item_path.exists(),
        })

    report = {
        "event_type": "best_artifact_repair",
        "repair_timestamp": repair_timestamp or time.time(),
        "output_dir": str(output_path),
        "repair_ok": bool(repair_result.get("ok")),
        "best_record": repair_result.get("best_record"),
        "post_repair_best_record": attestation_summary.get("best_record"),
        "written": written_entries,
        "removed": removed_entries,
        "missing_sources": list(repair_result.get("missing_sources", [])),
        "notes": list(repair_result.get("notes", [])),
        "post_repair_audit": {
            "path": audit_path.name,
            "ok": audit_report.get("ok"),
            "core_sha256": sha256_json(_audit_report_repair_core(audit_report)),
        },
        "post_repair_attestation": {
            "path": attestation_path.name,
            "ok": attestation_summary.get("ok"),
            "core_sha256": sha256_json(attestation_core),
        },
    }
    signature = sign_json_payload_if_configured(report)
    report["post_repair_repair_history_summary"] = _project_repair_history_headline_summary(
        output_path,
        require_signature=bool(audit_report.get("require_signature") or attestation_summary.get("require_signature")),
        signature_present=signature is not None,
        attestation_summary=attestation_summary,
    )
    signature = sign_json_payload_if_configured(report)
    if signature is not None:
        report["signature"] = signature
    return report


def write_repair_report(
    output_dir: str | Path,
    *,
    repair_result: dict[str, Any],
    audit_report: dict[str, Any],
    attestation_summary: dict[str, Any],
    repair_timestamp: float | None = None,
) -> tuple[dict[str, Any], Path]:
    output_path = Path(output_dir)
    report = build_repair_report(
        output_path,
        repair_result=repair_result,
        audit_report=audit_report,
        attestation_summary=attestation_summary,
        repair_timestamp=repair_timestamp,
    )
    report_path = output_path / "repair_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    return report, report_path


def verify_repair_report(path: str | Path, *, require_signature: bool) -> dict[str, Any]:
    report_path = Path(path)
    payload = json.loads(report_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Repair report at {report_path} must be a JSON object")

    signature = payload.get("signature")
    core = {key: value for key, value in payload.items() if key != "signature"}
    checks: dict[str, bool] = {}
    check_details: dict[str, Any] = {}
    attestation_payload: dict[str, Any] | None = None
    computed_repair_history: dict[str, Any] | None = None

    output_dir = Path(str(core.get("output_dir") or report_path.parent))
    checks["event_type"] = core.get("event_type") == "best_artifact_repair"
    checks["output_dir"] = output_dir.resolve() == report_path.parent.resolve()

    for entry in core.get("written", []):
        if not isinstance(entry, dict):
            checks[f"written:{entry}"] = False
            continue
        rel_path = entry.get("path")
        target_path = output_dir / str(rel_path)
        key = f"written:{rel_path}"
        checks[key] = target_path.exists()
        if checks[key] and entry.get("sha256") is not None:
            checks[key] = sha256_file(target_path) == entry.get("sha256")

    for entry in core.get("removed", []):
        if not isinstance(entry, dict):
            checks[f"removed:{entry}"] = False
            continue
        rel_path = entry.get("path")
        target_path = output_dir / str(rel_path)
        checks[f"removed:{rel_path}"] = not target_path.exists()

    for label, report_key in (
        ("audit", "post_repair_audit"),
        ("attestation", "post_repair_attestation"),
    ):
        entry = core.get(report_key) or {}
        rel_path = entry.get("path")
        if rel_path is None:
            checks[label] = False
            continue
        target_path = output_dir / str(rel_path)
        checks[f"{label}:exists"] = target_path.exists()
        if not target_path.exists():
            continue
        expected_sha = entry.get("sha256")
        expected_core_sha = entry.get("core_sha256")
        checks[f"{label}:sha256"] = expected_sha is None or sha256_file(target_path) == expected_sha
        try:
            target_payload = json.loads(target_path.read_text())
        except Exception:
            checks[f"{label}:ok"] = False
            checks[f"{label}:core_sha256"] = expected_core_sha is None
        else:
            checks[f"{label}:ok"] = target_payload.get("ok") == entry.get("ok")
            if expected_core_sha is None:
                checks[f"{label}:core_sha256"] = True
            elif label == "attestation":
                attestation_payload = target_payload
                checks[f"{label}:core_sha256"] = (
                    sha256_json(_attestation_summary_repair_core(target_payload)) == expected_core_sha
                )
                expected_best_record = core.get("post_repair_best_record")
                checks[f"{label}:best_record"] = expected_best_record == target_payload.get("best_record")
            elif label == "audit":
                checks[f"{label}:core_sha256"] = (
                    sha256_json(_audit_report_repair_core(target_payload)) == expected_core_sha
                )
            else:
                checks[f"{label}:core_sha256"] = sha256_json(target_payload) == expected_core_sha

    expected_repair_history_summary: dict[str, Any] | None
    actual_repair_history_summary = core.get("post_repair_repair_history_summary")
    if _repair_history_dir(output_dir).exists():
        computed_repair_history = build_repair_history_summary(
            output_dir,
            require_signature=require_signature,
        )
        expected_repair_history_summary = _repair_history_headline_summary(
            computed_repair_history
        )
    elif attestation_payload is not None:
        expected_repair_history_summary = _project_repair_history_headline_summary(
            output_dir,
            require_signature=require_signature,
            signature_present=signature is not None,
            attestation_summary=attestation_payload,
        )
    else:
        expected_repair_history_summary = None
    expected_live_snapshot_ok = None
    expected_signed_snapshot_ok = None
    actual_live_snapshot_ok = None
    actual_signed_snapshot_ok = None
    if isinstance(expected_repair_history_summary, dict):
        expected_live_snapshot_ok = expected_repair_history_summary.get(
            "repair_history_latest_repair_report_snapshot_ok"
        )
        expected_signed_snapshot_ok = expected_repair_history_summary.get(
            "repair_history_latest_signed_repair_report_snapshot_ok"
        )
    if isinstance(actual_repair_history_summary, dict):
        actual_live_snapshot_ok = actual_repair_history_summary.get(
            "repair_history_latest_repair_report_snapshot_ok"
        )
        actual_signed_snapshot_ok = actual_repair_history_summary.get(
            "repair_history_latest_signed_repair_report_snapshot_ok"
        )

    checks["repair_history_summary"] = (
        isinstance(actual_repair_history_summary, dict)
        and expected_repair_history_summary is not None
        and actual_repair_history_summary == expected_repair_history_summary
    )
    checks["repair_history_latest_repair_report_snapshot_status"] = (
        isinstance(expected_live_snapshot_ok, bool)
        and isinstance(actual_live_snapshot_ok, bool)
        and expected_live_snapshot_ok == actual_live_snapshot_ok
    )
    checks["repair_history_latest_signed_repair_report_snapshot_status"] = (
        isinstance(expected_signed_snapshot_ok, bool)
        and isinstance(actual_signed_snapshot_ok, bool)
        and expected_signed_snapshot_ok == actual_signed_snapshot_ok
    )
    check_details["repair_history_summary"] = {
        "actual": actual_repair_history_summary,
        "expected": expected_repair_history_summary,
    }
    check_details["repair_history_latest_repair_report_snapshot_status"] = {
        "actual": actual_live_snapshot_ok,
        "expected": expected_live_snapshot_ok,
    }
    check_details["repair_history_latest_signed_repair_report_snapshot_status"] = {
        "actual": actual_signed_snapshot_ok,
        "expected": expected_signed_snapshot_ok,
    }

    if signature is None:
        signature_valid = None if not require_signature else False
        signature_reason = "Signature not present"
    else:
        signature_valid, signature_reason = verify_json_signature(core, signature)
        if not require_signature and signature_reason == "No verification key configured":
            signature_valid = None
            signature_reason = "Verification key not configured"

    repair_history_classification = "summary_unavailable"
    if expected_repair_history_summary is not None and isinstance(actual_repair_history_summary, dict):
        live_status_ok = checks["repair_history_latest_repair_report_snapshot_status"]
        signed_status_ok = checks["repair_history_latest_signed_repair_report_snapshot_status"]
        if live_status_ok and signed_status_ok:
            repair_history_classification = (
                "match" if checks["repair_history_summary"] else "summary_mismatch"
            )
        elif not live_status_ok and not signed_status_ok:
            repair_history_classification = "live_and_signed_snapshot_mismatch"
        elif not live_status_ok:
            repair_history_classification = "live_snapshot_mismatch"
        else:
            repair_history_classification = "signed_snapshot_mismatch"

    computed_latest_issues = [
        str(item)
        for item in (
            []
            if not isinstance(expected_repair_history_summary, dict)
            else expected_repair_history_summary.get("repair_history_latest_issues") or []
        )
    ]
    recorded_latest_issues = [
        str(item)
        for item in (
            []
            if not isinstance(actual_repair_history_summary, dict)
            else actual_repair_history_summary.get("repair_history_latest_issues") or []
        )
    ]
    repair_history_primary_issue = None
    repair_history_primary_issue_origin = None
    if computed_latest_issues:
        repair_history_primary_issue = computed_latest_issues[0]
        repair_history_primary_issue_origin = "computed"
    elif recorded_latest_issues:
        repair_history_primary_issue = recorded_latest_issues[0]
        repair_history_primary_issue_origin = "recorded"
    elif not checks["repair_history_latest_repair_report_snapshot_status"]:
        repair_history_primary_issue = "latest_repair_report_snapshot"
        repair_history_primary_issue_origin = "derived"
    elif not checks["repair_history_latest_signed_repair_report_snapshot_status"]:
        repair_history_primary_issue = "post_repair_repair_history_snapshot_summary"
        repair_history_primary_issue_origin = "derived"
    elif not checks["repair_history_summary"]:
        repair_history_primary_issue = "repair_history_summary"
        repair_history_primary_issue_origin = "derived"

    repair_history_primary_issue_details = None
    if repair_history_primary_issue == "latest_repair_report" and isinstance(computed_repair_history, dict):
        repair_history_primary_issue_details = {
            "path": computed_repair_history.get("latest_repair_report_path"),
            "expected_core_sha256": computed_repair_history.get("latest_repair_report_expected_core_sha256"),
            "actual_core_sha256": computed_repair_history.get("latest_repair_report_actual_core_sha256"),
            "actual_parse_error": computed_repair_history.get("latest_repair_report_actual_parse_error"),
        }
    elif repair_history_primary_issue == "latest_repair_report_snapshot" and isinstance(computed_repair_history, dict):
        repair_history_primary_issue_details = {
            "path": computed_repair_history.get("latest_repair_report_snapshot_path"),
            "expected_sha256": computed_repair_history.get("latest_repair_report_snapshot_expected_sha256"),
            "actual_sha256": computed_repair_history.get("latest_repair_report_snapshot_actual_sha256"),
            "expected_best_record": computed_repair_history.get("latest_repair_report_snapshot_expected_best_record"),
            "actual_best_record": computed_repair_history.get("latest_repair_report_snapshot_actual_best_record"),
            "actual_parse_error": computed_repair_history.get("latest_repair_report_snapshot_actual_parse_error"),
        }
    elif (
        repair_history_primary_issue == "post_repair_repair_history_snapshot_summary"
        and isinstance(computed_repair_history, dict)
    ):
        repair_history_primary_issue_details = {
            "path": computed_repair_history.get("latest_signed_repair_report_snapshot_path"),
            "expected_sha256": computed_repair_history.get("latest_signed_repair_report_snapshot_expected_sha256"),
            "actual_sha256": computed_repair_history.get("latest_signed_repair_report_snapshot_actual_sha256"),
            "expected_best_record": computed_repair_history.get("latest_signed_repair_report_snapshot_expected_best_record"),
            "actual_best_record": computed_repair_history.get("latest_signed_repair_report_snapshot_actual_best_record"),
            "actual_parse_error": computed_repair_history.get("latest_signed_repair_report_snapshot_actual_parse_error"),
        }

    ok = all(checks.values()) and (signature_valid if signature_valid is not None else True)
    return {
        "ok": ok,
        "path": str(report_path),
        "signature_present": signature is not None,
        "signature_valid": signature_valid,
        "signature_reason": signature_reason,
        "checks": checks,
        "check_details": check_details,
        "repair_history_classification": repair_history_classification,
        "repair_history_status": {
            "classification": repair_history_classification,
            "expected_live_snapshot_ok": expected_live_snapshot_ok,
            "actual_live_snapshot_ok": actual_live_snapshot_ok,
            "expected_signed_snapshot_ok": expected_signed_snapshot_ok,
            "actual_signed_snapshot_ok": actual_signed_snapshot_ok,
        },
        "repair_history_computed_latest_issues": computed_latest_issues,
        "repair_history_recorded_latest_issues": recorded_latest_issues,
        "repair_history_primary_issue": repair_history_primary_issue,
        "repair_history_primary_issue_origin": repair_history_primary_issue_origin,
        "repair_history_primary_issue_details": repair_history_primary_issue_details,
    }


def summarize_repair_report_verification(result: dict[str, Any]) -> dict[str, Any]:
    repair_history_status = result.get("repair_history_status") or {}
    checks = result.get("checks") or {}
    failed_checks = sorted(name for name, ok in checks.items() if not ok)
    return {
        "path": result.get("path"),
        "ok": result.get("ok"),
        "signature_present": result.get("signature_present"),
        "signature_valid": result.get("signature_valid"),
        "repair_history_classification": result.get("repair_history_classification"),
        "repair_history_status": {
            "classification": repair_history_status.get("classification"),
            "expected_live_snapshot_ok": repair_history_status.get("expected_live_snapshot_ok"),
            "actual_live_snapshot_ok": repair_history_status.get("actual_live_snapshot_ok"),
            "expected_signed_snapshot_ok": repair_history_status.get("expected_signed_snapshot_ok"),
            "actual_signed_snapshot_ok": repair_history_status.get("actual_signed_snapshot_ok"),
        },
        "repair_history_computed_latest_issues": list(
            result.get("repair_history_computed_latest_issues") or []
        ),
        "repair_history_recorded_latest_issues": list(
            result.get("repair_history_recorded_latest_issues") or []
        ),
        "repair_history_primary_issue": result.get("repair_history_primary_issue"),
        "repair_history_primary_issue_origin": result.get("repair_history_primary_issue_origin"),
        "repair_history_primary_issue_details": result.get("repair_history_primary_issue_details"),
        "failed_checks": failed_checks,
    }



def _repair_history_dir(output_path: Path) -> Path:
    return output_path / "repair_history"


def write_repair_history_entry(
    output_dir: str | Path,
    *,
    repair_report: dict[str, Any],
    history_timestamp: float | None = None,
) -> tuple[dict[str, Any], Path]:
    output_path = Path(output_dir)
    history_dir = _repair_history_dir(output_path)
    history_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(history_dir.glob("repair_*.json"))
    sequence_number = len(existing) + 1
    previous_path = existing[-1] if existing else None
    previous_rel = None if previous_path is None else _relativize(output_path, previous_path)
    previous_sha256 = None if previous_path is None else sha256_file(previous_path)

    repair_report_path = output_path / "repair_report.json"
    repair_report_core = {
        key: value
        for key, value in repair_report.items()
        if key != "signature"
    }
    entry = {
        "event_type": "best_artifact_repair_history",
        "history_timestamp": history_timestamp or time.time(),
        "output_dir": str(output_path),
        "sequence_number": sequence_number,
        "repair_report_path": repair_report_path.name,
        "repair_report_sha256": sha256_file(repair_report_path),
        "repair_report_core_sha256": sha256_json(_repair_report_history_core(repair_report_core)),
        "repair_report_snapshot": repair_report_core,
        "post_repair_repair_history_summary": repair_report_core.get("post_repair_repair_history_summary") or {},
        "post_repair_repair_history_snapshot_summary": _signed_repair_history_snapshot_summary(
            output_path,
            repair_report_path=repair_report_path.name,
            repair_report_snapshot=repair_report_core,
        ),
        "repair_report_signature_present": repair_report.get("signature") is not None,
        "previous_entry_path": previous_rel,
        "previous_entry_sha256": previous_sha256,
    }
    signature = sign_json_payload_if_configured(entry)
    if signature is not None:
        entry["signature"] = signature

    entry_path = history_dir / f"repair_{sequence_number:04d}.json"
    entry_path.write_text(json.dumps(entry, indent=2, default=str))
    return entry, entry_path


def verify_repair_history(path: str | Path, *, require_signature: bool) -> dict[str, Any]:
    target = Path(path)
    if target.is_dir() and target.name == "repair_history":
        history_dir = target
        output_dir = target.parent
    elif target.is_dir():
        history_dir = _repair_history_dir(target)
        output_dir = target
    else:
        history_dir = target.parent
        output_dir = history_dir.parent if history_dir.name == "repair_history" else target.parent

    if target.is_file():
        entry_paths = [target]
    else:
        entry_paths = sorted(history_dir.glob("repair_*.json"))
        if not entry_paths:
            raise FileNotFoundError(f"No repair history found under {history_dir}")

    checks: dict[str, bool] = {}
    entries: list[dict[str, Any]] = []
    previous_path: Path | None = None
    previous_sha256: str | None = None

    for index, entry_path in enumerate(entry_paths, start=1):
        payload = json.loads(entry_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Repair history entry at {entry_path} must be a JSON object")

        signature = payload.get("signature")
        core = {key: value for key, value in payload.items() if key != "signature"}
        entry_checks: dict[str, bool] = {}
        entry_check_details: dict[str, Any] = {}
        entry_checks["event_type"] = core.get("event_type") == "best_artifact_repair_history"
        entry_checks["output_dir"] = Path(str(core.get("output_dir") or output_dir)).resolve() == output_dir.resolve()
        entry_checks["sequence_number"] = int(core.get("sequence_number", 0)) == index

        previous_entry_path = core.get("previous_entry_path")
        previous_entry_sha256 = core.get("previous_entry_sha256")
        if index == 1:
            entry_checks["previous_entry"] = previous_entry_path is None and previous_entry_sha256 is None
        else:
            expected_prev_path = None if previous_path is None else _relativize(output_dir, previous_path)
            entry_checks["previous_entry"] = (
                previous_entry_path == expected_prev_path
                and previous_entry_sha256 == previous_sha256
            )

        repair_report_snapshot = core.get("repair_report_snapshot") or {}
        normalized_repair_report_snapshot = None
        repair_report_path_value = str(core.get("repair_report_path") or "repair_report.json")
        signed_snapshot_summary = core.get("post_repair_repair_history_snapshot_summary") or {}
        entry_checks["repair_report_snapshot"] = (
            isinstance(repair_report_snapshot, dict)
            and repair_report_snapshot.get("event_type") == "best_artifact_repair"
        )
        if entry_checks["repair_report_snapshot"]:
            normalized_repair_report_snapshot = _repair_report_history_core(repair_report_snapshot)
        entry_checks["post_repair_repair_history_summary"] = (
            (core.get("post_repair_repair_history_summary") or {})
            == (repair_report_snapshot.get("post_repair_repair_history_summary") or {})
        )
        expected_signed_snapshot_summary = None
        if entry_checks["repair_report_snapshot"]:
            expected_signed_snapshot_summary = _signed_repair_history_snapshot_summary(
                output_dir,
                repair_report_path=repair_report_path_value,
                repair_report_snapshot=repair_report_snapshot,
            )
        entry_checks["post_repair_repair_history_snapshot_summary"] = (
            isinstance(signed_snapshot_summary, dict)
            and expected_signed_snapshot_summary is not None
            and signed_snapshot_summary == expected_signed_snapshot_summary
        )
        entry_check_details["post_repair_repair_history_snapshot_summary"] = {
            "expected": expected_signed_snapshot_summary,
            "actual": signed_snapshot_summary,
        }

        if index == len(entry_paths):
            latest_report_path = output_dir / str(core.get("repair_report_path") or "repair_report.json")
            actual_report_sha256 = None
            expected_report_core_sha256 = (
                core.get("repair_report_core_sha256")
                or (None if normalized_repair_report_snapshot is None else sha256_json(normalized_repair_report_snapshot))
            )
            entry_checks["latest_repair_report"] = latest_report_path.exists()
            if latest_report_path.exists():
                actual_report_sha256 = sha256_file(latest_report_path)
                try:
                    latest_report_payload = json.loads(latest_report_path.read_text())
                except Exception as exc:
                    entry_checks["latest_repair_report"] = False
                    entry_checks["latest_repair_report_snapshot"] = False
                    entry_check_details["latest_repair_report"] = {
                        "path": str(latest_report_path),
                        "expected_core_sha256": expected_report_core_sha256,
                        "actual_core_sha256": None,
                        "actual_parse_error": str(exc),
                    }
                    entry_check_details["latest_repair_report_snapshot"] = {
                        "path": str(latest_report_path),
                        "expected_sha256": expected_report_core_sha256,
                        "actual_sha256": actual_report_sha256,
                        "expected_post_repair_best_record": repair_report_snapshot.get("post_repair_best_record"),
                        "actual_post_repair_best_record": None,
                        "actual_parse_error": str(exc),
                    }
                else:
                    latest_report_core = {key: value for key, value in latest_report_payload.items() if key != "signature"}
                    normalized_latest_report_core = _repair_report_history_core(latest_report_core)
                    actual_report_core_sha256 = sha256_json(normalized_latest_report_core)
                    entry_checks["latest_repair_report"] = (
                        expected_report_core_sha256 is not None
                        and actual_report_core_sha256 == expected_report_core_sha256
                    )
                    entry_checks["latest_repair_report_snapshot"] = (
                        normalized_latest_report_core == normalized_repair_report_snapshot
                    )
                    entry_check_details["latest_repair_report"] = {
                        "path": str(latest_report_path),
                        "expected_core_sha256": expected_report_core_sha256,
                        "actual_core_sha256": actual_report_core_sha256,
                        "actual_parse_error": None,
                    }
                    entry_check_details["latest_repair_report_snapshot"] = {
                        "path": str(latest_report_path),
                        "expected_sha256": expected_report_core_sha256,
                        "actual_sha256": actual_report_core_sha256,
                        "expected_post_repair_best_record": repair_report_snapshot.get("post_repair_best_record"),
                        "actual_post_repair_best_record": latest_report_core.get("post_repair_best_record"),
                    }
            else:
                entry_checks["latest_repair_report_snapshot"] = False
                entry_check_details["latest_repair_report"] = {
                    "path": str(latest_report_path),
                    "expected_core_sha256": expected_report_core_sha256,
                    "actual_core_sha256": None,
                    "actual_parse_error": None,
                }
                entry_check_details["latest_repair_report_snapshot"] = {
                    "path": str(latest_report_path),
                    "expected_sha256": expected_report_core_sha256,
                    "actual_sha256": None,
                    "expected_post_repair_best_record": repair_report_snapshot.get("post_repair_best_record"),
                    "actual_post_repair_best_record": None,
                }
        else:
            entry_checks["latest_repair_report"] = True
            entry_checks["latest_repair_report_snapshot"] = True

        if signature is None:
            signature_valid = None if not require_signature else False
            signature_reason = "Signature not present"
        else:
            signature_valid, signature_reason = verify_json_signature(core, signature)
            if not require_signature and signature_reason == "No verification key configured":
                signature_valid = None
                signature_reason = "Verification key not configured"

        entries.append(
            {
                "path": str(entry_path),
                "sequence_number": core.get("sequence_number"),
                "ok": all(entry_checks.values()) and (signature_valid if signature_valid is not None else True),
                "signature_present": signature is not None,
                "signature_valid": signature_valid,
                "signature_reason": signature_reason,
                "checks": entry_checks,
                "check_details": entry_check_details,
            }
        )
        for name, value in entry_checks.items():
            checks[f"{entry_path.name}:{name}"] = value
        checks[f"{entry_path.name}:signature"] = signature_valid if signature_valid is not None else True

        previous_path = entry_path
        previous_sha256 = sha256_file(entry_path)

    ok = all(checks.values())
    return {
        "ok": ok,
        "path": str(target),
        "history_dir": str(history_dir),
        "entry_count": len(entries),
        "entries": entries,
        "checks": checks,
    }


def build_repair_history_summary(
    output_dir: str | Path,
    *,
    require_signature: bool,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    history_dir = _repair_history_dir(output_path)
    if not history_dir.exists():
        return {
            "present": False,
            "ok": True,
            "history_dir": str(history_dir),
            "entry_count": 0,
            "latest_entry_path": None,
            "latest_sequence_number": None,
            "latest_history_timestamp": None,
            "latest_repair_report_path": None,
            "latest_repair_ok": None,
            "latest_repair_report_integrity_ok": None,
            "latest_repair_report_expected_core_sha256": None,
            "latest_repair_report_actual_core_sha256": None,
            "latest_repair_report_actual_parse_error": None,
            "latest_repair_report_snapshot_ok": None,
            "latest_signed_repair_report_snapshot_ok": None,
            "latest_repair_report_snapshot_path": None,
            "latest_repair_report_snapshot_expected_sha256": None,
            "latest_repair_report_snapshot_actual_sha256": None,
            "latest_repair_report_snapshot_expected_best_record": None,
            "latest_repair_report_snapshot_actual_best_record": None,
            "latest_repair_report_snapshot_actual_parse_error": None,
            "latest_signed_repair_report_snapshot_path": None,
            "latest_signed_repair_report_snapshot_expected_sha256": None,
            "latest_signed_repair_report_snapshot_actual_sha256": None,
            "latest_signed_repair_report_snapshot_expected_best_record": None,
            "latest_signed_repair_report_snapshot_actual_best_record": None,
            "latest_signed_repair_report_snapshot_actual_parse_error": None,
            "latest_issue_count": 0,
            "latest_issues": [],
            "latest_primary_issue": None,
            "latest_best_record_experiment_id": None,
            "latest_best_record_audit_best_artifacts_ok": None,
            "latest_best_record_audit_best_artifact_issue_count": None,
            "latest_best_record_audit_best_artifact_issues": [],
            "latest_signature_present": False,
            "latest_signature_valid": None,
        }

    verification = verify_repair_history(output_path, require_signature=require_signature)
    latest_entry_summary = verification["entries"][-1] if verification["entries"] else None
    latest_payload: dict[str, Any] = {}
    latest_checks: dict[str, Any] = {}
    latest_check_details: dict[str, Any] = {}
    latest_snapshot_detail: dict[str, Any] = {}
    latest_report_detail: dict[str, Any] = {}
    latest_signed_snapshot_detail: dict[str, Any] = {}
    latest_issues: list[str] = []
    if latest_entry_summary is not None:
        latest_payload = json.loads(Path(latest_entry_summary["path"]).read_text())
        latest_checks = latest_entry_summary.get("checks") or {}
        latest_check_details = latest_entry_summary.get("check_details") or {}
        latest_snapshot_detail = latest_check_details.get("latest_repair_report_snapshot") or {}
        latest_report_detail = latest_check_details.get("latest_repair_report") or {}
        latest_signed_snapshot_detail = latest_payload.get("post_repair_repair_history_snapshot_summary") or {}
        latest_issues = sorted(name for name, value in latest_checks.items() if not value)
        if latest_entry_summary.get("signature_valid") is False:
            latest_issues.append("signature")
    repair_snapshot = latest_payload.get("repair_report_snapshot") or {}
    best_record = repair_snapshot.get("best_record") or {}
    post_repair_best_record = repair_snapshot.get("post_repair_best_record") or {}

    return {
        "present": True,
        "ok": verification["ok"],
        "history_dir": verification["history_dir"],
        "entry_count": verification["entry_count"],
        "latest_entry_path": None if latest_entry_summary is None else latest_entry_summary["path"],
        "latest_sequence_number": latest_payload.get("sequence_number"),
        "latest_history_timestamp": latest_payload.get("history_timestamp"),
        "latest_repair_report_path": latest_payload.get("repair_report_path"),
        "latest_repair_ok": repair_snapshot.get("repair_ok"),
        "latest_repair_report_integrity_ok": latest_checks.get("latest_repair_report"),
        "latest_repair_report_expected_core_sha256": latest_report_detail.get("expected_core_sha256"),
        "latest_repair_report_actual_core_sha256": latest_report_detail.get("actual_core_sha256"),
        "latest_repair_report_actual_parse_error": latest_report_detail.get("actual_parse_error"),
        "latest_repair_report_snapshot_ok": latest_checks.get("latest_repair_report_snapshot"),
        "latest_signed_repair_report_snapshot_ok": latest_checks.get("post_repair_repair_history_snapshot_summary"),
        "latest_repair_report_snapshot_path": latest_snapshot_detail.get("path"),
        "latest_repair_report_snapshot_expected_sha256": latest_snapshot_detail.get("expected_sha256"),
        "latest_repair_report_snapshot_actual_sha256": latest_snapshot_detail.get("actual_sha256"),
        "latest_repair_report_snapshot_expected_best_record": latest_snapshot_detail.get("expected_post_repair_best_record"),
        "latest_repair_report_snapshot_actual_best_record": latest_snapshot_detail.get("actual_post_repair_best_record"),
        "latest_repair_report_snapshot_actual_parse_error": latest_snapshot_detail.get("actual_parse_error"),
        "latest_signed_repair_report_snapshot_path": latest_signed_snapshot_detail.get("repair_history_latest_repair_report_snapshot_path"),
        "latest_signed_repair_report_snapshot_expected_sha256": latest_signed_snapshot_detail.get("repair_history_latest_repair_report_snapshot_expected_sha256"),
        "latest_signed_repair_report_snapshot_actual_sha256": latest_signed_snapshot_detail.get("repair_history_latest_repair_report_snapshot_actual_sha256"),
        "latest_signed_repair_report_snapshot_expected_best_record": latest_signed_snapshot_detail.get("repair_history_latest_repair_report_snapshot_expected_best_record"),
        "latest_signed_repair_report_snapshot_actual_best_record": latest_signed_snapshot_detail.get("repair_history_latest_repair_report_snapshot_actual_best_record"),
        "latest_signed_repair_report_snapshot_actual_parse_error": latest_signed_snapshot_detail.get("repair_history_latest_repair_report_snapshot_actual_parse_error"),
        "latest_issue_count": len(latest_issues),
        "latest_issues": latest_issues,
        "latest_primary_issue": None if not latest_issues else latest_issues[0],
        "latest_best_record_experiment_id": best_record.get("experiment_id"),
        "latest_best_record_audit_best_artifacts_ok": post_repair_best_record.get("audit_best_artifacts_ok"),
        "latest_best_record_audit_best_artifact_issue_count": post_repair_best_record.get("audit_best_artifact_issue_count"),
        "latest_best_record_audit_best_artifact_issues": list(post_repair_best_record.get("audit_best_artifact_issues") or []),
        "latest_signature_present": None if latest_entry_summary is None else latest_entry_summary["signature_present"],
        "latest_signature_valid": None if latest_entry_summary is None else latest_entry_summary["signature_valid"],
    }


def write_audit_report(
    output_dir: str | Path,
    *,
    require_signature: bool,
    report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    output_path = Path(output_dir)
    resolved = report or build_audit_report(output_path, require_signature=require_signature)
    audit_path = output_path / "audit_report.json"
    audit_path.write_text(json.dumps(resolved, indent=2, default=str))
    return resolved, audit_path


def _repair_history_headline_summary(repair_history: dict[str, Any]) -> dict[str, Any]:
    present = repair_history.get("present", False)
    latest_issues = set(repair_history.get("latest_issues") or [])

    live_snapshot_ok = repair_history.get("latest_repair_report_snapshot_ok")
    if live_snapshot_ok is None and present:
        live_snapshot_ok = "latest_repair_report_snapshot" not in latest_issues

    signed_snapshot_ok = repair_history.get("latest_signed_repair_report_snapshot_ok")
    if signed_snapshot_ok is None and present:
        signed_snapshot_ok = "post_repair_repair_history_snapshot_summary" not in latest_issues

    return {
        "repair_history_present": present,
        "repair_history_ok": repair_history.get("ok", True),
        "repair_history_entry_count": repair_history.get("entry_count", 0),
        "repair_history_latest_issue_count": repair_history.get("latest_issue_count", 0),
        "repair_history_latest_issues": list(repair_history.get("latest_issues") or []),
        "repair_history_latest_primary_issue": repair_history.get("latest_primary_issue"),
        "repair_history_latest_repair_report_snapshot_ok": live_snapshot_ok,
        "repair_history_latest_signed_repair_report_snapshot_ok": signed_snapshot_ok,
        "repair_history_latest_best_artifact_issue_count": (
            repair_history.get("latest_best_record_audit_best_artifact_issue_count") or 0
        ),
        "repair_history_latest_best_artifact_issues": list(
            repair_history.get("latest_best_record_audit_best_artifact_issues") or []
        ),
    }


def _repair_history_snapshot_summary(repair_history: dict[str, Any]) -> dict[str, Any]:
    return {
        "repair_history_latest_repair_report_snapshot_path": repair_history.get("latest_repair_report_snapshot_path"),
        "repair_history_latest_repair_report_snapshot_expected_sha256": repair_history.get("latest_repair_report_snapshot_expected_sha256"),
        "repair_history_latest_repair_report_snapshot_actual_sha256": repair_history.get("latest_repair_report_snapshot_actual_sha256"),
        "repair_history_latest_repair_report_snapshot_expected_best_record": repair_history.get("latest_repair_report_snapshot_expected_best_record"),
        "repair_history_latest_repair_report_snapshot_actual_best_record": repair_history.get("latest_repair_report_snapshot_actual_best_record"),
        "repair_history_latest_repair_report_snapshot_actual_parse_error": repair_history.get("latest_repair_report_snapshot_actual_parse_error"),
    }


def _signed_repair_history_snapshot_summary(
    output_dir: str | Path,
    *,
    repair_report_path: str,
    repair_report_snapshot: dict[str, Any],
) -> dict[str, Any]:
    output_path = Path(output_dir)
    latest_report_path = output_path / repair_report_path
    report_sha256 = sha256_json(repair_report_snapshot)
    best_record = repair_report_snapshot.get("post_repair_best_record")
    return {
        "repair_history_latest_repair_report_snapshot_path": str(latest_report_path),
        "repair_history_latest_repair_report_snapshot_expected_sha256": report_sha256,
        "repair_history_latest_repair_report_snapshot_actual_sha256": report_sha256,
        "repair_history_latest_repair_report_snapshot_expected_best_record": best_record,
        "repair_history_latest_repair_report_snapshot_actual_best_record": best_record,
        "repair_history_latest_repair_report_snapshot_actual_parse_error": None,
    }


def _project_repair_history_headline_summary(
    output_dir: str | Path,
    *,
    require_signature: bool,
    signature_present: bool,
    attestation_summary: dict[str, Any],
) -> dict[str, Any]:
    output_path = Path(output_dir)
    existing = build_repair_history_summary(output_path, require_signature=require_signature)
    best_record = attestation_summary.get("best_record") or {}
    latest_issues: list[str] = []
    if require_signature and not signature_present:
        latest_issues.append("signature")
    projected = {
        "present": True,
        "ok": bool(existing.get("ok", True)) and not latest_issues,
        "entry_count": int(existing.get("entry_count") or 0) + 1,
        "latest_issue_count": len(latest_issues),
        "latest_issues": latest_issues,
        "latest_primary_issue": None if not latest_issues else latest_issues[0],
        "latest_best_record_audit_best_artifact_issue_count": (
            best_record.get("audit_best_artifact_issue_count") or 0
        ),
        "latest_best_record_audit_best_artifact_issues": list(
            best_record.get("audit_best_artifact_issues") or []
        ),
    }
    return _repair_history_headline_summary(projected)


def build_attestation_summary(
    output_dir: str | Path,
    *,
    require_signature: bool,
    audit_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    tracker = ExperimentTracker.load(output_path)
    report = audit_report or build_audit_report(output_path, require_signature=require_signature)
    repair_history = build_repair_history_summary(
        output_path,
        require_signature=require_signature,
    )
    summary_payload = dict(report.get("summary") or {})
    summary_payload.update(_repair_history_headline_summary(repair_history))
    summary_payload.update(_repair_history_snapshot_summary(repair_history))

    best_record = tracker.best_record
    best_audit = None
    if best_record is not None:
        best_audit = next(
            (
                item
                for item in report["experiments"]
                if item["experiment_id"] == best_record.experiment_id
            ),
            None,
        )

    best_artifacts = report.get("best_artifacts") or {}
    best_artifact_issues = _best_artifact_issues(best_artifacts)

    best_record_summary = None
    if best_record is not None:
        best_record_summary = {
            "experiment_id": best_record.experiment_id,
            "objective_value": best_record.objective_value,
            "status": best_record.status,
            "provenance_path": best_record.provenance_path,
            "proof_path": best_record.proof_path,
            "audit_provenance_status": None if best_audit is None else best_audit["provenance_status"],
            "audit_proof_status": None if best_audit is None else best_audit["proof_status"],
            "audit_best_artifacts_ok": best_artifacts.get("ok"),
            "audit_best_artifact_issue_count": len(best_artifact_issues),
            "audit_best_artifact_issues": best_artifact_issues,
        }

    return {
        "ok": report["ok"],
        "output_dir": str(output_path),
        "require_signature": require_signature,
        "audit_report_path": str(output_path / "audit_report.json"),
        "summary": summary_payload,
        "best_record": best_record_summary,
        "best_artifacts": report.get("best_artifacts") or {},
        "repair_history": repair_history,
    }


def write_attestation_summary(
    output_dir: str | Path,
    *,
    require_signature: bool,
    audit_report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    output_path = Path(output_dir)
    summary = build_attestation_summary(
        output_path,
        require_signature=require_signature,
        audit_report=audit_report,
    )
    summary_path = output_path / "attestation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    repair_report_path = output_path / "repair_report.json"
    if repair_report_path.exists():
        summary["repair_report_verification"] = summarize_repair_report_verification(
            verify_repair_report(repair_report_path, require_signature=require_signature)
        )
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    return summary, summary_path
