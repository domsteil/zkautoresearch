"""Autonomous Research entrypoint with subprocess and in-process runners."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from attestation_audit import (
    build_audit_report,
    build_repair_history_summary,
    rebuild_best_artifacts,
    summarize_repair_report_verification,
    verify_proof_artifact,
    verify_repair_history,
    verify_repair_report,
    write_attestation_summary,
    write_audit_report,
    write_repair_history_entry,
    write_repair_report,
)
from experiment_runtime import detect_device_info
from prepare import EVAL_SCENARIOS, LOCKED_SELECTION_METRIC_NAME, TRAIN_SCENARIOS
from subprocess_auto_research import SubprocessAutoResearchRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("auto_research")

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert customer service agent. Follow these rules strictly: "
    "1) Greet the customer warmly. 2) Acknowledge their issue with empathy. "
    "3) Ask clarifying questions. 4) Provide a clear solution or next steps. "
    "5) Confirm resolution. Always be polite, professional, and concise."
)
DEFAULT_BASELINE = {
    "learning_rate": 5e-6,
    "num_generations": 2,
    "num_outer_iterations": 1,
    "generations_per_iteration": 1,
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
RUNTIME_PROFILES: dict[str, dict[str, Any]] = {
    "standard": {},
    "smoke": {
        "model_name": "Qwen/Qwen1.5-0.5B-Chat",
        "max_new_tokens": 4,
        "eval_episodes": 1,
        "train_scenario_limit": 1,
        "eval_scenario_limit": 1,
        "runtime_overrides": {
            "num_generations": 2,
            "num_outer_iterations": 1,
            "generations_per_iteration": 1,
            "lora_r": 4,
            "run_post_eval": False,
            "post_eval_samples": 0,
            "post_eval_detailed": False,
        },
    },
}


def _resolve_device_info(config_data: dict[str, Any]):
    requested_use_gpu = config_data.get("use_gpu")
    requested_bf16 = config_data.get("bf16", False)

    if requested_use_gpu is None:
        return detect_device_info(
            prefer_gpu=True,
            require_gpu=False,
            requested_bf16=requested_bf16,
        )

    use_gpu = bool(requested_use_gpu)
    return detect_device_info(
        prefer_gpu=use_gpu,
        require_gpu=use_gpu,
        requested_bf16=requested_bf16,
    )


def _load_yaml_config(path: str) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_profile(name: str) -> dict[str, Any]:
    try:
        return dict(RUNTIME_PROFILES[name])
    except KeyError as exc:
        available = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(f"Unknown runtime profile {name!r}. Available: {available}") from exc


def _resolve_with_profile(
    explicit_value: Any,
    config_value: Any,
    profile_value: Any,
    default: Any,
) -> Any:
    if explicit_value is not None:
        return explicit_value
    if profile_value is not None:
        return profile_value
    if config_value is not None:
        return config_value
    return default


def _slice_scenarios(
    scenarios: list[dict[str, Any]],
    limit: int | None,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if limit is None:
        return list(scenarios)
    if limit < 1:
        raise ValueError(f"{label} scenario limit must be >= 1, got {limit}")
    return list(scenarios[:limit])


def _best_artifact_issues(best_artifacts: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    checks = (best_artifacts or {}).get("checks") or {}
    issues: list[tuple[str, str, str | None]] = []
    for name, check in checks.items():
        if not isinstance(check, dict):
            continue
        status = str(check.get("status") or "")
        if status in {"fail", "missing"}:
            reason = check.get("reason")
            issues.append((name, status, None if reason is None else str(reason)))
    return issues


def _bool_status_label(
    value: bool | None,
    *,
    true_label: str = "ok",
    false_label: str = "fail",
    none_label: str = "unknown",
) -> str:
    if value is True:
        return true_label
    if value is False:
        return false_label
    return none_label


def _repair_report_verification_summary(result: dict[str, Any]) -> dict[str, Any]:
    return summarize_repair_report_verification(result)


def _repair_report_primary_issue_label(summary: dict[str, Any]) -> str:
    issue = summary.get("repair_history_primary_issue")
    origin = summary.get("repair_history_primary_issue_origin")
    if issue is None:
        return "none"
    if origin:
        return f"{issue}({origin})"
    return str(issue)


def _print_repair_history_summary(repair_history: dict[str, Any]) -> None:
    if not repair_history.get("present"):
        return

    latest_issues = set(repair_history.get("latest_issues") or [])

    print(f"\nRepair history: {'OK' if repair_history.get('ok') else 'FAIL'}")
    print(f"  entries: {repair_history.get('entry_count')}")
    primary_issue = repair_history.get("latest_primary_issue")
    primary_issue_label = "none" if primary_issue is None else str(primary_issue)
    print(
        "  summary: "
        f"latest_issues={repair_history.get('latest_issue_count', 0)} "
        f"primary_issue={primary_issue_label} "
        "latest_best_artifact_issues="
        f"{repair_history.get('latest_best_record_audit_best_artifact_issue_count', 0)}"
    )
    latest_artifacts_label = _bool_status_label(
        repair_history.get("latest_best_record_audit_best_artifacts_ok")
    )
    latest_report_label = _bool_status_label(
        repair_history.get("latest_repair_report_integrity_ok")
    )
    latest_snapshot_label = _bool_status_label(
        repair_history.get("latest_repair_report_snapshot_ok")
    )
    latest_signed_snapshot_label = _bool_status_label(
        repair_history.get("latest_signed_repair_report_snapshot_ok")
    )
    print(
        "  latest: "
        f"sequence={repair_history.get('latest_sequence_number')} "
        f"best={repair_history.get('latest_best_record_experiment_id')} "
        f"repair_ok={repair_history.get('latest_repair_ok')} "
        f"artifacts={latest_artifacts_label} "
        f"report={latest_report_label} "
        f"report_snapshot={latest_snapshot_label} "
        f"signed_snapshot={latest_signed_snapshot_label}"
    )
    if repair_history.get("latest_repair_report_integrity_ok") is False:
        print(
            "  latest_repair_report_path: "
            f"{repair_history.get('latest_repair_report_path')}"
        )
        print(
            "  latest_repair_report_expected_core_sha256: "
            f"{repair_history.get('latest_repair_report_expected_core_sha256')}"
        )
        print(
            "  latest_repair_report_actual_core_sha256: "
            f"{repair_history.get('latest_repair_report_actual_core_sha256')}"
        )
        parse_error = repair_history.get("latest_repair_report_actual_parse_error")
        if parse_error:
            print(f"  latest_repair_report_actual_parse_error: {parse_error}")
    if repair_history.get("latest_repair_report_snapshot_ok") is False:
        print(
            "  latest_repair_report_snapshot_expected_sha256: "
            f"{repair_history.get('latest_repair_report_snapshot_expected_sha256')}"
        )
        print(
            "  latest_repair_report_snapshot_actual_sha256: "
            f"{repair_history.get('latest_repair_report_snapshot_actual_sha256')}"
        )
        print(
            "  latest_repair_report_snapshot_expected_best_record: "
            f"{json.dumps(repair_history.get('latest_repair_report_snapshot_expected_best_record'), sort_keys=True)}"
        )
        print(
            "  latest_repair_report_snapshot_actual_best_record: "
            f"{json.dumps(repair_history.get('latest_repair_report_snapshot_actual_best_record'), sort_keys=True)}"
        )
        parse_error = repair_history.get("latest_repair_report_snapshot_actual_parse_error")
        if parse_error:
            print(f"  latest_repair_report_snapshot_actual_parse_error: {parse_error}")
    if "post_repair_repair_history_snapshot_summary" in latest_issues:
        print(
            "  latest_signed_repair_report_snapshot_path: "
            f"{repair_history.get('latest_signed_repair_report_snapshot_path')}"
        )
        print(
            "  latest_signed_repair_report_snapshot_expected_sha256: "
            f"{repair_history.get('latest_signed_repair_report_snapshot_expected_sha256')}"
        )
        print(
            "  latest_signed_repair_report_snapshot_actual_sha256: "
            f"{repair_history.get('latest_signed_repair_report_snapshot_actual_sha256')}"
        )
        print(
            "  latest_signed_repair_report_snapshot_expected_best_record: "
            f"{json.dumps(repair_history.get('latest_signed_repair_report_snapshot_expected_best_record'), sort_keys=True)}"
        )
        print(
            "  latest_signed_repair_report_snapshot_actual_best_record: "
            f"{json.dumps(repair_history.get('latest_signed_repair_report_snapshot_actual_best_record'), sort_keys=True)}"
        )
        signed_parse_error = repair_history.get("latest_signed_repair_report_snapshot_actual_parse_error")
        if signed_parse_error:
            print(f"  latest_signed_repair_report_snapshot_actual_parse_error: {signed_parse_error}")
    for issue in repair_history.get("latest_issues") or []:
        print(f"  repair_history_issue: {issue}")
    for item in repair_history.get("latest_best_record_audit_best_artifact_issues") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        status = item.get("status")
        reason = item.get("reason")
        detail = "" if reason is None else f" ({reason})"
        print(f"  repair_history_best_artifact_issue {name}: {status}{detail}")
    if repair_history.get("latest_signature_present"):
        signature_label = _bool_status_label(
            repair_history.get("latest_signature_valid"),
            false_label="invalid",
            none_label="unverified",
        )
        print(f"  latest_signature: {signature_label}")

def _run_analyze(output_dir: str) -> None:
    from stateset_agents.training.auto_research import ExperimentTracker
    from stateset_agents.training.auto_research.analysis import generate_report

    output_path = Path(output_dir)

    if (output_path / "experiments.jsonl").exists():
        tracker = ExperimentTracker.load(output_path)
        source = "platform results"
    elif Path("results.tsv").exists():
        tracker = ExperimentTracker.from_legacy_tsv("results.tsv")
        source = "legacy results.tsv"
    else:
        print(f"No results found in {output_dir} or results.tsv")
        sys.exit(1)

    print(f"Loaded {tracker.num_experiments} experiments from {source}")
    tracker.print_summary()

    if tracker.num_experiments >= 6:
        report = generate_report(
            tracker.records,
            objective_metric=tracker.objective_metric,
            direction=tracker.direction,
        )
        print(report)

    analysis = tracker.get_analysis()
    analysis_payload = dict(analysis)
    repair_history = build_repair_history_summary(output_path, require_signature=False)
    if repair_history.get("present"):
        analysis_payload["repair_history"] = repair_history
    repair_report_verification = None
    repair_report_path = output_path / "repair_report.json"
    if repair_report_path.exists():
        repair_report_verification = _repair_report_verification_summary(
            verify_repair_report(repair_report_path, require_signature=False)
        )
        analysis_payload["repair_report_verification"] = repair_report_verification
    analysis_path = output_path / "analysis.json" if output_path.exists() else Path("analysis.json")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis_payload, indent=2, default=str))
    print(f"Analysis saved to {analysis_path}")

    preflight_path = output_path / "preflight.json"
    if preflight_path.exists():
        preflight = json.loads(preflight_path.read_text())
        recommended = preflight.get("recommended_min_budget_seconds") or {}
        budget_recommendation = preflight.get("budget_recommendation") or {}
        print("\nPreflight:")
        print(f"  configured_time_budget_seconds: {preflight.get('configured_time_budget_seconds')}")
        print(f"  allow_undersized_budget: {preflight.get('allow_undersized_budget')}")
        print(
            "  recommended_min_budget_seconds: "
            f"baseline={recommended.get('baseline')} experiment={recommended.get('experiment')}"
        )
        baseline_detail = budget_recommendation.get("baseline") or {}
        experiment_detail = budget_recommendation.get("experiment") or {}
        if baseline_detail:
            print(
                "  baseline_budget_detail: "
                f"heuristic={baseline_detail.get('heuristic_seconds')} "
                f"observed_phase={baseline_detail.get('observed_phase_seconds')} "
                f"phase_samples={baseline_detail.get('historical_phase_samples')}"
            )
        if experiment_detail:
            print(
                "  experiment_budget_detail: "
                f"heuristic={experiment_detail.get('heuristic_seconds')} "
                f"observed_phase={experiment_detail.get('observed_phase_seconds')} "
                f"phase_samples={experiment_detail.get('historical_phase_samples')}"
            )

    attestation_summary_path = output_path / "attestation_summary.json"
    if attestation_summary_path.exists():
        attestation = json.loads(attestation_summary_path.read_text())
        attestation_summary = attestation.get("summary") or {}
        best_record = attestation.get("best_record") or {}
        print(f"\nAttestation: {'OK' if attestation.get('ok') else 'FAIL'}")
        print(f"  audit_report: {attestation.get('audit_report_path')}")
        print(
            "  provenance: "
            f"ok={attestation_summary.get('provenance_ok', 0)} "
            f"fail={attestation_summary.get('provenance_fail', 0)} "
            f"missing={attestation_summary.get('provenance_missing', 0)}"
        )
        print(
            "  proofs: "
            f"ok={attestation_summary.get('proofs_ok', 0)} "
            f"fail={attestation_summary.get('proofs_fail', 0)} "
            f"missing_required={attestation_summary.get('proofs_missing_required', 0)}"
        )
        print(
            "  best_artifacts: "
            f"ok={attestation_summary.get('best_artifacts_ok', 0)} "
            f"fail={attestation_summary.get('best_artifacts_fail', 0)} "
            f"missing={attestation_summary.get('best_artifacts_missing', 0)} "
            f"not_required={attestation_summary.get('best_artifacts_not_required', 0)}"
        )
        for name, status, reason in _best_artifact_issues(attestation.get("best_artifacts") or {}):
            detail = "" if reason is None else f" ({reason})"
            print(f"  best_artifact {name}: {status}{detail}")
        if best_record:
            best_artifacts_label = _bool_status_label(best_record.get("audit_best_artifacts_ok"))
            print(
                "  best: "
                f"{best_record.get('experiment_id')} "
                f"provenance={best_record.get('audit_provenance_status')} "
                f"proof={best_record.get('audit_proof_status')} "
                f"artifacts={best_artifacts_label}"
            )

    _print_repair_history_summary(repair_history)

    if repair_report_verification is not None:
        repair_report_status = repair_report_verification.get("repair_history_status") or {}
        print(
            f"\nRepair report verification: {'OK' if repair_report_verification.get('ok') else 'FAIL'}"
        )
        print(
            "  summary: "
            f"classification={repair_report_verification.get('repair_history_classification')} "
            f"primary_issue={_repair_report_primary_issue_label(repair_report_verification)} "
            f"live=expected:{_bool_status_label(repair_report_status.get('expected_live_snapshot_ok'))} "
            f"actual:{_bool_status_label(repair_report_status.get('actual_live_snapshot_ok'))} "
            f"signed=expected:{_bool_status_label(repair_report_status.get('expected_signed_snapshot_ok'))} "
            f"actual:{_bool_status_label(repair_report_status.get('actual_signed_snapshot_ok'))}"
        )
        if repair_report_verification.get("failed_checks"):
            print(
                "  failed_checks: "
                f"{', '.join(repair_report_verification.get('failed_checks') or [])}"
            )

    try:
        df = tracker.to_dataframe()
        print(f"\nDataFrame: {len(df)} rows, {len(df.columns)} columns")
        print(df[["id", "objective", "status", "description"]].to_string())
    except ImportError:
        pass


def _run_compare(run_dirs: list[str]) -> None:
    from stateset_agents.training.auto_research import compare_runs

    print(compare_runs(*run_dirs))


def _iter_provenance_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    direct = target / "provenance.json"
    if direct.exists():
        return [direct]

    candidates = sorted(target.glob("runs/*/provenance.json"))
    if candidates:
        return candidates

    raise FileNotFoundError(f"No provenance envelopes found under {target}")


def _run_verify_provenance(target: str, *, require_signature: bool) -> None:
    from experiment_runtime import verify_provenance_envelope

    target_path = Path(target)
    results = [
        verify_provenance_envelope(path, require_signature=require_signature)
        for path in _iter_provenance_paths(target_path)
    ]

    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        print(f"{status} {result['path']}")
        checks = result["checks"]
        failed_checks = [name for name, ok in checks.items() if not ok]
        if failed_checks:
            print(f"  failed_checks: {', '.join(failed_checks)}")
        if result["signature_present"]:
            print(f"  signature: {result['signature_reason']}")
        elif require_signature:
            print("  signature: missing")

    if not all(result["ok"] for result in results):
        sys.exit(1)


def _iter_proof_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    direct = target / "proof.json"
    if direct.exists():
        return [direct]

    run_candidates = sorted(target.glob("runs/*/proof.json"))
    if run_candidates:
        return run_candidates

    best = target / "best_proof.json"
    if best.exists():
        return [best]

    raise FileNotFoundError(f"No proof artifacts found under {target}")



def _verify_proof_artifact(path: Path, *, require_signature: bool) -> dict[str, Any]:
    return verify_proof_artifact(path, require_signature=require_signature)



def _run_verify_proof(target: str, *, require_signature: bool) -> None:
    target_path = Path(target)
    results = [
        _verify_proof_artifact(path, require_signature=require_signature)
        for path in _iter_proof_paths(target_path)
    ]

    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        print(f"{status} {result['path']}")
        print(f"  proof_bytes_sha256: {'ok' if result['proof_bytes_sha256_ok'] else 'mismatch'}")
        if result["provenance_path"]:
            print(f"  provenance: {result['provenance_path']}")
        print(f"  verify: {result['verification_message']}")

    if not all(result["ok"] for result in results):
        sys.exit(1)


def _build_audit_report(output_dir: str | Path, *, require_signature: bool) -> dict[str, Any]:
    return build_audit_report(output_dir, require_signature=require_signature)



def _run_audit_run(output_dir: str, *, require_signature: bool) -> None:
    report, audit_path = write_audit_report(output_dir, require_signature=require_signature)
    repair_history = report.get("repair_history")
    if not isinstance(repair_history, dict):
        repair_history = build_repair_history_summary(output_dir, require_signature=require_signature)

    summary = report["summary"]
    print(f"Loaded {summary['total_experiments']} experiments from platform results")
    print(f"Audit status: {'OK' if report['ok'] else 'FAIL'}")
    print(
        "Provenance: "
        f"ok={summary['provenance_ok']} fail={summary['provenance_fail']} missing={summary['provenance_missing']}"
    )
    print(
        "Proofs: "
        f"ok={summary['proofs_ok']} present={summary['proofs_present']} "
        f"missing_required={summary['proofs_missing_required']} not_required={summary['proofs_not_required']}"
    )
    print(
        "Best artifacts: "
        f"ok={summary['best_artifacts_ok']} fail={summary['best_artifacts_fail']} "
        f"missing={summary['best_artifacts_missing']} not_required={summary['best_artifacts_not_required']}"
    )
    for name, status, reason in _best_artifact_issues(report.get("best_artifacts") or {}):
        detail = "" if reason is None else f" ({reason})"
        print(f"  best_artifact {name}: {status}{detail}")

    _print_repair_history_summary(repair_history)

    failures = [
        item for item in report['experiments']
        if item['provenance_status'] != 'ok'
        or item['proof_status'] in {'fail', 'missing'}
    ]
    for item in failures:
        print(
            f"  {item['experiment_id']}: provenance={item['provenance_status']} "
            f"proof={item['proof_status']}"
        )

    print(f"Audit report saved to {audit_path}")
    if not report['ok']:
        sys.exit(1)


def _iter_repair_report_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    report_path = target / "repair_report.json"
    if report_path.exists():
        return [report_path]

    raise FileNotFoundError(f"No repair_report.json found under {target}")


def _iter_repair_history_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    history_dir = target if target.name == "repair_history" else target / "repair_history"
    entries = sorted(history_dir.glob("repair_*.json"))
    if entries:
        return entries

    raise FileNotFoundError(f"No repair history found under {history_dir}")


def _run_repair_best_artifacts(output_dir: str, *, require_signature: bool) -> None:
    output_path = Path(output_dir)
    repair = rebuild_best_artifacts(output_path)
    report, audit_path = write_audit_report(output_path, require_signature=require_signature)
    attestation_summary, summary_path = write_attestation_summary(
        output_path,
        require_signature=require_signature,
        audit_report=report,
    )
    repair_report, repair_report_path = write_repair_report(
        output_path,
        repair_result=repair,
        audit_report=report,
        attestation_summary=attestation_summary,
    )
    history_entry, history_entry_path = write_repair_history_entry(
        output_path,
        repair_report=repair_report,
    )
    attestation_summary, summary_path = write_attestation_summary(
        output_path,
        require_signature=require_signature,
        audit_report=report,
    )

    best_record = repair.get("best_record") or {}
    print(f"Repair status: {'OK' if repair['ok'] else 'FAIL'}")
    if best_record:
        print(f"Best record: {best_record.get('experiment_id')}")
    if repair["written"]:
        print("Written:")
        for item in repair["written"]:
            print(f"  {item}")
    if repair["removed"]:
        print("Removed:")
        for item in repair["removed"]:
            print(f"  {item}")
    if repair["notes"]:
        print("Notes:")
        for item in repair["notes"]:
            print(f"  {item}")
    if repair["missing_sources"]:
        print("Missing sources:")
        for item in repair["missing_sources"]:
            print(f"  {item}")

    print(f"Audit status: {'OK' if report['ok'] else 'FAIL'}")
    print(f"Audit report saved to {audit_path}")
    print(f"Attestation summary saved to {summary_path}")
    print(f"Repair report saved to {repair_report_path}")
    print(f"Repair history entry saved to {history_entry_path}")
    if repair_report.get("signature") is not None:
        print("Repair report signature: present")
    if history_entry.get("signature") is not None:
        print("Repair history signature: present")
    if not repair["ok"] or not report["ok"]:
        sys.exit(1)


def _run_verify_repair_report(target: str, *, require_signature: bool) -> None:
    target_path = Path(target)
    results = [
        verify_repair_report(path, require_signature=require_signature)
        for path in _iter_repair_report_paths(target_path)
    ]

    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        print(f"{status} {result['path']}")
        failed_checks = [name for name, ok in result["checks"].items() if not ok]
        if failed_checks:
            print(f"  failed_checks: {', '.join(failed_checks)}")
            repair_history_status = result.get("repair_history_status") or {}
            check_details = result.get("check_details") or {}
            repair_history_detail = check_details.get("repair_history_summary")
            if any(
                name in failed_checks
                for name in (
                    "repair_history_summary",
                    "repair_history_latest_repair_report_snapshot_status",
                    "repair_history_latest_signed_repair_report_snapshot_status",
                )
            ):
                print(
                    "  repair_history_classification: "
                    f"{result.get('repair_history_classification')}"
                )
                print(
                    "  repair_history_status: "
                    f"live=expected:{_bool_status_label(repair_history_status.get('expected_live_snapshot_ok'))} "
                    f"actual:{_bool_status_label(repair_history_status.get('actual_live_snapshot_ok'))} "
                    f"signed=expected:{_bool_status_label(repair_history_status.get('expected_signed_snapshot_ok'))} "
                    f"actual:{_bool_status_label(repair_history_status.get('actual_signed_snapshot_ok'))}"
                )
                print(
                    "  repair_history_primary_issue: "
                    f"{_repair_report_primary_issue_label(result)}"
                )
                print(
                    "  repair_history_computed_latest_issues: "
                    f"{json.dumps(result.get('repair_history_computed_latest_issues') or [], sort_keys=True)}"
                )
                print(
                    "  repair_history_recorded_latest_issues: "
                    f"{json.dumps(result.get('repair_history_recorded_latest_issues') or [], sort_keys=True)}"
                )
                primary_issue_details = result.get("repair_history_primary_issue_details")
                if primary_issue_details is not None:
                    print(
                        "  repair_history_primary_issue_details: "
                        f"{json.dumps(primary_issue_details, sort_keys=True)}"
                    )
            if "repair_history_summary" in failed_checks and isinstance(repair_history_detail, dict):
                expected_summary = repair_history_detail.get("expected")
                actual_summary = repair_history_detail.get("actual")
                expected_snapshot_ok = None if not isinstance(expected_summary, dict) else expected_summary.get(
                    "repair_history_latest_repair_report_snapshot_ok"
                )
                actual_snapshot_ok = None if not isinstance(actual_summary, dict) else actual_summary.get(
                    "repair_history_latest_repair_report_snapshot_ok"
                )
                expected_signed_snapshot_ok = None if not isinstance(expected_summary, dict) else expected_summary.get(
                    "repair_history_latest_signed_repair_report_snapshot_ok"
                )
                actual_signed_snapshot_ok = None if not isinstance(actual_summary, dict) else actual_summary.get(
                    "repair_history_latest_signed_repair_report_snapshot_ok"
                )
                print(
                    "  repair_history_summary_expected: "
                    f"{json.dumps(expected_summary, sort_keys=True)}"
                )
                print(
                    "  repair_history_summary_actual: "
                    f"{json.dumps(actual_summary, sort_keys=True)}"
                )
                print(
                    "  repair_history_latest_repair_report_snapshot: "
                    f"expected={_bool_status_label(expected_snapshot_ok)} "
                    f"actual={_bool_status_label(actual_snapshot_ok)}"
                )
                print(
                    "  repair_history_latest_signed_repair_report_snapshot: "
                    f"expected={_bool_status_label(expected_signed_snapshot_ok)} "
                    f"actual={_bool_status_label(actual_signed_snapshot_ok)}"
                )
        if result["signature_present"]:
            print(f"  signature: {result['signature_reason']}")
        elif require_signature:
            print("  signature: missing")

    if not all(result["ok"] for result in results):
        sys.exit(1)


def _run_verify_repair_history(target: str, *, require_signature: bool) -> None:
    target_path = Path(target)
    result = verify_repair_history(target_path, require_signature=require_signature)
    status = "OK" if result["ok"] else "FAIL"
    print(f"{status} {result['history_dir']}")
    print(f"  entries: {result['entry_count']}")
    latest_entry = result["entries"][-1] if result["entries"] else None
    if latest_entry is not None:
        latest_checks = latest_entry.get("checks") or {}
        print(
            "  latest_repair_report: "
            f"{_bool_status_label(latest_checks.get('latest_repair_report'))}"
        )
        print(
            "  latest_repair_report_snapshot: "
            f"{_bool_status_label(latest_checks.get('latest_repair_report_snapshot'))}"
        )
        latest_details = latest_entry.get("check_details") or {}
        report_detail = latest_details.get("latest_repair_report")
        if latest_checks.get("latest_repair_report") is False and isinstance(report_detail, dict):
            print(
                "  latest_repair_report_path: "
                f"{report_detail.get('path')}"
            )
            print(
                "  latest_repair_report_expected_core_sha256: "
                f"{report_detail.get('expected_core_sha256')}"
            )
            print(
                "  latest_repair_report_actual_core_sha256: "
                f"{report_detail.get('actual_core_sha256')}"
            )
            parse_error = report_detail.get("actual_parse_error")
            if parse_error:
                print(f"  latest_repair_report_actual_parse_error: {parse_error}")
        snapshot_detail = latest_details.get("latest_repair_report_snapshot")
        if latest_checks.get("latest_repair_report_snapshot") is False and isinstance(snapshot_detail, dict):
            print(
                "  latest_repair_report_snapshot_expected_sha256: "
                f"{snapshot_detail.get('expected_sha256')}"
            )
            print(
                "  latest_repair_report_snapshot_actual_sha256: "
                f"{snapshot_detail.get('actual_sha256')}"
            )
            print(
                "  latest_repair_report_snapshot_expected_best_record: "
                f"{json.dumps(snapshot_detail.get('expected_post_repair_best_record'), sort_keys=True)}"
            )
            print(
                "  latest_repair_report_snapshot_actual_best_record: "
                f"{json.dumps(snapshot_detail.get('actual_post_repair_best_record'), sort_keys=True)}"
            )
        signed_snapshot_detail = latest_details.get("post_repair_repair_history_snapshot_summary")
        if latest_checks.get("post_repair_repair_history_snapshot_summary") is False and isinstance(signed_snapshot_detail, dict):
            print(
                "  post_repair_repair_history_snapshot_summary_expected: "
                f"{json.dumps(signed_snapshot_detail.get('expected'), sort_keys=True)}"
            )
            print(
                "  post_repair_repair_history_snapshot_summary_actual: "
                f"{json.dumps(signed_snapshot_detail.get('actual'), sort_keys=True)}"
            )
    failed_checks = [name for name, ok in result['checks'].items() if not ok]
    if failed_checks:
        print(f"  failed_checks: {', '.join(failed_checks)}")
    if not result['ok']:
        sys.exit(1)


def _resolve_runtime_settings(
    args: argparse.Namespace,
    config_data: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    str,
    int,
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    profile = _resolve_profile(args.runtime_profile)
    model_name = _resolve_with_profile(
        None,
        config_data.get("model_name"),
        profile.get("model_name"),
        DEFAULT_MODEL,
    )
    system_prompt = config_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    baseline = dict(DEFAULT_BASELINE)
    baseline.update(config_data.get("baseline_params", {}))
    reward_domain = config_data.get("reward_domain", "customer_service")
    max_new_tokens = int(
        _resolve_with_profile(
            args.max_new_tokens,
            config_data.get("max_new_tokens"),
            profile.get("max_new_tokens"),
            64,
        )
    )
    eval_episodes = int(
        _resolve_with_profile(
            args.eval_episodes,
            config_data.get("eval_episodes"),
            profile.get("eval_episodes"),
            4,
        )
    )
    train_scenario_limit = _resolve_with_profile(
        args.train_scenario_limit,
        config_data.get("train_scenario_limit"),
        profile.get("train_scenario_limit"),
        None,
    )
    eval_scenario_limit = _resolve_with_profile(
        args.eval_scenario_limit,
        config_data.get("eval_scenario_limit"),
        profile.get("eval_scenario_limit"),
        None,
    )
    train_scenarios = _slice_scenarios(
        list(config_data.get("train_scenarios", TRAIN_SCENARIOS)),
        int(train_scenario_limit) if train_scenario_limit is not None else None,
        label="train",
    )
    eval_scenarios = _slice_scenarios(
        list(config_data.get("eval_scenarios", EVAL_SCENARIOS)),
        int(eval_scenario_limit) if eval_scenario_limit is not None else None,
        label="eval",
    )
    runtime_overrides = dict(config_data.get("runtime_overrides", {}))
    runtime_overrides.update(profile.get("runtime_overrides", {}))
    return (
        baseline,
        model_name,
        system_prompt,
        reward_domain,
        args.runtime_profile,
        max_new_tokens,
        eval_episodes,
        train_scenarios,
        eval_scenarios,
        runtime_overrides,
    )


def _build_auto_research_config(
    args: argparse.Namespace,
    *,
    eval_episodes: int,
    config_data: dict[str, Any] | None = None,
):
    from stateset_agents.training.auto_research import AutoResearchConfig

    config_data = config_data or {}
    selection_promotion_zscore = args.selection_promotion_zscore
    if selection_promotion_zscore is None:
        selection_promotion_zscore = config_data.get("selection_promotion_zscore", 1.0)
    experiment_isolation = args.experiment_isolation
    if experiment_isolation is None:
        experiment_isolation = config_data.get("experiment_isolation", "worktree")
    runtime_environment = args.runtime_environment
    if runtime_environment is None:
        runtime_environment = config_data.get("runtime_environment", "venv")

    return AutoResearchConfig(
        time_budget=args.time_budget,
        max_experiments=args.max_experiments,
        max_wall_clock=args.max_wall_clock,
        objective_metric=LOCKED_SELECTION_METRIC_NAME,
        proposer=args.proposer,
        search_space_name=args.search_space,
        trainer_algorithm=args.algorithm,
        improvement_patience=args.patience,
        eval_episodes=eval_episodes,
        eval_seed=42,
        eval_concurrency=1,
        selection_promotion_zscore=float(selection_promotion_zscore),
        experiment_isolation=str(experiment_isolation),
        runtime_environment=str(runtime_environment),
        output_dir=args.output_dir,
        save_checkpoints=True,
        log_to_wandb=args.wandb,
        wandb_project="autoresearch",
    )


def _log_run_header(
    *,
    runner: str,
    runtime_profile: str,
    model_name: str,
    device_label: str,
    proposer: str,
    search_space: str,
    time_budget: int,
    eval_episodes: int,
    max_new_tokens: int,
    max_experiments: int,
    output_dir: str,
    train_scenario_count: int,
    eval_scenario_count: int,
) -> None:
    logger.info("=" * 60)
    logger.info("Autonomous Research")
    logger.info("=" * 60)
    logger.info("  Runner:          %s", runner)
    logger.info("  Runtime profile: %s", runtime_profile)
    logger.info("  Model:           %s", model_name)
    logger.info("  Device:          %s", device_label)
    logger.info("  Proposer:        %s", proposer)
    logger.info("  Search space:    %s", search_space)
    logger.info("  Time budget:     %ss", time_budget)
    logger.info("  Eval episodes:   %s", eval_episodes)
    logger.info("  Max new tokens:  %s", max_new_tokens)
    logger.info("  Train scenarios: %s", train_scenario_count)
    logger.info("  Eval scenarios:  %s", eval_scenario_count)
    logger.info("  Max experiments: %s", max_experiments or "unlimited")
    logger.info("  Output:          %s", output_dir)
    logger.info("")


async def _run_training_inprocess(
    args: argparse.Namespace,
    config_data: dict[str, Any],
) -> None:
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.rewards import create_domain_reward
    from stateset_agents.training.auto_research import run_auto_research

    (
        baseline,
        model_name,
        system_prompt,
        reward_domain,
        runtime_profile,
        max_new_tokens,
        eval_episodes,
        train_scenarios,
        eval_scenarios,
        runtime_overrides,
    ) = _resolve_runtime_settings(args, config_data)
    device_info = _resolve_device_info(config_data)
    ar_config = _build_auto_research_config(
        args,
        eval_episodes=eval_episodes,
        config_data=config_data,
    )
    ar_config.base_config_overrides = {
        "model_name": model_name,
        "num_episodes": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "use_cpu": device_info.use_cpu,
        "gradient_checkpointing": config_data.get("gradient_checkpointing", False),
        "bf16": device_info.bf16_enabled,
        "output_dir": "./outputs",
        "report_to": "none",
    }

    if runtime_overrides:
        logger.info(
            "In-process runner does not hard-clamp proposer outputs; runtime profile %s still reduces scenario and token budgets only.",
            runtime_profile,
        )

    _log_run_header(
        runner="inprocess",
        runtime_profile=runtime_profile,
        model_name=model_name,
        device_label=device_info.device_name or device_info.accelerator,
        proposer=args.proposer,
        search_space=args.search_space,
        time_budget=args.time_budget,
        eval_episodes=eval_episodes,
        max_new_tokens=max_new_tokens,
        max_experiments=args.max_experiments,
        output_dir=args.output_dir,
        train_scenario_count=len(train_scenarios),
        eval_scenario_count=len(eval_scenarios),
    )

    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=baseline.get("temperature", 0.7),
        attn_implementation="sdpa",
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    environment = ConversationEnvironment(scenarios=train_scenarios, max_turns=8)
    reward_fn = create_domain_reward(reward_domain)

    tracker = await run_auto_research(
        agent=agent,
        environment=environment,
        eval_scenarios=eval_scenarios,
        reward_fn=reward_fn,
        config=ar_config,
        baseline_params=baseline,
    )

    if tracker.best_record:
        logger.info("Best %s: %.6f", ar_config.objective_metric, tracker.best_value)
        logger.info("Best config:")
        for key, value in sorted(tracker.best_record.params.items()):
            logger.info("  %s: %s", key, value)


def _run_training_subprocess(
    args: argparse.Namespace,
    config_data: dict[str, Any],
) -> None:
    from stateset_agents.training.auto_research import ExperimentTracker

    if args.algorithm not in {"gspo", "auto"}:
        raise ValueError(
            "The subprocess runner currently supports only --algorithm gspo or auto"
        )
    if args.search_space == "multi_algorithm":
        raise ValueError(
            "The subprocess runner does not support --search-space multi_algorithm. "
            "Use --runner inprocess for algorithm search."
        )

    (
        baseline,
        model_name,
        system_prompt,
        reward_domain,
        runtime_profile,
        max_new_tokens,
        eval_episodes,
        train_scenarios,
        eval_scenarios,
        runtime_overrides,
    ) = _resolve_runtime_settings(args, config_data)
    device_info = _resolve_device_info(config_data)
    ar_config = _build_auto_research_config(
        args,
        eval_episodes=eval_episodes,
        config_data=config_data,
    )

    base_run_config = {
        "model_name": model_name,
        "system_prompt": system_prompt,
        "reward_domain": reward_domain,
        "max_new_tokens": max_new_tokens,
        "gradient_checkpointing": config_data.get("gradient_checkpointing", False),
        "bf16": device_info.bf16_enabled,
        "runtime_profile": runtime_profile,
        "train_scenarios": train_scenarios,
        "eval_scenarios": eval_scenarios,
    }
    if "custom_reward_weights" in config_data:
        base_run_config["custom_reward_weights"] = config_data["custom_reward_weights"]
    if args.algorithm != "auto":
        base_run_config["algorithm"] = args.algorithm

    _log_run_header(
        runner="subprocess",
        runtime_profile=runtime_profile,
        model_name=model_name,
        device_label=device_info.device_name or device_info.accelerator,
        proposer=args.proposer,
        search_space=args.search_space,
        time_budget=args.time_budget,
        eval_episodes=eval_episodes,
        max_new_tokens=max_new_tokens,
        max_experiments=args.max_experiments,
        output_dir=args.output_dir,
        train_scenario_count=len(train_scenarios),
        eval_scenario_count=len(eval_scenarios),
    )

    runner = SubprocessAutoResearchRunner(
        config=ar_config,
        baseline_params=baseline,
        base_run_config=base_run_config,
        device_info=device_info,
        runtime_overrides=runtime_overrides,
        allow_undersized_budget=args.allow_undersized_budget,
    )
    tracker: ExperimentTracker = runner.run()

    if tracker.best_record:
        logger.info("Best %s: %.6f", ar_config.objective_metric, tracker.best_value)
        logger.info("Best config:")
        for key, value in sorted(tracker.best_record.params.items()):
            logger.info("  %s: %s", key, value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous RL Research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--analyze", action="store_true",
        help="Analyze results from a previous run (no training)",
    )
    mode.add_argument(
        "--compare", nargs="+", metavar="DIR",
        help="Compare multiple run directories",
    )
    mode.add_argument(
        "--verify-provenance", type=str, metavar="PATH",
        help="Verify provenance envelope(s) in a run directory or a single provenance.json",
    )
    mode.add_argument(
        "--verify-proof", type=str, metavar="PATH",
        help="Verify STARK proof artifact(s) in a run directory or a single proof.json",
    )
    mode.add_argument(
        "--audit-run", type=str, metavar="DIR",
        help="Audit a platform run directory and write audit_report.json",
    )
    mode.add_argument(
        "--repair-best-artifacts", type=str, metavar="DIR",
        help="Rebuild root best_* artifacts from the best recorded run, then refresh audit_report.json and attestation_summary.json",
    )
    mode.add_argument(
        "--verify-repair-report", type=str, metavar="PATH",
        help="Verify a repair_report.json file or a run directory containing one",
    )
    mode.add_argument(
        "--verify-repair-history", type=str, metavar="PATH",
        help="Verify append-only repair history entries for a run directory or repair_history entry",
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file (overrides defaults for model, baseline, etc.)",
    )
    parser.add_argument(
        "--runner", type=str, default="subprocess",
        choices=["subprocess", "inprocess"],
        help="Experiment execution backend",
    )
    parser.add_argument(
        "--runtime-profile", type=str, default="standard",
        choices=sorted(RUNTIME_PROFILES),
        help="Runtime profile for CPU smoke runs or full evaluation",
    )
    parser.add_argument(
        "--proposer", type=str, default="perturbation",
        choices=["perturbation", "smart", "adaptive", "random", "grid", "bayesian", "llm"],
        help="Experiment proposal strategy",
    )
    parser.add_argument(
        "--search-space", type=str, default="auto_research",
        help="Search space name",
    )
    parser.add_argument(
        "--time-budget", type=int, default=300,
        help="Wall-clock seconds per experiment",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=None,
        help="Override evaluation episodes per experiment",
    )
    parser.add_argument(
        "--train-scenario-limit", type=int, default=None,
        help="Limit the number of training scenarios per experiment",
    )
    parser.add_argument(
        "--eval-scenario-limit", type=int, default=None,
        help="Limit the number of evaluation scenarios per experiment",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Override the agent response length for training and evaluation",
    )
    parser.add_argument(
        "--max-experiments", type=int, default=0,
        help="Maximum experiments to run (0 = unlimited)",
    )
    parser.add_argument(
        "--max-wall-clock", type=int, default=0,
        help="Total wall-clock budget in seconds (0 = unlimited)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./auto_research_results",
        help="Directory for results, checkpoints, and logs",
    )
    parser.add_argument(
        "--algorithm", type=str, default="gspo",
        choices=["gspo", "grpo", "dapo", "vapo", "auto"],
        help="Training algorithm (auto = let proposer choose)",
    )
    parser.add_argument(
        "--patience", type=int, default=0,
        help="Stop after N consecutive non-improvements (0 = disabled)",
    )
    parser.add_argument(
        "--selection-promotion-zscore", type=float, default=None,
        help="Require selection_score - z*bootstrap_std to beat the current best before promotion",
    )
    parser.add_argument(
        "--experiment-isolation", type=str, default=None,
        choices=["shared", "worktree"],
        help="Run subprocess experiments in the shared repo checkout or in a per-run git worktree",
    )
    parser.add_argument(
        "--runtime-environment", type=str, default=None,
        choices=["venv", "uv"],
        help="Launch training with the repo virtualenv when available or through uv",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log experiments to Weights & Biases",
    )
    parser.add_argument(
        "--require-signature", action="store_true",
        help="Require a valid provenance signature when using --verify-provenance, --verify-proof, --verify-repair-report, --verify-repair-history, or --audit-run",
    )
    parser.add_argument(
        "--allow-undersized-budget", action="store_true",
        help="Run experiments even when the configured time budget is below the platform preflight recommendation",
    )

    args = parser.parse_args()

    if args.compare:
        _run_compare(args.compare)
        return

    if args.analyze:
        _run_analyze(args.output_dir)
        return

    if args.verify_provenance:
        _run_verify_provenance(
            args.verify_provenance,
            require_signature=args.require_signature,
        )
        return

    if args.verify_proof:
        _run_verify_proof(
            args.verify_proof,
            require_signature=args.require_signature,
        )
        return

    if args.audit_run:
        _run_audit_run(
            args.audit_run,
            require_signature=args.require_signature,
        )
        return

    if args.repair_best_artifacts:
        _run_repair_best_artifacts(
            args.repair_best_artifacts,
            require_signature=args.require_signature,
        )
        return

    if args.verify_repair_report:
        _run_verify_repair_report(
            args.verify_repair_report,
            require_signature=args.require_signature,
        )
        return

    if args.verify_repair_history:
        _run_verify_repair_history(
            args.verify_repair_history,
            require_signature=args.require_signature,
        )
        return

    config_data: dict[str, Any] = {}
    if args.config:
        config_data = _load_yaml_config(args.config)

    if args.runner == "subprocess":
        _run_training_subprocess(args, config_data)
        return

    asyncio.run(_run_training_inprocess(args, config_data))


if __name__ == "__main__":
    main()
