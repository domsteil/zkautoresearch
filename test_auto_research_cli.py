from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from attestation_audit import (
    rebuild_best_artifacts,
    verify_repair_history,
    verify_repair_report,
    write_attestation_summary,
    write_audit_report,
    write_repair_report,
)
from auto_research import (
    _build_audit_report,
    _build_auto_research_config,
    _run_analyze,
    _run_audit_run,
    _run_repair_best_artifacts,
    _run_verify_proof,
    _run_verify_repair_history,
    _run_verify_repair_report,
    _verify_proof_artifact,
)
from experiment_runtime import (
    collect_file_digests,
    sha256_file,
    sha256_json,
    sign_json_payload,
    verify_provenance_envelope,
)
from stark_autoresearch.proof import MetricImprovementProver
from stateset_agents.training.auto_research.experiment_tracker import (
    ExperimentRecord,
    ExperimentTracker,
)


class AutoResearchProofCliTests(unittest.TestCase):
    def _write_signed_run_artifacts(
        self,
        root: Path,
        *,
        objective_value: float = 0.55,
        claimed_best_reward: float = 0.52,
    ) -> tuple[Path, Path]:
        run_dir = root / "runs" / "exp_0001"
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / "config.json"
        config_payload = {"learning_rate": 1e-5}
        config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True))

        summary_path = run_dir / "summary.json"
        summary_payload = {
            "config": config_payload,
            "metrics": {"eval_reward": objective_value},
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

        log_path = run_dir / "train.log"
        log_path.write_text("training ok\n")

        runtime_artifact_path = run_dir / "runtime_environment.json"
        runtime_artifact_payload = {
            "mode": "custom_command",
            "command": ["python", "train.py"],
            "environment_snapshot": {
                "status": "ok",
                "python_version": "3.10.0",
                "package_count": 2,
                "packages_sha256": sha256_json([["alpha", "1.0"], ["beta", "2.0"]]),
                "packages": [["alpha", "1.0"], ["beta", "2.0"]],
            },
        }
        runtime_artifact_path.write_text(json.dumps(runtime_artifact_payload, indent=2, sort_keys=True))

        model_dir = run_dir / "outputs" / "final_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "weights.bin").write_bytes(b"weights")
        artifact_files = [digest.to_dict() for digest in collect_file_digests(model_dir)]

        experiment_id = str(uuid.uuid4())
        envelope_core = {
            "experiment_id": experiment_id,
            "description": "cli verification fixture",
            "process_status": "ok",
            "record_status": "keep",
            "elapsed_seconds": 1.0,
            "objective_metric": "eval_reward",
            "objective_value": objective_value,
            "git": {
                "repo_root": str(root),
                "commit": "test",
                "branch": "main",
                "is_dirty": False,
            },
            "config_path": "config.json",
            "config_sha256": sha256_file(config_path),
            "summary_path": "summary.json",
            "summary_sha256": sha256_file(summary_path),
            "log_path": "train.log",
            "log_sha256": sha256_file(log_path),
            "runtime_artifact_path": "runtime_environment.json",
            "runtime_artifact_sha256": sha256_file(runtime_artifact_path),
            "summary_config_sha256": sha256_json(summary_payload["config"]),
            "summary_metrics_sha256": sha256_json(summary_payload["metrics"]),
            "artifacts": {
                "model_dir": "outputs/final_model",
                "manifest_sha256": sha256_json(artifact_files),
                "files": artifact_files,
            },
        }
        envelope = dict(envelope_core)
        envelope["envelope_sha256"] = sha256_json(envelope_core)
        envelope["signature"] = sign_json_payload(envelope, key=b"secret-key", key_id="unit-test")

        provenance_path = run_dir / "provenance.json"
        provenance_path.write_text(json.dumps(envelope, indent=2, sort_keys=True))

        prover = MetricImprovementProver()
        with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
            proof = prover.prove_improvement_from_provenance(
                provenance_path=provenance_path,
                best_reward=claimed_best_reward,
                agent_id="cli-test",
                require_signature=True,
            )

        proof_bin_path = run_dir / "proof.bin"
        proof_bin_path.write_bytes(proof.proof_bytes)
        proof_payload = proof.to_dict()
        proof_payload.update(
            {
                "proof_path": "proof.bin",
                "provenance_path": "provenance.json",
                "proof_bytes_sha256": sha256_file(proof_bin_path),
                "claimed_best_reward": claimed_best_reward,
                "objective_value": objective_value,
                "witness_commitment": proof.witness_commitment,
            }
        )
        proof_json_path = run_dir / "proof.json"
        proof_json_path.write_text(json.dumps(proof_payload, indent=2, sort_keys=True))
        return run_dir, proof_json_path


    def _write_signed_provenance_artifacts(
        self,
        root: Path,
        *,
        run_name: str,
        objective_value: float,
    ) -> tuple[Path, Path]:
        run_dir = root / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / "config.json"
        config_payload = {"learning_rate": 1e-5}
        config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True))

        summary_path = run_dir / "summary.json"
        summary_payload = {
            "config": config_payload,
            "metrics": {"eval_reward": objective_value},
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

        log_path = run_dir / "train.log"
        log_path.write_text("training ok\n")

        runtime_artifact_path = run_dir / "runtime_environment.json"
        runtime_artifact_payload = {
            "mode": "custom_command",
            "command": ["python", "train.py"],
            "environment_snapshot": {
                "status": "ok",
                "python_version": "3.10.0",
                "package_count": 2,
                "packages_sha256": sha256_json([["alpha", "1.0"], ["beta", "2.0"]]),
                "packages": [["alpha", "1.0"], ["beta", "2.0"]],
            },
        }
        runtime_artifact_path.write_text(json.dumps(runtime_artifact_payload, indent=2, sort_keys=True))

        model_dir = run_dir / "outputs" / "final_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "weights.bin").write_bytes(b"weights")
        artifact_files = [digest.to_dict() for digest in collect_file_digests(model_dir)]

        experiment_id = str(uuid.uuid4())
        envelope_core = {
            "experiment_id": experiment_id,
            "description": f"fixture {run_name}",
            "process_status": "ok",
            "record_status": "keep",
            "elapsed_seconds": 1.0,
            "objective_metric": "eval_reward",
            "objective_value": objective_value,
            "git": {
                "repo_root": str(root),
                "commit": "test",
                "branch": "main",
                "is_dirty": False,
            },
            "config_path": "config.json",
            "config_sha256": sha256_file(config_path),
            "summary_path": "summary.json",
            "summary_sha256": sha256_file(summary_path),
            "log_path": "train.log",
            "log_sha256": sha256_file(log_path),
            "runtime_artifact_path": "runtime_environment.json",
            "runtime_artifact_sha256": sha256_file(runtime_artifact_path),
            "summary_config_sha256": sha256_json(summary_payload["config"]),
            "summary_metrics_sha256": sha256_json(summary_payload["metrics"]),
            "artifacts": {
                "model_dir": "outputs/final_model",
                "manifest_sha256": sha256_json(artifact_files),
                "files": artifact_files,
            },
        }
        envelope = dict(envelope_core)
        envelope["envelope_sha256"] = sha256_json(envelope_core)
        envelope["signature"] = sign_json_payload(envelope, key=b"secret-key", key_id="unit-test")

        provenance_path = run_dir / "provenance.json"
        provenance_path.write_text(json.dumps(envelope, indent=2, sort_keys=True))
        return run_dir, provenance_path

    def _write_best_artifacts(
        self,
        root: Path,
        *,
        run_dir: Path,
        proof_json_path: Path | None,
    ) -> None:
        (root / "best_config.json").write_text((run_dir / "config.json").read_text())
        (root / "best_summary.json").write_text((run_dir / "summary.json").read_text())
        (root / "best_provenance.json").write_text((run_dir / "provenance.json").read_text())
        source_runtime_artifact = run_dir / "runtime_environment.json"
        best_runtime_artifact = root / "best_runtime_environment.json"
        if source_runtime_artifact.exists():
            best_runtime_artifact.write_text(source_runtime_artifact.read_text())
        elif best_runtime_artifact.exists():
            best_runtime_artifact.unlink()

        bundle_root = root / "best_bundle"
        if bundle_root.exists():
            shutil.rmtree(bundle_root)
        bundle_root.mkdir(parents=True, exist_ok=True)
        (bundle_root / "config.json").write_text((run_dir / "config.json").read_text())
        (bundle_root / "summary.json").write_text((run_dir / "summary.json").read_text())
        (bundle_root / "train.log").write_text((run_dir / "train.log").read_text())
        (bundle_root / "provenance.json").write_text((run_dir / "provenance.json").read_text())
        if source_runtime_artifact.exists():
            (bundle_root / "runtime_environment.json").write_text(source_runtime_artifact.read_text())
        shutil.copytree(run_dir / "outputs" / "final_model", bundle_root / "outputs" / "final_model")

        best_proof_json = root / "best_proof.json"
        best_proof_bin = root / "best_proof.bin"
        if proof_json_path is None:
            if best_proof_json.exists():
                best_proof_json.unlink()
            if best_proof_bin.exists():
                best_proof_bin.unlink()
            return

        proof_payload = json.loads(proof_json_path.read_text())
        source_bin = proof_json_path.parent / str(proof_payload.get("proof_path") or "proof.bin")
        best_proof_bin.write_bytes(source_bin.read_bytes())
        proof_payload["proof_path"] = "best_proof.bin"
        proof_payload["provenance_path"] = "best_provenance.json"
        proof_payload["proof_bytes_sha256"] = sha256_file(best_proof_bin)
        best_proof_json.write_text(json.dumps(proof_payload, indent=2, sort_keys=True))

        bundle_proof_bin = bundle_root / "proof.bin"
        bundle_proof_bin.write_bytes(source_bin.read_bytes())
        bundle_proof_payload = json.loads(proof_json_path.read_text())
        bundle_proof_payload["proof_path"] = "proof.bin"
        bundle_proof_payload["provenance_path"] = "provenance.json"
        bundle_proof_payload["proof_bytes_sha256"] = sha256_file(bundle_proof_bin)
        (bundle_root / "proof.json").write_text(json.dumps(bundle_proof_payload, indent=2, sort_keys=True))

    def test_verify_proof_artifact_and_directory_succeed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, proof_json_path = self._write_signed_run_artifacts(root)

            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                result = _verify_proof_artifact(proof_json_path, require_signature=True)
            self.assertTrue(result['ok'], result['verification_message'])
            self.assertTrue(result['proof_bytes_sha256_ok'])
            self.assertEqual(result['provenance_path'], str(proof_json_path.parent / 'provenance.json'))

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                    _run_verify_proof(str(root), require_signature=True)
            output = stdout.getvalue()
            self.assertIn('OK', output)
            self.assertIn('proof_bytes_sha256: ok', output)
            self.assertIn('provenance:', output)

    def test_verify_proof_artifact_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            proof_bin_path = run_dir / 'proof.bin'
            proof_bytes = bytearray(proof_bin_path.read_bytes())
            proof_bytes[64] ^= 0xFF
            proof_bin_path.write_bytes(bytes(proof_bytes))

            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                result = _verify_proof_artifact(proof_json_path, require_signature=True)
            self.assertFalse(result['ok'])
            self.assertFalse(result['proof_bytes_sha256_ok'])

            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()):
                    with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                        _run_verify_proof(str(proof_json_path), require_signature=True)


    def test_build_audit_report_and_cli_succeed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)

            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                report = _build_audit_report(root, require_signature=True)
            self.assertTrue(report['ok'])
            self.assertTrue(report['best_artifacts']['ok'])
            self.assertEqual(report['summary']['provenance_ok'], 2)
            self.assertEqual(report['summary']['proofs_ok'], 1)
            self.assertEqual(report['summary']['proofs_missing_required'], 0)
            self.assertEqual(report['summary']['best_artifacts_ok'], 7)
            self.assertEqual(report['best_artifacts']['checks']['best_bundle']['status'], 'ok')
            self.assertTrue(report['best_artifacts']['checks']['best_bundle']['ok'])

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                    _run_audit_run(str(root), require_signature=True)
            output = stdout.getvalue()
            self.assertIn('Audit status: OK', output)
            self.assertIn('Best artifacts: ok=7 fail=0 missing=0 not_required=0', output)
            self.assertTrue((root / 'audit_report.json').exists())

    def test_audit_run_detects_tampered_best_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )
            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)
            (root / "best_config.json").write_text(json.dumps({"learning_rate": 9e-5}, indent=2, sort_keys=True))

            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                report = _build_audit_report(root, require_signature=True)
            self.assertFalse(report['ok'])
            self.assertFalse(report['best_artifacts']['ok'])
            self.assertEqual(report['summary']['best_artifacts_fail'], 1)
            self.assertEqual(report['best_artifacts']['checks']['best_config']['status'], 'fail')

    def test_audit_run_detects_missing_best_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)
            shutil.rmtree(root / "best_bundle")

            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                report = _build_audit_report(root, require_signature=True)
            self.assertFalse(report['ok'])
            self.assertFalse(report['best_artifacts']['ok'])
            self.assertFalse(report['repair_history']['present'])
            self.assertTrue(report['repair_history']['ok'])
            self.assertFalse(report['summary']['repair_history_present'])
            self.assertTrue(report['summary']['repair_history_ok'])
            self.assertEqual(report['summary']['repair_history_entry_count'], 0)
            self.assertEqual(report['summary']['repair_history_latest_issue_count'], 0)
            self.assertEqual(report['summary']['repair_history_latest_issues'], [])
            self.assertIsNone(report['summary']['repair_history_latest_primary_issue'])
            self.assertIsNone(report['summary']['repair_history_latest_repair_report_snapshot_ok'])
            self.assertIsNone(report['summary']['repair_history_latest_signed_repair_report_snapshot_ok'])
            self.assertEqual(report['summary']['repair_history_latest_best_artifact_issue_count'], 0)
            self.assertEqual(report['summary']['repair_history_latest_best_artifact_issues'], [])
            self.assertEqual(report['summary']['best_artifacts_missing'], 1)
            self.assertEqual(report['best_artifacts']['checks']['best_bundle']['status'], 'missing')
            self.assertFalse(report['best_artifacts']['checks']['best_bundle']['ok'])

            attestation_summary, _ = write_attestation_summary(root, require_signature=True, audit_report=report)
            self.assertFalse(attestation_summary['best_record']['audit_best_artifacts_ok'])
            self.assertEqual(attestation_summary['best_record']['audit_best_artifact_issue_count'], 1)
            self.assertEqual(
                attestation_summary['best_record']['audit_best_artifact_issues'],
                [{'name': 'best_bundle', 'status': 'missing', 'reason': 'Portable best bundle provenance missing'}],
            )

            stdout = io.StringIO()
            with self.assertRaises(SystemExit):
                with redirect_stdout(stdout):
                    with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                        _run_audit_run(str(root), require_signature=True)
            output = stdout.getvalue()
            self.assertIn('Best artifacts: ok=6 fail=0 missing=1 not_required=0', output)
            self.assertIn('best_artifact best_bundle: missing', output)

    def test_repair_best_artifacts_restores_tampered_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)
            (root / "best_config.json").write_text(json.dumps({"learning_rate": 9e-5}, indent=2, sort_keys=True))
            (root / "best_proof.bin").write_bytes(b"tampered")

            env = {
                'AUTORESEARCH_SIGNING_KEY': 'secret-key',
                'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                'AUTORESEARCH_VERIFY_KEY': 'secret-key',
            }
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with patch.dict('os.environ', env, clear=False):
                    _run_repair_best_artifacts(str(root), require_signature=True)
            output = stdout.getvalue()
            self.assertIn('Repair status: OK', output)
            self.assertIn('Audit status: OK', output)
            self.assertIn('Repair report saved to', output)
            self.assertIn('Repair history entry saved to', output)
            self.assertIn('Repair report signature: present', output)
            self.assertIn('Repair history signature: present', output)

            self.assertEqual((root / "best_config.json").read_text(), (exp_run_dir / "config.json").read_text())
            self.assertEqual(
                (root / "best_runtime_environment.json").read_text(),
                (exp_run_dir / "runtime_environment.json").read_text(),
            )
            repaired_proof = json.loads((root / "best_proof.json").read_text())
            self.assertEqual(repaired_proof["proof_path"], "best_proof.bin")
            self.assertEqual(repaired_proof["provenance_path"], "best_provenance.json")
            self.assertEqual(sha256_file(root / "best_proof.bin"), repaired_proof["proof_bytes_sha256"])

            best_bundle = root / "best_bundle"
            self.assertTrue((best_bundle / "provenance.json").exists())
            self.assertTrue((best_bundle / "proof.json").exists())
            self.assertTrue((best_bundle / "proof.bin").exists())
            self.assertTrue((best_bundle / "outputs" / "final_model").exists())
            self.assertEqual(
                (best_bundle / "runtime_environment.json").read_text(),
                (exp_run_dir / "runtime_environment.json").read_text(),
            )
            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                bundle_provenance = verify_provenance_envelope(best_bundle / "provenance.json", require_signature=True)
                bundle_proof = _verify_proof_artifact(best_bundle / "proof.json", require_signature=True)
            self.assertTrue(bundle_provenance['ok'], bundle_provenance['signature_reason'])
            self.assertTrue(bundle_proof['ok'], bundle_proof['verification_message'])

            repair_report = json.loads((root / 'repair_report.json').read_text())
            self.assertEqual(repair_report['event_type'], 'best_artifact_repair')
            self.assertIn('signature', repair_report)
            self.assertEqual(
                repair_report['post_repair_repair_history_summary'],
                {
                    'repair_history_present': True,
                    'repair_history_ok': True,
                    'repair_history_entry_count': 1,
                    'repair_history_latest_issue_count': 0,
                    'repair_history_latest_issues': [],
                    'repair_history_latest_primary_issue': None,
                    'repair_history_latest_repair_report_snapshot_ok': True,
                    'repair_history_latest_signed_repair_report_snapshot_ok': True,
                    'repair_history_latest_best_artifact_issue_count': 0,
                    'repair_history_latest_best_artifact_issues': [],
                },
            )
            history_entry = json.loads((root / 'repair_history' / 'repair_0001.json').read_text())
            self.assertEqual(history_entry['event_type'], 'best_artifact_repair_history')
            self.assertEqual(history_entry['sequence_number'], 1)
            self.assertIn('signature', history_entry)
            self.assertEqual(
                history_entry['post_repair_repair_history_summary'],
                repair_report['post_repair_repair_history_summary'],
            )
            self.assertEqual(
                history_entry['repair_report_snapshot']['post_repair_repair_history_summary'],
                repair_report['post_repair_repair_history_summary'],
            )
            self.assertIsNotNone(history_entry['post_repair_repair_history_snapshot_summary']['repair_history_latest_repair_report_snapshot_path'])
            self.assertIsNotNone(history_entry['post_repair_repair_history_snapshot_summary']['repair_history_latest_repair_report_snapshot_expected_sha256'])
            self.assertIsNotNone(history_entry['post_repair_repair_history_snapshot_summary']['repair_history_latest_repair_report_snapshot_actual_sha256'])
            self.assertTrue(history_entry['post_repair_repair_history_snapshot_summary']['repair_history_latest_repair_report_snapshot_expected_best_record']['audit_best_artifacts_ok'])
            self.assertTrue(history_entry['post_repair_repair_history_snapshot_summary']['repair_history_latest_repair_report_snapshot_actual_best_record']['audit_best_artifacts_ok'])
            self.assertIsNone(history_entry['post_repair_repair_history_snapshot_summary']['repair_history_latest_repair_report_snapshot_actual_parse_error'])
            self.assertTrue(history_entry['repair_report_snapshot']['post_repair_best_record']['audit_best_artifacts_ok'])
            self.assertEqual(history_entry['repair_report_snapshot']['post_repair_best_record']['audit_best_artifact_issue_count'], 0)
            self.assertEqual(history_entry['repair_report_snapshot']['post_repair_best_record']['audit_best_artifact_issues'], [])

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with patch.dict('os.environ', env, clear=False):
                    _run_verify_repair_report(str(root), require_signature=True)
            self.assertIn('OK', stdout.getvalue())
            self.assertIn('signature: ok', stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with patch.dict('os.environ', env, clear=False):
                    _run_verify_repair_history(str(root), require_signature=True)
            verify_history_output = stdout.getvalue()
            self.assertIn('OK', verify_history_output)
            self.assertIn('entries: 1', verify_history_output)
            self.assertIn('latest_repair_report_snapshot: ok', verify_history_output)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with patch.dict('os.environ', env, clear=False):
                    _run_audit_run(str(root), require_signature=True)
            audit_output = stdout.getvalue()
            self.assertIn('Repair history: OK', audit_output)
            self.assertIn('summary: latest_issues=0 primary_issue=none latest_best_artifact_issues=0', audit_output)
            self.assertIn('report_snapshot=ok signed_snapshot=ok', audit_output)

            audit_report = json.loads((root / 'audit_report.json').read_text())
            self.assertTrue(audit_report['ok'])
            self.assertTrue(audit_report['best_artifacts']['ok'])
            self.assertTrue(audit_report['repair_history']['present'])
            self.assertTrue(audit_report['repair_history']['ok'])
            self.assertTrue(audit_report['repair_history']['latest_repair_report_snapshot_ok'])
            self.assertTrue(audit_report['summary']['repair_history_present'])
            self.assertTrue(audit_report['summary']['repair_history_ok'])
            self.assertEqual(audit_report['summary']['repair_history_entry_count'], 1)
            self.assertEqual(audit_report['summary']['repair_history_latest_issue_count'], 0)
            self.assertEqual(audit_report['summary']['repair_history_latest_issues'], [])
            self.assertIsNone(audit_report['summary']['repair_history_latest_primary_issue'])
            self.assertTrue(audit_report['summary']['repair_history_latest_repair_report_snapshot_ok'])
            self.assertTrue(audit_report['summary']['repair_history_latest_signed_repair_report_snapshot_ok'])
            self.assertEqual(audit_report['summary']['repair_history_latest_best_artifact_issue_count'], 0)
            self.assertEqual(audit_report['summary']['repair_history_latest_best_artifact_issues'], [])
            summary = json.loads((root / 'attestation_summary.json').read_text())
            self.assertTrue(summary['ok'])
            self.assertTrue(summary['best_record']['audit_best_artifacts_ok'])
            self.assertEqual(summary['best_record']['audit_best_artifact_issue_count'], 0)
            self.assertEqual(summary['best_record']['audit_best_artifact_issues'], [])
            self.assertTrue(summary['summary']['repair_history_present'])
            self.assertTrue(summary['summary']['repair_history_ok'])
            self.assertEqual(summary['summary']['repair_history_entry_count'], 1)
            self.assertEqual(summary['summary']['repair_history_latest_issue_count'], 0)
            self.assertEqual(summary['summary']['repair_history_latest_issues'], [])
            self.assertIsNone(summary['summary']['repair_history_latest_primary_issue'])
            self.assertTrue(summary['summary']['repair_history_latest_repair_report_snapshot_ok'])
            self.assertTrue(summary['summary']['repair_history_latest_signed_repair_report_snapshot_ok'])
            self.assertIsNotNone(summary['summary']['repair_history_latest_repair_report_snapshot_path'])
            self.assertIsNotNone(summary['summary']['repair_history_latest_repair_report_snapshot_expected_sha256'])
            self.assertIsNotNone(summary['summary']['repair_history_latest_repair_report_snapshot_actual_sha256'])
            self.assertTrue(summary['summary']['repair_history_latest_repair_report_snapshot_expected_best_record']['audit_best_artifacts_ok'])
            self.assertTrue(summary['summary']['repair_history_latest_repair_report_snapshot_actual_best_record']['audit_best_artifacts_ok'])
            self.assertIsNone(summary['summary']['repair_history_latest_repair_report_snapshot_actual_parse_error'])
            self.assertEqual(summary['summary']['repair_history_latest_best_artifact_issue_count'], 0)
            self.assertEqual(summary['summary']['repair_history_latest_best_artifact_issues'], [])
            self.assertTrue(summary['repair_history']['present'])
            self.assertTrue(summary['repair_history']['ok'])
            self.assertEqual(summary['repair_history']['entry_count'], 1)
            self.assertEqual(summary['repair_history']['latest_sequence_number'], 1)
            self.assertTrue(summary['repair_history']['latest_repair_report_integrity_ok'])
            self.assertTrue(summary['repair_history']['latest_repair_report_snapshot_ok'])
            self.assertIsNotNone(summary['repair_history']['latest_signed_repair_report_snapshot_path'])
            self.assertIsNotNone(summary['repair_history']['latest_signed_repair_report_snapshot_expected_sha256'])
            self.assertIsNotNone(summary['repair_history']['latest_signed_repair_report_snapshot_actual_sha256'])
            self.assertTrue(summary['repair_history']['latest_signed_repair_report_snapshot_expected_best_record']['audit_best_artifacts_ok'])
            self.assertTrue(summary['repair_history']['latest_signed_repair_report_snapshot_actual_best_record']['audit_best_artifacts_ok'])
            self.assertIsNone(summary['repair_history']['latest_signed_repair_report_snapshot_actual_parse_error'])
            self.assertTrue(summary['repair_history']['latest_best_record_audit_best_artifacts_ok'])
            self.assertEqual(summary['repair_history']['latest_best_record_audit_best_artifact_issue_count'], 0)
            self.assertEqual(summary['repair_history']['latest_best_record_audit_best_artifact_issues'], [])
            self.assertEqual(summary['repair_report_verification']['repair_history_classification'], 'match')
            self.assertEqual(summary['repair_report_verification']['failed_checks'], [])
            self.assertTrue(summary['repair_report_verification']['ok'])
            self.assertTrue(summary['repair_report_verification']['repair_history_status']['expected_live_snapshot_ok'])
            self.assertTrue(summary['repair_report_verification']['repair_history_status']['actual_live_snapshot_ok'])
            self.assertTrue(summary['repair_report_verification']['repair_history_status']['expected_signed_snapshot_ok'])
            self.assertTrue(summary['repair_report_verification']['repair_history_status']['actual_signed_snapshot_ok'])
            self.assertEqual(summary['repair_report_verification']['repair_history_computed_latest_issues'], [])
            self.assertEqual(summary['repair_report_verification']['repair_history_recorded_latest_issues'], [])
            self.assertIsNone(summary['repair_report_verification']['repair_history_primary_issue'])
            self.assertIsNone(summary['repair_report_verification']['repair_history_primary_issue_origin'])
            self.assertIsNone(summary['repair_report_verification']['repair_history_primary_issue_details'])

            stale_audit_report = json.loads(json.dumps(audit_report))
            for key in [
                'repair_history_present',
                'repair_history_ok',
                'repair_history_entry_count',
                'repair_history_latest_issue_count',
                'repair_history_latest_issues',
                'repair_history_latest_repair_report_snapshot_ok',
                'repair_history_latest_signed_repair_report_snapshot_ok',
                'repair_history_latest_repair_report_snapshot_path',
                'repair_history_latest_repair_report_snapshot_expected_sha256',
                'repair_history_latest_repair_report_snapshot_actual_sha256',
                'repair_history_latest_repair_report_snapshot_expected_best_record',
                'repair_history_latest_repair_report_snapshot_actual_best_record',
                'repair_history_latest_repair_report_snapshot_actual_parse_error',
                'repair_history_latest_best_artifact_issue_count',
                'repair_history_latest_best_artifact_issues',
            ]:
                stale_audit_report['summary'].pop(key, None)
            with patch.dict('os.environ', env, clear=False):
                rebuilt_summary, _ = write_attestation_summary(root, require_signature=True, audit_report=stale_audit_report)
            self.assertTrue(rebuilt_summary['summary']['repair_history_present'])
            self.assertTrue(rebuilt_summary['summary']['repair_history_ok'])
            self.assertEqual(rebuilt_summary['summary']['repair_history_entry_count'], 1)
            self.assertEqual(rebuilt_summary['summary']['repair_history_latest_issue_count'], 0)
            self.assertEqual(rebuilt_summary['summary']['repair_history_latest_issues'], [])
            self.assertIsNone(rebuilt_summary['summary']['repair_history_latest_primary_issue'])
            self.assertTrue(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_ok'])
            self.assertTrue(rebuilt_summary['summary']['repair_history_latest_signed_repair_report_snapshot_ok'])
            self.assertIsNotNone(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_path'])
            self.assertIsNotNone(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_expected_sha256'])
            self.assertIsNotNone(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_actual_sha256'])
            self.assertTrue(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_expected_best_record']['audit_best_artifacts_ok'])
            self.assertTrue(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_actual_best_record']['audit_best_artifacts_ok'])
            self.assertIsNone(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_actual_parse_error'])
            self.assertEqual(rebuilt_summary['summary']['repair_history_latest_best_artifact_issue_count'], 0)
            self.assertEqual(rebuilt_summary['summary']['repair_history_latest_best_artifact_issues'], [])

    def test_verify_repair_report_detects_post_repair_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)
            env = {
                'AUTORESEARCH_SIGNING_KEY': 'secret-key',
                'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                'AUTORESEARCH_VERIFY_KEY': 'secret-key',
            }
            with redirect_stdout(io.StringIO()):
                with patch.dict('os.environ', env, clear=False):
                    _run_repair_best_artifacts(str(root), require_signature=True)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with patch.dict('os.environ', env, clear=False):
                    _run_audit_run(str(root), require_signature=True)
            self.assertIn('Audit status: OK', stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with patch.dict('os.environ', env, clear=False):
                    _run_verify_repair_report(str(root), require_signature=True)
            self.assertIn('OK', stdout.getvalue())

            (root / 'best_config.json').write_text(json.dumps({'learning_rate': 9e-5}, indent=2, sort_keys=True))

            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()):
                    with patch.dict('os.environ', env, clear=False):
                        _run_verify_repair_report(str(root), require_signature=True)

    def test_verify_repair_history_accepts_repair_report_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)
            env = {
                'AUTORESEARCH_SIGNING_KEY': 'secret-key',
                'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                'AUTORESEARCH_VERIFY_KEY': 'secret-key',
            }
            with redirect_stdout(io.StringIO()):
                with patch.dict('os.environ', env, clear=False):
                    _run_repair_best_artifacts(str(root), require_signature=True)

            with patch.dict('os.environ', env, clear=False):
                repair = rebuild_best_artifacts(root)
                audit_report, _ = write_audit_report(root, require_signature=True)
                attestation_summary, _ = write_attestation_summary(
                    root,
                    require_signature=True,
                    audit_report=audit_report,
                )
                write_repair_report(
                    root,
                    repair_result=repair,
                    audit_report=audit_report,
                    attestation_summary=attestation_summary,
                )

            with patch.dict('os.environ', env, clear=False):
                verify_history = verify_repair_history(root, require_signature=True)
            self.assertTrue(verify_history['ok'])
            latest_entry = verify_history['entries'][-1]
            self.assertTrue(latest_entry['checks']['latest_repair_report'])
            self.assertTrue(latest_entry['checks']['latest_repair_report_snapshot'])

            with patch.dict('os.environ', env, clear=False):
                audit_report = _build_audit_report(root, require_signature=True)
            self.assertTrue(audit_report['repair_history']['latest_repair_report_snapshot_ok'])

    def test_verify_repair_report_detects_repair_history_summary_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)
            env = {
                'AUTORESEARCH_SIGNING_KEY': 'secret-key',
                'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                'AUTORESEARCH_VERIFY_KEY': 'secret-key',
            }
            with redirect_stdout(io.StringIO()):
                with patch.dict('os.environ', env, clear=False):
                    _run_repair_best_artifacts(str(root), require_signature=True)

            repair_report_path = root / 'repair_report.json'
            repair_report = json.loads(repair_report_path.read_text())
            self.assertEqual(
                repair_report['post_repair_repair_history_summary']['repair_history_entry_count'],
                1,
            )
            repair_report['post_repair_repair_history_summary'] = {
                **repair_report['post_repair_repair_history_summary'],
                'repair_history_ok': False,
                'repair_history_latest_issue_count': 1,
                'repair_history_latest_issues': ['forged'],
                'repair_history_latest_primary_issue': 'forged',
            }
            repair_report_core = {key: value for key, value in repair_report.items() if key != 'signature'}
            repair_report['signature'] = sign_json_payload(
                repair_report_core,
                key=b'secret-key',
                key_id='unit-test',
            )
            repair_report_path.write_text(json.dumps(repair_report, indent=2, sort_keys=True))

            stdout = io.StringIO()
            with self.assertRaises(SystemExit):
                with redirect_stdout(stdout):
                    with patch.dict('os.environ', env, clear=False):
                        _run_verify_repair_report(str(root), require_signature=True)
            verify_report_output = stdout.getvalue()
            self.assertIn('repair_history_summary', verify_report_output)
            self.assertNotIn('repair_history_latest_repair_report_snapshot_status', verify_report_output)
            self.assertNotIn('repair_history_latest_signed_repair_report_snapshot_status', verify_report_output)
            self.assertIn('repair_history_classification: summary_mismatch', verify_report_output)
            self.assertIn('repair_history_status: live=expected:ok actual:ok signed=expected:ok actual:ok', verify_report_output)
            self.assertIn('repair_history_primary_issue: forged(recorded)', verify_report_output)
            self.assertIn('repair_history_computed_latest_issues: []', verify_report_output)
            self.assertIn('repair_history_recorded_latest_issues: ["forged"]', verify_report_output)
            with patch.dict('os.environ', env, clear=False):
                result = verify_repair_report(root / 'repair_report.json', require_signature=True)
            self.assertEqual(result['repair_history_classification'], 'summary_mismatch')
            self.assertEqual(result['repair_history_status']['classification'], 'summary_mismatch')
            self.assertEqual(result['repair_history_primary_issue'], 'forged')
            self.assertEqual(result['repair_history_primary_issue_origin'], 'recorded')
            self.assertEqual(result['repair_history_computed_latest_issues'], [])
            self.assertEqual(result['repair_history_recorded_latest_issues'], ['forged'])
            self.assertIsNone(result['repair_history_primary_issue_details'])
            self.assertIn('repair_history_summary_expected:', verify_report_output)
            self.assertIn('repair_history_summary_actual:', verify_report_output)
            self.assertIn('"repair_history_ok": true', verify_report_output)
            self.assertIn('"repair_history_latest_issue_count": 0', verify_report_output)
            self.assertIn('"repair_history_latest_issues": []', verify_report_output)
            self.assertIn('"repair_history_latest_primary_issue": null', verify_report_output)
            self.assertIn('"repair_history_latest_issue_count": 1', verify_report_output)
            self.assertIn('"repair_history_latest_issues": ["forged"]', verify_report_output)
            self.assertIn('"repair_history_latest_primary_issue": "forged"', verify_report_output)
            self.assertIn('repair_history_latest_repair_report_snapshot: expected=ok actual=ok', verify_report_output)
            self.assertIn('repair_history_latest_signed_repair_report_snapshot: expected=ok actual=ok', verify_report_output)

    def test_verify_repair_history_detects_signed_snapshot_summary_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)
            env = {
                'AUTORESEARCH_SIGNING_KEY': 'secret-key',
                'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                'AUTORESEARCH_VERIFY_KEY': 'secret-key',
            }
            with redirect_stdout(io.StringIO()):
                with patch.dict('os.environ', env, clear=False):
                    _run_repair_best_artifacts(str(root), require_signature=True)

            history_path = root / 'repair_history' / 'repair_0001.json'
            history_payload = json.loads(history_path.read_text())
            history_payload['post_repair_repair_history_snapshot_summary'] = {
                **history_payload['post_repair_repair_history_snapshot_summary'],
                'repair_history_latest_repair_report_snapshot_expected_best_record': {
                    **history_payload['post_repair_repair_history_snapshot_summary']['repair_history_latest_repair_report_snapshot_expected_best_record'],
                    'audit_best_artifacts_ok': False,
                    'audit_best_artifact_issue_count': 1,
                    'audit_best_artifact_issues': [
                        {'name': 'best_bundle', 'status': 'missing', 'reason': 'forged-signed-snapshot'}
                    ],
                },
                'repair_history_latest_repair_report_snapshot_actual_best_record': {
                    **history_payload['post_repair_repair_history_snapshot_summary']['repair_history_latest_repair_report_snapshot_actual_best_record'],
                    'audit_best_artifacts_ok': False,
                    'audit_best_artifact_issue_count': 1,
                    'audit_best_artifact_issues': [
                        {'name': 'best_bundle', 'status': 'missing', 'reason': 'forged-signed-snapshot'}
                    ],
                },
            }
            history_core = {key: value for key, value in history_payload.items() if key != 'signature'}
            history_payload['signature'] = sign_json_payload(history_core, key=b'secret-key', key_id='unit-test')
            history_path.write_text(json.dumps(history_payload, indent=2, sort_keys=True))

            stdout = io.StringIO()
            with self.assertRaises(SystemExit):
                with redirect_stdout(stdout):
                    with patch.dict('os.environ', env, clear=False):
                        _run_verify_repair_history(str(root), require_signature=True)
            verify_history_output = stdout.getvalue()
            self.assertIn('post_repair_repair_history_snapshot_summary', verify_history_output)
            self.assertIn('post_repair_repair_history_snapshot_summary_expected:', verify_history_output)
            self.assertIn('post_repair_repair_history_snapshot_summary_actual:', verify_history_output)
            self.assertIn('"reason": "forged-signed-snapshot"', verify_history_output)
            self.assertIn('"audit_best_artifacts_ok": false', verify_history_output)
            self.assertIn('"audit_best_artifacts_ok": true', verify_history_output)

            stdout = io.StringIO()
            with self.assertRaises(SystemExit):
                with redirect_stdout(stdout):
                    with patch.dict('os.environ', env, clear=False):
                        _run_verify_repair_report(str(root), require_signature=True)
            verify_report_output = stdout.getvalue()
            self.assertIn('repair_history_summary_expected:', verify_report_output)
            self.assertIn('repair_history_summary_actual:', verify_report_output)
            self.assertIn('repair_history_latest_signed_repair_report_snapshot_status', verify_report_output)
            self.assertNotIn('repair_history_latest_repair_report_snapshot_status', verify_report_output)
            self.assertIn('repair_history_classification: signed_snapshot_mismatch', verify_report_output)
            self.assertIn('repair_history_status: live=expected:ok actual:ok signed=expected:fail actual:ok', verify_report_output)
            result = verify_repair_report(root / 'repair_report.json', require_signature=True)
            self.assertEqual(result['repair_history_classification'], 'signed_snapshot_mismatch')
            self.assertEqual(result['repair_history_status']['classification'], 'signed_snapshot_mismatch')
            self.assertIn('repair_history_latest_repair_report_snapshot: expected=ok actual=ok', verify_report_output)
            self.assertIn('repair_history_latest_signed_repair_report_snapshot: expected=fail actual=ok', verify_report_output)
            self.assertIn('"repair_history_latest_issues": ["post_repair_repair_history_snapshot_summary"]', verify_report_output)

            stdout = io.StringIO()
            with self.assertRaises(SystemExit):
                with redirect_stdout(stdout):
                    with patch.dict('os.environ', env, clear=False):
                        _run_audit_run(str(root), require_signature=True)
            audit_output = stdout.getvalue()
            self.assertIn('Repair history: FAIL', audit_output)
            self.assertIn('summary: latest_issues=1 primary_issue=post_repair_repair_history_snapshot_summary latest_best_artifact_issues=0', audit_output)
            self.assertIn('report_snapshot=ok signed_snapshot=fail', audit_output)
            self.assertIn('latest_signed_repair_report_snapshot_path:', audit_output)
            self.assertIn('latest_signed_repair_report_snapshot_expected_sha256:', audit_output)
            self.assertIn('latest_signed_repair_report_snapshot_actual_sha256:', audit_output)
            self.assertIn('latest_signed_repair_report_snapshot_expected_best_record:', audit_output)
            self.assertIn('latest_signed_repair_report_snapshot_actual_best_record:', audit_output)
            self.assertIn('repair_history_issue: post_repair_repair_history_snapshot_summary', audit_output)
            self.assertIn('forged-signed-snapshot', audit_output)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                    _run_analyze(str(root))
            analyze_output = stdout.getvalue()
            self.assertIn('Repair history: FAIL', analyze_output)
            self.assertIn('report_snapshot=ok signed_snapshot=fail', analyze_output)
            self.assertIn('latest_signed_repair_report_snapshot_expected_best_record:', analyze_output)
            self.assertIn('latest_signed_repair_report_snapshot_actual_best_record:', analyze_output)
            self.assertIn('forged-signed-snapshot', analyze_output)
            self.assertIn('Repair report verification: FAIL', analyze_output)
            self.assertIn(
                'classification=signed_snapshot_mismatch primary_issue=post_repair_repair_history_snapshot_summary(computed)',
                analyze_output,
            )
            self.assertIn('repair_history_latest_signed_repair_report_snapshot_status', analyze_output)

            analysis_payload = json.loads((root / 'analysis.json').read_text())
            self.assertEqual(
                analysis_payload['repair_report_verification']['repair_history_classification'],
                'signed_snapshot_mismatch',
            )
            self.assertEqual(
                analysis_payload['repair_report_verification']['repair_history_primary_issue'],
                'post_repair_repair_history_snapshot_summary',
            )
            self.assertEqual(
                analysis_payload['repair_report_verification']['repair_history_primary_issue_origin'],
                'computed',
            )
            self.assertIn(
                'repair_history_latest_signed_repair_report_snapshot_status',
                analysis_payload['repair_report_verification']['failed_checks'],
            )

            audit_report = json.loads((root / 'audit_report.json').read_text())
            self.assertFalse(audit_report['ok'])
            self.assertEqual(
                audit_report['summary']['repair_history_latest_issues'],
                ['post_repair_repair_history_snapshot_summary'],
            )
            self.assertEqual(
                audit_report['summary']['repair_history_latest_primary_issue'],
                'post_repair_repair_history_snapshot_summary',
            )
            self.assertTrue(audit_report['summary']['repair_history_latest_repair_report_snapshot_ok'])
            self.assertFalse(audit_report['summary']['repair_history_latest_signed_repair_report_snapshot_ok'])
            self.assertFalse(
                audit_report['repair_history']['latest_signed_repair_report_snapshot_expected_best_record']['audit_best_artifacts_ok']
            )
            self.assertFalse(
                audit_report['repair_history']['latest_signed_repair_report_snapshot_actual_best_record']['audit_best_artifacts_ok']
            )
            with patch.dict('os.environ', env, clear=False):
                rebuilt_summary, _ = write_attestation_summary(
                    root,
                    require_signature=True,
                    audit_report=audit_report,
                )
            self.assertEqual(
                rebuilt_summary['repair_report_verification']['repair_history_classification'],
                'signed_snapshot_mismatch',
            )
            self.assertEqual(
                rebuilt_summary['repair_report_verification']['repair_history_primary_issue'],
                'post_repair_repair_history_snapshot_summary',
            )
            self.assertIn(
                'repair_history_latest_signed_repair_report_snapshot_status',
                rebuilt_summary['repair_report_verification']['failed_checks'],
            )
            persisted_summary = json.loads((root / 'attestation_summary.json').read_text())
            self.assertEqual(
                persisted_summary['repair_report_verification']['repair_history_classification'],
                'signed_snapshot_mismatch',
            )
            self.assertEqual(
                persisted_summary['repair_report_verification']['repair_history_primary_issue'],
                'post_repair_repair_history_snapshot_summary',
            )

    def test_verify_repair_history_detects_best_record_snapshot_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)
            env = {
                'AUTORESEARCH_SIGNING_KEY': 'secret-key',
                'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                'AUTORESEARCH_VERIFY_KEY': 'secret-key',
            }
            with redirect_stdout(io.StringIO()):
                with patch.dict('os.environ', env, clear=False):
                    _run_repair_best_artifacts(str(root), require_signature=True)

            history_path = root / 'repair_history' / 'repair_0001.json'
            history_payload = json.loads(history_path.read_text())
            self.assertEqual(
                history_payload['post_repair_repair_history_summary'],
                history_payload['repair_report_snapshot']['post_repair_repair_history_summary'],
            )
            history_payload['repair_report_snapshot']['post_repair_best_record']['audit_best_artifacts_ok'] = False
            history_payload['repair_report_snapshot']['post_repair_best_record']['audit_best_artifact_issue_count'] = 1
            history_payload['repair_report_snapshot']['post_repair_best_record']['audit_best_artifact_issues'] = [
                {'name': 'best_bundle', 'status': 'missing', 'reason': 'forged'}
            ]
            history_payload['post_repair_repair_history_snapshot_summary'] = {
                **history_payload['post_repair_repair_history_snapshot_summary'],
                'repair_history_latest_repair_report_snapshot_expected_sha256': sha256_json(history_payload['repair_report_snapshot']),
                'repair_history_latest_repair_report_snapshot_actual_sha256': sha256_json(history_payload['repair_report_snapshot']),
                'repair_history_latest_repair_report_snapshot_expected_best_record': history_payload['repair_report_snapshot']['post_repair_best_record'],
                'repair_history_latest_repair_report_snapshot_actual_best_record': history_payload['repair_report_snapshot']['post_repair_best_record'],
                'repair_history_latest_repair_report_snapshot_actual_parse_error': None,
            }
            history_core = {key: value for key, value in history_payload.items() if key != 'signature'}
            history_payload['signature'] = sign_json_payload(history_core, key=b'secret-key', key_id='unit-test')
            history_path.write_text(json.dumps(history_payload, indent=2, sort_keys=True))

            stdout = io.StringIO()
            with self.assertRaises(SystemExit):
                with redirect_stdout(stdout):
                    with patch.dict('os.environ', env, clear=False):
                        _run_verify_repair_history(str(root), require_signature=True)
            verify_history_output = stdout.getvalue()
            self.assertIn('latest_repair_report_snapshot: fail', verify_history_output)
            self.assertIn('latest_repair_report_snapshot_expected_sha256:', verify_history_output)
            self.assertIn('latest_repair_report_snapshot_actual_sha256:', verify_history_output)
            self.assertIn('latest_repair_report_snapshot_expected_best_record:', verify_history_output)
            self.assertIn('latest_repair_report_snapshot_actual_best_record:', verify_history_output)
            self.assertIn('"audit_best_artifacts_ok": false', verify_history_output)
            self.assertIn('"audit_best_artifacts_ok": true', verify_history_output)
            self.assertIn('"reason": "forged"', verify_history_output)
            self.assertIn('repair_0001.json:latest_repair_report_snapshot', verify_history_output)

            stdout = io.StringIO()
            with self.assertRaises(SystemExit):
                with redirect_stdout(stdout):
                    with patch.dict('os.environ', env, clear=False):
                        _run_audit_run(str(root), require_signature=True)
            audit_output = stdout.getvalue()
            self.assertIn('Audit status: FAIL', audit_output)
            self.assertIn('Repair history: FAIL', audit_output)
            self.assertIn('summary: latest_issues=1 primary_issue=latest_repair_report_snapshot latest_best_artifact_issues=1', audit_output)
            self.assertIn('report_snapshot=fail signed_snapshot=ok', audit_output)
            self.assertIn('latest_repair_report_snapshot_expected_sha256:', audit_output)
            self.assertIn('latest_repair_report_snapshot_actual_sha256:', audit_output)
            self.assertIn('latest_repair_report_snapshot_expected_best_record:', audit_output)
            self.assertIn('latest_repair_report_snapshot_actual_best_record:', audit_output)
            self.assertIn('repair_history_issue: latest_repair_report_snapshot', audit_output)
            self.assertIn('repair_history_best_artifact_issue best_bundle: missing (forged)', audit_output)

            audit_report = json.loads((root / 'audit_report.json').read_text())
            self.assertFalse(audit_report['ok'])
            self.assertTrue(audit_report['repair_history']['present'])
            self.assertFalse(audit_report['repair_history']['ok'])
            self.assertFalse(audit_report['repair_history']['latest_repair_report_snapshot_ok'])
            self.assertIsNotNone(audit_report['repair_history']['latest_repair_report_snapshot_expected_sha256'])
            self.assertIsNotNone(audit_report['repair_history']['latest_repair_report_snapshot_actual_sha256'])
            self.assertFalse(audit_report['repair_history']['latest_repair_report_snapshot_expected_best_record']['audit_best_artifacts_ok'])
            self.assertTrue(audit_report['repair_history']['latest_repair_report_snapshot_actual_best_record']['audit_best_artifacts_ok'])
            self.assertIsNotNone(audit_report['repair_history']['latest_signed_repair_report_snapshot_path'])
            self.assertIsNotNone(audit_report['repair_history']['latest_signed_repair_report_snapshot_expected_sha256'])
            self.assertIsNotNone(audit_report['repair_history']['latest_signed_repair_report_snapshot_actual_sha256'])
            self.assertFalse(audit_report['repair_history']['latest_signed_repair_report_snapshot_expected_best_record']['audit_best_artifacts_ok'])
            self.assertFalse(audit_report['repair_history']['latest_signed_repair_report_snapshot_actual_best_record']['audit_best_artifacts_ok'])
            self.assertTrue(audit_report['summary']['repair_history_present'])
            self.assertFalse(audit_report['summary']['repair_history_ok'])
            self.assertEqual(audit_report['summary']['repair_history_entry_count'], 1)
            self.assertEqual(audit_report['summary']['repair_history_latest_issue_count'], 1)
            self.assertEqual(audit_report['summary']['repair_history_latest_issues'], ['latest_repair_report_snapshot'])
            self.assertEqual(audit_report['summary']['repair_history_latest_primary_issue'], 'latest_repair_report_snapshot')
            self.assertFalse(audit_report['summary']['repair_history_latest_repair_report_snapshot_ok'])
            self.assertTrue(audit_report['summary']['repair_history_latest_signed_repair_report_snapshot_ok'])
            self.assertEqual(audit_report['summary']['repair_history_latest_best_artifact_issue_count'], 1)
            self.assertEqual(
                audit_report['summary']['repair_history_latest_best_artifact_issues'],
                [{'name': 'best_bundle', 'status': 'missing', 'reason': 'forged'}],
            )

            stale_audit_report = json.loads(json.dumps(audit_report))
            for key in [
                'repair_history_present',
                'repair_history_ok',
                'repair_history_entry_count',
                'repair_history_latest_issue_count',
                'repair_history_latest_issues',
                'repair_history_latest_repair_report_snapshot_ok',
                'repair_history_latest_signed_repair_report_snapshot_ok',
                'repair_history_latest_repair_report_snapshot_path',
                'repair_history_latest_repair_report_snapshot_expected_sha256',
                'repair_history_latest_repair_report_snapshot_actual_sha256',
                'repair_history_latest_repair_report_snapshot_expected_best_record',
                'repair_history_latest_repair_report_snapshot_actual_best_record',
                'repair_history_latest_repair_report_snapshot_actual_parse_error',
                'repair_history_latest_best_artifact_issue_count',
                'repair_history_latest_best_artifact_issues',
            ]:
                stale_audit_report['summary'].pop(key, None)
            with patch.dict('os.environ', env, clear=False):
                rebuilt_summary, _ = write_attestation_summary(root, require_signature=True, audit_report=stale_audit_report)
            self.assertTrue(rebuilt_summary['summary']['repair_history_present'])
            self.assertFalse(rebuilt_summary['summary']['repair_history_ok'])
            self.assertEqual(rebuilt_summary['summary']['repair_history_entry_count'], 1)
            self.assertEqual(rebuilt_summary['summary']['repair_history_latest_issue_count'], 1)
            self.assertEqual(rebuilt_summary['summary']['repair_history_latest_issues'], ['latest_repair_report_snapshot'])
            self.assertEqual(rebuilt_summary['summary']['repair_history_latest_primary_issue'], 'latest_repair_report_snapshot')
            self.assertFalse(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_ok'])
            self.assertTrue(rebuilt_summary['summary']['repair_history_latest_signed_repair_report_snapshot_ok'])
            self.assertIsNotNone(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_path'])
            self.assertIsNotNone(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_expected_sha256'])
            self.assertIsNotNone(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_actual_sha256'])
            self.assertFalse(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_expected_best_record']['audit_best_artifacts_ok'])
            self.assertTrue(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_actual_best_record']['audit_best_artifacts_ok'])
            self.assertIsNone(rebuilt_summary['summary']['repair_history_latest_repair_report_snapshot_actual_parse_error'])
            self.assertEqual(rebuilt_summary['summary']['repair_history_latest_best_artifact_issue_count'], 1)
            self.assertEqual(
                rebuilt_summary['summary']['repair_history_latest_best_artifact_issues'],
                [{'name': 'best_bundle', 'status': 'missing', 'reason': 'forged'}],
            )

    def test_verify_repair_history_detects_repair_report_digest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)
            env = {
                'AUTORESEARCH_SIGNING_KEY': 'secret-key',
                'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                'AUTORESEARCH_VERIFY_KEY': 'secret-key',
            }
            with redirect_stdout(io.StringIO()):
                with patch.dict('os.environ', env, clear=False):
                    _run_repair_best_artifacts(str(root), require_signature=True)

            history_path = root / 'repair_history' / 'repair_0001.json'
            history_payload = json.loads(history_path.read_text())
            history_payload['repair_report_core_sha256'] = 'forged-core-digest'
            history_core = {key: value for key, value in history_payload.items() if key != 'signature'}
            history_payload['signature'] = sign_json_payload(history_core, key=b'secret-key', key_id='unit-test')
            history_path.write_text(json.dumps(history_payload, indent=2, sort_keys=True))

            stdout = io.StringIO()
            with self.assertRaises(SystemExit):
                with redirect_stdout(stdout):
                    with patch.dict('os.environ', env, clear=False):
                        _run_verify_repair_history(str(root), require_signature=True)
            verify_history_output = stdout.getvalue()
            self.assertIn('latest_repair_report: fail', verify_history_output)
            self.assertIn('latest_repair_report_snapshot: ok', verify_history_output)
            self.assertIn('latest_repair_report_expected_core_sha256: forged-core-digest', verify_history_output)
            self.assertIn('latest_repair_report_actual_core_sha256:', verify_history_output)
            self.assertIn('repair_0001.json:latest_repair_report', verify_history_output)

            stdout = io.StringIO()
            with self.assertRaises(SystemExit):
                with redirect_stdout(stdout):
                    with patch.dict('os.environ', env, clear=False):
                        _run_audit_run(str(root), require_signature=True)
            audit_output = stdout.getvalue()
            self.assertIn('Repair history: FAIL', audit_output)
            self.assertIn('summary: latest_issues=1 primary_issue=latest_repair_report latest_best_artifact_issues=0', audit_output)
            self.assertIn('report=fail report_snapshot=ok signed_snapshot=ok', audit_output)
            self.assertIn('latest_repair_report_expected_core_sha256: forged-core-digest', audit_output)
            self.assertIn('latest_repair_report_actual_core_sha256:', audit_output)
            self.assertIn('repair_history_issue: latest_repair_report', audit_output)

            audit_report = json.loads((root / 'audit_report.json').read_text())
            self.assertFalse(audit_report['repair_history']['latest_repair_report_integrity_ok'])
            self.assertEqual(audit_report['repair_history']['latest_repair_report_expected_core_sha256'], 'forged-core-digest')
            self.assertIsNotNone(audit_report['repair_history']['latest_repair_report_actual_core_sha256'])
            self.assertEqual(audit_report['summary']['repair_history_latest_issues'], ['latest_repair_report'])
            self.assertEqual(audit_report['summary']['repair_history_latest_primary_issue'], 'latest_repair_report')
            with patch.dict('os.environ', env, clear=False):
                rebuilt_summary, _ = write_attestation_summary(
                    root,
                    require_signature=True,
                    audit_report=audit_report,
                )
            self.assertEqual(
                rebuilt_summary['repair_report_verification']['repair_history_classification'],
                'summary_mismatch',
            )
            self.assertEqual(
                rebuilt_summary['repair_report_verification']['repair_history_primary_issue'],
                'latest_repair_report',
            )
            self.assertEqual(
                rebuilt_summary['repair_report_verification']['repair_history_primary_issue_origin'],
                'computed',
            )
            self.assertEqual(
                rebuilt_summary['repair_report_verification']['repair_history_computed_latest_issues'],
                ['latest_repair_report'],
            )
            self.assertEqual(
                rebuilt_summary['repair_report_verification']['repair_history_recorded_latest_issues'],
                [],
            )
            self.assertEqual(
                rebuilt_summary['repair_report_verification']['repair_history_primary_issue_details']['expected_core_sha256'],
                'forged-core-digest',
            )
            self.assertIsNotNone(
                rebuilt_summary['repair_report_verification']['repair_history_primary_issue_details']['actual_core_sha256']
            )
            persisted_summary = json.loads((root / 'attestation_summary.json').read_text())
            self.assertEqual(
                persisted_summary['repair_report_verification']['repair_history_primary_issue'],
                'latest_repair_report',
            )

    def test_verify_repair_history_detects_history_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)
            env = {
                'AUTORESEARCH_SIGNING_KEY': 'secret-key',
                'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                'AUTORESEARCH_VERIFY_KEY': 'secret-key',
            }
            with redirect_stdout(io.StringIO()):
                with patch.dict('os.environ', env, clear=False):
                    _run_repair_best_artifacts(str(root), require_signature=True)

            history_path = root / 'repair_history' / 'repair_0001.json'
            history_payload = json.loads(history_path.read_text())
            history_payload['sequence_number'] = 9
            history_path.write_text(json.dumps(history_payload, indent=2, sort_keys=True))

            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()):
                    with patch.dict('os.environ', env, clear=False):
                        _run_verify_repair_history(str(root), require_signature=True)

    def test_audit_run_detects_missing_required_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, _ = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved without proof path",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=None,
                )
            )

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=None)

            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                report = _build_audit_report(root, require_signature=True)
            self.assertFalse(report['ok'])
            self.assertEqual(report['summary']['proofs_missing_required'], 1)

            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()):
                    with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                        _run_audit_run(str(root), require_signature=True)
            self.assertTrue((root / 'audit_report.json').exists())

    def test_analyze_prints_attestation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_run_dir, baseline_provenance = self._write_signed_provenance_artifacts(
                root,
                run_name="baseline",
                objective_value=0.52,
            )
            exp_run_dir, proof_json_path = self._write_signed_run_artifacts(root)
            exp_provenance = exp_run_dir / "provenance.json"
            baseline_payload = json.loads(baseline_provenance.read_text())
            exp_payload = json.loads(exp_provenance.read_text())

            tracker = ExperimentTracker(root, objective_metric="eval_reward", direction="maximize")
            tracker.record(
                ExperimentRecord(
                    experiment_id=baseline_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.52},
                    objective_value=0.52,
                    training_time=1.0,
                    status="keep",
                    description="baseline",
                    checkpoint_path=str(baseline_run_dir),
                    provenance_path=str(baseline_provenance),
                )
            )
            tracker.record(
                ExperimentRecord(
                    experiment_id=exp_payload["experiment_id"],
                    params={"learning_rate": 1e-5},
                    metrics={"eval_reward": 0.55},
                    objective_value=0.55,
                    training_time=1.0,
                    status="keep",
                    description="improved",
                    checkpoint_path=str(exp_run_dir),
                    provenance_path=str(exp_provenance),
                    proof_path=str(proof_json_path),
                )
            )

            (root / 'preflight.json').write_text(json.dumps({
                'configured_time_budget_seconds': 12,
                'allow_undersized_budget': False,
                'recommended_min_budget_seconds': {'baseline': 10, 'experiment': 18},
                'budget_recommendation': {
                    'baseline': {'heuristic_seconds': 10, 'observed_phase_seconds': 9, 'historical_phase_samples': 1},
                    'experiment': {'heuristic_seconds': 18, 'observed_phase_seconds': 16, 'historical_phase_samples': 1},
                },
            }, indent=2))

            self._write_best_artifacts(root, run_dir=exp_run_dir, proof_json_path=proof_json_path)

            env = {
                'AUTORESEARCH_SIGNING_KEY': 'secret-key',
                'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                'AUTORESEARCH_VERIFY_KEY': 'secret-key',
            }
            with redirect_stdout(io.StringIO()):
                with patch.dict('os.environ', env, clear=False):
                    _run_repair_best_artifacts(str(root), require_signature=True)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                    _run_analyze(str(root))
            output = stdout.getvalue()
            self.assertIn('Preflight:', output)
            self.assertIn('experiment_budget_detail:', output)
            self.assertIn('Attestation: OK', output)
            self.assertIn('best_artifacts: ok=7 fail=0 missing=0 not_required=0', output)
            self.assertIn('proof=ok artifacts=ok', output)
            self.assertIn('Repair history: OK', output)
            self.assertIn('summary: latest_issues=0 primary_issue=none latest_best_artifact_issues=0', output)
            self.assertIn('repair_ok=True artifacts=ok report=ok report_snapshot=ok signed_snapshot=ok', output)
            self.assertIn('entries: 1', output)
            self.assertIn('latest_signature: ok', output)
            self.assertIn('Repair report verification: OK', output)
            self.assertIn('classification=match primary_issue=none', output)

            analysis_payload = json.loads((root / 'analysis.json').read_text())
            self.assertIn('repair_history', analysis_payload)
            self.assertEqual(analysis_payload['repair_history']['entry_count'], 1)
            self.assertTrue(analysis_payload['repair_history']['ok'])
            self.assertIsNone(analysis_payload['repair_history']['latest_primary_issue'])
            self.assertEqual(
                analysis_payload['repair_report_verification']['repair_history_classification'],
                'match',
            )
            self.assertEqual(analysis_payload['repair_report_verification']['failed_checks'], [])
            self.assertIsNone(analysis_payload['repair_report_verification']['repair_history_primary_issue'])


class AutoResearchConfigDefaultsTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = {
            "time_budget": 300,
            "max_experiments": 0,
            "max_wall_clock": 0,
            "proposer": "perturbation",
            "search_space": "auto_research",
            "algorithm": "gspo",
            "patience": 0,
            "output_dir": "./auto_research_results",
            "wandb": False,
            "selection_promotion_zscore": None,
            "experiment_isolation": None,
            "runtime_environment": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_cli_uses_locked_selection_metric(self) -> None:
        config = _build_auto_research_config(self._args(), eval_episodes=4)

        self.assertEqual(config.objective_metric, "selection_score")
        self.assertEqual(config.selection_promotion_zscore, 1.0)
        self.assertEqual(config.experiment_isolation, "worktree")
        self.assertEqual(config.runtime_environment, "venv")

    def test_cli_selection_gate_override_wins_over_config(self) -> None:
        config = _build_auto_research_config(
            self._args(selection_promotion_zscore=1.75),
            eval_episodes=4,
            config_data={"selection_promotion_zscore": 0.5},
        )

        self.assertEqual(config.selection_promotion_zscore, 1.75)

    def test_config_file_can_set_selection_gate_when_cli_omits_it(self) -> None:
        config = _build_auto_research_config(
            self._args(),
            eval_episodes=4,
            config_data={"selection_promotion_zscore": 1.5},
        )

        self.assertEqual(config.selection_promotion_zscore, 1.5)

    def test_cli_experiment_isolation_override_wins_over_config(self) -> None:
        config = _build_auto_research_config(
            self._args(experiment_isolation="shared"),
            eval_episodes=4,
            config_data={"experiment_isolation": "worktree"},
        )

        self.assertEqual(config.experiment_isolation, "shared")

    def test_config_file_can_set_experiment_isolation_when_cli_omits_it(self) -> None:
        config = _build_auto_research_config(
            self._args(),
            eval_episodes=4,
            config_data={"experiment_isolation": "shared"},
        )

        self.assertEqual(config.experiment_isolation, "shared")

    def test_cli_runtime_environment_override_wins_over_config(self) -> None:
        config = _build_auto_research_config(
            self._args(runtime_environment="uv"),
            eval_episodes=4,
            config_data={"runtime_environment": "venv"},
        )

        self.assertEqual(config.runtime_environment, "uv")

    def test_config_file_can_set_runtime_environment_when_cli_omits_it(self) -> None:
        config = _build_auto_research_config(
            self._args(),
            eval_episodes=4,
            config_data={"runtime_environment": "uv"},
        )

        self.assertEqual(config.runtime_environment, "uv")

if __name__ == '__main__':
    unittest.main()
