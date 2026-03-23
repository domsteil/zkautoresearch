from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment_runtime import (
    DeviceInfo,
    RunSummary,
    build_time_budget_recommendation,
    collect_file_digests,
    detect_device_info,
    getenv_bool,
    sha256_file,
    sha256_json,
    sign_json_payload,
    verify_json_signature,
    verify_provenance_envelope,
    write_run_summary,
)


class _FakeCuda:
    def __init__(self, available: bool, name: str = 'Fake GPU') -> None:
        self._available = available
        self._name = name

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, index: int) -> str:
        return self._name


class _FakeTorch:
    __version__ = 'fake'

    def __init__(self, available: bool) -> None:
        self.cuda = _FakeCuda(available)


class ExperimentRuntimeTests(unittest.TestCase):
    def test_getenv_bool_parses_known_values(self) -> None:
        with patch.dict('os.environ', {'AUTORESEARCH_USE_GPU': 'true'}, clear=True):
            self.assertTrue(getenv_bool('AUTORESEARCH_USE_GPU', False))

        with patch.dict('os.environ', {'AUTORESEARCH_USE_GPU': '0'}, clear=True):
            self.assertFalse(getenv_bool('AUTORESEARCH_USE_GPU', True))

    def test_detect_device_info_prefers_cuda_when_available(self) -> None:
        fake_torch = _FakeTorch(True)
        with patch.dict(sys.modules, {'torch': fake_torch}):
            info = detect_device_info(prefer_gpu=True, requested_bf16=True)

        self.assertEqual(info.accelerator, 'cuda')
        self.assertFalse(info.use_cpu)
        self.assertTrue(info.cuda_available)
        self.assertTrue(info.bf16_enabled)
        self.assertEqual(info.device_name, 'Fake GPU')

    def test_detect_device_info_requires_gpu(self) -> None:
        fake_torch = _FakeTorch(False)
        with patch.dict(sys.modules, {'torch': fake_torch}):
            with self.assertRaises(RuntimeError):
                detect_device_info(prefer_gpu=True, require_gpu=True)

    def test_write_run_summary_round_trips_json(self) -> None:
        summary = RunSummary(
            status='ok',
            started_at=1.0,
            completed_at=2.0,
            time_budget_seconds=300,
            accelerator='cpu',
            cuda_available=False,
            device_name=None,
            torch_version='fake',
            bf16_enabled=False,
            training_seconds=1.5,
            total_seconds=2.0,
            peak_vram_mb=0.0,
            metrics={'eval_reward': 0.5},
            config={'learning_rate': 5e-6},
            phase_timings={'startup_seconds': 0.5, 'evaluation_seconds': 0.5},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_run_summary(Path(tmpdir) / 'latest_run.json', summary)
            payload = json.loads(path.read_text())

        self.assertEqual(payload['status'], 'ok')
        self.assertEqual(payload['metrics']['eval_reward'], 0.5)
        self.assertEqual(payload['config']['learning_rate'], 5e-06)
        self.assertEqual(payload['phase_timings']['startup_seconds'], 0.5)

    def test_build_time_budget_recommendation_uses_phase_history(self) -> None:
        recommendation = build_time_budget_recommendation(
            runtime_profile='smoke',
            accelerator='cpu',
            skip_train=False,
            train_scenario_count=1,
            eval_scenario_count=1,
            eval_episodes=1,
            max_new_tokens=4,
            historical_durations=[12.0, 14.0],
            historical_phase_timings=[
                {'startup_seconds': 6.0, 'evaluation_seconds': 4.0, 'training_seconds': 7.0},
                {'startup_seconds': 5.0, 'evaluation_seconds': 3.0, 'training_seconds': 8.0},
            ],
        )
        self.assertGreaterEqual(recommendation['recommended_seconds'], 18)
        self.assertEqual(recommendation['phase_budget_seconds']['startup_seconds'], 8)
        self.assertEqual(recommendation['historical_phase_samples'], 2)

    def test_sign_and_verify_json_payload(self) -> None:
        payload = {'experiment_id': 'exp_0001', 'objective': 0.5}
        signature = sign_json_payload(payload, key=b'secret-key', key_id='unit-test')

        ok, reason = verify_json_signature(payload, signature, key=b'secret-key')
        self.assertTrue(ok)
        self.assertEqual(reason, 'ok')

        ok, reason = verify_json_signature({'experiment_id': 'exp_0002'}, signature, key=b'secret-key')
        self.assertFalse(ok)
        self.assertIn('hash mismatch', reason.lower())

    def test_verify_provenance_envelope_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            output_dir = run_dir / 'outputs' / 'final_model'
            output_dir.mkdir(parents=True)

            config_path = run_dir / 'config.json'
            config_payload = {'learning_rate': 5e-6, 'runtime_profile': 'smoke'}
            config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True))

            summary_path = run_dir / 'summary.json'
            summary_payload = {
                'status': 'ok',
                'config': config_payload,
                'metrics': {'eval_reward': 0.5},
            }
            summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

            log_path = run_dir / 'train.log'
            log_path.write_text('training ok\n')

            runtime_artifact_path = run_dir / 'runtime_environment.json'
            runtime_artifact_payload = {
                'mode': 'custom_command',
                'command': ['python', 'train.py'],
                'environment_snapshot': {
                    'status': 'ok',
                    'python_version': '3.10.0',
                    'package_count': 2,
                    'packages_sha256': sha256_json([['alpha', '1.0'], ['beta', '2.0']]),
                    'packages': [['alpha', '1.0'], ['beta', '2.0']],
                },
            }
            runtime_artifact_path.write_text(json.dumps(runtime_artifact_payload, indent=2, sort_keys=True))

            artifact_path = output_dir / 'weights.bin'
            artifact_path.write_bytes(b'weights')
            artifact_files = [digest.to_dict() for digest in collect_file_digests(output_dir)]

            envelope_core = {
                'experiment_id': 'exp_0001',
                'description': 'unit test envelope',
                'process_status': 'ok',
                'record_status': 'keep',
                'elapsed_seconds': 1.0,
                'objective_metric': 'eval_reward',
                'objective_value': 0.5,
                'git': {'repo_root': str(run_dir), 'commit': 'test', 'branch': 'main', 'is_dirty': False},
                'config_path': 'config.json',
                'config_sha256': sha256_file(config_path),
                'summary_path': 'summary.json',
                'summary_sha256': sha256_file(summary_path),
                'log_path': 'train.log',
                'log_sha256': sha256_file(log_path),
                'runtime_artifact_path': 'runtime_environment.json',
                'runtime_artifact_sha256': sha256_file(runtime_artifact_path),
                'summary_config_sha256': sha256_json(summary_payload['config']),
                'summary_metrics_sha256': sha256_json(summary_payload['metrics']),
                'artifacts': {
                    'model_dir': 'outputs/final_model',
                    'manifest_sha256': sha256_json(artifact_files),
                    'files': artifact_files,
                },
            }
            envelope = dict(envelope_core)
            envelope['envelope_sha256'] = sha256_json(envelope_core)
            envelope['signature'] = sign_json_payload(envelope, key=b'secret-key', key_id='unit-test')

            provenance_path = run_dir / 'provenance.json'
            provenance_path.write_text(json.dumps(envelope, indent=2, sort_keys=True))

            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'secret-key'}, clear=False):
                result = verify_provenance_envelope(provenance_path, require_signature=True)
                self.assertTrue(result['ok'])
                self.assertTrue(result['signature_valid'])

                log_path.write_text('tampered\n')
                result = verify_provenance_envelope(provenance_path, require_signature=True)
                self.assertFalse(result['ok'])
                self.assertFalse(result['checks']['log_sha256'])

                log_path.write_text('training ok\n')
                runtime_artifact_path.write_text(json.dumps({'tampered': True}, indent=2, sort_keys=True))
                result = verify_provenance_envelope(provenance_path, require_signature=True)
                self.assertFalse(result['ok'])
                self.assertFalse(result['checks']['runtime_artifact_sha256'])


if __name__ == '__main__':
    unittest.main()
