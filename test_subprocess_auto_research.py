from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_research import _resolve_runtime_settings, _verify_proof_artifact
from experiment_runtime import DeviceInfo, estimate_min_time_budget_seconds, sha256_file, verify_provenance_envelope
from stark_autoresearch.proof import MetricImprovementProof, verify_improvement
from subprocess_auto_research import (
    DEFAULT_TRAIN_COMMAND,
    SubprocessAutoResearchRunner,
    run_train_subprocess,
)
from stateset_agents.training.auto_research.config import AutoResearchConfig
from stateset_agents.training.auto_research.experiment_tracker import ExperimentRecord, ExperimentTracker
from stateset_agents.training.auto_research.proposer import ExperimentProposer


class FixedProposer(ExperimentProposer):
    def __init__(self, proposals: list[tuple[dict[str, object], str]]):
        self._proposals = proposals
        self._index = 0

    def propose(self, current_best, history):
        params, description = self._proposals[self._index]
        self._index += 1
        return dict(params), description


class RuntimeSettingsTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = {
            "runtime_profile": "standard",
            "max_new_tokens": None,
            "eval_episodes": None,
            "train_scenario_limit": None,
            "eval_scenario_limit": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_smoke_profile_applies_cpu_friendly_defaults(self) -> None:
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
        ) = _resolve_runtime_settings(self._args(runtime_profile="smoke"), {})

        self.assertEqual(runtime_profile, "smoke")
        self.assertEqual(model_name, "Qwen/Qwen1.5-0.5B-Chat")
        self.assertEqual(max_new_tokens, 4)
        self.assertEqual(eval_episodes, 1)
        self.assertEqual(len(train_scenarios), 1)
        self.assertEqual(len(eval_scenarios), 1)
        self.assertEqual(runtime_overrides["num_generations"], 2)
        self.assertEqual(runtime_overrides["lora_r"], 4)
        self.assertFalse(runtime_overrides["run_post_eval"])
        self.assertEqual(runtime_overrides["post_eval_samples"], 0)
        self.assertEqual(baseline["num_generations"], 2)
        self.assertEqual(reward_domain, "customer_service")
        self.assertTrue(model_name)
        self.assertTrue(system_prompt)

    def test_explicit_cli_values_override_smoke_profile(self) -> None:
        (
            _,
            _,
            _,
            _,
            _,
            max_new_tokens,
            eval_episodes,
            train_scenarios,
            eval_scenarios,
            _,
        ) = _resolve_runtime_settings(
            self._args(
                runtime_profile="smoke",
                max_new_tokens=7,
                eval_episodes=3,
                train_scenario_limit=2,
                eval_scenario_limit=3,
            ),
            {},
        )

        self.assertEqual(max_new_tokens, 7)
        self.assertEqual(eval_episodes, 3)
        self.assertEqual(len(train_scenarios), 2)
        self.assertEqual(len(eval_scenarios), 3)


class SubprocessAutoResearchTests(unittest.TestCase):
    def _device_info(self) -> DeviceInfo:
        return DeviceInfo(
            accelerator="cpu",
            use_cpu=True,
            cuda_available=False,
            device_name=None,
            torch_version="test",
            bf16_enabled=False,
        )

    def _init_git_repo(self, repo_path: Path) -> None:
        subprocess.run(["git", "init", str(repo_path)], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(repo_path), "config", "user.email", "unit@test.invalid"], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(repo_path), "config", "user.name", "Unit Test"], check=True, capture_output=True, text=True)
        (repo_path / "tracked.txt").write_text("tracked\n")
        subprocess.run(["git", "-C", str(repo_path), "add", "tracked.txt"], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(repo_path), "commit", "-m", "init"], check=True, capture_output=True, text=True)

    def test_runner_prefers_repo_venv_for_default_train_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            python_path = repo_path / ".venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("#!/bin/sh\nexit 0\n")
            python_path.chmod(0o755)

            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=1,
                output_dir=str(repo_path / "results"),
                search_space_name="quick",
                runtime_environment="venv",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                allow_undersized_budget=True,
                source_repo_root=repo_path,
            )

            invocation = runner._resolve_train_invocation(
                runner._shared_execution_context(requested_mode="shared")
            )

            self.assertEqual(invocation.command, (str(python_path), "train.py"))
            self.assertEqual(invocation.runtime["mode"], "venv")
            self.assertEqual(invocation.runtime["python_executable"], str(python_path))
            self.assertEqual(invocation.runtime["venv_root"], str(repo_path / ".venv"))

    def test_runner_falls_back_to_uv_when_repo_venv_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=1,
                output_dir=str(repo_path / "results"),
                search_space_name="quick",
                runtime_environment="venv",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                allow_undersized_budget=True,
                source_repo_root=repo_path,
            )

            invocation = runner._resolve_train_invocation(
                runner._shared_execution_context(requested_mode="shared")
            )

            self.assertEqual(invocation.command, DEFAULT_TRAIN_COMMAND)
            self.assertEqual(invocation.runtime["mode"], "uv")
            self.assertIn("falling back to uv", invocation.runtime["fallback_reason"])

    def test_runner_executes_baseline_in_isolated_worktree(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()
            self._init_git_repo(repo_path)

            script_path = tmp_path / "capture_cwd.py"
            script_path.write_text(
                textwrap.dedent(
                    """
                    import json
                    import os
                    from pathlib import Path

                    config = json.loads(Path(os.environ["AUTORESEARCH_RUN_CONFIG"]).read_text())
                    summary_path = Path(os.environ["AUTORESEARCH_RUN_SUMMARY"])

                    summary = {
                        "status": "ok",
                        "total_seconds": 0.1,
                        "phase_timings": {
                            "startup_seconds": 0.02,
                            "training_seconds": 0.0,
                            "evaluation_seconds": 0.05,
                        },
                        "metrics": {"eval_reward": 0.5, "selection_score": 0.5},
                        "config": config,
                    }
                    summary_path.write_text(json.dumps(summary))
                    """
                ).strip()
                + "\n"
            )

            output_dir = tmp_path / "results"
            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=1,
                output_dir=str(output_dir),
                search_space_name="quick",
                experiment_isolation="worktree",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                train_command=(sys.executable, str(script_path)),
                allow_undersized_budget=True,
                source_repo_root=repo_path,
            )

            tracker = runner.run()

            self.assertEqual([record.status for record in tracker.records], ["keep"])
            provenance = json.loads(Path(tracker.records[0].provenance_path).read_text())
            execution = provenance["execution"]
            self.assertEqual(execution["requested_mode"], "worktree")
            self.assertEqual(execution["mode"], "worktree")
            self.assertEqual(execution["source_repo_root"], str(repo_path.resolve()))
            self.assertNotEqual(Path(execution["cwd"]), repo_path.resolve())
            self.assertFalse(Path(execution["cwd"]).exists())
            self.assertEqual(provenance["git"]["repo_root"], str(repo_path.resolve()))
            self.assertEqual(execution["git"]["commit"], provenance["git"]["commit"])
            self.assertEqual(provenance["runtime"]["mode"], "custom_command")
            self.assertTrue(provenance["runtime"]["fingerprint"])
            self.assertEqual(provenance["runtime"]["environment_snapshot"]["status"], "ok")
            self.assertGreater(provenance["runtime"]["environment_snapshot"]["package_count"], 0)
            self.assertTrue(provenance["runtime"]["environment_snapshot"]["packages_sha256"])
            run_runtime_artifact_path = Path(tracker.records[0].checkpoint_path) / provenance["runtime_artifact_path"]
            self.assertTrue(run_runtime_artifact_path.exists())
            self.assertEqual(sha256_file(run_runtime_artifact_path), provenance["runtime_artifact_sha256"])
            run_runtime_artifact = json.loads(run_runtime_artifact_path.read_text())
            self.assertGreater(len(run_runtime_artifact["environment_snapshot"]["packages"]), 0)
            self.assertEqual(
                run_runtime_artifact["environment_snapshot"]["packages_sha256"],
                provenance["runtime"]["environment_snapshot"]["packages_sha256"],
            )

            preflight = json.loads((output_dir / "preflight.json").read_text())
            self.assertEqual(preflight["execution"]["requested_mode"], "worktree")
            self.assertEqual(preflight["runtime"]["mode"], "custom_command")
            self.assertTrue(preflight["runtime"]["fingerprint"])
            self.assertEqual(preflight["runtime"]["environment_snapshot"]["status"], "ok")
            self.assertGreater(preflight["runtime"]["environment_snapshot"]["package_count"], 0)
            preflight_runtime_artifact_path = output_dir / preflight["runtime_artifact_path"]
            self.assertTrue(preflight_runtime_artifact_path.exists())
            self.assertEqual(sha256_file(preflight_runtime_artifact_path), preflight["runtime_artifact_sha256"])
            preflight_runtime_artifact = json.loads(preflight_runtime_artifact_path.read_text())
            self.assertGreater(len(preflight_runtime_artifact["environment_snapshot"]["packages"]), 0)
            self.assertEqual(
                preflight["runtime"]["environment_snapshot"]["packages_sha256"],
                provenance["runtime"]["environment_snapshot"]["packages_sha256"],
            )
            self.assertEqual(
                preflight_runtime_artifact["environment_snapshot"]["packages_sha256"],
                provenance["runtime"]["environment_snapshot"]["packages_sha256"],
            )

    def test_runner_falls_back_to_shared_execution_when_worktree_setup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_path = tmp_path / "not_a_repo"
            repo_path.mkdir()

            script_path = tmp_path / "capture_shared.py"
            script_path.write_text(
                textwrap.dedent(
                    """
                    import json
                    import os
                    from pathlib import Path

                    config = json.loads(Path(os.environ["AUTORESEARCH_RUN_CONFIG"]).read_text())
                    summary_path = Path(os.environ["AUTORESEARCH_RUN_SUMMARY"])

                    summary = {
                        "status": "ok",
                        "total_seconds": 0.1,
                        "phase_timings": {
                            "startup_seconds": 0.02,
                            "training_seconds": 0.0,
                            "evaluation_seconds": 0.05,
                        },
                        "metrics": {"eval_reward": 0.5, "selection_score": 0.5},
                        "config": config,
                    }
                    summary_path.write_text(json.dumps(summary))
                    """
                ).strip()
                + "\n"
            )

            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=1,
                output_dir=str(tmp_path / "results"),
                search_space_name="quick",
                experiment_isolation="worktree",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                train_command=(sys.executable, str(script_path)),
                allow_undersized_budget=True,
                source_repo_root=repo_path,
            )

            tracker = runner.run()

            self.assertEqual([record.status for record in tracker.records], ["keep"])
            provenance = json.loads(Path(tracker.records[0].provenance_path).read_text())
            execution = provenance["execution"]
            self.assertEqual(execution["requested_mode"], "worktree")
            self.assertEqual(execution["mode"], "shared")
            self.assertEqual(execution["cwd"], str(repo_path.resolve()))
            self.assertTrue(execution["fallback_reason"])

    def test_runner_rejects_reward_search_space(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=1,
                output_dir=tmp,
                search_space_name="reward",
            )

            with self.assertRaises(ValueError):
                SubprocessAutoResearchRunner(
                    config=config,
                    baseline_params={"learning_rate": 5e-6},
                    base_run_config={"runtime_profile": "smoke"},
                    device_info=self._device_info(),
                )

    def test_subprocess_runner_rejects_locked_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lock_path = tmp_path / ".autoresearch.lock"
            with open(lock_path, "w+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_file.write('{"pid": 999, "output_dir": "%s"}' % tmp_path)
                lock_file.flush()

                config = AutoResearchConfig(
                    time_budget=5,
                    max_experiments=1,
                    output_dir=tmp,
                    search_space_name="quick",
                )
                runner = SubprocessAutoResearchRunner(
                    config=config,
                    baseline_params={"learning_rate": 5e-6},
                    base_run_config={"runtime_profile": "smoke"},
                    device_info=self._device_info(),
                )

                with self.assertRaises(RuntimeError):
                    runner.run()

    def test_run_train_subprocess_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "sleep_script.py"
            script_path.write_text("import time\ntime.sleep(5)\n")
            summary_path = tmp_path / "summary.json"
            log_path = tmp_path / "train.log"

            result = run_train_subprocess(
                command=(sys.executable, str(script_path)),
                cwd=tmp_path,
                env=os.environ.copy(),
                timeout_seconds=0.2,
                summary_path=summary_path,
                log_path=log_path,
            )

            self.assertEqual(result.status, "timeout")
            self.assertIsNone(result.returncode)
            self.assertTrue(log_path.exists())

    def test_estimate_min_time_budget_seconds_uses_history_floor(self) -> None:
        estimated = estimate_min_time_budget_seconds(
            runtime_profile="smoke",
            accelerator="cpu",
            skip_train=False,
            train_scenario_count=1,
            eval_scenario_count=1,
            eval_episodes=1,
            max_new_tokens=4,
            historical_durations=[12.0, 18.0],
        )
        self.assertGreaterEqual(estimated, 18)

    def test_build_run_overrides_maps_aliases_and_applies_runtime_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoResearchConfig(
                time_budget=1,
                max_experiments=1,
                output_dir=tmp,
                search_space_name="quick",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={},
                base_run_config={"model_name": "test-model"},
                device_info=self._device_info(),
                runtime_overrides={"num_generations": 2, "lora_r": 4},
            )
            overrides = runner._build_run_overrides(
                experiment_id="exp_0001",
                params={
                    "max_completion_length": 16,
                    "empathy_weight": 0.2,
                    "professionalism_weight": 0.3,
                    "num_generations": 12,
                    "lora_r": 32,
                },
                skip_train=False,
                output_dir=Path(tmp) / "run_outputs",
            )

            self.assertEqual(overrides["max_new_tokens"], 16)
            self.assertNotIn("max_completion_length", overrides)
            self.assertEqual(
                overrides["custom_reward_weights"],
                {"empathy": 0.2, "professionalism": 0.3},
            )
            self.assertEqual(overrides["num_generations"], 2)
            self.assertEqual(overrides["lora_r"], 4)

    def test_subprocess_runner_fails_fast_for_undersized_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "should_not_run.py"
            script_path.write_text("raise SystemExit(99)\n")

            config = AutoResearchConfig(
                time_budget=2,
                max_experiments=1,
                output_dir=tmp,
                search_space_name="quick",
                proposer="perturbation",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={
                    "runtime_profile": "smoke",
                    "train_scenarios": [{"topic": "t"}],
                    "eval_scenarios": [{"topic": "e"}],
                    "max_new_tokens": 4,
                },
                device_info=self._device_info(),
                proposer=FixedProposer([]),
                train_command=(sys.executable, str(script_path)),
            )

            with patch.dict(
                'os.environ',
                {
                    'AUTORESEARCH_SIGNING_KEY': 'unit-test-secret',
                    'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                    'AUTORESEARCH_VERIFY_KEY': 'unit-test-secret',
                },
                clear=False,
            ):
                tracker = runner.run()

            self.assertEqual(tracker.num_experiments, 1)
            self.assertEqual(tracker.records[0].status, "crash")
            self.assertEqual(tracker.records[0].params["runtime_profile"], "smoke")
            self.assertTrue(tracker.records[0].params["skip_train"])
            self.assertIn("below recommended minimum", tracker.records[0].description)
            self.assertTrue((tmp_path / "runs" / "baseline" / "config.json").exists())
            self.assertTrue((tmp_path / "preflight.json").exists())
            preflight = json.loads((tmp_path / "preflight.json").read_text())
            self.assertEqual(preflight["configured_time_budget_seconds"], 2)
            self.assertFalse(preflight["allow_undersized_budget"])
            self.assertGreaterEqual(preflight["recommended_min_budget_seconds"]["baseline"], 1)
            self.assertIn("budget_recommendation", preflight)
            self.assertIn("heuristic_seconds", preflight["budget_recommendation"]["baseline"])
            self.assertTrue(preflight["attestation"]["signatures_required"])
            self.assertTrue(preflight["attestation"]["signing_key_available"])
            self.assertTrue((tmp_path / "audit_report.json").exists())
            self.assertTrue((tmp_path / "attestation_summary.json").exists())

    def test_runner_requires_signing_key_when_signature_verification_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=1,
                output_dir=tmp,
                search_space_name="quick",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                allow_undersized_budget=True,
            )

            with patch.dict('os.environ', {}, clear=True):
                os.environ['AUTORESEARCH_VERIFY_KEY'] = 'unit-test-secret'
                with self.assertRaises(RuntimeError):
                    runner.run()

            preflight = json.loads((Path(tmp) / "preflight.json").read_text())
            self.assertTrue(preflight["attestation"]["signatures_required"])
            self.assertFalse(preflight["attestation"]["signing_key_available"])

    def test_runner_rejects_invalid_existing_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = ExperimentTracker(Path(tmp), objective_metric="eval_reward", direction="maximize")
            tracker.record(ExperimentRecord(
                experiment_id="baseline",
                params={"learning_rate": 5e-6},
                metrics={"eval_reward": 0.5},
                objective_value=0.5,
                training_time=0.0,
                status="keep",
                description="baseline",
                checkpoint_path=str(Path(tmp) / "runs" / "baseline"),
                provenance_path=None,
            ))

            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=2,
                output_dir=tmp,
                search_space_name="quick",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                allow_undersized_budget=True,
            )

            with self.assertRaises(RuntimeError):
                runner.run()

            self.assertTrue((Path(tmp) / "preflight.json").exists())

    def test_runner_rejects_existing_history_with_runtime_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "results"
            script_one = tmp_path / "runtime_one.py"
            script_two = tmp_path / "runtime_two.py"
            script_template = textwrap.dedent(
                """
                import json
                import os
                from pathlib import Path

                config = json.loads(Path(os.environ["AUTORESEARCH_RUN_CONFIG"]).read_text())
                summary_path = Path(os.environ["AUTORESEARCH_RUN_SUMMARY"])

                summary = {
                    "status": "ok",
                    "total_seconds": 0.1,
                    "phase_timings": {
                        "startup_seconds": 0.02,
                        "training_seconds": 0.0,
                        "evaluation_seconds": 0.05,
                    },
                    "metrics": {"eval_reward": 0.5, "selection_score": 0.5},
                    "config": config,
                }
                model_dir = summary_path.parent / "outputs" / "final_model"
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "weights.bin").write_bytes(b"weights")
                summary_path.write_text(json.dumps(summary))
                """
            ).strip() + "\n"
            script_one.write_text(script_template)
            script_two.write_text(script_template)

            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=1,
                output_dir=str(output_dir),
                search_space_name="quick",
            )
            first_runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                train_command=(sys.executable, str(script_one)),
                allow_undersized_budget=True,
            )

            tracker = first_runner.run()
            self.assertEqual([record.status for record in tracker.records], ["keep"])

            second_runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                train_command=(sys.executable, str(script_two)),
                allow_undersized_budget=True,
            )

            with self.assertRaises(RuntimeError) as exc_info:
                second_runner.run()

            self.assertIn("different runtime environment", str(exc_info.exception))
            self.assertIn("mismatched runtime fingerprint", str(exc_info.exception))
            self.assertTrue((output_dir / "preflight.json").exists())

    def test_improved_result_without_proof_is_downgraded_to_discard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "dummy_train.py"
            script_path.write_text(
                textwrap.dedent(
                    '''
                    import json
                    import os
                    from pathlib import Path

                    config = json.loads(Path(os.environ["AUTORESEARCH_RUN_CONFIG"]).read_text())
                    summary_path = Path(os.environ["AUTORESEARCH_RUN_SUMMARY"])

                    reward = 0.5 if config.get("skip_train") else 0.7
                    summary = {
                        "status": "ok",
                        "total_seconds": 0.1,
                        "phase_timings": {
                            "startup_seconds": 0.02,
                            "training_seconds": 0.03 if not config.get("skip_train") else 0.0,
                            "evaluation_seconds": 0.05,
                        },
                        "metrics": {"eval_reward": reward},
                        "config": config,
                    }
                    model_dir = summary_path.parent / "outputs" / "final_model"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    (model_dir / "weights.bin").write_bytes(b"weights")
                    summary_path.write_text(json.dumps(summary))
                    '''
                ).strip()
                + "\n"
            )

            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=2,
                output_dir=tmp,
                search_space_name="quick",
                proposer="perturbation",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                proposer=FixedProposer([({"learning_rate": 2e-4}, "better config")]),
                train_command=(sys.executable, str(script_path)),
                runtime_overrides={"num_generations": 2},
                allow_undersized_budget=True,
            )

            with patch.object(runner, '_write_stark_proof_artifact', return_value=None):
                with patch.dict(
                    'os.environ',
                    {
                        'AUTORESEARCH_SIGNING_KEY': 'unit-test-secret',
                        'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                        'AUTORESEARCH_VERIFY_KEY': 'unit-test-secret',
                    },
                    clear=False,
                ):
                    tracker = runner.run()

            self.assertEqual([record.status for record in tracker.records], ["keep", "discard"])
            self.assertEqual(tracker.best_record.experiment_id, "baseline")
            self.assertIsNone(tracker.records[1].proof_path)
            self.assertIn("attestation rejected", tracker.records[1].description)
            self.assertIn("Required proof missing", tracker.records[1].description)

            provenance = json.loads(Path(tracker.records[1].provenance_path).read_text())
            self.assertEqual(provenance["record_status"], "discard")

            audit_report = json.loads((tmp_path / "audit_report.json").read_text())
            self.assertTrue(audit_report["ok"])
            self.assertEqual(audit_report["summary"]["proofs_missing_required"], 0)
            self.assertFalse((tmp_path / "best_proof.json").exists())

    def test_runner_cleans_stale_best_proof_artifacts_when_best_has_no_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "best_proof.json").write_text('{"stale": true}')
            (tmp_path / "best_proof.bin").write_bytes(b"stale")
            script_path = tmp_path / "dummy_train.py"
            script_path.write_text(
                textwrap.dedent(
                    '''
                    import json
                    import os
                    from pathlib import Path

                    config = json.loads(Path(os.environ["AUTORESEARCH_RUN_CONFIG"]).read_text())
                    summary_path = Path(os.environ["AUTORESEARCH_RUN_SUMMARY"])

                    reward = 0.5 if config.get("skip_train") else 0.7
                    summary = {
                        "status": "ok",
                        "total_seconds": 0.1,
                        "phase_timings": {
                            "startup_seconds": 0.02,
                            "training_seconds": 0.03 if not config.get("skip_train") else 0.0,
                            "evaluation_seconds": 0.05,
                        },
                        "metrics": {"eval_reward": reward},
                        "config": config,
                    }
                    model_dir = summary_path.parent / "outputs" / "final_model"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    (model_dir / "weights.bin").write_bytes(b"weights")
                    summary_path.write_text(json.dumps(summary))
                    '''
                ).strip()
                + "\n"
            )

            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=1,
                output_dir=tmp,
                search_space_name="quick",
                proposer="perturbation",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                train_command=(sys.executable, str(script_path)),
                runtime_overrides={"num_generations": 2},
                allow_undersized_budget=True,
            )

            with patch.dict(
                'os.environ',
                {
                    'AUTORESEARCH_SIGNING_KEY': 'unit-test-secret',
                    'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                    'AUTORESEARCH_VERIFY_KEY': 'unit-test-secret',
                },
                clear=False,
            ):
                tracker = runner.run()

            self.assertEqual([record.status for record in tracker.records], ["keep"])
            self.assertFalse((tmp_path / "best_proof.json").exists())
            self.assertFalse((tmp_path / "best_proof.bin").exists())

            audit_report = json.loads((tmp_path / "audit_report.json").read_text())
            self.assertTrue(audit_report["ok"])
            self.assertEqual(audit_report["summary"]["best_artifacts_ok"], 5)
            self.assertEqual(audit_report["summary"]["best_artifacts_not_required"], 2)

    def test_subprocess_runner_records_keep_and_discard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "dummy_train.py"
            script_path.write_text(
                textwrap.dedent(
                    '''
                    import json
                    import os
                    from pathlib import Path

                    config = json.loads(Path(os.environ["AUTORESEARCH_RUN_CONFIG"]).read_text())
                    summary_path = Path(os.environ["AUTORESEARCH_RUN_SUMMARY"])

                    if config.get("skip_train"):
                        reward = 0.5
                    else:
                        reward = 0.7 if float(config.get("learning_rate", 0.0)) > 1e-4 else 0.4

                    summary = {
                        "status": "ok",
                        "total_seconds": 0.1,
                        "phase_timings": {
                            "startup_seconds": 0.02,
                            "training_seconds": 0.03 if not config.get("skip_train") else 0.0,
                            "evaluation_seconds": 0.05,
                        },
                        "metrics": {"eval_reward": reward},
                        "config": config,
                    }
                    model_dir = summary_path.parent / "outputs" / "final_model"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    (model_dir / "weights.bin").write_bytes(b"weights")
                    summary_path.write_text(json.dumps(summary))
                    '''
                ).strip()
                + "\n"
            )

            proposer = FixedProposer(
                [
                    ({"learning_rate": 2e-4}, "better config"),
                    ({"learning_rate": 1e-6}, "worse config"),
                ]
            )
            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=3,
                output_dir=tmp,
                search_space_name="quick",
                proposer="perturbation",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                proposer=proposer,
                train_command=(sys.executable, str(script_path)),
                runtime_overrides={"num_generations": 2},
                allow_undersized_budget=True,
            )

            with patch.dict(
                'os.environ',
                {
                    'AUTORESEARCH_SIGNING_KEY': 'unit-test-secret',
                    'AUTORESEARCH_SIGNING_KEY_ID': 'unit-test',
                    'AUTORESEARCH_VERIFY_KEY': 'unit-test-secret',
                },
                clear=False,
            ):
                tracker = runner.run()

            self.assertEqual(tracker.num_experiments, 3)
            self.assertEqual([record.status for record in tracker.records], ["keep", "keep", "discard"])
            self.assertAlmostEqual(tracker.best_value or 0.0, 0.7)
            self.assertEqual(tracker.best_record.params["num_generations"], 2)
            self.assertEqual(tracker.best_record.params["runtime_profile"], "smoke")
            self.assertIsNotNone(tracker.best_record.provenance_path)

            provenance = json.loads(Path(tracker.best_record.provenance_path).read_text())
            self.assertEqual(provenance["record_status"], "keep")
            self.assertEqual(provenance["process_status"], "ok")
            self.assertIsNotNone(provenance["config_sha256"])
            self.assertIsNotNone(provenance["summary_config_sha256"])
            self.assertEqual(provenance["signature"]["algorithm"], "hmac-sha256")

            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'unit-test-secret'}, clear=False):
                verification = verify_provenance_envelope(tracker.best_record.provenance_path, require_signature=True)
            self.assertTrue(verification["ok"])
            self.assertTrue(verification["signature_valid"])
            self.assertIsNotNone(tracker.best_record.proof_path)

            proof_payload = json.loads(Path(tracker.best_record.proof_path).read_text())
            proof_binary_path = Path(tracker.best_record.proof_path).parent / proof_payload["proof_path"]
            self.assertTrue(proof_binary_path.exists())
            self.assertEqual(proof_payload["experiment_id"], tracker.best_record.experiment_id)
            self.assertEqual(proof_payload["envelope_sha256"], provenance["envelope_sha256"])

            proof = MetricImprovementProof(
                experiment_id=proof_payload["experiment_id"],
                agent_id=proof_payload["agent_id"],
                best_reward_scaled=proof_payload["best_reward_scaled"],
                sequence_number=proof_payload["sequence_number"],
                proof_bytes=proof_binary_path.read_bytes(),
                proof_hash=proof_payload["proof_hash"],
                witness_commitment=list(proof_payload["witness_commitment"]),
                witness_commitment_hex=proof_payload["witness_commitment_hex"],
                policy_hash=proof_payload["policy_hash"],
                proving_time_ms=proof_payload["proving_time_ms"],
                proof_size=proof_payload["proof_size"],
                timestamp=proof_payload["timestamp"],
                envelope_sha256=proof_payload.get("envelope_sha256"),
                amount_binding_hash=proof_payload.get("amount_binding_hash"),
            )
            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'unit-test-secret'}, clear=False):
                proof_verification = verify_improvement(
                    proof,
                    claimed_best_reward=tracker.records[0].objective_value,
                    provenance_path=tracker.best_record.provenance_path,
                    require_provenance_signature=True,
                )
            self.assertTrue(proof_verification.valid, proof_verification.message)

            best_config = json.loads((tmp_path / "best_config.json").read_text())
            self.assertEqual(best_config["num_generations"], 2)
            self.assertEqual(best_config["runtime_profile"], "smoke")
            self.assertTrue((tmp_path / "best_summary.json").exists())
            self.assertTrue((tmp_path / "best_provenance.json").exists())
            self.assertTrue((tmp_path / "best_runtime_environment.json").exists())
            self.assertEqual(
                (tmp_path / "best_runtime_environment.json").read_text(),
                (Path(tracker.best_record.checkpoint_path) / "runtime_environment.json").read_text(),
            )
            self.assertTrue((tmp_path / "best_proof.json").exists())
            self.assertTrue((tmp_path / "best_proof.bin").exists())
            best_bundle = tmp_path / "best_bundle"
            self.assertTrue((best_bundle / "provenance.json").exists())
            self.assertTrue((best_bundle / "proof.json").exists())
            self.assertTrue((best_bundle / "proof.bin").exists())
            self.assertTrue((best_bundle / "outputs" / "final_model").exists())
            self.assertEqual(
                (best_bundle / "runtime_environment.json").read_text(),
                (Path(tracker.best_record.checkpoint_path) / "runtime_environment.json").read_text(),
            )
            with patch.dict('os.environ', {'AUTORESEARCH_VERIFY_KEY': 'unit-test-secret'}, clear=False):
                bundle_provenance = verify_provenance_envelope(best_bundle / "provenance.json", require_signature=True)
                bundle_proof_verification = _verify_proof_artifact(best_bundle / "proof.json", require_signature=True)
            self.assertTrue(bundle_provenance["ok"], bundle_provenance["signature_reason"])
            self.assertTrue(bundle_proof_verification["ok"], bundle_proof_verification["verification_message"])
            self.assertTrue((tmp_path / "audit_report.json").exists())
            self.assertTrue((tmp_path / "attestation_summary.json").exists())

            audit_report = json.loads((tmp_path / "audit_report.json").read_text())
            self.assertTrue(audit_report["ok"])
            self.assertEqual(audit_report["summary"]["provenance_ok"], 3)
            self.assertEqual(audit_report["summary"]["proofs_ok"], 1)

            attestation_summary = json.loads((tmp_path / "attestation_summary.json").read_text())
            self.assertTrue(attestation_summary["ok"])
            self.assertEqual(
                attestation_summary["best_record"]["experiment_id"],
                tracker.best_record.experiment_id,
            )
            self.assertEqual(attestation_summary["best_record"]["audit_provenance_status"], "ok")
            self.assertEqual(attestation_summary["best_record"]["audit_proof_status"], "ok")
            self.assertTrue(attestation_summary["best_record"]["audit_best_artifacts_ok"])
            self.assertEqual(attestation_summary["best_record"]["audit_best_artifact_issue_count"], 0)
            self.assertEqual(attestation_summary["best_record"]["audit_best_artifact_issues"], [])

            resume_runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke", "max_new_tokens": 4},
                device_info=self._device_info(),
                proposer=proposer,
                train_command=(sys.executable, str(script_path)),
                runtime_overrides={"num_generations": 2},
                allow_undersized_budget=True,
            )
            preflight = json.loads(resume_runner._write_run_preflight().read_text())
            self.assertGreaterEqual(
                preflight["budget_recommendation"]["experiment"]["historical_phase_samples"],
                1,
            )
            self.assertIn(
                "startup_seconds",
                preflight["budget_recommendation"]["experiment"]["phase_budget_seconds"],
            )


    def test_runner_uses_selection_score_for_keep_discard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "selection_metric_train.py"
            script_path.write_text(
                textwrap.dedent(
                    '''
                    import json
                    import os
                    from pathlib import Path

                    config = json.loads(Path(os.environ["AUTORESEARCH_RUN_CONFIG"]).read_text())
                    summary_path = Path(os.environ["AUTORESEARCH_RUN_SUMMARY"])

                    if config.get("skip_train"):
                        metrics = {"eval_reward": 0.5, "selection_score": 0.6}
                    else:
                        metrics = {"eval_reward": 0.9, "selection_score": 0.4}

                    summary = {
                        "status": "ok",
                        "total_seconds": 0.1,
                        "phase_timings": {
                            "startup_seconds": 0.02,
                            "training_seconds": 0.03 if not config.get("skip_train") else 0.0,
                            "evaluation_seconds": 0.05,
                        },
                        "metrics": metrics,
                        "config": config,
                    }
                    summary_path.write_text(json.dumps(summary))
                    '''
                ).strip()
                + "\n"
            )

            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=2,
                output_dir=tmp,
                search_space_name="quick",
                proposer="perturbation",
                objective_metric="selection_score",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                proposer=FixedProposer([({"learning_rate": 2e-4}, "higher reward, worse selection")]),
                train_command=(sys.executable, str(script_path)),
                runtime_overrides={"num_generations": 2},
                allow_undersized_budget=True,
            )

            tracker = runner.run()

            self.assertEqual([record.status for record in tracker.records], ["keep", "discard"])
            self.assertAlmostEqual(tracker.best_value or 0.0, 0.6)
            self.assertEqual(tracker.best_record.experiment_id, "baseline")

    def test_runner_discards_noisy_selection_score_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "selection_gate_train.py"
            script_path.write_text(
                textwrap.dedent(
                    """
                    import json
                    import os
                    from pathlib import Path

                    config = json.loads(Path(os.environ["AUTORESEARCH_RUN_CONFIG"]).read_text())
                    summary_path = Path(os.environ["AUTORESEARCH_RUN_SUMMARY"])

                    if config.get("skip_train"):
                        metrics = {
                            "eval_reward": 0.5,
                            "selection_score": 0.6,
                            "selection_score_bootstrap_std": 0.01,
                        }
                    else:
                        metrics = {
                            "eval_reward": 0.55,
                            "selection_score": 0.65,
                            "selection_score_bootstrap_std": 0.06,
                        }

                    summary = {
                        "status": "ok",
                        "total_seconds": 0.1,
                        "phase_timings": {
                            "startup_seconds": 0.02,
                            "training_seconds": 0.03 if not config.get("skip_train") else 0.0,
                            "evaluation_seconds": 0.05,
                        },
                        "metrics": metrics,
                        "config": config,
                    }
                    summary_path.write_text(json.dumps(summary))
                    """
                ).strip()
                + "\n"
            )

            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=2,
                output_dir=tmp,
                search_space_name="quick",
                proposer="perturbation",
                objective_metric="selection_score",
                selection_promotion_zscore=1.0,
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                proposer=FixedProposer([({"learning_rate": 2e-4}, "noisy selection improvement")]),
                train_command=(sys.executable, str(script_path)),
                runtime_overrides={"num_generations": 2},
                allow_undersized_budget=True,
            )

            tracker = runner.run()

            self.assertEqual([record.status for record in tracker.records], ["keep", "discard"])
            self.assertAlmostEqual(tracker.best_value or 0.0, 0.6)
            self.assertEqual(tracker.best_record.experiment_id, "baseline")
            self.assertIn("selection gate", tracker.records[1].description)
            self.assertIn("does not beat current best", tracker.records[1].description)

            provenance = json.loads(Path(tracker.records[1].provenance_path).read_text())
            self.assertEqual(provenance["record_status"], "discard")

    def test_runner_rejects_proposals_that_mutate_locked_judge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script_path = tmp_path / "writes_marker.py"
            marker_path = tmp_path / "executed.txt"
            script_path.write_text(
                textwrap.dedent(
                    f'''
                    import json
                    import os
                    from pathlib import Path

                    config = json.loads(Path(os.environ["AUTORESEARCH_RUN_CONFIG"]).read_text())
                    summary_path = Path(os.environ["AUTORESEARCH_RUN_SUMMARY"])
                    marker_path = Path(r"{marker_path}")
                    existing = marker_path.read_text() if marker_path.exists() else ""
                    marker_path.write_text(existing + config.get("experiment_id", "unknown") + "\\n")

                    summary = {{
                        "status": "ok",
                        "total_seconds": 0.1,
                        "phase_timings": {{
                            "startup_seconds": 0.02,
                            "training_seconds": 0.0 if config.get("skip_train") else 0.03,
                            "evaluation_seconds": 0.05,
                        }},
                        "metrics": {{"eval_reward": 0.5, "selection_score": 0.5}},
                        "config": config,
                    }}
                    summary_path.write_text(json.dumps(summary))
                    '''
                ).strip()
                + "\n"
            )

            config = AutoResearchConfig(
                time_budget=5,
                max_experiments=2,
                output_dir=tmp,
                search_space_name="quick",
                proposer="perturbation",
                objective_metric="selection_score",
            )
            runner = SubprocessAutoResearchRunner(
                config=config,
                baseline_params={"learning_rate": 5e-6},
                base_run_config={"runtime_profile": "smoke"},
                device_info=self._device_info(),
                proposer=FixedProposer([({"reward_domain": "sales", "eval_episodes": 1}, "mutate judge")]),
                train_command=(sys.executable, str(script_path)),
                runtime_overrides={"num_generations": 2},
                allow_undersized_budget=True,
            )

            tracker = runner.run()

            self.assertEqual([record.status for record in tracker.records], ["keep", "crash"])
            self.assertIn("locked reward or evaluation parameters", tracker.records[1].description)
            self.assertEqual(marker_path.read_text().strip().splitlines(), ["baseline"])


if __name__ == "__main__":
    unittest.main()
