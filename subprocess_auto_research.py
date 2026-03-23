"""Subprocess-backed autonomous research loop for hard-bounded experiments."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
import fcntl
from pathlib import Path
from typing import Any

from attestation_audit import build_audit_report, evaluate_record_attestation, rebuild_best_artifacts
from experiment_runtime import (
    DeviceInfo,
    build_time_budget_recommendation,
    collect_file_digests,
    detect_git_metadata,
    load_hmac_secret,
    sha256_file,
    sha256_json,
    sign_json_payload_if_configured,
)
from stateset_agents.training.auto_research.config import AutoResearchConfig
from stateset_agents.training.auto_research.experiment_tracker import (
    ExperimentRecord,
    ExperimentTracker,
)
from stateset_agents.training.auto_research.proposer import (
    BayesianProposer,
    ExperimentProposer,
    create_proposer,
)
from stateset_agents.training.auto_research.search_spaces import (
    AUTO_RESEARCH_SPACES,
    validate_params_against_space,
)

logger = logging.getLogger("subprocess_auto_research")
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_COMMAND = ("uv", "run", "train.py")
FORBIDDEN_SEARCH_SPACE_DIMENSIONS = {
    "reward_domain",
    "custom_reward_weights",
    "empathy_weight",
    "professionalism_weight",
    "action_oriented_weight",
    "reasoning_weight",
    "length_weight",
}
FORBIDDEN_PROPOSAL_KEYS = FORBIDDEN_SEARCH_SPACE_DIMENSIONS | {
    "train_scenarios",
    "eval_scenarios",
    "train_scenario_limit",
    "eval_scenario_limit",
    "eval_episodes",
}
RUNTIME_SNAPSHOT_PROBE = """
import json
import platform
import sys
from importlib import metadata as importlib_metadata

packages = []
for dist in importlib_metadata.distributions():
    try:
        name = dist.metadata.get("Name") or getattr(dist, "name", None)
        version = dist.version
    except Exception:
        continue
    if not name or not version:
        continue
    packages.append([str(name), str(version)])

packages.sort(key=lambda item: item[0].lower())
print(
    json.dumps(
        {
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "sys_prefix": sys.prefix,
            "base_prefix": getattr(sys, "base_prefix", None),
            "packages": packages,
        },
        sort_keys=True,
    )
)
""".strip()


def _search_space_dimension_names(search_space: Any) -> set[str]:
    return {str(dim.name) for dim in getattr(search_space, "dimensions", [])}


def _forbidden_proposal_params(params: dict[str, Any]) -> list[str]:
    disallowed = sorted(key for key in params if key in FORBIDDEN_PROPOSAL_KEYS)
    custom_weights = params.get("custom_reward_weights")
    if isinstance(custom_weights, dict) and custom_weights:
        if "custom_reward_weights" not in disallowed:
            disallowed.append("custom_reward_weights")
    elif custom_weights is not None and "custom_reward_weights" not in disallowed:
        disallowed.append("custom_reward_weights")
    return disallowed


@dataclass(frozen=True)
class ResolvedTrainInvocation:
    command: tuple[str, ...]
    runtime: dict[str, Any]


@dataclass(frozen=True)
class ExperimentExecutionContext:
    requested_mode: str
    mode: str
    cwd: Path
    repo_root: Path
    source_repo_root: Path
    worktree_path: Path | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "mode": self.mode,
            "cwd": str(self.cwd),
            "repo_root": str(self.repo_root),
            "source_repo_root": str(self.source_repo_root),
            "worktree_path": None if self.worktree_path is None else str(self.worktree_path),
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class ExperimentProcessResult:
    status: str
    returncode: int | None
    elapsed_seconds: float
    summary: dict[str, Any] | None
    summary_path: Path
    log_path: Path
    error: str | None = None
    execution: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Summary at {path} must be a JSON object")
    return payload


def _tail_log(path: Path, max_lines: int = 10) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    tail = lines[-max_lines:]
    return " | ".join(line.strip() for line in tail if line.strip())


def _resolve_search_space(name: str | None) -> Any:
    if name and name in AUTO_RESEARCH_SPACES:
        return AUTO_RESEARCH_SPACES[name]()

    from stateset_agents.training.hpo.search_spaces import (
        create_grpo_search_space,
        get_search_space,
    )

    if name:
        try:
            return get_search_space(name)
        except (KeyError, ValueError):
            logger.warning(
                "Search space %r not found, falling back to 'grpo'",
                name,
            )
    return create_grpo_search_space()


def run_train_subprocess(
    *,
    command: tuple[str, ...],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    summary_path: Path,
    log_path: Path,
) -> ExperimentProcessResult:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        summary_path.unlink()

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=max(0.1, timeout_seconds))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            elapsed = time.time() - start
            summary = _load_summary(summary_path)
            return ExperimentProcessResult(
                status="timeout",
                returncode=None,
                elapsed_seconds=elapsed,
                summary=summary,
                summary_path=summary_path,
                log_path=log_path,
                error=f"hard timeout after {timeout_seconds:.1f}s",
            )

    elapsed = time.time() - start
    summary = _load_summary(summary_path)

    if returncode == 0 and summary and summary.get("status") == "ok":
        return ExperimentProcessResult(
            status="ok",
            returncode=returncode,
            elapsed_seconds=elapsed,
            summary=summary,
            summary_path=summary_path,
            log_path=log_path,
        )

    error = None
    status = "error"
    if summary:
        summary_status = str(summary.get("status", "error"))
        if summary_status == "timeout":
            status = "timeout"
        error = str(summary.get("error") or f"train.py exited with code {returncode}")
    else:
        error = f"train.py exited with code {returncode} without writing a summary"

    log_tail = _tail_log(log_path)
    if log_tail:
        error = f"{error} | {log_tail}"

    return ExperimentProcessResult(
        status=status,
        returncode=returncode,
        elapsed_seconds=elapsed,
        summary=summary,
        summary_path=summary_path,
        log_path=log_path,
        error=error,
    )


class SubprocessAutoResearchRunner:
    """Autonomous research loop that runs each experiment in a subprocess."""

    def __init__(
        self,
        *,
        config: AutoResearchConfig,
        baseline_params: dict[str, Any],
        base_run_config: dict[str, Any],
        device_info: DeviceInfo,
        proposer: ExperimentProposer | None = None,
        train_command: tuple[str, ...] = DEFAULT_TRAIN_COMMAND,
        runtime_overrides: dict[str, Any] | None = None,
        allow_undersized_budget: bool = False,
        source_repo_root: str | Path = REPO_ROOT,
    ):
        self.config = config
        self.baseline_params = dict(baseline_params)
        self.base_run_config = dict(base_run_config)
        self.device_info = device_info
        self.train_command = train_command
        self.runtime_overrides = dict(runtime_overrides or {})
        self.allow_undersized_budget = allow_undersized_budget
        self.source_repo_root = Path(source_repo_root).expanduser().resolve()
        self.output_path = Path(config.output_dir).expanduser().resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._worktree_root = (
            Path(tempfile.gettempdir())
            / "autoresearch-worktrees"
            / sha256_json({"repo": str(self.source_repo_root), "output": str(self.output_path)})[:16]
        )

        self.search_space = _resolve_search_space(config.search_space_name)
        self._validate_search_space()
        self.proposer = proposer or create_proposer(
            strategy=config.proposer,
            search_space=self.search_space,
            direction=config.direction,
        )

        if (self.output_path / "experiments.jsonl").exists():
            self.tracker = ExperimentTracker.load(
                self.output_path,
                objective_metric=config.objective_metric,
                direction=config.direction,
            )
        else:
            self.tracker = ExperimentTracker(
                self.output_path,
                objective_metric=config.objective_metric,
                direction=config.direction,
            )

        self._loop_start_time = 0.0
        self._stark_prover: Any | None = None
        self._stark_prover_import_failed = False
        self._lock_path = self.output_path / ".autoresearch.lock"
        self._lock_fd: int | None = None
        self._runtime_snapshot_cache: dict[str, dict[str, Any]] = {}

        for warning in validate_params_against_space(self.baseline_params, self.search_space):
            logger.warning("Baseline param out of search space: %s", warning)

    def _validate_search_space(self) -> None:
        forbidden = sorted(_search_space_dimension_names(self.search_space) & FORBIDDEN_SEARCH_SPACE_DIMENSIONS)
        if not forbidden:
            return
        raise ValueError(
            "Search space mutates locked reward or evaluation settings: "
            + ", ".join(forbidden)
            + ". Use a model/training search space instead."
        )

    def _validate_proposed_params(self, params: dict[str, Any]) -> str | None:
        forbidden = _forbidden_proposal_params(params)
        if not forbidden:
            return None
        return (
            "experiment proposals may not modify locked reward or evaluation parameters: "
            + ", ".join(forbidden)
        )

    def _requested_experiment_isolation(self) -> str:
        mode = str(getattr(self.config, "experiment_isolation", "worktree") or "worktree").lower()
        if mode not in {"shared", "worktree"}:
            return "worktree"
        return mode

    def _requested_runtime_environment(self) -> str:
        mode = str(getattr(self.config, "runtime_environment", "venv") or "venv").lower()
        if mode not in {"uv", "venv"}:
            return "venv"
        return mode

    def _runtime_manifest(self) -> dict[str, Any]:
        pyproject_path = self.source_repo_root / "pyproject.toml"
        uv_lock_path = self.source_repo_root / "uv.lock"
        return {
            "pyproject_path": str(pyproject_path) if pyproject_path.exists() else None,
            "pyproject_sha256": self._hash_if_exists(pyproject_path),
            "uv_lock_path": str(uv_lock_path) if uv_lock_path.exists() else None,
            "uv_lock_sha256": self._hash_if_exists(uv_lock_path),
        }

    def _runtime_snapshot_probe_command(
        self,
        *,
        execution: ExperimentExecutionContext,
        runtime: dict[str, Any],
        command: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        mode = str(runtime.get("mode") or "")
        if mode == "uv":
            if shutil.which("uv") is None:
                return None
            return ("uv", "run", "python", "-c", RUNTIME_SNAPSHOT_PROBE)

        python_executable = str(runtime.get("python_executable") or "")
        if python_executable:
            executable_name = Path(python_executable).name.lower()
            if "python" in executable_name:
                return (python_executable, "-c", RUNTIME_SNAPSHOT_PROBE)

        if command:
            executable_name = Path(str(command[0])).name.lower()
            if "python" in executable_name:
                return (str(command[0]), "-c", RUNTIME_SNAPSHOT_PROBE)

        return None

    def _runtime_environment_snapshot(
        self,
        *,
        execution: ExperimentExecutionContext,
        runtime: dict[str, Any],
        command: tuple[str, ...],
    ) -> dict[str, Any]:
        probe_command = self._runtime_snapshot_probe_command(
            execution=execution,
            runtime=runtime,
            command=command,
        )
        if probe_command is None:
            return {
                "status": "unsupported",
                "reason": "runtime is not probeable via python",
            }

        cache_key = sha256_json({"cwd": str(execution.cwd), "command": list(probe_command)})
        cached = self._runtime_snapshot_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        try:
            probe = subprocess.run(
                probe_command,
                cwd=str(execution.cwd),
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            snapshot = {
                "status": "unavailable",
                "reason": "environment probe timed out",
            }
            self._runtime_snapshot_cache[cache_key] = dict(snapshot)
            return dict(snapshot)
        except OSError as exc:
            snapshot = {
                "status": "unavailable",
                "reason": str(exc),
            }
            self._runtime_snapshot_cache[cache_key] = dict(snapshot)
            return dict(snapshot)

        if probe.returncode != 0:
            stderr = (probe.stderr or probe.stdout or "").strip()
            snapshot = {
                "status": "unavailable",
                "reason": f"environment probe exited with code {probe.returncode}",
            }
            if stderr:
                snapshot["stderr"] = stderr[:200]
            self._runtime_snapshot_cache[cache_key] = dict(snapshot)
            return dict(snapshot)

        try:
            payload = json.loads(probe.stdout)
        except json.JSONDecodeError:
            snapshot = {
                "status": "unavailable",
                "reason": "environment probe returned invalid JSON",
            }
            self._runtime_snapshot_cache[cache_key] = dict(snapshot)
            return dict(snapshot)

        raw_packages = payload.get("packages")
        packages: list[list[str]] = []
        if isinstance(raw_packages, list):
            for item in raw_packages:
                if not isinstance(item, list) or len(item) != 2:
                    continue
                name, version = item
                packages.append([str(name), str(version)])

        snapshot = {
            "status": "ok",
            "python_executable": payload.get("python_executable"),
            "python_implementation": payload.get("python_implementation"),
            "python_version": payload.get("python_version"),
            "sys_prefix": payload.get("sys_prefix"),
            "base_prefix": payload.get("base_prefix"),
            "package_count": len(packages),
            "packages_sha256": sha256_json(packages),
            "packages": packages,
        }
        self._runtime_snapshot_cache[cache_key] = dict(snapshot)
        return dict(snapshot)

    @staticmethod
    def _runtime_fingerprint_payload(runtime: dict[str, Any]) -> dict[str, Any]:
        manifest = dict(runtime.get("manifest") or {})
        environment_snapshot = dict(runtime.get("environment_snapshot") or {})
        return {
            "requested_environment": runtime.get("requested_environment"),
            "mode": runtime.get("mode"),
            "command": list(runtime.get("command") or []),
            "python_executable": runtime.get("python_executable"),
            "venv_root": runtime.get("venv_root"),
            "manifest": {
                "pyproject_sha256": manifest.get("pyproject_sha256"),
                "uv_lock_sha256": manifest.get("uv_lock_sha256"),
            },
            "environment_snapshot": {
                "status": environment_snapshot.get("status"),
                "python_version": environment_snapshot.get("python_version"),
                "python_implementation": environment_snapshot.get("python_implementation"),
                "package_count": environment_snapshot.get("package_count"),
                "packages_sha256": environment_snapshot.get("packages_sha256"),
            },
        }

    def _finalize_runtime_metadata(self, runtime: dict[str, Any]) -> dict[str, Any]:
        finalized = dict(runtime)
        finalized["fingerprint"] = sha256_json(self._runtime_fingerprint_payload(finalized))
        return finalized

    @staticmethod
    def _runtime_public_payload(runtime: dict[str, Any]) -> dict[str, Any]:
        payload = dict(runtime)
        environment_snapshot = dict(payload.get("environment_snapshot") or {})
        if environment_snapshot:
            environment_snapshot.pop("packages", None)
            payload["environment_snapshot"] = environment_snapshot
        return payload

    def _write_runtime_artifact(self, artifact_path: Path, runtime: dict[str, Any]) -> Path:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(runtime, indent=2, sort_keys=True))
        return artifact_path

    def _repo_python_executable(self) -> Path | None:
        candidates = [
            self.source_repo_root / ".venv" / "bin" / "python",
            self.source_repo_root / ".venv" / "Scripts" / "python.exe",
        ]
        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
        return None

    def _resolve_train_invocation(
        self,
        execution: ExperimentExecutionContext,
    ) -> ResolvedTrainInvocation:
        requested_environment = self._requested_runtime_environment()
        runtime = {
            "requested_environment": requested_environment,
            "manifest": self._runtime_manifest(),
        }

        if self.train_command != DEFAULT_TRAIN_COMMAND:
            command = tuple(self.train_command)
            runtime.update(
                {
                    "mode": "custom_command",
                    "command": list(command),
                    "python_executable": command[0] if command else None,
                    "venv_root": None,
                    "fallback_reason": None,
                }
            )
            runtime["environment_snapshot"] = self._runtime_environment_snapshot(
                execution=execution,
                runtime=runtime,
                command=command,
            )
            return ResolvedTrainInvocation(command=command, runtime=self._finalize_runtime_metadata(runtime))

        if requested_environment == "venv":
            repo_python = self._repo_python_executable()
            if repo_python is not None:
                command = (str(repo_python), "train.py")
                runtime.update(
                    {
                        "mode": "venv",
                        "command": list(command),
                        "python_executable": str(repo_python),
                        "venv_root": str(repo_python.parent.parent),
                        "fallback_reason": None,
                    }
                )
                runtime["environment_snapshot"] = self._runtime_environment_snapshot(
                    execution=execution,
                    runtime=runtime,
                    command=command,
                )
                return ResolvedTrainInvocation(command=command, runtime=self._finalize_runtime_metadata(runtime))

            runtime["fallback_reason"] = "repo virtualenv not found; falling back to uv"

        command = DEFAULT_TRAIN_COMMAND
        runtime.update(
            {
                "mode": "uv",
                "command": list(command),
                "python_executable": sys.executable if requested_environment == "uv" else None,
                "venv_root": None,
                "fallback_reason": runtime.get("fallback_reason"),
            }
        )
        runtime["environment_snapshot"] = self._runtime_environment_snapshot(
            execution=execution,
            runtime=runtime,
            command=command,
        )
        return ResolvedTrainInvocation(command=command, runtime=self._finalize_runtime_metadata(runtime))

    def _shared_execution_context(
        self,
        *,
        requested_mode: str,
        fallback_reason: str | None = None,
    ) -> ExperimentExecutionContext:
        return ExperimentExecutionContext(
            requested_mode=requested_mode,
            mode="shared",
            cwd=self.source_repo_root,
            repo_root=self.source_repo_root,
            source_repo_root=self.source_repo_root,
            worktree_path=None,
            fallback_reason=fallback_reason,
        )

    def _remove_worktree_path(self, worktree_path: Path) -> None:
        with suppress(FileNotFoundError, subprocess.CalledProcessError):
            subprocess.run(
                ["git", "-C", str(self.source_repo_root), "worktree", "remove", "--force", str(worktree_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        with suppress(FileNotFoundError, subprocess.CalledProcessError):
            subprocess.run(
                ["git", "-C", str(self.source_repo_root), "worktree", "prune"],
                check=True,
                capture_output=True,
                text=True,
            )
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)

    def _prepare_execution_context(self, *, experiment_id: str) -> ExperimentExecutionContext:
        requested_mode = self._requested_experiment_isolation()
        if requested_mode != "worktree":
            return self._shared_execution_context(requested_mode=requested_mode)

        worktree_path = self._worktree_root / experiment_id
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_worktree_path(worktree_path)

        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.source_repo_root),
                    "worktree",
                    "add",
                    "--detach",
                    str(worktree_path),
                    "HEAD",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            reason = None
            if isinstance(exc, subprocess.CalledProcessError):
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                reason = stderr or stdout or str(exc)
            else:
                reason = str(exc)
            logger.warning(
                "Falling back to shared execution for %s because git worktree setup failed: %s",
                experiment_id,
                reason,
            )
            return self._shared_execution_context(
                requested_mode=requested_mode,
                fallback_reason=reason,
            )

        if completed.stdout.strip():
            logger.debug("git worktree add output for %s: %s", experiment_id, completed.stdout.strip())

        return ExperimentExecutionContext(
            requested_mode=requested_mode,
            mode="worktree",
            cwd=worktree_path,
            repo_root=worktree_path,
            source_repo_root=self.source_repo_root,
            worktree_path=worktree_path,
            fallback_reason=None,
        )

    def _cleanup_execution_context(self, result: ExperimentProcessResult) -> None:
        execution = result.execution or {}
        if execution.get("mode") != "worktree":
            return
        worktree_value = execution.get("worktree_path") or execution.get("cwd")
        if not worktree_value:
            return
        self._remove_worktree_path(Path(str(worktree_value)))

    def run(self) -> ExperimentTracker:
        self._acquire_output_lock()
        try:
            self._loop_start_time = time.time()
            next_experiment_num = self._next_experiment_num()

            logger.info("Starting subprocess-backed autonomous research loop")
            logger.info("  Objective: %s (%s)", self.config.objective_metric, self.config.direction)
            logger.info("  Time budget per experiment: %ss", self.config.time_budget)
            logger.info("  Max experiments: %s", self.config.max_experiments or "unlimited")
            logger.info("  Proposer: %s", self.config.proposer)
            logger.info("  Search space: %s", self.config.search_space_name)
            logger.info("  Output: %s", self.output_path)
            preflight_path = self._write_run_preflight()
            logger.info("  Preflight: %s", preflight_path)
            self._validate_attestation_runtime()
            self._validate_existing_history()

            if next_experiment_num == 1:
                self._run_baseline()
            else:
                logger.info("Resuming from experiment %d", next_experiment_num)

            while not self._should_stop():
                experiment_id = f"exp_{next_experiment_num:04d}"
                self._run_experiment(experiment_id)
                next_experiment_num += 1

            self._write_analysis()
            self._write_attestation_artifacts()
            self.tracker.print_summary()
            return self.tracker
        finally:
            self._release_output_lock()

    def _acquire_output_lock(self) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            with suppress(OSError):
                os.close(lock_fd)
            owner = self._lock_path.read_text().strip() if self._lock_path.exists() else "unknown"
            raise RuntimeError(
                f"Another autoresearch run is already using {self.output_path}: {owner}"
            ) from exc

        payload = json.dumps({
            "pid": os.getpid(),
            "output_dir": str(self.output_path),
            "time": time.time(),
        }, indent=2, sort_keys=True)
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, payload.encode("utf-8"))
        os.fsync(lock_fd)
        self._lock_fd = lock_fd

    def _release_output_lock(self) -> None:
        if self._lock_fd is None:
            return
        with suppress(OSError):
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        with suppress(OSError):
            os.close(self._lock_fd)
        self._lock_fd = None



    def _validate_attestation_runtime(self) -> None:
        if not self._attestation_requires_signature():
            return
        if load_hmac_secret("sign") is None:
            raise RuntimeError(
                "AUTORESEARCH_VERIFY_KEY is set but AUTORESEARCH_SIGNING_KEY is missing; "
                "signed provenance is required for this run."
            )

    def _validate_existing_history(self) -> None:
        if not self.tracker.records:
            return
        report = build_audit_report(
            self.output_path,
            require_signature=self._attestation_requires_signature(),
        )
        if not report["ok"]:
            summary = report["summary"]
            raise RuntimeError(
                "Existing run history failed attestation audit: "
                f"provenance_fail={summary['provenance_fail']} "
                f"provenance_missing={summary['provenance_missing']} "
                f"proofs_fail={summary['proofs_fail']} "
                f"proofs_missing_required={summary['proofs_missing_required']}"
            )
        self._validate_runtime_history()

    def _validate_runtime_history(self) -> None:
        if not self.tracker.records:
            return

        expected_runtime = self._resolve_train_invocation(
            self._shared_execution_context(requested_mode=self._requested_experiment_isolation())
        ).runtime
        expected_fingerprint = str(expected_runtime.get("fingerprint") or "")
        if not expected_fingerprint:
            return

        missing_runtime: list[str] = []
        mismatched_runtime: list[str] = []
        for record in self.tracker.records:
            if not record.provenance_path:
                missing_runtime.append(record.experiment_id)
                continue

            provenance_payload = self._load_json_object(Path(record.provenance_path))
            runtime_payload = provenance_payload.get("runtime") if provenance_payload else None
            if not isinstance(runtime_payload, dict):
                missing_runtime.append(record.experiment_id)
                continue

            actual_fingerprint = runtime_payload.get("fingerprint")
            if not actual_fingerprint:
                actual_fingerprint = sha256_json(self._runtime_fingerprint_payload(runtime_payload))

            if actual_fingerprint != expected_fingerprint:
                runtime_mode = str(runtime_payload.get("mode") or "unknown")
                mismatched_runtime.append(
                    f"{record.experiment_id}:{runtime_mode}:{str(actual_fingerprint)[:12]}"
                )

        if not missing_runtime and not mismatched_runtime:
            return

        details: list[str] = []
        if missing_runtime:
            details.append("missing runtime metadata for " + ", ".join(sorted(missing_runtime)))
        if mismatched_runtime:
            details.append("mismatched runtime fingerprint for " + ", ".join(mismatched_runtime))
        raise RuntimeError(
            "Existing run history uses a different runtime environment. "
            f"expected={expected_runtime.get('mode')}:{expected_fingerprint[:12]} "
            + "; ".join(details)
        )

    def _next_experiment_num(self) -> int:
        last_num = 0
        for record in self.tracker.records:
            if record.experiment_id.startswith("exp_"):
                try:
                    last_num = max(last_num, int(record.experiment_id.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        return last_num + 1

    def _run_baseline(self) -> None:
        logger.info("Running baseline experiment via subprocess...")
        result = self._execute_run(
            experiment_id="baseline",
            params=self.baseline_params,
            description="baseline (no training)",
            skip_train=True,
        )
        self._record_result(
            experiment_id="baseline",
            params=self.baseline_params,
            description="baseline (no training)",
            result=result,
            allow_keep=True,
        )

    def _run_experiment(self, experiment_id: str) -> None:
        current_best = (
            self.tracker.best_record.params
            if self.tracker.best_record is not None
            else self.baseline_params
        )
        proposed_params, description = self.proposer.propose(
            current_best=current_best,
            history=self.tracker.get_history_for_proposer(),
        )
        logger.info("Experiment %s: %s", experiment_id, description)
        result = self._execute_run(
            experiment_id=experiment_id,
            params=proposed_params,
            description=description,
            skip_train=False,
        )
        self._record_result(
            experiment_id=experiment_id,
            params=proposed_params,
            description=description,
            result=result,
            allow_keep=False,
        )

    def _execute_run(
        self,
        *,
        experiment_id: str,
        params: dict[str, Any],
        description: str,
        skip_train: bool,
    ) -> ExperimentProcessResult:
        run_dir = self.output_path / "runs" / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        log_path = run_dir / "train.log"
        config_path = run_dir / "config.json"

        invalid_params_error = self._validate_proposed_params(params)
        if invalid_params_error is not None:
            return ExperimentProcessResult(
                status="error",
                returncode=None,
                elapsed_seconds=0.0,
                summary=None,
                summary_path=summary_path,
                log_path=log_path,
                error=invalid_params_error,
            )

        overrides = self._build_run_overrides(
            experiment_id=experiment_id,
            params=params,
            skip_train=skip_train,
            output_dir=run_dir / "outputs",
        )
        config_path.write_text(json.dumps(overrides, indent=2, sort_keys=True))

        timeout_seconds = self._current_timeout()
        if timeout_seconds <= 0:
            return ExperimentProcessResult(
                status="timeout",
                returncode=None,
                elapsed_seconds=0.0,
                summary=None,
                summary_path=summary_path,
                log_path=log_path,
                error="max_wall_clock exhausted before experiment start",
            )

        budget_error = self._check_budget_feasibility(
            experiment_id=experiment_id,
            timeout_seconds=timeout_seconds,
            skip_train=skip_train,
        )
        if budget_error is not None:
            return ExperimentProcessResult(
                status="budget_too_low",
                returncode=None,
                elapsed_seconds=0.0,
                summary=None,
                summary_path=summary_path,
                log_path=log_path,
                error=budget_error,
            )

        algorithm = str(overrides.get("algorithm", "gspo")).lower()
        if algorithm != "gspo":
            return ExperimentProcessResult(
                status="error",
                returncode=None,
                elapsed_seconds=0.0,
                summary=None,
                summary_path=summary_path,
                log_path=log_path,
                error=(
                    f"subprocess runner supports only algorithm='gspo', got {algorithm!r}"
                ),
            )

        execution = self._prepare_execution_context(experiment_id=experiment_id)
        invocation = self._resolve_train_invocation(execution)

        env = os.environ.copy()
        env.update(
            {
                "AUTORESEARCH_RUN_CONFIG": str(config_path),
                "AUTORESEARCH_RUN_SUMMARY": str(summary_path),
                "AUTORESEARCH_TIME_BUDGET": str(max(1, int(timeout_seconds))),
                "AUTORESEARCH_EVAL_EPISODES": str(self.config.eval_episodes),
                "AUTORESEARCH_USE_GPU": "0" if self.device_info.use_cpu else "1",
                "AUTORESEARCH_REQUIRE_GPU": "0" if self.device_info.use_cpu else "1",
                "AUTORESEARCH_EXPERIMENT_ISOLATION": execution.mode,
                "AUTORESEARCH_RUNTIME_ENVIRONMENT": invocation.runtime["mode"],
                "AUTORESEARCH_SOURCE_REPO_ROOT": str(self.source_repo_root),
                "PYTHONUNBUFFERED": "1",
            }
        )

        result = run_train_subprocess(
            command=invocation.command,
            cwd=execution.cwd,
            env=env,
            timeout_seconds=timeout_seconds,
            summary_path=summary_path,
            log_path=log_path,
        )
        return ExperimentProcessResult(
            status=result.status,
            returncode=result.returncode,
            elapsed_seconds=result.elapsed_seconds,
            summary=result.summary,
            summary_path=result.summary_path,
            log_path=result.log_path,
            error=result.error,
            execution=execution.to_dict(),
            runtime=invocation.runtime,
        )

    def _matching_runtime_history(self, *, skip_train: bool) -> tuple[list[float], list[dict[str, float]]]:
        runtime_profile = str(self.base_run_config.get("runtime_profile", "standard"))
        durations: list[float] = []
        phase_timings: list[dict[str, float]] = []
        for record in self.tracker.records:
            record_skip_train = bool((record.params or {}).get("skip_train", False))
            record_profile = str((record.params or {}).get("runtime_profile", "standard"))
            if record_skip_train != skip_train or record_profile != runtime_profile:
                continue
            if record.training_time > 0.0:
                durations.append(float(record.training_time))
            summary_path = None
            if record.checkpoint_path:
                candidate = Path(record.checkpoint_path) / "summary.json"
                if candidate.exists():
                    summary_path = candidate
            if summary_path is None:
                continue
            try:
                payload = json.loads(summary_path.read_text())
            except Exception:
                continue
            raw_timings = payload.get("phase_timings")
            if isinstance(raw_timings, dict):
                normalized = {
                    str(name): float(value)
                    for name, value in raw_timings.items()
                    if isinstance(value, (int, float)) and float(value) > 0.0
                }
                if normalized:
                    phase_timings.append(normalized)
        return durations, phase_timings

    def _time_budget_recommendation(self, *, skip_train: bool) -> dict[str, Any]:
        runtime_profile = str(self.base_run_config.get("runtime_profile", "standard"))
        train_scenarios = list(self.base_run_config.get("train_scenarios") or [])
        eval_scenarios = list(self.base_run_config.get("eval_scenarios") or [])
        max_new_tokens = int(self.base_run_config.get("max_new_tokens", 64))
        durations, phase_timings = self._matching_runtime_history(skip_train=skip_train)
        return build_time_budget_recommendation(
            runtime_profile=runtime_profile,
            accelerator=self.device_info.accelerator,
            skip_train=skip_train,
            train_scenario_count=len(train_scenarios),
            eval_scenario_count=len(eval_scenarios),
            eval_episodes=self.config.eval_episodes,
            max_new_tokens=max_new_tokens,
            historical_durations=durations,
            historical_phase_timings=phase_timings,
        )

    def _write_run_preflight(self) -> Path:
        require_signature = self._attestation_requires_signature()
        baseline_recommendation = self._time_budget_recommendation(skip_train=True)
        experiment_recommendation = self._time_budget_recommendation(skip_train=False)
        invocation = self._resolve_train_invocation(
            self._shared_execution_context(requested_mode=self._requested_experiment_isolation())
        )
        runtime_artifact_path = self._write_runtime_artifact(
            self.output_path / "runtime_environment.json",
            invocation.runtime,
        )
        preflight = {
            "configured_time_budget_seconds": int(self.config.time_budget),
            "max_wall_clock_seconds": int(self.config.max_wall_clock),
            "allow_undersized_budget": self.allow_undersized_budget,
            "runtime_profile": str(self.base_run_config.get("runtime_profile", "standard")),
            "device": self.device_info.to_dict(),
            "attestation": {
                "signatures_required": require_signature,
                "signing_key_available": load_hmac_secret("sign") is not None,
            },
            "execution": {
                "requested_mode": self._requested_experiment_isolation(),
                "source_repo_root": str(self.source_repo_root),
                "source_git": detect_git_metadata(self.source_repo_root),
            },
            "runtime": self._runtime_public_payload(invocation.runtime),
            "runtime_artifact_path": self._relative_path(self.output_path, runtime_artifact_path),
            "runtime_artifact_sha256": self._hash_if_exists(runtime_artifact_path),
            "budget_recommendation": {
                "baseline": baseline_recommendation,
                "experiment": experiment_recommendation,
            },
            "recommended_min_budget_seconds": {
                "baseline": baseline_recommendation["recommended_seconds"],
                "experiment": experiment_recommendation["recommended_seconds"],
            },
        }
        preflight_path = self.output_path / "preflight.json"
        preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True))
        return preflight_path

    def _check_budget_feasibility(
        self,
        *,
        experiment_id: str,
        timeout_seconds: float,
        skip_train: bool,
    ) -> str | None:
        recommendation = self._time_budget_recommendation(skip_train=skip_train)
        recommended = int(recommendation["recommended_seconds"])
        if timeout_seconds >= recommended or self.allow_undersized_budget:
            return None
        mode = "baseline" if skip_train else "experiment"
        source = "heuristic"
        if recommendation.get("historical_phase_samples", 0) > 0:
            source = "phase history"
        elif recommendation.get("historical_duration_samples", 0) > 0:
            source = "duration history"
        return (
            f"Configured time budget {timeout_seconds:.1f}s is below recommended minimum "
            f"{recommended}s for {mode} runs on {self.device_info.accelerator} "
            f"with runtime_profile={self.base_run_config.get('runtime_profile', 'standard')} "
            f"based on {source}. Re-run with a larger --time-budget or pass "
            "--allow-undersized-budget to force execution."
        )


    @staticmethod
    def _load_json_object(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _planned_effective_params(
        self,
        *,
        experiment_id: str,
        params: dict[str, Any],
        result: ExperimentProcessResult,
    ) -> dict[str, Any]:
        if result.summary and isinstance(result.summary.get("config"), dict):
            return dict(result.summary["config"])

        config_payload = self._load_json_object(result.summary_path.parent / "config.json")
        if config_payload is not None:
            return config_payload

        return self._build_run_overrides(
            experiment_id=experiment_id,
            params=params,
            skip_train=experiment_id == "baseline",
            output_dir=result.summary_path.parent / "outputs",
        )


    def _build_run_overrides(
        self,
        *,
        experiment_id: str,
        params: dict[str, Any],
        skip_train: bool,
        output_dir: Path,
    ) -> dict[str, Any]:
        overrides = dict(self.baseline_params)
        overrides.update(self.base_run_config)
        overrides.update(params)
        overrides.update(self.runtime_overrides)
        overrides["experiment_id"] = experiment_id
        overrides["skip_train"] = skip_train
        overrides["output_dir"] = str(output_dir.resolve())

        if "max_completion_length" in overrides and "max_new_tokens" not in overrides:
            overrides["max_new_tokens"] = overrides.pop("max_completion_length")

        reward_aliases = {
            "empathy_weight": "empathy",
            "professionalism_weight": "professionalism",
            "action_oriented_weight": "action_oriented",
            "reasoning_weight": "reasoning",
            "length_weight": "length",
        }
        custom_reward_weights = dict(overrides.get("custom_reward_weights") or {})
        for source_key, target_key in reward_aliases.items():
            if source_key in overrides:
                custom_reward_weights[target_key] = float(overrides.pop(source_key))
        if custom_reward_weights:
            overrides["custom_reward_weights"] = custom_reward_weights

        return overrides

    @staticmethod
    def _relative_path(base: Path, path: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    @staticmethod
    def _hash_if_exists(path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        return sha256_file(path)

    def _write_provenance_envelope(
        self,
        *,
        experiment_id: str,
        description: str,
        result: ExperimentProcessResult,
        record_status: str,
        objective: float,
        metrics: dict[str, Any],
        effective_params: dict[str, Any],
    ) -> Path:
        run_dir = result.summary_path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        artifact_dir = run_dir / "outputs" / "final_model"
        artifact_files = [digest.to_dict() for digest in collect_file_digests(artifact_dir)]
        artifact_manifest_sha256 = sha256_json(artifact_files) if artifact_files else None

        summary_payload = result.summary or {}
        summary_config = summary_payload.get("config") or effective_params
        summary_metrics = summary_payload.get("metrics") or metrics

        execution_info = dict(result.execution or {})
        raw_runtime_payload = dict(result.runtime or {})
        runtime_artifact_path = self._write_runtime_artifact(
            run_dir / "runtime_environment.json",
            raw_runtime_payload,
        )
        runtime_payload = self._runtime_public_payload(raw_runtime_payload)
        execution_repo_root = Path(str(execution_info.get("repo_root") or self.source_repo_root))
        execution_payload = {
            "requested_mode": execution_info.get("requested_mode", self._requested_experiment_isolation()),
            "mode": execution_info.get("mode", "shared"),
            "cwd": execution_info.get("cwd", str(self.source_repo_root)),
            "repo_root": str(execution_repo_root),
            "source_repo_root": execution_info.get("source_repo_root", str(self.source_repo_root)),
            "fallback_reason": execution_info.get("fallback_reason"),
            "git": detect_git_metadata(execution_repo_root),
        }

        envelope: dict[str, Any] = {
            "experiment_id": experiment_id,
            "description": description,
            "process_status": result.status,
            "record_status": record_status,
            "elapsed_seconds": result.elapsed_seconds,
            "objective_metric": self.config.objective_metric,
            "objective_value": objective,
            "git": detect_git_metadata(self.source_repo_root),
            "execution": execution_payload,
            "runtime": runtime_payload,
            "runtime_artifact_path": self._relative_path(run_dir, runtime_artifact_path),
            "runtime_artifact_sha256": self._hash_if_exists(runtime_artifact_path),
            "config_path": self._relative_path(run_dir, config_path) if config_path.exists() else None,
            "config_sha256": self._hash_if_exists(config_path),
            "summary_path": self._relative_path(run_dir, result.summary_path) if result.summary_path.exists() else None,
            "summary_sha256": self._hash_if_exists(result.summary_path),
            "log_path": self._relative_path(run_dir, result.log_path) if result.log_path.exists() else None,
            "log_sha256": self._hash_if_exists(result.log_path),
            "summary_config_sha256": sha256_json(summary_config) if summary_config else None,
            "summary_metrics_sha256": sha256_json(summary_metrics) if summary_metrics else None,
            "artifacts": {
                "model_dir": self._relative_path(run_dir, artifact_dir) if artifact_dir.exists() else None,
                "manifest_sha256": artifact_manifest_sha256,
                "files": artifact_files,
            },
        }
        envelope["envelope_sha256"] = sha256_json(envelope)
        signature = sign_json_payload_if_configured(envelope)
        if signature is not None:
            envelope["signature"] = signature

        provenance_path = run_dir / "provenance.json"
        provenance_path.write_text(json.dumps(envelope, indent=2, sort_keys=True))
        return provenance_path

    def _get_stark_prover(self) -> Any | None:
        if self._stark_prover_import_failed:
            return None
        if self._stark_prover is not None:
            return self._stark_prover
        try:
            from stark_autoresearch.proof import MetricImprovementProver
        except Exception as exc:
            logger.warning("Skipping STARK proof emission: %s", exc)
            self._stark_prover_import_failed = True
            return None

        self._stark_prover = MetricImprovementProver()
        return self._stark_prover

    def _write_stark_proof_artifact(
        self,
        *,
        provenance_path: Path,
        previous_best: float | None,
        objective: float,
    ) -> Path | None:
        if previous_best is None or objective <= previous_best:
            return None

        prover = self._get_stark_prover()
        if prover is None:
            return None

        try:
            proof = prover.prove_improvement_from_provenance(
                provenance_path=provenance_path,
                best_reward=previous_best,
                agent_id="autoresearch-subprocess",
                require_signature=True,
            )
        except Exception as exc:
            logger.warning(
                "Skipping STARK proof for %s: %s",
                provenance_path.parent.name,
                exc,
            )
            return None

        proof_bin_path = provenance_path.parent / "proof.bin"
        proof_bin_path.write_bytes(proof.proof_bytes)
        proof_payload = proof.to_dict()
        proof_payload.update(
            {
                "proof_path": "proof.bin",
                "provenance_path": "provenance.json",
                "proof_bytes_sha256": sha256_file(proof_bin_path),
                "claimed_best_reward": previous_best,
                "objective_value": objective,
                "witness_commitment": proof.witness_commitment,
            }
        )
        proof_json_path = provenance_path.parent / "proof.json"
        proof_json_path.write_text(json.dumps(proof_payload, indent=2, sort_keys=True))
        return proof_json_path


    @staticmethod
    def _drop_proof_artifacts(proof_path: Path | None) -> None:
        if proof_path is None:
            return
        with suppress(OSError):
            proof_path.unlink()
        with suppress(OSError):
            (proof_path.parent / "proof.bin").unlink()

    def _selection_stability_rejection(
        self,
        *,
        metrics: dict[str, Any],
        previous_best: float | None,
    ) -> str | None:
        if self.config.objective_metric != "selection_score":
            return None
        if previous_best is None:
            return None

        bootstrap_std = metrics.get("selection_score_bootstrap_std")
        if bootstrap_std is None:
            return None

        zscore = float(getattr(self.config, "selection_promotion_zscore", 0.0) or 0.0)
        if zscore <= 0.0:
            return None

        selection_score = float(metrics.get("selection_score", 0.0))
        lower_bound = selection_score - zscore * float(bootstrap_std)
        if lower_bound > previous_best:
            return None

        return (
            f"selection lower bound {lower_bound:.6f} does not beat current best "
            f"{previous_best:.6f} at z={zscore:.2f}"
        )

    def _record_result(
        self,
        *,
        experiment_id: str,
        params: dict[str, Any],
        description: str,
        result: ExperimentProcessResult,
        allow_keep: bool,
    ) -> None:
        try:
            if result.status != "ok" or not result.summary:
                crash_params = self._planned_effective_params(
                    experiment_id=experiment_id,
                    params=params,
                    result=result,
                )
                provenance_path = self._write_provenance_envelope(
                    experiment_id=experiment_id,
                    description=description,
                    result=result,
                    record_status="crash",
                    objective=0.0,
                    metrics={},
                    effective_params=crash_params,
                )
                crash_record = ExperimentRecord(
                    experiment_id=experiment_id,
                    params=crash_params,
                    metrics={},
                    objective_value=0.0,
                    training_time=result.elapsed_seconds,
                    status="crash",
                    description=result.error or f"{description} failed",
                    checkpoint_path=str(result.summary_path.parent),
                    provenance_path=str(provenance_path),
                )
                self.tracker.record(crash_record)
                self._report_to_proposer(0.0, crashed=True)
                return

            metrics = dict(result.summary.get("metrics") or {})
            effective_params = dict(result.summary.get("config") or params)
            objective = float(metrics.get(self.config.objective_metric, 0.0))
            duration = float(result.summary.get("total_seconds", result.elapsed_seconds))
            previous_best = self.tracker.best_value
            record_description = description

            if allow_keep:
                tentative_status = "keep"
            else:
                tentative_status = "keep" if self._should_keep(objective) else "discard"
                if tentative_status == "keep":
                    selection_gate_reason = self._selection_stability_rejection(
                        metrics=metrics,
                        previous_best=previous_best,
                    )
                    if selection_gate_reason is not None:
                        tentative_status = "discard"
                        record_description = f"{description} [selection gate: {selection_gate_reason}]"

            provenance_path = self._write_provenance_envelope(
                experiment_id=experiment_id,
                description=record_description,
                result=result,
                record_status=tentative_status,
                objective=objective,
                metrics=metrics,
                effective_params=effective_params,
            )
            proof_path = None
            if tentative_status == "keep":
                proof_path = self._write_stark_proof_artifact(
                    provenance_path=provenance_path,
                    previous_best=previous_best,
                    objective=objective,
                )

            require_signature = self._attestation_requires_signature()
            attestation = evaluate_record_attestation(
                record_status=tentative_status,
                objective_value=objective,
                current_best=previous_best,
                direction=self.config.direction,
                provenance_path=provenance_path,
                proof_path=proof_path,
                require_signature=require_signature,
            )

            status = tentative_status
            if not attestation["ok"]:
                status = "discard"
                reasons: list[str] = []
                if attestation["provenance_status"] != "ok":
                    reasons.append(
                        f"provenance {attestation['provenance_status']}: "
                        f"{attestation['provenance_message'] or 'verification failed'}"
                    )
                if attestation["proof_status"] not in {"ok", "not_required"}:
                    reasons.append(
                        f"proof {attestation['proof_status']}: "
                        f"{attestation['proof_message'] or 'verification failed'}"
                    )
                if not reasons:
                    reasons.append("verification failed")
                record_description = (
                    f"{record_description} [attestation rejected: {'; '.join(reasons)}]"
                )
                if proof_path is not None:
                    self._drop_proof_artifacts(proof_path)
                    proof_path = None
                if status != tentative_status:
                    provenance_path = self._write_provenance_envelope(
                        experiment_id=experiment_id,
                        description=record_description,
                        result=result,
                        record_status=status,
                        objective=objective,
                        metrics=metrics,
                        effective_params=effective_params,
                    )

            record = ExperimentRecord(
                experiment_id=experiment_id,
                params=effective_params,
                metrics=metrics,
                objective_value=objective,
                training_time=duration,
                status=status,
                description=record_description,
                checkpoint_path=str(result.summary_path.parent),
                provenance_path=str(provenance_path),
                proof_path=str(proof_path) if proof_path else None,
            )
            self.tracker.record(record)
            self._report_to_proposer(objective, crashed=False)

            if status == "keep":
                self._persist_best_artifacts(record, result.summary, provenance_path, proof_path)
        finally:
            self._cleanup_execution_context(result)

    def _persist_best_artifacts(
        self,
        record: ExperimentRecord,
        summary: dict[str, Any],
        provenance_path: Path,
        proof_path: Path | None,
    ) -> None:
        result = rebuild_best_artifacts(self.output_path, tracker=self.tracker)
        if not result["ok"]:
            logger.warning(
                "Best artifact export incomplete for %s: missing source artifacts: %s",
                record.experiment_id,
                ", ".join(result["missing_sources"]),
            )

    def _should_keep(self, objective: float) -> bool:
        if not self.tracker.is_improvement(objective):
            return False
        if self.tracker.best_value is None:
            return True
        if self.config.improvement_threshold <= 0.0:
            return True

        best = self.tracker.best_value
        if best == 0.0:
            return True

        if self.config.direction == "maximize":
            relative_gain = (objective - best) / abs(best)
        else:
            relative_gain = (best - objective) / abs(best)
        return relative_gain >= self.config.improvement_threshold

    def _report_to_proposer(self, objective: float, crashed: bool) -> None:
        if isinstance(self.proposer, BayesianProposer):
            self.proposer.report_result(objective, crashed=crashed)

    def _remaining_wall_clock(self) -> float | None:
        if self.config.max_wall_clock <= 0:
            return None
        elapsed = time.time() - self._loop_start_time
        return max(0.0, self.config.max_wall_clock - elapsed)

    def _current_timeout(self) -> float:
        budget = float(self.config.time_budget)
        remaining = self._remaining_wall_clock()
        if remaining is not None:
            budget = min(budget, remaining)
        return max(0.0, budget)

    def _should_stop(self) -> bool:
        if (
            self.config.max_experiments > 0
            and self.tracker.num_experiments >= self.config.max_experiments
        ):
            logger.info("Reached max_experiments=%d", self.config.max_experiments)
            return True

        remaining = self._remaining_wall_clock()
        if remaining is not None and remaining <= 0:
            logger.info("Reached max_wall_clock=%ds", self.config.max_wall_clock)
            return True

        patience = self.config.improvement_patience
        if patience > 0 and len(self.tracker.records) > patience:
            recent = self.tracker.records[-patience:]
            if all(record.status != "keep" for record in recent):
                logger.info(
                    "Plateau detected: last %d experiments had no improvement.",
                    patience,
                )
                return True

        return False

    def _write_analysis(self) -> None:
        analysis = self.tracker.get_analysis()
        (self.output_path / "analysis.json").write_text(
            json.dumps(analysis, indent=2, default=str)
        )

    @staticmethod
    def _attestation_requires_signature() -> bool:
        return load_hmac_secret("verify") is not None

    def _write_attestation_artifacts(self) -> None:
        from attestation_audit import write_attestation_summary, write_audit_report

        require_signature = self._attestation_requires_signature()
        report, audit_path = write_audit_report(
            self.output_path,
            require_signature=require_signature,
        )
        summary, summary_path = write_attestation_summary(
            self.output_path,
            require_signature=require_signature,
            audit_report=report,
        )
        logger.info(
            "Attestation audit: %s",
            "OK" if report["ok"] else "FAIL",
        )
        logger.info("  Signatures required: %s", require_signature)
        logger.info("  Audit report: %s", audit_path)
        logger.info("  Attestation summary: %s", summary_path)
        best_record = summary.get("best_record") or {}
        if best_record:
            logger.info(
                "  Best attestation: %s provenance=%s proof=%s",
                best_record.get("experiment_id"),
                best_record.get("audit_provenance_status"),
                best_record.get("audit_proof_status"),
            )
