from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def getenv_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with strict parsing."""
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False

    raise ValueError(
        f"Environment variable {name} must be one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw!r}"
    )


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved runtime device settings for an experiment."""

    accelerator: str
    use_cpu: bool
    cuda_available: bool
    device_name: str | None
    torch_version: str | None
    bf16_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FileDigest:
    """Digest for a single file within an artifact tree."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSummary:
    """Structured run summary written after each experiment."""

    status: str
    started_at: float
    completed_at: float
    time_budget_seconds: int
    accelerator: str
    cuda_available: bool
    device_name: str | None
    torch_version: str | None
    bf16_enabled: bool
    training_seconds: float | None
    total_seconds: float
    peak_vram_mb: float
    metrics: dict[str, Any]
    config: dict[str, Any]
    phase_timings: dict[str, float]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON deterministically for hashing/signing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json_dumps(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_digests(root: str | Path) -> list[FileDigest]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    digests: list[FileDigest] = []
    for file_path in sorted(path for path in root_path.rglob("*") if path.is_file()):
        digests.append(
            FileDigest(
                path=str(file_path.relative_to(root_path)),
                sha256=sha256_file(file_path),
                size_bytes=file_path.stat().st_size,
            )
        )
    return digests


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    output = completed.stdout.strip()
    return output or None


def detect_git_metadata(repo_root: str | Path) -> dict[str, Any]:
    repo_path = Path(repo_root)
    commit = _run_git(repo_path, "rev-parse", "HEAD")
    branch = _run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    status_output = _run_git(repo_path, "status", "--porcelain")
    is_dirty = None if status_output is None else bool(status_output)

    return {
        "repo_root": str(repo_path),
        "commit": commit,
        "branch": branch,
        "is_dirty": is_dirty,
    }


def _decode_secret_value(raw: str) -> bytes:
    value = raw.strip()
    if value.startswith("hex:"):
        return bytes.fromhex(value[4:])
    return value.encode("utf-8")


def _load_secret(prefix: str) -> tuple[bytes, str | None] | None:
    raw = os.getenv(f"{prefix}_KEY")
    path = os.getenv(f"{prefix}_KEY_FILE")
    key_id = os.getenv(f"{prefix}_KEY_ID")

    if raw and path:
        raise ValueError(f"Set only one of {prefix}_KEY or {prefix}_KEY_FILE")
    if raw is not None:
        return _decode_secret_value(raw), key_id
    if path is not None:
        return Path(path).read_bytes().rstrip(b"\r\n"), key_id
    return None


def load_hmac_secret(role: str = "sign") -> tuple[bytes, str] | None:
    if role not in {"sign", "verify"}:
        raise ValueError(f"Unknown HMAC secret role: {role!r}")

    if role == "sign":
        loaded = _load_secret("AUTORESEARCH_SIGNING")
    else:
        loaded = _load_secret("AUTORESEARCH_VERIFY") or _load_secret("AUTORESEARCH_SIGNING")

    if loaded is None:
        return None

    key, key_id = loaded
    return key, key_id or "default-hmac"


def sign_json_payload(
    payload: Any,
    *,
    key: bytes,
    key_id: str = "default-hmac",
) -> dict[str, str]:
    encoded = canonical_json_dumps(payload).encode("utf-8")
    payload_sha256 = sha256_bytes(encoded)
    signature = hmac.new(key, encoded, hashlib.sha256).hexdigest()
    return {
        "algorithm": "hmac-sha256",
        "key_id": key_id,
        "signed_payload_sha256": payload_sha256,
        "value": signature,
    }


def sign_json_payload_if_configured(payload: Any) -> dict[str, str] | None:
    loaded = load_hmac_secret("sign")
    if loaded is None:
        return None
    key, key_id = loaded
    return sign_json_payload(payload, key=key, key_id=key_id)


def verify_json_signature(
    payload: Any,
    signature: dict[str, Any],
    *,
    key: bytes | None = None,
) -> tuple[bool, str]:
    algorithm = signature.get("algorithm")
    if algorithm != "hmac-sha256":
        return False, f"Unsupported signature algorithm: {algorithm!r}"

    encoded = canonical_json_dumps(payload).encode("utf-8")
    payload_sha256 = sha256_bytes(encoded)
    expected_payload_sha256 = signature.get("signed_payload_sha256")
    if payload_sha256 != expected_payload_sha256:
        return False, "Signed payload hash mismatch"

    if key is None:
        loaded = load_hmac_secret("verify")
        if loaded is None:
            return False, "No verification key configured"
        key, _ = loaded

    expected_signature = hmac.new(key, encoded, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_signature, str(signature.get("value", ""))):
        return False, "Signature mismatch"

    return True, "ok"


def verify_provenance_envelope(
    path: str | Path,
    *,
    require_signature: bool = False,
) -> dict[str, Any]:
    envelope_path = Path(path)
    payload = json.loads(envelope_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Provenance envelope at {envelope_path} must be a JSON object")

    signature = payload.get("signature")
    core = {key: value for key, value in payload.items() if key != "signature"}
    unhashed_core = {key: value for key, value in core.items() if key != "envelope_sha256"}
    run_dir = envelope_path.parent

    checks: dict[str, bool] = {}
    checks["envelope_sha256"] = sha256_json(unhashed_core) == core.get("envelope_sha256")

    def _check_optional_file_hash(path_key: str, hash_key: str) -> bool:
        relative_path = core.get(path_key)
        expected_hash = core.get(hash_key)
        if relative_path is None and expected_hash is None:
            return True
        if not relative_path or not expected_hash:
            return False
        candidate = run_dir / str(relative_path)
        return candidate.is_file() and sha256_file(candidate) == expected_hash

    checks["config_sha256"] = _check_optional_file_hash("config_path", "config_sha256")
    checks["summary_sha256"] = _check_optional_file_hash("summary_path", "summary_sha256")
    checks["log_sha256"] = _check_optional_file_hash("log_path", "log_sha256")
    checks["runtime_artifact_sha256"] = _check_optional_file_hash(
        "runtime_artifact_path",
        "runtime_artifact_sha256",
    )

    summary_ok = True
    summary_path = core.get("summary_path")
    if summary_path:
        summary_payload = json.loads((run_dir / str(summary_path)).read_text())
        summary_ok = (
            sha256_json(summary_payload.get("config") or {}) == core.get("summary_config_sha256")
            and sha256_json(summary_payload.get("metrics") or {}) == core.get("summary_metrics_sha256")
        )
    checks["summary_payload"] = summary_ok

    artifacts = core.get("artifacts") or {}
    artifact_dir = artifacts.get("model_dir")
    artifact_files = artifacts.get("files") or []
    artifact_manifest_sha256 = artifacts.get("manifest_sha256")
    if artifact_dir is None:
        checks["artifact_manifest"] = artifact_manifest_sha256 is None and artifact_files == []
        checks["artifact_files"] = artifact_files == []
    else:
        actual_files = [digest.to_dict() for digest in collect_file_digests(run_dir / str(artifact_dir))]
        checks["artifact_files"] = actual_files == artifact_files
        checks["artifact_manifest"] = sha256_json(actual_files) == artifact_manifest_sha256

    if signature is None:
        signature_valid = None if not require_signature else False
        signature_reason = "Signature not present"
    else:
        signature_valid, signature_reason = verify_json_signature(core, signature)

    ok = all(checks.values()) and (signature_valid if signature_valid is not None else True)
    return {
        "ok": ok,
        "path": str(envelope_path),
        "checks": checks,
        "signature_present": signature is not None,
        "signature_valid": signature_valid,
        "signature_reason": signature_reason,
        "envelope_sha256": core.get("envelope_sha256"),
    }


def estimate_min_time_budget_seconds(
    *,
    runtime_profile: str,
    accelerator: str,
    skip_train: bool,
    train_scenario_count: int,
    eval_scenario_count: int,
    eval_episodes: int,
    max_new_tokens: int,
    historical_durations: list[float] | None = None,
) -> int:
    profile = runtime_profile.strip().lower() or "standard"
    accelerator_name = accelerator.strip().lower() or "cpu"
    train_count = max(1, int(train_scenario_count))
    eval_count = max(1, int(eval_scenario_count))
    episodes = max(1, int(eval_episodes))
    token_count = max(1, int(max_new_tokens))

    if accelerator_name == "cpu":
        base = 8 if skip_train else 16
        if profile == "standard":
            base += 18 if skip_train else 34
    else:
        base = 4 if skip_train else 8
        if profile == "standard":
            base += 6 if skip_train else 12

    scenario_weight = train_count + (eval_count * episodes)
    heuristic = base + scenario_weight + max(0, token_count // 8)

    if historical_durations:
        positive = sorted(value for value in historical_durations if value > 0.0)
        if positive:
            observed = positive[-1]
            median = positive[len(positive) // 2]
            heuristic = max(heuristic, int(round(max(observed, median * 1.25))))

    return max(1, int(heuristic))


def _summarize_phase_history(
    historical_phase_timings: list[dict[str, float]] | None,
) -> tuple[dict[str, int], int]:
    if not historical_phase_timings:
        return {}, 0

    phase_budget: dict[str, int] = {}
    phase_names = sorted({name for sample in historical_phase_timings for name in sample})
    for name in phase_names:
        values = sorted(
            float(sample[name])
            for sample in historical_phase_timings
            if sample.get(name, 0.0) > 0.0
        )
        if not values:
            continue
        median = values[len(values) // 2]
        observed = values[-1]
        phase_budget[name] = int(round(max(observed, median * 1.25)))

    return phase_budget, len(historical_phase_timings)



def build_time_budget_recommendation(
    *,
    runtime_profile: str,
    accelerator: str,
    skip_train: bool,
    train_scenario_count: int,
    eval_scenario_count: int,
    eval_episodes: int,
    max_new_tokens: int,
    historical_durations: list[float] | None = None,
    historical_phase_timings: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    heuristic_seconds = estimate_min_time_budget_seconds(
        runtime_profile=runtime_profile,
        accelerator=accelerator,
        skip_train=skip_train,
        train_scenario_count=train_scenario_count,
        eval_scenario_count=eval_scenario_count,
        eval_episodes=eval_episodes,
        max_new_tokens=max_new_tokens,
        historical_durations=historical_durations,
    )

    phase_budget_seconds, phase_history_samples = _summarize_phase_history(historical_phase_timings)
    observed_seconds = sum(phase_budget_seconds.values()) if phase_budget_seconds else None
    recommended_seconds = heuristic_seconds if observed_seconds is None else max(heuristic_seconds, observed_seconds)

    return {
        "recommended_seconds": int(recommended_seconds),
        "heuristic_seconds": int(heuristic_seconds),
        "observed_phase_seconds": observed_seconds,
        "phase_budget_seconds": phase_budget_seconds,
        "historical_duration_samples": len(historical_durations or []),
        "historical_phase_samples": phase_history_samples,
    }


def detect_device_info(
    *,
    prefer_gpu: bool = True,
    require_gpu: bool = False,
    requested_bf16: bool = False,
) -> DeviceInfo:
    """Resolve whether this run should use CPU or CUDA."""
    try:
        import torch  # type: ignore
    except ImportError as exc:
        if require_gpu:
            raise RuntimeError(
                "PyTorch is required and no GPU runtime is available."
            ) from exc
        return DeviceInfo(
            accelerator="cpu",
            use_cpu=True,
            cuda_available=False,
            device_name=None,
            torch_version=None,
            bf16_enabled=False,
        )

    cuda_available = bool(torch.cuda.is_available())
    if require_gpu and not cuda_available:
        raise RuntimeError("GPU was requested but CUDA is not available.")

    use_cpu = not (prefer_gpu and cuda_available)
    device_name = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "cuda:0"

    return DeviceInfo(
        accelerator="cpu" if use_cpu else "cuda",
        use_cpu=use_cpu,
        cuda_available=cuda_available,
        device_name=device_name,
        torch_version=getattr(torch, "__version__", None),
        bf16_enabled=requested_bf16 and not use_cpu,
    )


def write_run_summary(path: str | Path, summary: RunSummary) -> Path:
    """Persist a run summary as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return output_path
