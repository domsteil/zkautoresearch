import builtins
import importlib
import json
import sys

import stark_autoresearch
from stark_autoresearch.environment import EnvironmentReport, DependencyStatus, format_environment_report


def test_root_package_import_is_lazy_for_non_crypto_exports(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ves_stark":
            raise AssertionError("root package import should not import ves_stark")
        return real_import(name, globals, locals, fromlist, level)

    modules_to_clear = [
        name for name in list(sys.modules)
        if name == "stark_autoresearch" or name.startswith("stark_autoresearch.")
    ]
    for name in modules_to_clear:
        sys.modules.pop(name, None)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("stark_autoresearch")

    assert module.ExperimentConfig.__name__ == "ExperimentConfig"
    assert "stark_autoresearch.proof" not in sys.modules


def test_environment_report_formatting() -> None:
    report = EnvironmentReport(
        python="3.10.0",
        platform="TestOS-1.0",
        dependencies=[
            DependencyStatus(
                name="ves_stark",
                available=False,
                detail="Missing for test",
            )
        ],
    )
    rendered = format_environment_report(report)
    assert "Python:   3.10.0" in rendered
    assert "ves_stark: missing" in rendered
    assert "Missing for test" in rendered


def test_environment_report_json_shape() -> None:
    report = EnvironmentReport(
        python="3.10.0",
        platform="TestOS-1.0",
        dependencies=[DependencyStatus(name="ves_stark", available=True, location="/tmp/ves_stark.so")],
    )
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["ready"] is True
    assert payload["dependencies"][0]["name"] == "ves_stark"
    assert payload["dependencies"][0]["location"] == "/tmp/ves_stark.so"
