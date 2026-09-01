from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_cosmos3_research_server.py"
SPEC = importlib.util.spec_from_file_location("run_cosmos3_research_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
resolve_donor_path = MODULE.resolve_donor_path


def test_donor_path_must_stay_under_configured_result_root(tmp_path, monkeypatch) -> None:
    allowed = tmp_path / "results"
    allowed.mkdir()
    monkeypatch.setenv("IMAGINED_FUTURE_DONOR_ROOTS", str(allowed))

    assert resolve_donor_path(str(allowed / "unit" / "future.npz")) == (
        allowed / "unit" / "future.npz"
    ).resolve()
    with pytest.raises(ValueError, match="donor path must be under"):
        resolve_donor_path(str(tmp_path / "outside.npz"))


def test_multiple_donor_mount_aliases_are_supported(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv(
        "IMAGINED_FUTURE_DONOR_ROOTS", f"{first}:{second}"
    )

    assert resolve_donor_path(str(second / "future.npz")) == (
        second / "future.npz"
    ).resolve()
