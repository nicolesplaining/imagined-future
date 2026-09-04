from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from imagined_future.fastwam_cache_factorial import (
    CACHE_FACTORIAL_CONDITIONS,
    build_cache_factorial_body,
    expand_cache_factorial_runs,
    file_sha256,
    freeze_cache_factorial_manifest,
    load_cache_factorial_manifest,
    validate_factorial_parent,
    write_cache_factorial_manifest,
)
from imagined_future.fastwam_optional_idm import (
    FastWAMCondition,
    FastWAMStateSpec,
    atomic_write_json,
    atomic_write_npz,
    build_manifest_body,
    expand_run_specs,
    freeze_manifest,
    make_branches,
    write_frozen_manifest,
)


def _fixtures(tmp_path: Path):
    state = FastWAMStateSpec(
        suite="libero_spatial",
        task_id=0,
        initial_state_index=4,
        wait_steps=30,
        branches=make_branches((101, 211, 307, 401), (1009, 2017, 3019, 4021)),
    )
    base = freeze_manifest(
        build_manifest_body(
            study_name="base",
            states=(state,),
            inference={
                "num_inference_steps": 20,
                "sigma_shift": 1.0,
                "num_video_frames": 9,
                "action_horizon": 32,
                "rand_device": "cpu",
            },
        )
    )
    base_path = tmp_path / "base.json"
    write_frozen_manifest(base_path, base)
    body = build_cache_factorial_body(
        study_name="factorial",
        base_manifest=base,
        base_manifest_sha256=file_sha256(base_path),
        design={"execution_rule": "after base completion"},
    )
    return state, base_path, freeze_cache_factorial_manifest(body)


def test_cache_factorial_is_separate_complete_2x2(tmp_path: Path) -> None:
    state, base_path, manifest = _fixtures(tmp_path)
    runs = expand_cache_factorial_runs(manifest["manifest_id"], state)
    assert len(runs) == 48
    assert len({run.run_id for run in runs}) == 48
    assert {run.condition for run in runs} == set(CACHE_FACTORIAL_CONDITIONS)
    assert all(run.recipient_id != run.donor_id for run in runs)
    missing = [
        run
        for run in runs
        if run.condition == "future_donor_cache_recipient"
    ]
    assert len(missing) == 12
    assert all(run.future_source_id == run.donor_id for run in missing)
    assert all(run.cache_source_id == run.recipient_id for run in missing)

    path = tmp_path / "factorial.json"
    write_cache_factorial_manifest(path, manifest)
    assert load_cache_factorial_manifest(path) == manifest
    assert validate_factorial_parent(manifest, base_path)["manifest_id"] == manifest[
        "base_manifest_id"
    ]


def test_cache_factorial_rejects_changed_parent(tmp_path: Path) -> None:
    _, base_path, manifest = _fixtures(tmp_path)
    changed = base_path.read_text(encoding="utf-8").replace('"base"', '"changed"')
    base_path.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_factorial_parent(manifest, base_path)


def test_cache_factorial_analyzer_requires_and_summarizes_complete_matrix(
    tmp_path: Path,
) -> None:
    state, base_path, manifest = _fixtures(tmp_path)
    factorial_path = tmp_path / "factorial.json"
    write_cache_factorial_manifest(factorial_path, manifest)
    base = validate_factorial_parent(manifest, base_path)
    base_output = tmp_path / "base_output" / base["manifest_id"]
    native_actions = {
        branch.branch_id: np.asarray([[float(index), 0.0]], dtype=np.float32)
        for index, branch in enumerate(state.branches)
    }
    for spec in expand_run_specs(base["manifest_id"], state):
        run_dir = base_output / state.state_id / "runs"
        arrays = {
            "action_model": native_actions[spec.recipient_id],
            "action_env": native_actions[spec.recipient_id],
        }
        if spec.condition == FastWAMCondition.NATIVE.value:
            arrays["video_latent"] = np.full(
                (1, 2, 2), int(spec.recipient_id[1:])
            )
        atomic_write_npz(
            run_dir / f"{spec.run_id}.npz",
            arrays,
        )
        atomic_write_json(
            run_dir / f"{spec.run_id}.json",
            {
                "status": "complete",
                "manifest_id": base["manifest_id"],
                "run": spec.to_dict(),
                "array_file": f"{spec.run_id}.npz",
            },
        )
    atomic_write_json(
        base_output / state.state_id / "summary.json",
        {
            "status": "complete",
            "manifest_id": base["manifest_id"],
            "state_id": state.state_id,
            "completed_conditions": [
                condition.value for condition in FastWAMCondition
            ],
        },
    )

    factorial_output = tmp_path / "factorial_output" / manifest["manifest_id"]
    run_dir = factorial_output / state.state_id / "runs"
    for spec in expand_cache_factorial_runs(manifest["manifest_id"], state):
        action = native_actions[spec.cache_source_id]
        atomic_write_npz(
            run_dir / f"{spec.run_id}.npz",
            {"action_model": action, "action_env": action},
        )
        atomic_write_json(
            run_dir / f"{spec.run_id}.json",
            {
                "status": "complete",
                "manifest_id": manifest["manifest_id"],
                "base_manifest_id": manifest["base_manifest_id"],
                "run": spec.to_dict(),
                "array_file": f"{spec.run_id}.npz",
                "base_reference_max_abs_error": 0.0,
                "same_recipient_cache_future_swap_max_abs_error": 0.0,
                "same_donor_cache_future_swap_max_abs_error": 0.0,
            },
        )
    atomic_write_json(
        factorial_output / state.state_id / "summary.json",
        {
            "status": "complete",
            "manifest_id": manifest["manifest_id"],
            "base_manifest_id": manifest["base_manifest_id"],
            "state_id": state.state_id,
            "completed_conditions": list(CACHE_FACTORIAL_CONDITIONS),
            "run_count": 48,
            "native_action_regeneration_max_abs_error": 0.0,
            "stored_native_latent_regeneration_max_abs_error": 0.0,
            "base_reference_max_abs_error": {
                "future_recipient_cache_recipient": 0.0,
                "future_recipient_cache_donor": 0.0,
                "future_donor_cache_donor": 0.0,
            },
            "recipient_cache_future_swap_global_max_abs_error": 0.0,
            "donor_cache_future_swap_global_max_abs_error": 0.0,
        },
    )
    project_root = Path(__file__).resolve().parents[1]
    summary_dir = tmp_path / "analysis"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "summarize_fastwam_cache_factorial.py"),
            "--factorial-manifest",
            str(factorial_path),
            "--factorial-output-root",
            str(tmp_path / "factorial_output"),
            "--base-manifest",
            str(base_path),
            "--base-output-root",
            str(tmp_path / "base_output"),
            "--summary-dir",
            str(summary_dir),
            "--required-state-count",
            "1",
            "--bootstrap-samples",
            "100",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(
        (summary_dir / "fastwam_cache_factorial_results.json").read_text()
    )
    assert report["evidence_gate"]["passed"] is True
    assert report["audit"]["valid_run_count"] == 48
    assert report["conditions"]["future_donor_cache_recipient"][
        "correct_donor_retrieval_rate"
    ]["mean"] == pytest.approx(0.0)
    assert report["conditions"]["future_recipient_cache_donor"][
        "correct_donor_retrieval_rate"
    ]["mean"] == pytest.approx(1.0)
