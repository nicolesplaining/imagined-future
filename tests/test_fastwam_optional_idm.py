from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from imagined_future.fastwam_optional_idm import (
    FASTWAM_UPSTREAM_COMMIT,
    FastWAMCondition,
    FastWAMStateSpec,
    action_metrics,
    atomic_write_json,
    atomic_write_npz,
    build_manifest_body,
    expand_run_specs,
    freeze_manifest,
    load_frozen_manifest,
    make_branches,
    shuffled_kv_cache,
    state_from_dict,
    write_frozen_manifest,
)
from imagined_future.fastwam_analysis import analyze_fastwam_smoke, hierarchical_bootstrap


def _manifest_and_state() -> tuple[dict, FastWAMStateSpec]:
    state = FastWAMStateSpec(
        suite="libero_spatial",
        task_id=0,
        initial_state_index=3,
        wait_steps=30,
        branches=make_branches((11, 22, 33, 44), (101, 202, 303, 404)),
    )
    body = build_manifest_body(
        study_name="unit_test",
        states=(state,),
        inference={
            "num_inference_steps": 20,
            "sigma_shift": 1.0,
            "num_video_frames": 9,
            "action_horizon": 32,
            "rand_device": "cpu",
        },
    )
    return freeze_manifest(body), state


def test_manifest_id_is_deterministic_and_commit_pinned(tmp_path: Path) -> None:
    manifest, _ = _manifest_and_state()
    repeated, _ = _manifest_and_state()
    assert manifest["manifest_id"] == repeated["manifest_id"]
    assert manifest["upstream"]["commit"] == FASTWAM_UPSTREAM_COMMIT

    destination = tmp_path / "manifest.json"
    write_frozen_manifest(destination, manifest)
    assert load_frozen_manifest(destination) == manifest
    write_frozen_manifest(destination, manifest)


def test_manifest_detects_post_freeze_change(tmp_path: Path) -> None:
    manifest, _ = _manifest_and_state()
    manifest["states"][0]["task_id"] = 1
    destination = tmp_path / "tampered.json"
    destination.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_frozen_manifest(destination)


def test_complete_four_branch_matrix_has_unique_frozen_run_ids() -> None:
    manifest, state = _manifest_and_state()
    runs = expand_run_specs(manifest["manifest_id"], state)
    assert len(runs) == 72
    assert len({run.run_id for run in runs}) == len(runs)
    counts = {
        condition.value: sum(run.condition == condition.value for run in runs)
        for condition in FastWAMCondition
    }
    assert counts == {
        "native": 4,
        "self_latent": 4,
        "self_cache": 4,
        "donor_latent": 12,
        "donor_cache": 12,
        "wrong_latent": 12,
        "shuffled_cache": 12,
        "first_frame": 12,
    }
    for run in runs:
        if run.condition == FastWAMCondition.WRONG_LATENT.value:
            assert run.source_id not in {run.recipient_id, run.donor_id}
        if run.condition == FastWAMCondition.FIRST_FRAME.value:
            assert run.donor_id is not None
            assert run.source_id == run.donor_id
            assert run.source_id != run.recipient_id


def test_action_metrics_separate_projection_from_orthogonal_error() -> None:
    recipient = np.asarray([0.0, 0.0])
    donor = np.asarray([2.0, 0.0])
    patched = np.asarray([2.0, 3.0])
    metrics = action_metrics(
        patched,
        recipient,
        donor,
        {"recipient": recipient, "donor": donor, "third": np.asarray([2.0, 2.5])},
        donor_id="donor",
    )
    assert metrics["donor_projection"] == pytest.approx(1.0)
    assert metrics["orthogonal_residual"] == pytest.approx(3.0)
    assert metrics["orthogonal_residual_ratio"] == pytest.approx(1.5)
    assert metrics["distance_to_donor"] == pytest.approx(3.0)
    assert not metrics["correct_donor_retrieval"]


def test_action_metrics_preserve_degenerate_native_axis() -> None:
    recipient = np.asarray([1.0, 2.0])
    metrics = action_metrics(
        np.asarray([2.0, 2.0]),
        recipient,
        recipient.copy(),
        {"recipient": recipient, "donor": recipient.copy()},
        donor_id="donor",
    )
    assert metrics["axis_degenerate"] is True
    assert metrics["donor_projection"] is None
    assert metrics["donor_distance_reduction"] is None
    assert metrics["distance_to_donor"] == pytest.approx(1.0)


def test_shuffled_cache_is_deterministic_and_breaks_value_pairing() -> None:
    key = torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3)
    value = (100 + torch.arange(24, dtype=torch.float32)).reshape(1, 4, 2, 3)
    shuffled_a = shuffled_kv_cache([key], [value], seed=7)
    shuffled_b = shuffled_kv_cache([key], [value], seed=7)
    assert torch.equal(shuffled_a[0][0], key)
    assert torch.equal(shuffled_a[1][0], shuffled_b[1][0])
    assert not torch.equal(shuffled_a[1][0], value)
    assert torch.equal(
        shuffled_a[1][0].flatten().sort().values,
        value.flatten().sort().values,
    )


def test_atomic_json_and_npz_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "result.json"
    npz_path = tmp_path / "nested" / "result.npz"
    atomic_write_json(json_path, {"run_id": "run-1", "value": 3})
    atomic_write_npz(npz_path, {"action": np.arange(6).reshape(2, 3)})
    assert json.loads(json_path.read_text(encoding="utf-8"))["run_id"] == "run-1"
    with np.load(npz_path) as arrays:
        np.testing.assert_array_equal(arrays["action"], np.arange(6).reshape(2, 3))
    assert not list((tmp_path / "nested").glob(".*.tmp*"))


def _write_synthetic_complete_smoke(tmp_path: Path) -> tuple[Path, Path]:
    branches = make_branches((11, 22, 33, 44), (101, 202, 303, 404))
    states = tuple(
        FastWAMStateSpec(
            suite="libero_spatial",
            task_id=0,
            initial_state_index=index,
            wait_steps=30,
            branches=branches,
        )
        for index in range(8)
    )
    body = build_manifest_body(
        study_name="synthetic_complete_smoke",
        states=states,
        inference={
            "num_inference_steps": 20,
            "sigma_shift": 1.0,
            "num_video_frames": 9,
            "action_horizon": 32,
            "rand_device": "cpu",
        },
    )
    manifest = freeze_manifest(body)
    manifest_path = tmp_path / "manifest.json"
    write_frozen_manifest(manifest_path, manifest)
    output_root = tmp_path / "outputs"
    for state in states:
        state_root = output_root / manifest["manifest_id"] / state.state_id
        run_dir = state_root / "runs"
        native_actions = {
            branch.branch_id: np.asarray(
                [[float(branch_index), float(state.initial_state_index)]], dtype=np.float32
            )
            for branch_index, branch in enumerate(branches)
        }
        for run in expand_run_specs(manifest["manifest_id"], state):
            recipient = native_actions[run.recipient_id]
            if run.condition in {"native", "self_latent", "self_cache"}:
                action = recipient.copy()
            elif run.condition in {"donor_latent", "donor_cache"}:
                action = native_actions[run.donor_id].copy()
            elif run.condition == "wrong_latent":
                action = native_actions[run.source_id].copy()
            elif run.condition == "shuffled_cache":
                action = recipient.copy()
            elif run.condition == "first_frame":
                action = recipient + np.asarray([[0.0, 0.125]], dtype=np.float32)
            else:  # pragma: no cover - the enum is exhaustive
                raise AssertionError(run.condition)
            arrays = {
                "action_model": action,
                "action_env": action,
            }
            if run.condition == "native":
                branch_index = int(run.recipient_id[1:])
                arrays["video_latent"] = np.full(
                    (1, 2, 2), branch_index + 0.1 * state.initial_state_index,
                    dtype=np.float16,
                )
            atomic_write_npz(run_dir / f"{run.run_id}.npz", arrays)
            atomic_write_json(
                run_dir / f"{run.run_id}.json",
                {
                    "status": "complete",
                    "manifest_id": manifest["manifest_id"],
                    "upstream_commit": FASTWAM_UPSTREAM_COMMIT,
                    "run": run.to_dict(),
                    "array_file": f"{run.run_id}.npz",
                },
            )
        atomic_write_json(
            state_root / "summary.json",
            {
                "status": "complete",
                "manifest_id": manifest["manifest_id"],
                "state_id": state.state_id,
                "completed_conditions": sorted(
                    condition.value for condition in FastWAMCondition
                ),
            },
        )
    return manifest_path, output_root


def test_fastwam_analyzer_requires_complete_eight_state_matrix(tmp_path: Path) -> None:
    manifest_path, output_root = _write_synthetic_complete_smoke(tmp_path)
    summary_dir = tmp_path / "summary"
    report, exit_code = analyze_fastwam_smoke(
        manifest_path=manifest_path,
        output_root=output_root,
        summary_dir=summary_dir,
        bootstrap_samples=100,
        make_plot=True,
    )
    assert exit_code == 0
    assert report["scale_gate"]["passed"] is True
    assert report["conditions"]["donor_latent"]["correct_donor_retrieval_rate"][
        "mean"
    ] == pytest.approx(1.0)
    assert report["conditions"]["donor_cache"]["correct_donor_retrieval_rate"][
        "mean"
    ] == pytest.approx(1.0)
    assert report["conditions"]["wrong_latent"]["correct_donor_retrieval_rate"][
        "mean"
    ] < 1.0
    assert report["conditions"]["donor_latent"]["degenerate_axes"] == 0
    for filename in (
        "fastwam_results.json",
        "fastwam_run_metrics.csv",
        "fastwam_state_metrics.csv",
        "fastwam_aggregate_metrics.csv",
        "fastwam_leave_one_task_out_metrics.csv",
        "fastwam_results.tex",
        "fastwam_summary.png",
    ):
        assert (summary_dir / filename).is_file()

    hierarchical_dir = tmp_path / "hierarchical_summary"
    hierarchical, hierarchical_exit = analyze_fastwam_smoke(
        manifest_path=manifest_path,
        output_root=output_root,
        summary_dir=hierarchical_dir,
        bootstrap_samples=100,
        required_state_count=8,
        bootstrap_mode="hierarchical",
        gate_mode="powered_evidence",
        make_plot=False,
    )
    assert hierarchical_exit == 0
    assert hierarchical["gate"]["passed"] is True
    assert hierarchical["hierarchical_conditions"] is not None
    assert hierarchical["analysis"]["version"] == "source-grid-v2"
    assert hierarchical["source_grid_retrieval"]["latent"][
        "correct_source_retrieval_rate"
    ]["mean"] == pytest.approx(1.0)
    assert hierarchical["source_grid_retrieval"]["cache"][
        "correct_source_retrieval_rate"
    ]["mean"] == pytest.approx(1.0)
    assert hierarchical["source_grid_retrieval"]["latent"]["n_rows"] == 128
    assert hierarchical["source_grid_retrieval"]["cache"]["chance_rate"] == 0.25
    assert hierarchical["gate"]["thresholds"][
        "donor_only_three_label_reference_rate"
    ] == pytest.approx(1.0 / 3.0)
    assert (
        hierarchical["gate"]["criteria"][
            "latent_4x4_source_retrieval_hierarchical_lower_above_chance"
        ]
        is True
    )
    assert (hierarchical_dir / "fastwam_source_grid_state_metrics.csv").is_file()
    assert (hierarchical_dir / "fastwam_task_metrics.csv").is_file()
    assert (hierarchical_dir / "fastwam_suite_metrics.csv").is_file()

    manifest = load_frozen_manifest(manifest_path)
    first_state = state_from_dict(manifest["states"][0])
    missing_run = expand_run_specs(manifest["manifest_id"], first_state)[-1]
    (output_root / manifest["manifest_id"] / first_state.state_id / "runs" / f"{missing_run.run_id}.json").unlink()
    incomplete_dir = tmp_path / "incomplete_summary"
    incomplete, incomplete_exit = analyze_fastwam_smoke(
        manifest_path=manifest_path,
        output_root=output_root,
        summary_dir=incomplete_dir,
        bootstrap_samples=100,
        make_plot=False,
    )
    assert incomplete_exit == 2
    assert incomplete["scale_gate"]["decision"] == "unavailable_incomplete_matrix"
    assert (incomplete_dir / "fastwam_missingness.csv").is_file()
    assert not (incomplete_dir / "fastwam_state_metrics.csv").exists()


def test_hierarchical_bootstrap_is_deterministic_and_uses_nested_grid() -> None:
    rows = [
        {
            "suite": f"suite_{suite}",
            "task_id": task,
            "state_id": f"s{suite}-t{task}-x{state}",
            "value": float(100 * suite + 10 * task + state),
        }
        for suite in range(2)
        for task in range(3)
        for state in range(4)
    ]
    first = hierarchical_bootstrap(
        rows, value_key="value", samples=500, seed=17, key="unit"
    )
    second = hierarchical_bootstrap(
        rows, value_key="value", samples=500, seed=17, key="unit"
    )
    assert first == second
    assert first["n_suites"] == 2
    assert first["n_tasks"] == 6
    assert first["n_states_total"] == 24
    assert first["n_states_missing"] == 0
    assert first["mean"] == pytest.approx(np.mean([row["value"] for row in rows]))
    assert first["ci95_low"] < first["mean"] < first["ci95_high"]
