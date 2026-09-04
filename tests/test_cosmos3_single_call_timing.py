from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from imagined_future.cosmos3_single_call_timing import (
    ACTION_COORDINATE_COUNT,
    ACTION_SHAPE,
    BRANCH_SEEDS,
    EXPECTED_REQUEST_COUNT,
    EXPECTED_STATE_COUNT,
    REQUESTS_PER_STATE,
    SINGLE_CALL_CONDITIONS,
    TASKS,
    TIMING_CONDITIONS,
    directional_metrics,
    expected_request_labels,
    holm_adjust,
    make_hierarchical_draws,
    nearest_native_seed,
    ordered_off_diagonal_pairs,
    ordered_source_cells,
    separation_quartiles,
    state_estimands,
    summarize_state_values,
)


def test_frozen_design_counts_and_order() -> None:
    assert len(ordered_source_cells()) == 16
    assert len(ordered_off_diagonal_pairs()) == 12
    assert len(expected_request_labels()) == REQUESTS_PER_STATE == 108
    assert len(set(expected_request_labels())) == 108
    assert EXPECTED_REQUEST_COUNT == EXPECTED_STATE_COUNT * REQUESTS_PER_STATE == 3240
    assert [name for name, _ in TIMING_CONDITIONS] == [
        "none",
        "call_0_only",
        "call_1_only",
        "call_2_only",
        "call_3_only",
        "all_calls",
    ]


def test_nearest_native_uses_frozen_seed_order_for_exact_tie() -> None:
    native = {
        211: np.asarray([0.0]),
        223: np.asarray([2.0]),
        227: np.asarray([10.0]),
        229: np.asarray([20.0]),
    }
    nearest, distances, tied, margin = nearest_native_seed(np.asarray([1.0]), native)
    assert nearest == 211
    assert distances[211] == distances[223] == 1.0
    assert tied
    assert margin == 0.0


def test_directional_metrics_recover_on_axis_geometry() -> None:
    recipient = np.asarray([0.0, 0.0])
    donor = np.asarray([2.0, 0.0])
    result = directional_metrics(np.asarray([1.0, 0.0]), recipient, donor)
    assert result["native_separation"] == 2.0
    assert result["distance_reduction"] == 0.5
    assert result["donor_projection"] == 0.5
    assert result["cosine_alignment"] == 1.0
    assert result["orthogonal_residual_normalized"] == 0.0


def synthetic_report() -> dict[str, object]:
    def action_at(x: float, y: float) -> np.ndarray:
        action = np.zeros(ACTION_SHAPE, dtype=np.float64)
        action[0, 0] = x
        action[0, 1] = y
        return action

    native = {
        211: action_at(0.0, 0.0),
        223: action_at(1.0, 0.0),
        227: action_at(0.0, 2.0),
        229: action_at(3.0, 4.0),
    }
    rows = []
    sigmas = np.asarray(
        [0.9990000128746033, 0.9369999766349792, 0.8330000042915344, 0.6240000128746033],
        dtype=np.float32,
    )
    vision_shape = [1, 16, 9, 2, 2]
    future_frames = list(range(1, 9))
    default_unit_id = "BananaInBowlTask_seed_101_phase_middle_step_16"
    for timing, indices in TIMING_CONDITIONS:
        strength = {
            "none": 0.0,
            "call_0_only": 0.25,
            "call_1_only": 0.25,
            "call_2_only": 0.25,
            "call_3_only": 0.25,
            "all_calls": 1.0,
        }[timing]
        for recipient, source in ordered_source_cells():
            action = native[recipient] + strength * (native[source] - native[recipient])
            active = list(indices)
            inactive = [index for index in range(4) if index not in indices]
            rows.append(
                {
                    "timing_condition": timing,
                    "active_call_indices": active,
                    "recipient_seed": recipient,
                    "source_seed": source,
                    "action": action.tolist(),
                    "final_sampler_target_max_abs_error": 0.125,
                    "final_sampler_target_l2": 1.25,
                    "server": {
                        "research_sigmas": sigmas.tolist(),
                        "research_x0_sigmas": sigmas.tolist(),
                        "research_requested_active_call_indices": active,
                        "research_observed_active_call_indices": active,
                        "research_clamped_call_indices": active,
                        "research_inactive_call_indices": inactive,
                        "research_requested_active_sigmas": sigmas[active].tolist(),
                        "research_observed_active_sigmas": sigmas[active].tolist(),
                        "research_future_frame_indices": future_frames,
                        "research_vision_shape": vision_shape,
                        "research_vision_coordinate_count": 576,
                        "research_future_mask_coordinate_count": 512,
                        "research_future_mask_index_hash": "mask",
                        "research_model_input_future_clamp_errors": [0.0] * len(active),
                        "research_returned_future_velocity_overwrite_errors": [0.0]
                        * len(active),
                        "research_action_input_errors": [0.0] * 4,
                        "research_action_output_errors": [0.0] * 4,
                        "research_maximum_action_input_error": 0.0,
                        "research_maximum_action_output_error": 0.0,
                        "research_inactive_wrapper_write_count": 0,
                        "research_action_donor_projection": (
                            None if recipient == source else strength
                        ),
                        "research_action_donor_projection_applicable": (
                            recipient != source
                        ),
                        "research_attention_interface": {
                            "instrumented_server": False,
                            "intervention_requested": False,
                            "mode": "exclude",
                            "cache_id": None,
                        },
                        "research_state_hash": "state",
                        "research_parameter_probe_hash": "probe",
                        "research_recipient_id": (
                            f"fixture-{default_unit_id}-native-{recipient}"
                        ),
                        "research_donor_id": (
                            f"fixture-{default_unit_id}-native-{source}"
                        ),
                        "research_recipient_path_noise_hash": f"path-{recipient}",
                        "research_initial_state_hash": f"initial-{recipient}",
                        "research_recipient_future_hash": f"future-{recipient}",
                        "research_donor_future_hash": f"future-{source}",
                        "research_target_hash": (
                            f"future-{recipient}"
                            if timing == "none" or recipient == source
                            else f"future-{source}"
                        ),
                        "research_target_source": (
                            "recipient"
                            if timing == "none" or recipient == source
                            else "donor"
                        ),
                        "research_target_source_record_ids": [
                            (
                                f"fixture-{default_unit_id}-native-{recipient}"
                                if timing == "none" or recipient == source
                                else f"fixture-{default_unit_id}-native-{source}"
                            )
                        ],
                        "research_final_sampler_target_max_abs_error": 0.125,
                        "research_final_sampler_target_l2": 1.25,
                    },
                }
            )
    return {
        "unit_id": default_unit_id,
        "task": "BananaInBowlTask",
        "environment_seed": 101,
        "branch_seeds": list(BRANCH_SEEDS),
        "action_shape": list(ACTION_SHAPE),
        "action_coordinate_count": ACTION_COORDINATE_COUNT,
        "shape_valid_response_action_count": REQUESTS_PER_STATE,
        "action_shape_failure_count": 0,
        "native_actions": {str(seed): action.tolist() for seed, action in native.items()},
        "native_future_hashes": {
            str(seed): f"future-{seed}" for seed in BRANCH_SEEDS
        },
        "native_path_noise_hashes": {
            str(seed): f"path-{seed}" for seed in BRANCH_SEEDS
        },
        "native_initial_state_hashes": {
            str(seed): f"initial-{seed}" for seed in BRANCH_SEEDS
        },
        "input_fingerprints": ["state"],
        "parameter_probe_hashes": ["probe"],
        "timing_rows": rows,
    }


def test_state_estimands_use_timing_matched_self_comparator() -> None:
    result = state_estimands(synthetic_report())
    assert result["timing"]["none"]["complete_source_retrieval"] == 0.25
    assert result["timing"]["none"]["matched_retrieval_gain"] == 0.0
    assert result["timing"]["none"]["matched_distance_gain"] == 0.0
    assert result["timing"]["all_calls"]["complete_source_retrieval"] == 1.0
    assert result["timing"]["all_calls"]["matched_retrieval_gain"] == 1.0
    assert result["timing"]["all_calls"]["matched_distance_gain"] == 1.0
    assert result["average_single"]["matched_distance_gain"] == pytest.approx(0.25)
    assert result["sustained_minus_single"]["matched_distance_gain"] == pytest.approx(0.75)
    assert len(result["pair_rows"]) == 6 * 12


def test_state_estimands_refuse_missing_or_reordered_cell() -> None:
    report = synthetic_report()
    report["timing_rows"] = list(reversed(report["timing_rows"]))
    with pytest.raises(ValueError, match="frozen 96-row order"):
        state_estimands(report)


def test_analyzer_enforces_projection_null_and_none_zero_schema() -> None:
    analyzer = load_script_module(
        "timing_analyzer_projection_fixture",
        "summarize_cosmos3_single_call_timing.py",
    )
    manifest = {
        "manifest_id": "fixture",
        "design": {
            "future_frame_indices": list(range(1, 9)),
            "vision_shape": [1, 16, 9, 2, 2],
            "vision_coordinate_count": 576,
            "future_mask_coordinate_count": 512,
            "future_mask_index_hash": "mask",
        },
    }
    unit = {"unit_id": synthetic_report()["unit_id"]}
    valid = synthetic_report()
    analyzer.validate_timing_server_rows(valid, manifest, unit)

    diagonal_finite = synthetic_report()
    diagonal_finite["timing_rows"][0]["server"][
        "research_action_donor_projection"
    ] = 0.0
    with pytest.raises(ValueError, match="diagonal projection"):
        analyzer.validate_timing_server_rows(diagonal_finite, manifest, unit)

    off_diagonal_null = synthetic_report()
    off_diagonal_null["timing_rows"][1]["server"][
        "research_action_donor_projection"
    ] = None
    with pytest.raises(ValueError, match="required numeric scalar"):
        analyzer.validate_timing_server_rows(off_diagonal_null, manifest, unit)

    none_nonzero = synthetic_report()
    none_nonzero["timing_rows"][1]["server"][
        "research_action_donor_projection"
    ] = 0.25
    with pytest.raises(ValueError, match="off-diagonal none projection"):
        analyzer.validate_timing_server_rows(none_nonzero, manifest, unit)


def test_dedicated_timing_server_serializes_structural_projection_as_null() -> None:
    server = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_cosmos3_single_call_timing_server.py"
    ).read_text(encoding="utf-8")
    assert 'outputs["research_action_donor_projection_applicable"]' in server
    assert "if projection_applicable" in server
    assert "else None" in server
    assert 'float("nan")' not in server


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((32, 7)),
        np.zeros((32, 9)),
        np.zeros((8, 32)),
        [],
        None,
        [[0.0], [0.0, 1.0]],
        np.full((32, 8), np.nan),
        np.full((32, 8), np.inf),
    ],
    ids=("32x7", "32x9", "transposed", "empty", "null", "ragged", "nan", "inf"),
)
def test_runner_and_analyzer_fail_closed_on_malformed_action_shape(invalid) -> None:
    runner = load_script_module(
        "timing_runner_action_shape_fixture", "run_cosmos3_single_call_timing.py"
    )
    analyzer = load_script_module(
        "timing_analyzer_action_shape_fixture",
        "summarize_cosmos3_single_call_timing.py",
    )
    with pytest.raises((RuntimeError, ValueError)):
        runner.require_exact_action({"action": invalid}, "fixture")
    with pytest.raises(ValueError):
        analyzer.require_exact_action(invalid, "fixture")


def test_full_action_estimand_is_sensitive_to_gripper_coordinate() -> None:
    recipient = np.zeros(ACTION_SHAPE, dtype=np.float64)
    donor = recipient.copy()
    donor[:, 7] = 1.0
    metrics = directional_metrics(donor, recipient, donor)
    assert donor.size == ACTION_COORDINATE_COUNT
    assert metrics["native_separation"] == pytest.approx(np.sqrt(32.0))
    assert metrics["distance_reduction"] == 1.0
    assert metrics["donor_projection"] == 1.0


def thirty_state_rows() -> list[dict[str, object]]:
    return [
        {
            "task": task,
            "unit_id": f"{task}-state-{state_index}",
        }
        for task in TASKS
        for state_index in range(5)
    ]


def test_shared_hierarchical_draws_and_equal_task_point_estimate() -> None:
    rows = thirty_state_rows()
    values = {
        str(row["unit_id"]): float(TASKS.index(str(row["task"])))
        for row in rows
    }
    draws = make_hierarchical_draws(rows, samples=100, seed=7)
    result = summarize_state_values(rows, values, draws)
    assert result["mean"] == 2.5
    assert result["bootstrap_samples"] == 100
    assert set(result["task_means"]) == set(TASKS)
    matrix = np.asarray(
        [[values[state_id] for state_id in draws.state_ids_by_task[task]] for task in TASKS]
    )
    independent = []
    for draw_index, sampled_tasks in enumerate(draws.task_draws):
        independent.append(
            np.mean(
                [
                    matrix[task_index, draws.state_draws[draw_index, occurrence]].mean()
                    for occurrence, task_index in enumerate(sampled_tasks)
                ]
            )
        )
    assert result["bootstrap_values"] == pytest.approx(independent)
    expected_p = (1 + np.count_nonzero((np.asarray(independent) - 2.5) >= 2.5)) / 101
    assert result["one_sided_null_centered_p"] == expected_p


def test_holm_uses_stop_on_first_failure_and_monotone_adjustment() -> None:
    result = holm_adjust({"c0": 0.001, "c1": 0.02, "c2": 0.03, "c3": 0.04})
    assert result["c0"]["rejected"] is True
    assert result["c1"]["rejected"] is False
    assert result["c2"]["rejected"] is False
    assert result["c3"]["rejected"] is False
    ordered_adjusted = [result[name]["holm_adjusted_p"] for name in ("c0", "c1", "c2", "c3")]
    assert ordered_adjusted == sorted(ordered_adjusted)


def test_global_separation_quartiles_use_exactly_360_directed_pairs() -> None:
    pair_rows = [
        {
            "timing_condition": "all_calls",
            "unit_id": f"state-{index // 12}",
            "task": TASKS[(index // 60) % len(TASKS)],
            "native_separation": float(index + 1),
        }
        for index in range(360)
    ]
    result = separation_quartiles(pair_rows)
    assert result["boundaries"] == pytest.approx([90.75, 180.5, 270.25])
    assert sum(result["pair_counts"].values()) == 360
    assert all(value == 90 for value in result["pair_counts"].values())


def test_degenerate_native_axis_is_never_silently_dropped() -> None:
    report = synthetic_report()
    report["native_actions"]["223"] = report["native_actions"]["211"]
    with pytest.raises(ValueError, match="degenerate native axis"):
        state_estimands(report)


def test_complete_analyzer_fixture_emits_all_required_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = []
    reports = []
    for task in TASKS:
        for state_index, environment_seed in enumerate((101, 103, 107, 109, 113)):
            unit_id = f"{task}_seed_{environment_seed}_phase_middle_step_{16 + 32 * state_index}"
            state = {
                "unit_id": unit_id,
                "task": task,
                "environment_seed": environment_seed,
                "phase": "middle",
            }
            states.append(state)
            report = synthetic_report()
            report.update(
                {
                    "status": "complete",
                    "unit_id": unit_id,
                    "task": task,
                    "environment_seed": environment_seed,
                    "phase": "middle",
                    "request_count": REQUESTS_PER_STATE,
                    "request_labels": list(expected_request_labels()),
                    "runtime_gate": {
                        "passed": True,
                        "exact_schedule": True,
                        "exact_active_site_captures": True,
                        "exact_mask": True,
                        "zero_action_coordinate_writes": True,
                        "zero_inactive_wrapper_writes": True,
                        "exact_none_noop": True,
                        "exact_replays": True,
                        "exact_rng_and_target_hashes": True,
                        "all_finite": True,
                        "required_numeric_fields_finite": True,
                        "structural_null_census_exact": True,
                        "exact_projection_applicability_census": True,
                        "exact_action_shape_and_count": True,
                    },
                    "input_fingerprint_count": 1,
                    "input_fingerprints": ["state"],
                    "parameter_probe_hash_count": 1,
                    "parameter_probe_hashes": ["probe"],
                    "native_replay_max_action_error": 0.0,
                    "all_calls_diagonal_replay_max_action_error": 0.0,
                    "none_noop_max_action_error": 0.0,
                    "none_source_invariance_max_action_error": 0.0,
                    "maximum_action_input_error": 0.0,
                    "maximum_action_output_error": 0.0,
                    "maximum_active_model_input_future_clamp_error": 0.0,
                    "maximum_active_returned_future_velocity_error": 0.0,
                    "inactive_wrapper_write_count": 0,
                    "schedule_and_index_gate_exact": True,
                    "target_hash_gate_exact": True,
                    "rng_hash_gate_exact": True,
                    "replay_signature_gate_exact": True,
                    "structural_projection_null_count": 28,
                    "finite_off_diagonal_projection_count": 72,
                    "native_projection_absent_count": 8,
                    "native_replay_action_errors": {
                        str(seed): 0.0 for seed in BRANCH_SEEDS
                    },
                    "all_calls_diagonal_replay_action_errors": {
                        str(seed): 0.0 for seed in BRANCH_SEEDS
                    },
                    "none_noop_action_errors": {
                        str(seed): 0.0 for seed in BRANCH_SEEDS
                    },
                    "none_source_action_errors": {
                        str(seed): 0.0 for seed in BRANCH_SEEDS
                    },
                    "native_replay_signature_exact": {
                        str(seed): True for seed in BRANCH_SEEDS
                    },
                    "all_calls_diagonal_replay_signature_exact": {
                        str(seed): True for seed in BRANCH_SEEDS
                    },
                    "none_source_invariance_exact": {
                        str(seed): True for seed in BRANCH_SEEDS
                    },
                }
            )
            for row in report["timing_rows"]:
                recipient = int(row["recipient_seed"])
                source = int(row["source_seed"])
                timing = str(row["timing_condition"])
                server = row["server"]
                server["research_recipient_id"] = (
                    f"fixture-{unit_id}-native-{recipient}"
                )
                server["research_donor_id"] = f"fixture-{unit_id}-native-{source}"
                server["research_target_source_record_ids"] = [
                    (
                        f"fixture-{unit_id}-native-{recipient}"
                        if timing == "none" or recipient == source
                        else f"fixture-{unit_id}-native-{source}"
                    )
                ]
            reports.append(report)
    script = Path(__file__).parents[1] / "scripts" / "summarize_cosmos3_single_call_timing.py"
    analyzer_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    manifest = {
        "status": "frozen_before_model_outcomes",
        "study_name": "cosmos3-single-call-timing-v5",
        "manifest_id": "fixture",
        "states": states,
        "design": {
            "request_labels": list(expected_request_labels()),
            "future_frame_indices": list(range(1, 9)),
            "vision_shape": [1, 16, 9, 2, 2],
            "vision_coordinate_count": 576,
            "future_mask_coordinate_count": 512,
            "future_mask_index_hash": "mask",
            "action_shape": list(ACTION_SHAPE),
            "action_coordinate_count": ACTION_COORDINATE_COUNT,
        },
        "runtime": {
            "expected_parameter_probe_hash": "probe",
            "analyzer_sha256": analyzer_hash,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output_root = tmp_path / "states"
    output_root.mkdir()
    for report in reports:
        report["manifest_id"] = "fixture"
        report["manifest_sha256"] = manifest_hash
        (output_root / f"{report['unit_id']}.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
    spec = importlib.util.spec_from_file_location("timing_analyzer_fixture", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--manifest",
            str(manifest_path),
            "--expected-manifest-sha256",
            manifest_hash,
            "--output-root",
            str(output_root),
            "--summary-dir",
            str(summary_dir),
            "--bootstrap-samples",
            "10000",
        ],
    )
    module.main()
    expected = {
        "cosmos3_single_call_timing_states.csv",
        "cosmos3_single_call_timing_pairs.csv",
        "cosmos3_single_call_timing_quartiles.csv",
        "cosmos3_single_call_timing_residuals.csv",
        "cosmos3_single_call_timing_per_task.csv",
        "cosmos3_single_call_timing_leave_one_task_out.csv",
        "cosmos3_single_call_timing_aggregate.csv",
        "cosmos3_single_call_timing_results.json",
        "cosmos3_single_call_timing_results.tex",
        "cosmos3_single_call_timing_summary.png",
    }
    assert {path.name for path in summary_dir.iterdir()} == expected
    result = json.loads(
        (summary_dir / "cosmos3_single_call_timing_results.json").read_text()
    )
    assert result["audit"]["state_count"] == 30
    assert result["audit"]["request_count"] == 3240
    assert result["evidence_gates"]["runtime_and_completeness"] is True
    assert result["final_sampler_target_residual_distributions"]["call_0_only"][
        "final_sampler_target_l2"
    ]["count"] == 480


def load_script_module(name: str, filename: str):
    script = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_checkpoint_manifest_is_canonical_and_launcher_rehashes_every_file(
    tmp_path: Path,
) -> None:
    hasher = load_script_module(
        "checkpoint_hasher_fixture", "hash_checkpoint_content_manifest.py"
    )
    launcher = load_script_module(
        "timing_launcher_fixture", "launch_cosmos3_single_call_timing.py"
    )
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "nested").mkdir(parents=True)
    (checkpoint / "config.json").write_bytes(b"config\n")
    (checkpoint / "nested" / "weights.bin").write_bytes(b"weights\x00\x01")
    content = hasher.build_manifest(checkpoint)
    content_bytes = hasher.canonical_json_bytes(content)
    content_path = tmp_path / "checkpoint.content-manifest.json"
    content_path.write_bytes(content_bytes)
    content_hash = hashlib.sha256(content_bytes).hexdigest()
    runtime = {
        "checkpoint_identity_kind": (
            "sha256_of_canonical_full_file_content_manifest"
        ),
        "checkpoint_content_manifest": str(content_path),
        "checkpoint_content_manifest_sha256": content_hash,
        "checkpoint_content_manifest_file_count": 2,
        "checkpoint_content_manifest_total_size_bytes": 16,
        "checkpoint_root": str(checkpoint),
        "checkpoint_verification_root": str(checkpoint),
    }
    audit = launcher.validate_full_checkpoint_content({"runtime": runtime})
    assert audit["file_count"] == 2
    assert audit["total_size_bytes"] == 16
    assert audit["checkpoint_content_manifest_sha256"] == content_hash
    (checkpoint / "nested" / "weights.bin").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="size differs|SHA-256 differs"):
        launcher.validate_full_checkpoint_content({"runtime": runtime})


def test_excluded_smoke_manifest_preserves_snapshot_and_cannot_be_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = load_script_module(
        "timing_smoke_builder_fixture",
        "build_cosmos3_single_call_timing_smoke_manifest.py",
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    closure_file = snapshot / "runner.py"
    closure_file.write_text("# frozen\n", encoding="utf-8")
    closure_hash = hashlib.sha256(closure_file.read_bytes()).hexdigest()
    main = {
        "manifest_id": "main",
        "status": "frozen_before_model_outcomes",
        "study_name": "cosmos3-single-call-timing-v5",
        "admission": "frozen_single_call_timing_evaluation",
        "launch_authorization": "independent_outcome_blind_go_required",
        "selection_uses_model_or_intervention_outcomes": False,
        "scope": {},
        "source": {},
        "design": {},
        "runtime": {
            "snapshot_root": str(snapshot.resolve()),
            "snapshot_file_sha256": {"runner.py": closure_hash},
        },
        "states": [{"unit_id": f"main-{index}"} for index in range(30)],
    }
    main_path = tmp_path / "main.json"
    main_path.write_text(json.dumps(main), encoding="utf-8")
    main_hash = hashlib.sha256(main_path.read_bytes()).hexdigest()
    excluded = {
        "status": "frozen_before_model_outcomes",
        "admission": "excluded_development_smoke",
        "selection_uses_model_or_intervention_outcomes": False,
        "states": [
            {
                "unit_id": "excluded-bagels",
                "task": "BagelsOnPlateTask",
                "environment_seed": 101,
                "branch_seeds": list(BRANCH_SEEDS),
            }
        ],
    }
    excluded_path = tmp_path / "excluded.json"
    excluded_path.write_text(json.dumps(excluded), encoding="utf-8")
    output = tmp_path / "smoke.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--main-manifest",
            str(main_path),
            "--expected-main-manifest-sha256",
            main_hash,
            "--source-excluded-manifest",
            str(excluded_path),
            "--snapshot-root",
            str(snapshot),
            "--output",
            str(output),
        ],
    )
    builder.main()
    smoke = json.loads(output.read_text(encoding="utf-8"))
    assert smoke["study_name"] == "cosmos3-single-call-timing-v5"
    assert smoke["admission"] == "excluded_development_smoke"
    assert smoke["launch_authorization"] == "excluded_smoke_only_not_evaluation"
    assert smoke["scope"]["admitted_to_evaluation"] is False
    assert len(smoke["states"]) == 1
    assert smoke["design"]["total_request_count"] == 108


def test_required_numeric_audit_leaves_reject_none() -> None:
    analyzer = load_script_module(
        "timing_analyzer_numeric_fixture", "summarize_cosmos3_single_call_timing.py"
    )
    with pytest.raises(ValueError, match="missing required numeric scalar"):
        analyzer.require_finite_scalar({"value": None}, "value", "fixture")
    with pytest.raises(ValueError, match="missing required numeric array"):
        analyzer.require_exact_numeric_array(
            {"value": None}, "value", np.zeros(1, dtype=np.float64), "fixture"
        )
