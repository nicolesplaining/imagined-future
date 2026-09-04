#!/usr/bin/env python3
"""Strict frozen analysis for the 21-state Cosmos future x K/V factorial."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from launch_cosmos3_existing_kv_factorial import (
    EXPECTED_CELLS,
    FROZEN_MANIFEST_ID,
    FROZEN_MANIFEST_SHA256,
    FROZEN_RUNNER_SHA256,
    HOST_NFS_ROOT,
    ValidationError,
    host_to_container,
    read_json,
    require_mapping,
    sha256_file,
    validate_manifest,
    validate_receipt,
    validate_report,
)


CELL_ABBREVIATIONS = {
    "recipient_future_recipient_kv": "R future / R K/V",
    "donor_future_recipient_kv": "D future / R K/V",
    "donor_future_donor_kv": "D future / D K/V",
    "recipient_future_donor_kv": "R future / D K/V",
}
CELL_METRICS = (
    "donor_projection",
    "distance_to_recipient",
    "distance_to_donor",
)
CONTRAST_DEFINITIONS = {
    "kv_effect_at_recipient_future": (
        "recipient-future donor-K/V minus recipient-future recipient-K/V donor projection"
    ),
    "kv_effect_at_donor_future": (
        "donor-future donor-K/V minus donor-future recipient-K/V donor projection"
    ),
    "future_effect_at_recipient_kv": (
        "donor-future recipient-K/V minus recipient-future recipient-K/V donor projection"
    ),
    "future_effect_at_donor_kv": (
        "donor-future donor-K/V minus recipient-future donor-K/V donor projection"
    ),
    "future_by_kv_interaction": (
        "K/V effect at donor future minus K/V effect at recipient future"
    ),
}
FOLLOW_DEFINITIONS = {
    "donor_future_recipient_kv_follows_kv": (
        "1 when D-future/R-KV action is closer to recipient-native than donor-native"
    ),
    "recipient_future_donor_kv_follows_kv": (
        "1 when R-future/D-KV action is closer to donor-native than recipient-native"
    ),
    "both_crossed_arms_follow_kv": "1 when both crossed arms are closer to their K/V-native endpoint",
    "crossed_arm_kv_follow_fraction": "fraction of the two crossed arms closer to their K/V-native endpoint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Frozen v3 manifest; exact ID and SHA-256 are enforced.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Completed launcher output root containing manifest.json and states/.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        required=True,
        help="New directory to create atomically; an existing path is refused.",
    )
    return parser.parse_args()


def _under_host_root(path: Path, *, source: str) -> Path:
    if not path.is_absolute():
        raise ValidationError(f"{source} must be an absolute path")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(HOST_NFS_ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValidationError(f"{source} resolves outside {HOST_NFS_ROOT}: {resolved}") from error
    return resolved


def _validate_run_log(path: Path, *, state_id: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValidationError(f"{state_id}: missing/nonempty run.log proof")
    # The launcher writes its one-line exit record last, after the worker's JSON.
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        buffer = bytearray()
        while position > 0 and len(buffer) < 64 * 1024:
            position -= 1
            handle.seek(position)
            byte = handle.read(1)
            if byte == b"\n" and buffer:
                break
            if byte != b"\n" or buffer:
                buffer.extend(byte)
    try:
        last_line = bytes(reversed(buffer)).decode("utf-8")
        footer = json.loads(last_line)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{state_id}: run.log lacks a valid final exit record") from error
    if not isinstance(footer, Mapping):
        raise ValidationError(f"{state_id}: final run.log record must be an object")
    expected = {"event": "state_exit", "state_id": state_id, "return_code": 0}
    for key, value in expected.items():
        if footer.get(key) != value:
            raise ValidationError(
                f"{state_id}: run.log final {key} expected {value!r}, got {footer.get(key)!r}"
            )
    elapsed = footer.get("elapsed_seconds")
    if isinstance(elapsed, bool):
        raise ValidationError(f"{state_id}: invalid elapsed_seconds")
    try:
        elapsed_number = float(elapsed)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{state_id}: invalid elapsed_seconds") from error
    if not math.isfinite(elapsed_number) or elapsed_number < 0.0:
        raise ValidationError(f"{state_id}: invalid elapsed_seconds")


def inventory_complete_run(
    *, manifest_path: Path, manifest: Mapping[str, Any], manifest_sha: str, output_root: Path
) -> list[tuple[Mapping[str, Any], Path, Path, str]]:
    root = _under_host_root(output_root, source="output root")
    snapshot = root / "manifest.json"
    states_root = root / "states"
    if not snapshot.is_file() or sha256_file(snapshot) != manifest_sha:
        raise ValidationError("output root has no exact frozen-manifest snapshot")
    if snapshot.read_bytes() != manifest_path.read_bytes():
        raise ValidationError("output manifest snapshot is not byte-identical to supplied manifest")
    if not states_root.is_dir():
        raise ValidationError(f"missing states directory: {states_root}")
    root_entries = {path.name for path in root.iterdir()}
    if root_entries != {"manifest.json", "states"}:
        raise ValidationError(
            "output root must contain exactly manifest.json and states/: "
            f"found {sorted(root_entries)}"
        )
    expected_ids = {str(state["state_id"]) for state in manifest["states"]}
    actual_entries = list(states_root.iterdir())
    non_directories = [path.name for path in actual_entries if not path.is_dir()]
    actual_ids = {path.name for path in actual_entries if path.is_dir()}
    if non_directories or actual_ids != expected_ids:
        raise ValidationError(
            "state inventory is not exactly complete: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"unexpected={sorted(actual_ids - expected_ids)}, files={sorted(non_directories)}"
        )

    inventory: list[tuple[Mapping[str, Any], Path, Path, str]] = []
    for state in manifest["states"]:
        state_id = str(state["state_id"])
        state_dir = states_root / state_id
        entries = list(state_dir.iterdir())
        allowed_exact = {"report.json", "receipt.json", "run.log"}
        unexpected = [
            path.name
            for path in entries
            if path.name not in allowed_exact
            and not (
                path.is_file()
                and path.name.startswith("attempt-")
                and path.name.endswith(".failed.log")
            )
        ]
        missing = [name for name in sorted(allowed_exact) if not (state_dir / name).is_file()]
        if unexpected or missing:
            raise ValidationError(
                f"{state_id}: incomplete/unrecognized artifact set; missing={missing}, "
                f"unexpected={sorted(unexpected)}"
            )
        report_path = state_dir / "report.json"
        receipt_path = state_dir / "receipt.json"
        run_log = state_dir / "run.log"
        _validate_run_log(run_log, state_id=state_id)
        receipt = read_json(receipt_path, label=f"receipt[{state_id}]")
        report_sha = str(receipt.get("report_sha256", ""))
        if len(report_sha) != 64 or any(character not in "0123456789abcdef" for character in report_sha):
            raise ValidationError(f"{state_id}: receipt has invalid report_sha256")
        if sha256_file(report_path) != report_sha:
            raise ValidationError(f"{state_id}: report bytes differ from accepted receipt")
        inventory.append((state, report_path, receipt_path, report_sha))
    if len(inventory) != int(manifest["evaluation_state_count"]):
        raise ValidationError("complete inventory count differs from frozen manifest")
    del manifest_path
    return inventory


def validate_and_load_reports(
    *,
    manifest: Mapping[str, Any],
    manifest_sha: str,
    inventory: Sequence[tuple[Mapping[str, Any], Path, Path, str]],
) -> list[tuple[Mapping[str, Any], dict[str, Any], str]]:
    loaded: list[tuple[Mapping[str, Any], dict[str, Any], str]] = []
    for state, report_path, receipt_path, report_sha in inventory:
        receipt = validate_receipt(
            receipt_path,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            state=state,
            runner_sha256=FROZEN_RUNNER_SHA256,
            report_sha256=report_sha,
        )
        expected_receipt_keys = {
            "schema_version",
            "status",
            "manifest_id",
            "manifest_sha256",
            "state_id",
            "state_spec_sha256",
            "study_id",
            "runner_sha256",
            "server_image",
            "container",
            "report_sha256",
            "completed_at_utc",
            "validated_controls",
        }
        if set(receipt) != expected_receipt_keys:
            raise ValidationError(f"receipt[{state['state_id']}]: field set changed")
        if not str(receipt["container"]):
            raise ValidationError(f"receipt[{state['state_id']}]: empty container name")
        timestamp = str(receipt["completed_at_utc"])
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValidationError(
                f"receipt[{state['state_id']}]: invalid completion timestamp"
            ) from error
        if parsed_timestamp.tzinfo is None:
            raise ValidationError(
                f"receipt[{state['state_id']}]: completion timestamp lacks a timezone"
            )
        controls = require_mapping(
            receipt, "validated_controls", source=f"receipt[{state['state_id']}]"
        )
        expected_control_keys = {
            "exact_identity_and_replay",
            "single_state_fingerprint",
            "single_parameter_probe_fingerprint",
            "recipient_rng_repeat_exact_and_donor_rng_distinct",
            "fixed_recipient_rng_for_all_interventions",
            "future_signatures_fixed_within_source_and_distinct_between_sources",
            "no_action_coordinate_mutation",
            "cache_layers",
            "cache_calls_per_layer",
        }
        if set(controls) != expected_control_keys:
            raise ValidationError(
                f"receipt[{state['state_id']}]: validated-control key set changed"
            )
        report, observed_sha = validate_report(
            report_path,
            state=state,
            manifest=manifest,
            report_sha256=report_sha,
        )
        _validate_analyzer_only_controls(report, state_id=str(state["state_id"]))
        loaded.append((state, report, observed_sha))
    return loaded


def _validate_analyzer_only_controls(report: Mapping[str, Any], *, state_id: str) -> None:
    """Verify clean arms and receipt-adjacent fields beyond the launcher gate."""

    responses = require_mapping(report, "responses", source=f"report[{state_id}]")
    clean_labels = (
        "recipient-native",
        "recipient-repeat",
        "donor-native",
        "recipient-baseline",
        "donor-baseline",
    )
    for label in clean_labels:
        response = require_mapping(
            responses, label, source=f"report[{state_id}].responses"
        )
        if response.get("research_attention_exclude_layers") != []:
            raise ValidationError(f"{state_id}: clean arm {label} requested attention layers")
        if response.get("research_attention_exclude_scope") != "action":
            raise ValidationError(f"{state_id}: clean arm {label} has unexpected scope")
        interface = require_mapping(
            response,
            "research_attention_interface",
            source=f"report[{state_id}].responses.{label}",
        )
        expected = {
            "layers": 36,
            "excluded_keys_values": "future_video",
            "instrumented_server": True,
            "intervention_requested": False,
            "text_kv_reuse": False,
            "mode": "exclude",
            "cache_id": None,
            "cache_call_counts": {},
        }
        for key, value in expected.items():
            if interface.get(key) != value:
                raise ValidationError(
                    f"{state_id}: clean arm {label} interface.{key} expected "
                    f"{value!r}, got {interface.get(key)!r}"
                )


def _classify_closer(
    *, distance_to_recipient: float, distance_to_donor: float, donor_projection: float
) -> str:
    if distance_to_recipient == distance_to_donor:
        distance_class = "tie"
    elif distance_to_recipient < distance_to_donor:
        distance_class = "recipient"
    else:
        distance_class = "donor"
    if donor_projection == 0.5:
        projection_class = "tie"
    elif donor_projection < 0.5:
        projection_class = "recipient"
    else:
        projection_class = "donor"
    scale = max(1.0, abs(distance_to_recipient), abs(distance_to_donor))
    strong_distance_side = abs(distance_to_recipient - distance_to_donor) > 1e-7 * scale
    strong_projection_side = abs(donor_projection - 0.5) > 1e-7
    if (
        distance_class != projection_class
        and "tie" not in {distance_class, projection_class}
        and strong_distance_side
        and strong_projection_side
    ):
        raise ValidationError(
            "distance-based native-endpoint classification disagrees with donor projection"
        )
    return distance_class


def state_estimand_row(state: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    factorial = require_mapping(report, "factorial", source=f"report[{state['state_id']}]")
    row: dict[str, Any] = {
        "state_id": str(state["state_id"]),
        "task": str(state["task"]),
        "environment_seed": int(state["environment_seed"]),
        "branch_step": int(state["branch_step"]),
        "recipient_seed": int(state["recipient_seed"]),
        "donor_seed": int(state["donor_seed"]),
        "native_action_l2": float(report["native_action_l2"]),
    }
    for cell in EXPECTED_CELLS:
        payload = factorial[cell]
        for metric in CELL_METRICS:
            row[f"{cell}__{metric}"] = float(payload[metric])
    rr = row["recipient_future_recipient_kv__donor_projection"]
    dr = row["donor_future_recipient_kv__donor_projection"]
    dd = row["donor_future_donor_kv__donor_projection"]
    rd = row["recipient_future_donor_kv__donor_projection"]
    row["kv_effect_at_recipient_future"] = rd - rr
    row["kv_effect_at_donor_future"] = dd - dr
    row["future_effect_at_recipient_kv"] = dr - rr
    row["future_effect_at_donor_kv"] = dd - rd
    row["future_by_kv_interaction"] = (dd - dr) - (rd - rr)

    dr_class = _classify_closer(
        distance_to_recipient=row["donor_future_recipient_kv__distance_to_recipient"],
        distance_to_donor=row["donor_future_recipient_kv__distance_to_donor"],
        donor_projection=dr,
    )
    rd_class = _classify_closer(
        distance_to_recipient=row["recipient_future_donor_kv__distance_to_recipient"],
        distance_to_donor=row["recipient_future_donor_kv__distance_to_donor"],
        donor_projection=rd,
    )
    row["donor_future_recipient_kv_closer_to"] = dr_class
    row["recipient_future_donor_kv_closer_to"] = rd_class
    row["donor_future_recipient_kv_follows_kv"] = int(dr_class == "recipient")
    row["recipient_future_donor_kv_follows_kv"] = int(rd_class == "donor")
    row["donor_future_recipient_kv_tie"] = int(dr_class == "tie")
    row["recipient_future_donor_kv_tie"] = int(rd_class == "tie")
    row["both_crossed_arms_follow_kv"] = int(
        dr_class == "recipient" and rd_class == "donor"
    )
    row["crossed_arm_kv_follow_fraction"] = 0.5 * (
        row["donor_future_recipient_kv_follows_kv"]
        + row["recipient_future_donor_kv_follows_kv"]
    )
    if "tie" in {dr_class, rd_class}:
        row["crossed_arm_state_class"] = "tie_in_at_least_one_crossing"
    elif row["both_crossed_arms_follow_kv"]:
        row["crossed_arm_state_class"] = "both_follow_kv"
    elif not row["donor_future_recipient_kv_follows_kv"] and not row[
        "recipient_future_donor_kv_follows_kv"
    ]:
        row["crossed_arm_state_class"] = "both_follow_visible_future"
    else:
        row["crossed_arm_state_class"] = "mixed"
    return row


def estimand_definitions() -> dict[str, dict[str, str]]:
    definitions: dict[str, dict[str, str]] = {
        "native_action_l2": {
            "definition": "Euclidean distance between recipient-native and donor-native actions",
            "scale": "action_l2",
        }
    }
    for cell in EXPECTED_CELLS:
        for metric in CELL_METRICS:
            if metric == "donor_projection":
                definition = (
                    f"{CELL_ABBREVIATIONS[cell]} action projection on the recipient-to-donor native axis"
                )
                scale = "donor_projection"
            elif metric == "distance_to_recipient":
                definition = f"{CELL_ABBREVIATIONS[cell]} Euclidean action distance to recipient-native"
                scale = "action_l2"
            else:
                definition = f"{CELL_ABBREVIATIONS[cell]} Euclidean action distance to donor-native"
                scale = "action_l2"
            definitions[f"{cell}__{metric}"] = {"definition": definition, "scale": scale}
    for name, definition in CONTRAST_DEFINITIONS.items():
        definitions[name] = {"definition": definition, "scale": "donor_projection_difference"}
    for name, definition in FOLLOW_DEFINITIONS.items():
        definitions[name] = {"definition": definition, "scale": "proportion"}
    return definitions


def hierarchical_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    if resamples != 10_000:
        raise ValidationError(f"frozen bootstrap requires exactly 10000 draws, got {resamples}")
    tasks = sorted({str(row["task"]) for row in rows})
    if len(tasks) < 2:
        raise ValidationError("hierarchical bootstrap requires at least two tasks")
    arrays: list[np.ndarray] = []
    total_states = 0
    for task in tasks:
        task_rows = [row for row in rows if str(row["task"]) == task]
        if not task_rows:
            raise ValidationError(f"task {task!r} has no states")
        matrix = np.asarray(
            [[float(row[name]) for name in metric_names] for row in task_rows],
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape[1] != len(metric_names) or not np.isfinite(matrix).all():
            raise ValidationError(f"task {task!r} has nonfinite/malformed estimands")
        arrays.append(matrix)
        total_states += len(task_rows)
    if total_states != len(rows):
        raise ValidationError("task grouping lost states")

    generator = np.random.default_rng(seed)
    sampled_tasks = generator.integers(0, len(tasks), size=(resamples, len(tasks)))
    draws = np.zeros((resamples, len(metric_names)), dtype=np.float64)
    for slot in range(len(tasks)):
        selected = sampled_tasks[:, slot]
        for task_index, values in enumerate(arrays):
            mask = selected == task_index
            count = int(mask.sum())
            if count == 0:
                continue
            sampled_states = generator.integers(
                0, len(values), size=(count, len(values))
            )
            draws[mask] += values[sampled_states].mean(axis=1)
    draws /= len(tasks)
    task_mean_matrix = np.vstack([values.mean(axis=0) for values in arrays])
    state_matrix = np.asarray(
        [[float(row[name]) for name in metric_names] for row in rows], dtype=np.float64
    )
    summaries: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(metric_names):
        summaries[name] = {
            "tasks": len(tasks),
            "states": len(rows),
            "equal_task_mean": float(task_mean_matrix[:, index].mean()),
            "state_weighted_mean": float(state_matrix[:, index].mean()),
            "ci95": [
                float(np.quantile(draws[:, index], 0.025)),
                float(np.quantile(draws[:, index], 0.975)),
            ],
            "bootstrap_samples": resamples,
            "bootstrap_seed": seed,
        }
    return summaries, draws


def per_task_results(
    rows: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    tasks = sorted({str(row["task"]) for row in rows})
    for task in tasks:
        task_rows = [row for row in rows if str(row["task"]) == task]
        result[task] = {
            "states": len(task_rows),
            "means": {
                name: float(np.mean([float(row[name]) for row in task_rows]))
                for name in metric_names
            },
        }
    return result


def leave_one_task_out_results(
    rows: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    tasks = sorted({str(row["task"]) for row in rows})
    per_task = per_task_results(rows, metric_names)
    output: dict[str, dict[str, Any]] = {}
    for held_out in tasks:
        remaining = [task for task in tasks if task != held_out]
        if not remaining:
            raise ValidationError("leave-one-task-out requires at least two tasks")
        output[held_out] = {
            "remaining_tasks": len(remaining),
            "remaining_states": sum(per_task[task]["states"] for task in remaining),
            "equal_task_means": {
                name: float(np.mean([per_task[task]["means"][name] for task in remaining]))
                for name in metric_names
            },
        }
    return output


def follow_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def counts_for(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        classifications = Counter(str(row["crossed_arm_state_class"]) for row in subset)
        return {
            "states": len(subset),
            "crossed_arms": 2 * len(subset),
            "donor_future_recipient_kv_follows_recipient_kv": int(
                sum(int(row["donor_future_recipient_kv_follows_kv"]) for row in subset)
            ),
            "recipient_future_donor_kv_follows_donor_kv": int(
                sum(int(row["recipient_future_donor_kv_follows_kv"]) for row in subset)
            ),
            "crossed_arms_follow_kv": int(
                sum(
                    int(row["donor_future_recipient_kv_follows_kv"])
                    + int(row["recipient_future_donor_kv_follows_kv"])
                    for row in subset
                )
            ),
            "donor_future_recipient_kv_ties": int(
                sum(int(row["donor_future_recipient_kv_tie"]) for row in subset)
            ),
            "recipient_future_donor_kv_ties": int(
                sum(int(row["recipient_future_donor_kv_tie"]) for row in subset)
            ),
            "state_class_counts": dict(sorted(classifications.items())),
        }

    tasks = sorted({str(row["task"]) for row in rows})
    return {
        "definition": (
            "A crossed arm follows K/V when its action is closer in Euclidean distance "
            "to the native action associated with the K/V source than to the native "
            "action associated with the visible-future source; exact numerical ties "
            "are reported separately. The donor-projection 0.5 boundary is checked "
            "for classification consistency."
        ),
        "overall": counts_for(rows),
        "per_task": {
            task: counts_for([row for row in rows if str(row["task"]) == task])
            for task in tasks
        },
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def write_latex(
    path: Path,
    aggregates: Mapping[str, Mapping[str, Any]],
    counts: Mapping[str, Any],
) -> None:
    rows = [
        ("R future / R K/V", "recipient_future_recipient_kv__donor_projection"),
        ("D future / R K/V", "donor_future_recipient_kv__donor_projection"),
        ("D future / D K/V", "donor_future_donor_kv__donor_projection"),
        ("R future / D K/V", "recipient_future_donor_kv__donor_projection"),
        ("K/V effect at R future", "kv_effect_at_recipient_future"),
        ("K/V effect at D future", "kv_effect_at_donor_future"),
        ("Future x K/V interaction", "future_by_kv_interaction"),
    ]
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Estimand & Equal-task mean & 95\% hierarchical bootstrap CI \\",
        r"\midrule",
    ]
    for label, name in rows:
        result = aggregates[name]
        low, high = result["ci95"]
        lines.append(
            f"{latex_escape(label)} & {result['equal_task_mean']:.3f} & "
            f"[{low:.3f}, {high:.3f}] \\\\" 
        )
    overall = counts["overall"]
    lines.extend(
        [
            r"\midrule",
            (
                r"Crossed arms following K/V & "
                f"{overall['crossed_arms_follow_kv']}/{overall['crossed_arms']} & -- \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.flush()
        os.fsync(handle.fileno())


def save_plot(
    path: Path,
    aggregates: Mapping[str, Mapping[str, Any]],
    counts: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell_names = [f"{cell}__donor_projection" for cell in EXPECTED_CELLS]
    cell_labels = ["R/R", "D/R", "D/D", "R/D"]
    contrast_names = [
        "kv_effect_at_recipient_future",
        "kv_effect_at_donor_future",
        "future_by_kv_interaction",
    ]
    contrast_labels = ["K/V @ R-future", "K/V @ D-future", "Interaction"]

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    for axis, names, labels, color in (
        (axes[0], cell_names, cell_labels, "#4C78A8"),
        (axes[1], contrast_names, contrast_labels, "#F58518"),
    ):
        means = np.asarray([aggregates[name]["equal_task_mean"] for name in names])
        lows = np.asarray([aggregates[name]["ci95"][0] for name in names])
        highs = np.asarray([aggregates[name]["ci95"][1] for name in names])
        positions = np.arange(len(names))
        axis.vlines(positions, lows, highs, color=color, linewidth=1.4)
        axis.scatter(positions, means, color=color, s=26, zorder=3)
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Four factorial cells")
    axes[0].set_ylabel("Donor projection (equal-task mean, 95% CI)")
    axes[1].set_title("K/V contrasts")
    overall = counts["overall"]
    figure.suptitle(
        "Cosmos predicted-future × future-K/V factorial\n"
        f"K/V-following crossed arms: {overall['crossed_arms_follow_kv']}/"
        f"{overall['crossed_arms']}",
        fontsize=10,
    )
    figure.savefig(path, dpi=220, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(figure)


def emit_outputs(
    *,
    summary_dir: Path,
    manifest: Mapping[str, Any],
    manifest_sha: str,
    output_root: Path,
    loaded: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], str]],
) -> list[str]:
    analysis = require_mapping(manifest, "analysis", source="manifest")
    bootstrap_samples = int(analysis["bootstrap_samples"])
    bootstrap_seed = int(analysis["bootstrap_seed"])
    rows = [state_estimand_row(state, report) for state, report, _ in loaded]
    definitions = estimand_definitions()
    metric_names = tuple(definitions)
    aggregates, _draws = hierarchical_bootstrap(
        rows,
        metric_names,
        resamples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    for name in metric_names:
        aggregates[name].update(definitions[name])
    per_task = per_task_results(rows, metric_names)
    loto = leave_one_task_out_results(rows, metric_names)
    counts = follow_counts(rows)

    summary_parent = summary_dir.parent.resolve(strict=True)
    try:
        summary_parent.relative_to(HOST_NFS_ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValidationError(f"summary directory parent must be under {HOST_NFS_ROOT}") from error
    if os.path.lexists(summary_dir):
        raise FileExistsError(f"refusing to overwrite summary directory: {summary_dir}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{summary_dir.name}.staging-", dir=summary_parent)
    )
    emitted: list[str] = []
    try:
        state_fields = list(rows[0])
        state_csv = staging / "cosmos3_existing_kv_state_estimands.csv"
        write_csv(state_csv, rows, state_fields)
        emitted.append(state_csv.name)

        aggregate_rows = [
            {
                "estimand": name,
                "scale": aggregates[name]["scale"],
                "definition": aggregates[name]["definition"],
                "tasks": aggregates[name]["tasks"],
                "states": aggregates[name]["states"],
                "equal_task_mean": aggregates[name]["equal_task_mean"],
                "state_weighted_mean": aggregates[name]["state_weighted_mean"],
                "ci95_low": aggregates[name]["ci95"][0],
                "ci95_high": aggregates[name]["ci95"][1],
                "bootstrap_samples": aggregates[name]["bootstrap_samples"],
                "bootstrap_seed": aggregates[name]["bootstrap_seed"],
            }
            for name in metric_names
        ]
        aggregate_csv = staging / "cosmos3_existing_kv_aggregate.csv"
        write_csv(aggregate_csv, aggregate_rows, list(aggregate_rows[0]))
        emitted.append(aggregate_csv.name)

        task_rows = [
            {"task": task, "states": payload["states"], **payload["means"]}
            for task, payload in per_task.items()
        ]
        task_csv = staging / "cosmos3_existing_kv_per_task.csv"
        write_csv(task_csv, task_rows, list(task_rows[0]))
        emitted.append(task_csv.name)

        loto_rows = [
            {
                "held_out_task": task,
                "remaining_tasks": payload["remaining_tasks"],
                "remaining_states": payload["remaining_states"],
                **payload["equal_task_means"],
            }
            for task, payload in loto.items()
        ]
        loto_csv = staging / "cosmos3_existing_kv_leave_one_task_out.csv"
        write_csv(loto_csv, loto_rows, list(loto_rows[0]))
        emitted.append(loto_csv.name)

        count_rows = [{"task": "ALL", **counts["overall"]}]
        count_rows.extend(
            {"task": task, **payload} for task, payload in counts["per_task"].items()
        )
        # Serialize the nested class census in the otherwise-flat CSV.
        flattened_count_rows = []
        for payload in count_rows:
            item = dict(payload)
            item["state_class_counts"] = json.dumps(
                item["state_class_counts"], sort_keys=True, separators=(",", ":")
            )
            flattened_count_rows.append(item)
        counts_csv = staging / "cosmos3_existing_kv_follow_counts.csv"
        write_csv(counts_csv, flattened_count_rows, list(flattened_count_rows[0]))
        emitted.append(counts_csv.name)

        latex_path = staging / "cosmos3_existing_kv_table.tex"
        write_latex(latex_path, aggregates, counts)
        emitted.append(latex_path.name)

        plot_path = staging / "cosmos3_existing_kv_summary.png"
        save_plot(plot_path, aggregates, counts)
        emitted.append(plot_path.name)

        report_hashes = {
            str(state["state_id"]): report_sha for state, _, report_sha in loaded
        }
        summary = {
            "status": "complete",
            "study": {
                "manifest_id": manifest["manifest_id"],
                "manifest_sha256": manifest_sha,
                "runner_sha256": FROZEN_RUNNER_SHA256,
                "scope": manifest["scope"],
                "output_root": str(output_root),
                "states_expected": int(manifest["evaluation_state_count"]),
                "states_analyzed": len(rows),
                "tasks_analyzed": len(per_task),
                "complete": True,
                "report_sha256": report_hashes,
            },
            "analysis_contract": {
                "independent_unit": analysis["independent_unit"],
                "within_state_measurements": analysis["within_state_measurements"],
                "bootstrap": analysis["bootstrap"],
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_implementation": (
                    "one shared deterministic schedule: sample T tasks with replacement; "
                    "for each sampled task, sample its n_t states with replacement; average "
                    "state means within sampled task slots, then average the T slots"
                ),
                "population_mean_weighting": "equal weight per task",
            },
            "estimand_definitions": definitions,
            "state_estimands": rows,
            "aggregate": aggregates,
            "per_task": per_task,
            "leave_one_task_out": loto,
            "kv_follow_counts": counts,
            "artifacts": sorted([*emitted, "cosmos3_existing_kv_summary.json"]),
        }
        json_path = staging / "cosmos3_existing_kv_summary.json"
        write_json(json_path, summary)
        emitted.append(json_path.name)
        for path in staging.iterdir():
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"summary artifact is missing/empty: {path}")
        descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if os.path.lexists(summary_dir):
            raise FileExistsError(f"refusing to overwrite summary directory: {summary_dir}")
        os.rename(staging, summary_dir)
        parent_descriptor = os.open(summary_parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return sorted(emitted)


def main() -> None:
    args = parse_args()
    manifest_path = _under_host_root(args.manifest, source="manifest")
    manifest, manifest_sha = validate_manifest(manifest_path, verify_inputs=True)
    if manifest["manifest_id"] != FROZEN_MANIFEST_ID or manifest_sha != FROZEN_MANIFEST_SHA256:
        raise ValidationError("analyzer only accepts the frozen v3 manifest")
    output_root = _under_host_root(args.output_root, source="output root")
    summary_dir = args.summary_dir
    if not summary_dir.is_absolute():
        raise ValidationError("--summary-dir must be absolute")
    if os.path.lexists(summary_dir):
        raise FileExistsError(f"refusing to overwrite summary directory: {summary_dir}")
    host_to_container(summary_dir, must_exist=False, source="summary directory")

    # Inventory and receipt hashes are checked for all 21 states before any
    # factorial value is loaded, preventing accidental partial summaries.
    inventory = inventory_complete_run(
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha=manifest_sha,
        output_root=output_root,
    )
    loaded = validate_and_load_reports(
        manifest=manifest,
        manifest_sha=manifest_sha,
        inventory=inventory,
    )
    emitted = emit_outputs(
        summary_dir=summary_dir,
        manifest=manifest,
        manifest_sha=manifest_sha,
        output_root=output_root,
        loaded=loaded,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest_id": manifest["manifest_id"],
                "states": len(loaded),
                "summary_dir": str(summary_dir),
                "artifacts": emitted,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, FileExistsError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
