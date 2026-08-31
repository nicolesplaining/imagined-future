"""Validated tabular summaries of semantic-clamp artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def summarize_clamp_runs(
    run_dirs: Iterable[Path], *, require_deterministic: bool = False
) -> list[dict[str, Any]]:
    """Load clamp runs and return one flat, validation-aware row per run."""

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        if summary.get("baseline_reference_max_abs_error") != 0.0:
            raise ValueError(f"{run_dir} does not exactly reproduce its baseline reference")
        deterministic = bool(summary.get("deterministic_tokenizer", False))
        if require_deterministic and not deterministic:
            raise ValueError(f"{run_dir} does not record deterministic tokenization")
        execution_candidates = (
            run_dir / "execution_analysis_with_proprio.json",
            run_dir / "execution_analysis.json",
        )
        execution_path = next((path for path in execution_candidates if path.exists()), None)
        execution = json.loads(execution_path.read_text()) if execution_path else None
        execution_effects = execution["donor_minus_recipient_clamp"] if execution else {}
        latent = summary.get("selected_clean_latent_l2", {})
        rows.append(
            {
                "run": run_dir.name,
                "recipient_seed": int(summary["recipient_seed"]),
                "donor_seed": int(summary["donor_seed"]),
                "modalities": "+".join(summary.get("modalities", ["unspecified"])),
                "future_noise_seed": int(summary["future_noise_seed"]),
                "deterministic_tokenizer": deterministic,
                "action_donor_steering_effect": float(summary["donor_steering_effect"]),
                "executed_state_donor_steering_effect": execution_effects.get("state_donor_steering"),
                "endpoint_primary_donor_preference_effect": execution_effects.get(
                    "primary_pixel_donor_preference"
                ),
                "endpoint_proprio_donor_steering_effect": execution_effects.get(
                    "proprio_donor_steering"
                ),
                "recipient_latent_norm": latent.get("recipient_norm"),
                "donor_latent_norm": latent.get("donor_norm"),
                "donor_minus_recipient_latent_l2": latent.get("donor_minus_recipient"),
            }
        )
    return rows
