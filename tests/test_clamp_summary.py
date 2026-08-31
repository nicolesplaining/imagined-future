from __future__ import annotations

import json

import pytest

from imagined_future.clamp_summary import summarize_clamp_runs


def _write_run(path, *, deterministic=True, baseline_error=0.0, with_execution=True) -> None:
    path.mkdir()
    (path / "summary.json").write_text(
        json.dumps(
            {
                "recipient_seed": 195,
                "donor_seed": 198,
                "modalities": ["wrist"],
                "future_noise_seed": 20195,
                "deterministic_tokenizer": deterministic,
                "baseline_reference_max_abs_error": baseline_error,
                "donor_steering_effect": 0.25,
                "selected_clean_latent_l2": {
                    "recipient_norm": 10.0,
                    "donor_norm": 10.1,
                    "donor_minus_recipient": 2.0,
                },
            }
        )
    )
    if with_execution:
        (path / "execution_analysis.json").write_text(
            json.dumps(
                {
                    "donor_minus_recipient_clamp": {
                        "state_donor_steering": 0.2,
                        "primary_pixel_donor_preference": 0.3,
                    }
                }
            )
        )


def test_summarizes_valid_deterministic_run(tmp_path) -> None:
    run = tmp_path / "run"
    _write_run(run)

    assert summarize_clamp_runs([run], require_deterministic=True) == [
        {
            "run": "run",
            "recipient_seed": 195,
            "donor_seed": 198,
            "modalities": "wrist",
            "future_noise_seed": 20195,
            "deterministic_tokenizer": True,
            "action_donor_steering_effect": 0.25,
            "executed_state_donor_steering_effect": 0.2,
            "endpoint_primary_donor_preference_effect": 0.3,
            "recipient_latent_norm": 10.0,
            "donor_latent_norm": 10.1,
            "donor_minus_recipient_latent_l2": 2.0,
        }
    ]


@pytest.mark.parametrize(
    ("deterministic", "baseline_error", "match"),
    [(False, 0.0, "deterministic"), (True, 0.01, "baseline")],
)
def test_rejects_ineligible_run(tmp_path, deterministic, baseline_error, match) -> None:
    run = tmp_path / "run"
    _write_run(run, deterministic=deterministic, baseline_error=baseline_error)

    with pytest.raises(ValueError, match=match):
        summarize_clamp_runs([run], require_deterministic=True)
