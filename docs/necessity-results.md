# Future-to-action attention necessity pilot

Status: exploratory localization at the selected task-4/state-2 unit; not a population estimate.

## Intervention

Cosmos Policy's DiT uses bidirectional self-attention over flattened temporal-spatial tokens. At selected public DiT blocks, the intervention recomputes only action-frame queries while excluding future-frame keys and values. Current-observation keys, action-frame keys, language cross-attention, tokenwise MLPs, diffusion noise, and every unselected query remain unchanged. The implementation calls the checkpoint's configured public [`MinimalA2AAttnOp`](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/_src/predict2/networks/a2a_cp.py) and output projection rather than substituting a new attention implementation.

For every ablation, an all-key control recomputes the same action queries with every key retained. Across all tested layer sets, this control reproduces the baseline action bitwise and produces identical simulator endpoints. This isolates key removal from query-batching or backend changes.

## Calibration and localization

Removing future keys in all 28 blocks is a destructive stress test: normalized action L2 from baseline is `1.3667`, compared with `0.0689` between the matched natural failure and success branches. The executed endpoint is correspondingly off-manifold, with proprio donor steering `-20.72`. Removing future keys from only blocks 21--27 remains too large (action L2 `1.2071`). Neither intervention is interpreted as localized mediation evidence.

Single-block interventions are commensurate with natural variation:

| Block | Action L2 from baseline | Action donor steering | Endpoint proprio steering | Full-state steering | Endpoint-image preference |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0218 | -0.1511 | -0.1261 | +0.0564 | +0.1206 |
| 27 | 0.0440 | -0.2215 | -0.2378 | -0.1890 | +0.1523 |
| 27, current-key control | 0.0885 | -0.2492 | +0.3699 | +0.5204 | +0.2050 |

The reference donor is the robust-success seed-198 branch; the baseline is the robust-failure seed-195 branch. Negative action and endpoint-proprioception scores mean that removing future-to-action communication moves behavior away from the successful donor. The final-block result is directionally consistent across normalized action, endpoint proprioception, and the full simulator state. The endpoint-image preference is positive for both blocks, illustrating why a single image-distance metric is insufficient for robot behavior.

The equal-count control removes the three current-state key frames at block 27 while retaining the three future frames. It causes a larger raw action displacement and a similar negative action projection, so future keys are not uniquely dominant in action space. Its executed endpoint nevertheless moves toward the successful donor in proprioception and full-state space, opposite to future-key removal. The adverse behavioral direction of the future ablation is therefore not explained by key count or raw action magnitude alone.

At the independent task-8 semantic-null state, final-block future-key removal changes the action (L2 `0.0692`) but has little donor-aligned action projection (`+0.0403`). Its executed proprio endpoint moves donorward, but that donor does not have a robust success/failure label. This confirms an active structural path while reinforcing that semantic and outcome-aligned use is state-dependent.

## Interpretation

This is localized necessity evidence for an active future-to-action computational path at the positive task-4 state. Together with semantic future clamping, it is harder to explain by a pure observation-to-action shortcut with an epiphenomenal prediction head. It does not by itself prove that the removed information is a coherent semantic plan: key removal also changes attention normalization and deletes all future-token content at the selected block. The layer was selected exploratorily, and the result requires replication on held-out states before a confirmatory claim.

The versioned table is [results/task4_state2_attention_ablation.csv](../results/task4_state2_attention_ablation.csv).
