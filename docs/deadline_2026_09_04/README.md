# DreamZero and LingBot-VA deadline handoff

## Outcome

Both external-model evaluations are positive for the narrow causal claim:

> With the present input and recipient action randomness fixed, replacing the
> native predicted-future representation redirects the predicted action chunk
> toward the action paired with that source representation.

This extends the directional action-control result beyond Cosmos 3 to two
released systems with different inference organizations. It does not establish
that the models naturally compare futures, plan over consequences, or improve
closed-loop task success.

| System | Frozen cohort | Primary off-diagonal source retrieval | Geometry | Dose response |
|---|---|---|---|---|
| DreamZero | 30 states from 30 unique DROID episodes and instruction strings in its released preprocessed training-data corpus; four branches/state | 360/360 (100%); within-state label-permutation `p=1/100001` | Distance reduction 0.917 [0.897, 0.933]; projection 0.990 [0.984, 0.995]; cosine 0.995 [0.991, 0.997]; orthogonal residual 0.080 [0.065, 0.099] | Interior responses 0.046/.465/.939 at alpha .25/.5/.75; 30/30 interior triples nondecreasing; slope 1.786 [1.707, 1.853], `p=1/100001` |
| LingBot-VA | 30 predetermined LIBERO-10 states across ten tasks; four branches/state | 269/360 = 0.747 [0.667, 0.819]; within-state label-permutation `p=1/100001` | Distance reduction 0.473 [0.421, 0.524]; projection 0.675 [0.624, 0.723]; cosine 0.814 [0.777, 0.847]; orthogonal residual 0.388 [0.359, 0.418] | Responses 0/.221/.487/.758/1 at alpha 0/.25/.5/.75/1; 30/30 interior triples nondecreasing; slope 1.073 [1.039, 1.109], `p=1/100001` |

DreamZero's result is for the full 24-step action chunk. A post-hoc secondary
first-step audit remains strong at 348/360 source retrieval (0.967), projection 0.950,
and distance reduction 0.775. Its inputs are analysis-frozen and selected
without model-output or outcome filtering, but come from the model's released
training-data corpus and therefore do not test model-held-out generalization.

LingBot's result is for three post-conditioning frame groups of the predicted
action chunk, not physical execution. In post-hoc secondary temporal audits, the first
post-conditioning frame group is near chance (98/360 = 0.272), and the literal
first low-level action is below chance (48/360 = 0.133); later steps drive its
chunk-level result. Its crossed future/cache cells are exact interface-routing
controls: the action follows the installed future-derived cache identity. They
are not a separately discovered hidden mechanism.

Both models are sensitive to norm-matched Gaussian replacement, so Gaussian is
not an inert control. The completed LingBot 30-state x 4-source Gaussian grid
sharpens the comparison: Gaussian-source retrieval is 42/360 = 0.117 [0.094,
0.139], and native-donor alignment is 78/360 = 0.217 [0.192, 0.239], neither
above 25% chance. Recipient identity is retained in only 126/360 = 0.350 cells.
Thus, in this exploratory cohort, arbitrary sources disrupted action but did
not recover source identity or native-donor alignment above chance, unlike the
source-specific routing observed for native futures.

The two dose curves use different normalizations. DreamZero projects onto the
native recipient-to-native donor action axis. LingBot projects onto the
observed alpha-0-to-alpha-1 predicted-action endpoint segment, so its displayed
endpoints are 0 and 1 by construction. Compare within-model graded response,
not the numerical slopes across models.

## Paper throughline

1. Cosmos 3 establishes broad, source-specific directional steering.
2. DreamZero shows that the result is not peculiar to Cosmos 3: it is nearly
   exact in a chunk-autoregressive model that jointly denoises video and action
   within each chunk.
3. LingBot-VA shows that simultaneous video/action denoising during the action
   stage is not required: its released inference first predicts video and then
   generates actions through a future-derived cache.
4. The graded DreamZero and LingBot responses show that control varies with
   intervention strength rather than appearing only at a full replacement.
5. Cosmos 3's crossed future x K/V experiment remains the internal-mechanism
   centerpiece. LingBot's cache cross validates its documented interface, not
   an analogous hidden-circuit discovery.

The safest concise cross-model sentence is:

> Across released world-action-model systems with joint and ordered
> video-to-action inference organizations, native predicted-future source
> identity exerts source-specific directional control over predicted action
> chunks under transplantation.

The external cohorts establish distinct native latent sources, not systematic
semantic or visual differences across all branches. The decoded examples are
descriptive media only.

## Verified artifacts

- Heavy-output location and Git/Lambda storage split:
  [lambda_storage.md](lambda_storage.md)
- Machine-readable deadline freeze:
  [final_deadline_receipt.json](../../output/deadline_2026_09_04/final_deadline_receipt.json)
- Deadline-level result and requirement synthesis:
  [noon_final_results.md](noon_final_results.md)
- Machine-readable cross-model tables:
  [core results](../../output/deadline_2026_09_04/cross_model_results_table.csv) and
  [dose response](../../output/deadline_2026_09_04/cross_model_dose_table.csv)
- Full scientific synthesis and copy-ready claim language:
  [claim_safe_synthesis.md](claim_safe_synthesis.md)
- Independent adversarial audit and GO/NO-GO boundaries:
  [final_science_adversarial_audit.md](final_science_adversarial_audit.md)
- Reproducible final consistency checker:
  [verify_deadline_2026_09_04_handoff.py](../../scripts/verify_deadline_2026_09_04_handoff.py)
- DreamZero results:
  [audit report](dreamzero_audit_results.md),
  [remaining future × K/V implementation protocol](dreamzero_future_kv_factorial_protocol.md),
  [core summary](../../output/deadline_2026_09_04/dreamzero/core_analysis_final/summary.json),
  [control summary](../../output/deadline_2026_09_04/dreamzero/control_analysis/summary.json),
  [provenance reconciliation](dreamzero_provenance_reconciliation.md), and
  [corrective runtime-provenance addendum](../../output/deadline_2026_09_04/dreamzero/provenance_addendum/runtime_provenance_addendum_v1.json), and
  [median-effect representative media](../../output/deadline_2026_09_04/dreamzero/representative_media/), and
  [exhaustive all-120 native media](../../output/deadline_2026_09_04/dreamzero/all_native_media/), and
  [selection-neutral exhaustive overview](../../output/deadline_2026_09_04/dreamzero/all_native_media_derived/)
- LingBot final immutable package:
  [package README](../../output/deadline_2026_09_04/lingbot_final_package_v1/README.md),
  [umbrella index](../../output/deadline_2026_09_04/lingbot_final_package_v1/artifact_index.json),
  [core summary](../../output/deadline_2026_09_04/lingbot_final_package_v1/core_artifacts/summary.json),
  [dose summary](../../output/deadline_2026_09_04/lingbot_final_package_v1/dose_analysis/summary.json), and
  [selection-neutral all-state video](../../output/deadline_2026_09_04/lingbot_final_package_v1/core_artifacts/media/all_states_overview.mp4)
- LingBot raw-result addendum:
  [read-only raw receipt](../../output/deadline_2026_09_04/lingbot_final_raw_addendum_v2/raw_addendum_receipt.json) and
  [recursive artifact index](../../output/deadline_2026_09_04/lingbot_final_raw_addendum_v2/artifact_index.json)
- LingBot exhaustive decoded-future audit:
  [provenance](../../output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1/provenance.json),
  [all-120 terminal-frame sheet](../../output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1/media/all_120_terminal_frames_contact_sheet.png), and
  [all-30-state overview video](../../output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1/media/all_30_states_all_4_branches.mp4)
- LingBot decoded-future execution-history addendum:
  [receipt](../../output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1_execution_addendum_v2/execution_receipt.json) and
  [artifact index](../../output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1_execution_addendum_v2/artifact_index.json)
- LingBot Gaussian-latent provenance addendum:
  [receipt](../../output/deadline_2026_09_04/lingbot_gaussian_latent_provenance_addendum_v1/receipt.json) and
  [artifact index](../../output/deadline_2026_09_04/lingbot_gaussian_latent_provenance_addendum_v1/artifact_index.json)
- LingBot complete four-source Gaussian routing audit:
  [summary](../../output/deadline_2026_09_04/lingbot_gaussian_grid_v2_final/summary.json),
  [execution receipt](../../output/deadline_2026_09_04/lingbot_gaussian_grid_v2_final/execution_receipt.json), and
  [artifact index](../../output/deadline_2026_09_04/lingbot_gaussian_grid_v2_final/artifact_index.json)

LingBot umbrella index SHA-256:
`698a818bce0f8dbeda22aee7df76752dce70f1d70089c6dd79cedf3e3faf273e`.
All 140 indexed files independently passed hash and size verification, and the
local package matches its NFS mirror exactly.

The LingBot raw-result addendum contains the previously omitted complete core
and dose raw trees. Its receipt SHA-256 is
`6208bd9109c29e00b1e5b8c6ce3b4f7c3c40fd8b5b87f4e2e14017370bfdb779`;
its 275-entry artifact-index SHA-256 is
`0e6ca53c3403434518a733fdc6d03532282be4941464a8d667b6ab1b3e9f62fb`.
All indexed files are read-only and were bytewise verified against their
preserved NFS sources.

The exhaustive LingBot decode includes all 120 native latent futures, with no
visual or outcome selection. Its umbrella artifact-index SHA-256 is
`f89a96b3b12c35e25cc121284c84dc83de1edb2c8cfc28b3e0faadcaa6c3b332`;
all 366 indexed shard files and seven umbrella files passed independent local
hash/size verification.

The Gaussian-latent addendum reconstructs and archives all 30 x 4 exact
norm-matched controls used by the core runner. All 120 present frames,
deterministic repeats, and saved reloads are bitwise exact, with zero
discrepancies. Receipt SHA-256:
`87fec43a24788f6aa4ca21f054c264383a42d9cb75fdf0ff0c8f9b8aae54393f`;
artifact-index SHA-256:
`77848b2450246dfa1b00d0ee3d0a8e182caab74c7c807da0c5ef0bc38c134983`.

DreamZero's post-analysis media receipt SHA-256:
`b51849ea718b25bb1474e1b7c8c2040bac1c6af9673674db3ea10348ed71c2a7`.
Each of the four decoded-video re-runs reproduced its frozen native action
bit-for-bit.

The exhaustive DreamZero media package contains every 30-state x four-branch
native future in manifest order. All 120 action arrays are bit-exact and all
120 regenerated latent traces are byte-exact to the frozen core. Receipt
SHA-256:
`89c7f4742b42698d2a2cb122f63c5792d6e53da070280c019ad18a1fb3acacdb`;
artifact-index SHA-256:
`8907b7f854f7ea5217cfdce842bb56d2cc86649fd2fa53eed11480966f5f5aa6`;
392 indexed files, zero failures.

The derived DreamZero overview includes all 30 x 4 branches in frozen manifest
order. Its 54-second overview-video SHA-256 is
`1f6b76c3c3de34956bfc290fba114190b11d7258e8ac57314920375cb9818e19`,
terminal-sheet SHA-256 is
`0a453a952a982041c7c3307576c921caadb99d7aa615a12229ad863190cc0a46`,
receipt SHA-256 is
`80d0180c3df8fa88e715c7869c1345661c2fd4d85dbd7d062dd9d326cf6b533b`,
and artifact-index SHA-256 is
`85b551a45f5561b3bc71370cf4d24c9fd07992eb6528651673872854cb106a0e`.

Canonical NFS roots:

- DreamZero raw core: `/lambda/nfs/imagined-future/results/deadline_2026_09_04/dreamzero_eval_v1`
- DreamZero controls: `/lambda/nfs/imagined-future/results/deadline_2026_09_04/dreamzero_gaussian_control_v1` and `/lambda/nfs/imagined-future/results/deadline_2026_09_04/dreamzero_dose_v1`
- DreamZero exhaustive native media: `/lambda/nfs/imagined-future/results/deadline_2026_09_04/dreamzero_all_native_media_v1`
- DreamZero exhaustive derived overview: `/lambda/nfs/imagined-future/results/deadline_2026_09_04/dreamzero_all_native_media_derived_v1`
- LingBot immutable package: `/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_final_package_v1`
- LingBot immutable raw addendum: `/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_final_raw_addendum_v2`
- LingBot Gaussian-latent provenance: `/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_gaussian_latent_provenance_addendum_v1`
- LingBot decoded-future execution-history addendum: `/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_native_future_decode_all120_v1_execution_addendum_v2`
- LingBot Gaussian-source shards and analysis: `/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_gaussian_grid_v2_shard0`, `/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_gaussian_grid_v2_shard1`, and `/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_gaussian_grid_v2_final`
- DreamZero clean-upstream parity audit: `/lambda/nfs/imagined-future/results/deadline_2026_09_04/dreamzero_upstream_native_parity_v1`; local [execution receipt](../../output/deadline_2026_09_04/dreamzero/upstream_native_parity/execution_receipt.json)
- [DreamZero future x K/V feasibility audit](dreamzero_kv_factorial_feasibility.md): exact intervention site, donor-action leakage confound, and audited follow-up design

## Compute ownership and repository scope

The only dedicated instances used for this work are:

- `if-overnight-external-wams` (`68.209.72.187`): DreamZero, 2 x H100
- `if-overnight-robolab-clients` (`68.209.75.174`): LingBot-VA, 2 x H100

None of the four `nla-*` instances shown in the user's screenshot belong to
this run, and they were not accessed or modified. The loaded Cosmos services
were not disrupted. Nicole's manuscript was not edited or pushed.

At 11:50 PT, all experimental workers were closed after local verification.
Both dedicated nodes remained running but showed zero GPU processes, 0 MiB
allocated on all four H100s, and 0% utilization. See the machine-readable
[compute closure receipt](../../output/deadline_2026_09_04/compute_closure.json).
