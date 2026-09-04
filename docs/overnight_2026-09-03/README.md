# Submission evidence package — 2026-09-03/04

Status: **complete.** Six experiments were frozen before outcome inspection,
completed, and independently audited. This package does not edit or push the
manuscript.

## Start here

1. `final_artifact_index.md` — exact paths and cryptographic identities.
2. `manuscript_ready_results.md` — compact tables and insert-ready prose.
3. `submission_update.md` — full designs, results, scope, and artifact hashes.
4. `reviewer_readout.md` — adversarial ranking and five-page placement advice.
5. `run_ledger.tsv` and `infrastructure_ledger.md` — execution provenance.

## Completed evidence

| Experiment | Population | Audited headline |
|---|---:|---|
| Cosmos 3 selection-free multi-donor | 90 fixed archival states, six tasks | 1,440/1,440 correct future-source cells; distance reduction 0.763 [0.733, 0.792] |
| Cosmos 3 future × K/V | 21 existing states, six tasks | All 42 crossed arms followed K/V; K/V effects 0.895 and 0.885 |
| FastWAM Optional-IDM replication | 120 untouched states, all 40 LIBERO tasks | 1,919/1,920 correct future-source cells; distance reduction 0.683 [0.638, 0.738] |
| FastWAM future × cache | Same 120 states | Donor cache rescued 1,439/1,440 donor comparisons; explicit-interface control |
| Cosmos 3 single-call timing | 30 fixed states, six tasks | Every isolated solver call steered; calls 2/3 were numerically similar to all-call steering |
| Cosmos 3 future-strength dose | 30 fixed states, six tasks | Distance-reduction slope 0.899 [0.852, 0.936]; mean increased at all four adjacent mixture steps |

Every interval uses the hierarchy frozen for its study. The Cosmos archival
selection-free study uses task→episode→state resampling; FastWAM uses
suite→task→state; the other Cosmos studies use task→state. Donor directions and
diffusion repeats remain within-state measurements.

## Five-page recommendation

Lead with the selection-free Cosmos evaluation and the Cosmos future × K/V
factorial. Add the dose slope and the FastWAM replication as one sentence each
if space permits. Put the full dose profile, timing study, FastWAM population
tables, FastWAM cache-interface control, per-task results, and provenance in the
appendix. This preserves a clean discovery arc:

1. coherent future identity selects the corresponding action without pair
   selection;
2. future-token K/V can rescue and redirect that action;
3. the mean response is graded with imposed future strength;
4. the phenomenon extends to a released future-before-action architecture.

## Scope and provenance

- The new cohorts measure predicted action chunks. Existing manuscript endpoint
  claims must stay attached to the paper's executed-endpoint experiments.
- The Cosmos selection-free cohort is fixed archival data with lossy observation
  reconstruction; it is not described as fresh held-out physical evaluation.
- The K/V factorial supports a large causal contribution and conditional rescue,
  not strict necessity or complete mediation.
- The dose result supports a graded task-weighted mean under imposed all-call
  interpolation; 257/360 individual ordered-pair profiles were nondecreasing.
- FastWAM latent and cache transplantation are two access points to one explicit
  future-to-action cache pathway, not independent replications.

The attempted fresh RoboLab execution lane is preserved in the ledgers; Isaac
was incompatible with the available H100 client, so it was not silently
substituted into the scientific cohort. All provisioned instances were
intentionally left running. No instance was terminated, no scientific raw
output was deleted, and no manuscript file was modified or pushed by this goal.

## Original frozen plan

The plan below is retained as provenance. Superseding decisions, failed gates,
and justified substitutions are recorded in `decision_ledger.md` and
`run_ledger.tsv`.

## Objective

Within 12--16 hours, produce the strongest submission-ready experimental update without editing the manuscript. The required evidence package is:

1. A frozen, selection-free, multi-donor Cosmos 3 evaluation.
2. A complete future-target x future-K/V suppression-and-rescue factorial in Cosmos 3.
3. A smoke-gated replication in a released non-Cosmos model that natively generates a future before predicting actions.

The frozen numerical contract is in `configs/overnight_2026-09-03.toml`. Outcome-dependent changes require a new study ID and cannot silently replace this design.

## Priority and stop rules

### P0: Cosmos 3 selection-free multi-donor evaluation

Run all 48 frozen task-by-environment-seed states. The four branch seeds are fixed by seed order rather than native separation. In action space, test all 12 ordered recipient-to-donor pairs and average those repeated measurements within state. The primary result is four-way donor identification, with distance reduction and orthogonal residual reported alongside the existing projection metric. Execute the original fixed recipient and three donors physically for robot/object endpoint outcomes.

No state may be removed for policy failure, small action separation, an unattractive future, or an unfavorable intervention result. Mechanical failures remain in the selection funnel and are not replaced.

### P0: Cosmos 3 K/V factorial

Complete both directions of the interface intervention:

Use 24 task-balanced states from the fresh frozen cohort: environment seeds 3554, 4828, 5017, and 5428 in every task. This subset is fixed by config order and does not depend on native or intervention outcomes.

| Future target | Future-token K/V | Role |
| --- | --- | --- |
| Recipient | Recipient | Recipient control |
| Donor | Donor | Donor control |
| Donor | Recipient | Suppression / necessity |
| Recipient | Donor | Rescue / conditional sufficiency |

Exact record/replay is required for both recipient and donor caches. A suppression result without rescue retains necessity language; it is not described as K/V sufficiency.

### P0: FastWAM Optional-IDM smoke test

Use the released Optional-IDM checkpoint because its native inference order is future-video generation followed by action denoising, and the same checkpoint exposes a `first_frame` path without the future route. Separate video and action random-number generators before interpreting any transplant.

The eight-state smoke test must finish by hour four. Scale only if all identity controls pass and coherent donor identity is recovered above the frozen gate. If the gate fails, preserve the negative result and immediately move the external lane to LingBot-VA bring-up; do not tune seeds or select visually appealing futures.

## Execution schedule

| Time | Cosmos lane A | Cosmos lane B | External lane |
| --- | --- | --- | --- |
| 0--2 h | Runner changes, tests, fresh-state recording | K/V factorial changes and smoke | Provision, install, checkpoint download, RNG split |
| 2--4 h | Eight-state selection-free smoke | Four-state K/V smoke | Eight-state Optional-IDM smoke |
| 4 h gate | Scale if exact controls pass | Scale if both replay checks pass | Scale FastWAM or switch once to LingBot-VA |
| 4--11 h | Remaining selection-free states | Remaining K/V states, then endpoint subset | Powered external replication |
| 11--14 h | Aggregate and audit missingness | Hierarchical analysis | Aggregate and first-frame comparison |
| 14--16 h | Tables, plots, reruns of mechanical failures only, claim-safe language | Same | Same |

## Deliverables

- Frozen manifests and full state-selection funnels.
- Atomic per-state JSON and NPZ outputs.
- Exact control and replay audit tables.
- State-level CSV files for every primary and secondary estimand.
- Task-to-state hierarchical bootstrap intervals and leave-one-task-out results.
- Donor-retrieval, distance-reduction, orthogonal-error, and separation-quartile plots.
- A concise evidence ledger distinguishing positive, negative, failed-control, and mechanically missing outcomes.
- Suggested manuscript sentences, delivered separately from the manuscript source.

## Explicitly deferred unless all P0 work finishes

- Head-by-head or broad layer localization.
- Full robot x object x background factorial.
- Closed-loop success/failure future mining.
- A third external model.
- Additional Gaussian baselines beyond the frozen controls.
