# Prospective Cosmos 3 single-call future-intervention audit

Status: **designed before the archival selection-free outcomes; not yet an
active frozen run.** A content-addressed manifest and runtime smoke are required
before execution.

## Question

Does a coherent donor future influence the action when it is introduced at only
one denoising call, or is donor steering detectable only when the clean target is
continuously imposed at every call and sampler boundary?

This is a targeted strength/timing audit of the paper's repeated-clamp
intervention. It does not establish the model's naturally occurring mediation
path, and it does not replace a future native-trajectory transplant.

## Frozen population rule

Use the middle-phase state from every episode in the archival Cosmos cohort:

- six fixed paper tasks;
- environment seeds 101, 103, 107, 109, and 113;
- 30 states total, one state per archived episode;
- branch seeds 211, 223, 227, and 229;
- no filtering by episode success, future appearance, native separation, or any
  main-study/timing-intervention outcome.

If this protocol is activated after the main archival result is visible, the
population and conditions remain fixed exactly as written here. Mechanical
failure rules must be set in the manifest and failed units may not be replaced.

## Conditions

For each state, recipient, and future source, hold the current input,
instruction, recipient action noise, solver, and sampler fixed. Run the complete
4×4 recipient × future-source grid under each timing condition:

1. `none`: target registered but no denoising call clamped;
2. `call_0_only`;
3. `call_1_only`;
4. `call_2_only`;
5. `call_3_only`;
6. `all_calls`: the current full intervention.

The four diagonal source cells are self-future controls and the twelve
off-diagonal cells are donor-future conditions. The six grids contain 96 arms
per state. Generate four native references per state and exact native repeats.

## Required runtime audits

- The active call index must equal the registered condition exactly.
- Every response must report the same frozen model-parameter probe and the same
  transformed-input fingerprint within state.
- `none` must reproduce the corresponding native action and native future
  exactly.
- Repeat at least the four diagonal `all_calls` cells and require exact action,
  target, output-future, sigma, clamp-index, and x0-trace signatures.
- No intervention may write action input or output coordinates.
- All four native futures must be reported as distinct/non-distinct without
  changing admission.
- All action arrays and directional metrics must be finite; degenerate native
  axes are retained, counted, and never silently dropped.
- Apply the existing target-error threshold only to `all_calls`. A partial-call
  arm is not expected to terminate at the donor target, so its final
  target-future error is descriptive rather than an admission criterion.

## Outcomes and analysis

For each timing condition, report:

- balanced four-source retrieval accuracy (chance expectation 0.25);
- off-diagonal donor retrieval;
- donor distance reduction;
- donor-axis projection;
- cosine alignment;
- orthogonal residual divided by native donor separation;
- final future distance to the registered target.

The independent hierarchy is task → episode/state. Donors, recipient directions,
and the 16 source cells are repeated measurements within state. Use an
equal-task point estimate and a 10,000-draw task→state hierarchical bootstrap.

Primary contrast: the within-state average across the four single-call
conditions minus `none`, for (a) correct-source retrieval and (b) donor distance
reduction. The evidence gate requires both hierarchical 95% lower bounds to be
strictly above zero, alongside all exact controls and complete admission.

Secondary contrasts:

- each single-call condition versus `none`, with Holm correction across the
  four calls;
- `all_calls` versus the mean single-call condition;
- the ordered call profile, reported without selecting a post-hoc “best” call;
- task and native-separation-quartile summaries.

No condition may be selected or removed based on its observed effect. A null
single-call result is retained and interpreted as evidence that the current
steering effect depends on sustained intervention under this implementation.

