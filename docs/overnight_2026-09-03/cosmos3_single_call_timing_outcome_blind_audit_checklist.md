# Cosmos 3 single-call timing v2: outcome-blind launch checklist

Verdict state: **NOT YET AUDITED / DO NOT LAUNCH.** Complete this checklist
against the final content-addressed manifest, immutable snapshot, and excluded
smoke without opening any evaluation outcome.

## Design and population

- [ ] The manifest cites the exact SHA-256 of
  `cosmos3_single_call_timing_protocol_v2.md`; v1 remains preserved.
- [ ] Exactly 30 middle states are enumerated: six named tasks x five fixed
  environment seeds, with all archived success/failure labels retained.
- [ ] State IDs, middle branch steps, assets, and hashes match the frozen
  archival cohort; there is no outcome-, separation-, or appearance-based
  filtering or replacement.
- [ ] Branch order is exactly 211, 223, 227, 229 and every state has all 16
  recipient-source cells under each of six timing conditions.
- [ ] Request accounting is exact: 4 native + 4 native replay + 96 grid + 4
  extra all-call diagonal replay = 108 calls/state = 3,240 total.
- [ ] Deterministic request order/output names and fail-closed completeness,
  overwrite, and resume rules are frozen.

## Timing and causal isolation

- [ ] Call map is exactly `none=[]`, singles `[0]`, `[1]`, `[2]`, `[3]`, and
  `all_calls=[0,1,2,3]`.
- [ ] Both sigma traces equal, in chronological float32 order,
  `[0.9990000128746033, 0.9369999766349792, 0.8330000042915344,
  0.6240000128746033]` on every request.
- [ ] Recipient seed, initial sampler state, path noise, state, prompt, image,
  joints/gripper, solver, and model are fixed across source labels and timing;
  donor seed is used only to identify its native target.
- [ ] Target hash equals the intended native source future; self targets equal
  recipient native futures. No source ID/seed/hash/path or `research_*` field
  reaches ordinary policy conditioning.
- [ ] At active sites, captured model inputs equal the donor RF path and
  captured sampler velocities equal the desired target velocity, on the exact
  future-frame mask; the audit captures the live tensors rather than intended
  metadata.
- [ ] Inactive calls make zero wrapper writes, action-coordinate input/output
  errors are exactly zero, active indices/counts are exact, and mask indices,
  shape, and cardinality are exact.
- [ ] `none` is a full native no-op for all four target labels, including
  action/future/x0/initial/path signatures; native and all-call diagonal replays
  are exact.
- [ ] Final target residuals are finite and fully reported but never used for
  arm/state admission, exclusion, or retry.

## Runtime closure and smoke

- [ ] Manifest and launcher verify exact hashes for runner, analyzer, launcher,
  server, interventions, attention/runtime dependencies, upstream sample
  builder, image, checkpoint, source commit, protocol, and inputs before the
  first call.
- [ ] Snapshot files are immutable; output root is new; server/registry is new
  and isolated or proven empty.
- [ ] Final-snapshot excluded smoke covers all six timing conditions plus
  diagonal/off-diagonal targets and passes input/probe/RNG, schedule, active-
  site, mask, action-no-write, no-op, replay, finite, and output-schema gates.
- [ ] Any post-smoke code/config/protocol change forces a new version and smoke.

## Analyzer and statistics

- [ ] Analyzer requires exactly 30 complete state files and the exact 3,240
  request records, rejecting duplicates, extras, missing rows, nonfinite data,
  hash mismatches, and silent `None`/NaN drops.
- [ ] Actions are flattened in float64; nearest-native ties use frozen branch
  order; all native separations are reported and must exceed `1e-12` without
  excluding a state.
- [ ] Primary pair metrics use all 12 ordered off-diagonal pairs and compare
  donor versus self clamp at the same timing. Formula/unit tests independently
  verify retrieval gain and normalized donor-distance gain.
- [ ] State first, five states/task second, six equal-weight tasks last; donor
  rows/cells are never treated as independent observations.
- [ ] One shared `PCG64(20260903)` 10,000-draw task->state resample table is used
  for all endpoints/contrasts; point estimates and 2.5/97.5 percentiles are
  reproduced by an independent test fixture.
- [ ] Primary average-single and sustained-minus-single statements each require
  both prespecified matched-metric lower bounds > 0; no favorable metric may be
  substituted.
- [ ] Call-local p-values use the frozen null-centered formula; each composite
  p is the maximum of its two component p-values; Holm is applied across exactly
  four call composites with stop-on-first-failure and monotone adjusted values.
- [ ] Complete 4x4 retrieval uses all 16 cells and reports the exact 0.25
  source-permutation expectation. It is not confused with the 12-pair donor
  estimand.
- [ ] Global native-separation quartiles use all 360 directed pairs before
  state aggregation, with frozen boundaries/tie behavior and all denominators.
- [ ] Per-task, all six leave-one-task-out, ties/margins, controls, residual
  distributions, and zero-exclusion counts are emitted in JSON/CSV/LaTeX/plot.

## Claim boundary

- [ ] Text calls this an imposed action-space timing/strength audit, not natural
  mediation, physical success, semantic planning, or a local direct effect.
- [ ] A null/failed positive gate is stated only as “not detected under this
  design.” With no frozen equivalence margin, it is never evidence of no effect,
  necessity, or “only sustained intervention works.”
- [ ] A sustained-strength statement is based on the positive, timing-matched
  `all_calls - mean(single_calls)` conjunction, not on a null single-call test.
- [ ] Exact intervention-site replacement is distinguished from approximate
  final sampled-future proximity.

Launch verdict may become **GO** only after every box is checked and the audit
records the exact manifest ID/SHA, snapshot hashes, smoke SHA, auditor report
SHA, and launch command. Otherwise the verdict is **NO-GO**.
