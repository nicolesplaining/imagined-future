# Submission evidence update — 2026-09-03

This document is an evidence package, not a manuscript edit. It records the
frozen designs, completed outcomes, provenance, and claim-safe language from the
overnight submission run.

## Executive summary

Six completed experiments materially strengthen the paper.

1. **Selection-free Cosmos 3 multi-donor evaluation (90 fixed states, 30
   episodes, six tasks).** Across a balanced four-source grid, the transplanted
   future selected its associated action in all 1,440 balanced source cells,
   including all 1,080 off-diagonal donor cells (100%; hierarchical 95% CI
   100–100%; exact conditional label-permutation expectation 25%).
   Off-diagonal donor actions moved 0.763 of the native distance toward their
   donors, with 0.966 donor-axis projection and 0.227 normalized orthogonal
   residual. The result held at early, middle, and late states and in every
   native-separation quartile.
2. **Cosmos 3 future × K/V factorial (21 existing states, six tasks).** Holding
   the realized visible future fixed while crossing the source of future-token
   keys/values redirected the predicted action toward the K/V source in all 42
   crossed arms. The K/V effect was 0.895 at a recipient future and 0.885 at a
   donor future; visible-future identity retained a smaller 0.120–0.130 effect.
3. **Powered FastWAM Optional-IDM replication (120 untouched states, all 40
   LIBERO tasks).** A transplanted coherent future selected its associated action
   in 1,919/1,920 cells of a balanced four-source grid (99.95%, hierarchical 95%
   CI 99.74–100%; exact conditional label-permutation expectation 25%).
   Donor-only action projection was 0.869, distance reduction 0.683, cosine
   alignment 0.943, and normalized orthogonal residual 0.284. Exact self-replay
   and no-future donor-seed invariance were bitwise exact.
4. **FastWAM future × video-cache factorial (the same 120 states).** Crossing
   registered recipient/donor future-latent identity with recipient/donor cache
   identity under an explicit cache override showed cache-controlled blocking
   and rescue at the model's future-to-action interface. Donor-cache rescue
   achieved 99.93% donor retrieval and 0.683 distance reduction under either
   future-latent source;
   changing the registered future latent while holding its cache fixed changed
   no directional metric. Under the explicit cache override, action denoising
   does not consume the latent content, so this validates the interface and
   intervention plumbing rather than a learned preference for cache over a
   simultaneously active latent.
5. **Cosmos 3 single-call timing audit (30 fixed states, six tasks).** Donor
   future content changed the action when imposed at any one of the four solver
   calls. Correct off-diagonal donor retrieval increased from 0.728 at call 0
   to 0.922 at call 1 and 1.000 at calls 2 and 3; calls 2 and 3 alone produced
   essentially the same donor-directed distance reduction as the all-call
   intervention (0.761, 0.761, and 0.760, respectively). Every call-local
   donor-specific test survived Holm correction. Thus the directional result
   does not require repeating the clamp at every solver call.
6. **Cosmos 3 future-strength dose response (30 fixed states, six tasks).**
   Interpolating the imposed future from recipient to donor produced a positive
   donor-directed distance-reduction slope of 0.899 [0.852, 0.936]. The
   task-weighted mean increased at every adjacent mixture step, from -0.022 at
   recipient-only to 0.763 at full donor; correct-donor retrieval rose from
   0.000 to 1.000. All six task slopes and every leave-one-task-out slope were
   positive.

## Evidence matrix

| Question | Model/cohort | Status | Main result | Scope |
|---|---|---:|---|---|
| Does future-token K/V carry donor steering? | Cosmos 3, 21 existing selected-pair states | Complete; adversarial audit PASS | Both crossed arms followed K/V in 42/42 comparisons; K/V effects 0.895 and 0.885 | Predicted actions only; selected-pair cohort |
| Do coherent futures select corresponding actions in a released external WAM? | FastWAM Optional-IDM, 120 untouched states, 40 tasks | Complete; powered gate and independent audit PASS | 1,919/1,920 correct future-source retrieval; projection 0.869 | Predicted 32-step action chunks; no executed success claim |
| Does Cosmos 3 steering survive selection-free multi-donor evaluation? | Cosmos 3 archival cohort, 90 fixed states, 30 episodes, six tasks | Complete; all frozen gates and independent raw audit PASS | 100% four-source and off-diagonal donor retrieval; distance reduction 0.763 [0.733, 0.792] | Lossy reconstructed observations; action-only |
| Does explicit cache selection block and rescue FastWAM steering? | FastWAM, same 120-state population | Complete; frozen gate and independent audit PASS | Same-cache future swaps were exactly invariant; donor cache rescued donor-source steering, with cache-effect retrieval 0.9993 and distance reduction 0.6828 | Explicit cache-override path; latent content is bypassed once a cache is supplied; predicted actions only |
| Does Cosmos 3 steering require clamping at every solver call? | Cosmos 3, 30 fixed middle-phase states across six tasks | Complete; frozen gates and independent raw audit PASS | Every isolated call had a donor-specific effect; calls 2 and 3 alone each achieved 1.000 donor retrieval and approximately 0.761 raw distance reduction, numerically similar to the all-call condition | Imposed action-space timing audit; not natural mediation or physical success |
| Does steering vary with partial future replacement? | Cosmos 3, 30 fixed middle-phase states across six tasks | Complete; frozen gates and independent raw audit PASS | Distance-reduction slope 0.899 [0.852, 0.936]; the task-weighted mean increased at all four adjacent alpha steps | Imposed all-call latent-future interpolation; action-only |

## 1. Cosmos 3 future × K/V factorial

### Frozen population and intervention

- Manifest: `cosmos3-kv-existing-bb8591311eda8a59`
- Manifest SHA-256:
  `972bcab77e2b999c703250f9e7ab17f3d854c858d99712bb7873d0bbf589714f`
- Population: 21 evaluation states across six tasks; the predeclared Banana
  seed-103 development state was excluded before evaluation.
- Four cells crossed recipient/donor visible-future identity with
  recipient/donor future-token K/V at all direct future-to-action attention
  interfaces.
- Exact cache record/replay, input identity, action-coordinate non-write, and
  realized-future identity controls passed.

### Results

| Visible future / future K/V | Equal-task mean donor projection | Hierarchical 95% CI |
|---|---:|---:|
| Recipient / recipient | -0.011 | [-0.022, 0.000] |
| Donor / recipient | 0.119 | [0.078, 0.162] |
| Recipient / donor | 0.884 | [0.848, 0.919] |
| Donor / donor | 1.004 | [0.988, 1.025] |

| Prespecified contrast | Estimate | Hierarchical 95% CI |
|---|---:|---:|
| K/V effect at recipient visible future | 0.895 | [0.859, 0.928] |
| K/V effect at donor visible future | 0.885 | [0.845, 0.921] |
| Visible-future effect at recipient K/V | 0.130 | [0.089, 0.174] |
| Visible-future effect at donor K/V | 0.120 | [0.084, 0.157] |
| Future × K/V interaction | -0.011 | [-0.032, 0.006] |

All 21 donor-future/recipient-K/V actions were closer to the recipient native
action, and all 21 recipient-future/donor-K/V actions were closer to the donor
native action. Leave-one-task-out estimates were stable.

### Claim-safe language

> In the existing 21-state Cosmos 3 cohort, replacing the future-token keys and
> values available to action queries redirected the predicted action toward the
> K/V source in every crossed comparison. Crossing visible-future and K/V
> identity showed a large K/V effect in both future conditions, while visible
> future identity retained a smaller positive effect.

This supports a **large causal contribution plus conditional
rescue/redirection** at the audited future-to-action attention interface. It
does not establish strict necessity: donor-visible-future/recipient-K/V retains
a positive projection effect, 0.119 [0.078, 0.162]. Nor does it imply that K/V
completely determines action, that visible-future identity is irrelevant, or
that this action-only intervention changed physical task success.

## 2. Powered external-WAM replication: FastWAM Optional-IDM

### Why this model

[FastWAM Optional-IDM](https://arxiv.org/abs/2603.16666) first generates a video
representation and then conditions action denoising on its deterministic video
K/V cache. The same [released checkpoint](https://huggingface.co/yuanty/fastwam)
exposes a `first_frame` route that does not consume future-video conditioning.
It is therefore a clean deadline-feasible test of whether directional
future-conditioned action steering generalizes beyond Cosmos-style joint
denoising. The appropriate description is **a recent, high-performing released
WAM with an explicit future-to-action interface**; this evidence package does
not assert a benchmark rank or SOTA status.

The [pinned official repository README](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/README.md)
(SHA-256
`1fbad9ad9407d54a224864a10ec9576fa731e4a8f291bc9a7e28f545647fef2a`)
reports an author-evaluated 98.55% average success for this checkpoint on the
full 40-task LIBERO benchmark. That number was not independently re-evaluated
in the overnight action-only study and is used only as model context.

### Frozen population and controls

- Manifest: `fastwam-813f0233b9a2c083`
- Manifest SHA-256:
  `d74edd650f32faf7a0907871ae43e7362b5be19e029bef0b17d055eb114d125a`
- Population: all 40 tasks from the four LIBERO suites, with untouched initial
  state indices 4, 5, and 6: 120 states total.
- Four fixed future/action seed pairs per state; all 12 ordered donor directions.
- Complete matrix: 8,640/8,640 arms, with no malformed, missing, invalid, or
  degenerate rows.
- Separate video and action RNG generators; recipient action noise remained
  fixed within every intervention comparison.
- Exact self-latent replay, exact self-cache replay, wrong-source latent,
  shuffled cache, and no-future donor-seed invariance controls.

### Results

| Estimand | Mean | Hierarchical 95% CI |
|---|---:|---:|
| Correct future-source retrieval, balanced 4×4 latent grid | 0.9995 | [0.9974, 1.0000] |
| Correct future-source retrieval, balanced 4×4 cache grid | 0.9995 | [0.9974, 1.0000] |
| Correct donor retrieval, off-diagonal donor latent | 0.9993 | [0.9965, 1.0000] |
| Donor-axis projection | 0.8690 | [0.8326, 0.9124] |
| Distance reduction toward donor | 0.6828 | [0.6375, 0.7380] |
| Cosine alignment with donor direction | 0.9433 | [0.9270, 0.9621] |
| Orthogonal residual / native donor separation | 0.2839 | [0.2431, 0.3175] |

Correct-source retrieval was 1,919/1,920 against an exact 0.25 chance
expectation. The wrong-latent condition had 0.0007 correct registered-donor
retrieval and -0.055 distance reduction. Shuffled-cache retrieval was 0.236 and
its actions were strongly out of distribution, so that control establishes a
need for coherent cache structure rather than a fine-grained semantic claim.
All leave-one-task-out estimates remained stable.

An independent audit reloaded all 8,640 registered raw JSON/NPZ pairs and
recomputed every directional metric. All 7,200 directional values matched the
reported analysis (maximum numerical difference
`2.13e-14`), as did the source-retrieval counts, hierarchical intervals,
task/suite summaries, and all 40 leave-one-task-out rows. The wrong-latent
source assignment was not balanced across labels, so it remains a secondary
paired control rather than a source-retrieval baseline; a balanced sensitivity
check did not change its conclusion.

Matching donor-latent and donor-cache interventions produced bitwise-identical
action arrays: in this architecture the generated future reaches action
denoising only through its deterministic cache. They are two access points to
one pathway, not independent replications; the latent and cache tensors
themselves are not being asserted to be identical.

### Claim-safe language

> In a frozen 120-state evaluation spanning all 40 LIBERO tasks, transplanting a
> coherent FastWAM future while holding recipient action noise fixed selected the
> action associated with that future in 1,919 of 1,920 four-source comparisons.
> The effect generalized directional future steering to a released architecture
> that generates its future before denoising actions.

This is a predicted-action result. It does not by itself establish improved
closed-loop task success, semantic planning, or comparison among outcomes.

## 3. FastWAM future × video-cache factorial

### Frozen population and intervention

- Manifest: `fastwam-kvfact-a09195568dd5a17f`
- Manifest SHA-256:
  `2cb76a6a4a6012caec72a976302f5dc91f75e15bb1dd9f3f4bc1699ae9a8f8fe`
- Population: the same 120 untouched states, four branches per state, and all
  twelve ordered recipient-to-donor directions as the powered replication.
- Four cells crossed recipient/donor future-latent identity with
  recipient/donor precomputed video-cache identity.
- Complete matrix: 5,760/5,760 registered arms and 120/120 state summaries;
  no missing, malformed, invalid, or degenerate rows.

### Results

| Future latent / video cache | Donor retrieval | Distance reduction | Projection | Orthogonal residual ratio |
|---|---:|---:|---:|---:|
| Recipient / recipient | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Donor / recipient | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Recipient / donor | 0.9993 | 0.6828 | 0.8690 | 0.2839 |
| Donor / donor | 0.9993 | 0.6828 | 0.8690 | 0.2839 |

The cache main effect was 0.9993 donor retrieval (hierarchical 95% CI
[0.9965, 1.0000]), 0.6828 distance reduction [0.6368, 0.7384], and 0.8690
donor-axis projection [0.8323, 0.9135]. The future-latent main effect and every
future-by-cache interaction were exactly zero for all directional metrics. All
native, stored-latent, and base-reference replays, plus both same-cache
future-swap invariance controls, were bitwise exact.

### Claim-safe language

> In FastWAM Optional-IDM, selecting recipient versus donor video cache under
> the explicit override blocked or rescued donor-source steering irrespective
> of the registered future latent: the recomputed action followed the
> video-cache source, with donor-cache rescue in 1,439 of 1,440 ordered donor
> comparisons.

This is an architecture-defined interface decomposition. FastWAM action
denoising intentionally consumes the generated future through this deterministic
cache, so the result validates that bottleneck rather than discovering an
unexpected internal circuit.

An independent audit recomputed all raw arms, state summaries, and 49
hierarchical estimates. Maximum discrepancies were `4.44e-16` at the run level,
`1.67e-16` at the state level, and `5.55e-17` for hierarchical results. The
audit also verified directly that the patched control flow bypasses latent
prefill when a cache override is supplied. Thus the zero future-latent effect
validates the explicit cache interface; it is not evidence that the learned
model prefers cache content over simultaneously available latent content.
The independent audit directly recomputed all parent-action references and
same-cache future-swap controls. The regenerated full-precision latent/cache
tensors were not persisted, so the native-action and stored-FP16-latent
regeneration zeros remain frozen-runner/atomic-summary evidence rather than
independently reconstructable raw-tensor checks.

## 4. Cosmos 3 archival selection-free multi-donor evaluation

### Frozen population and intervention

- Manifest: `cosmos3-archival-sf-507feb24297971eb`
- Manifest SHA-256:
  `8ab294c7a581424cfa02faae1db85941e3f43c1e28100e68e21fa5f533703b9e`
- Population: 90 fixed states = six tasks × five archived episodes ×
  early/middle/late phase, with no outcome-based state, pair, success,
  separation, or visual-quality filtering.
- Four fixed branch seeds per state, the full balanced 4×4 future-source grid,
  and all 12 ordered off-diagonal recipient→donor directions.
- Task→episode→state hierarchical bootstrap; donor directions were averaged
  within state. Phase, native-separation quartile, and leave-one-task-out
  analyses were fixed before outcomes were opened.
- Exactly 90/90 outputs completed with no resumed results. The raw
  package was hashed, changed to mode 0444, and rehashed before analysis.

### Results

| Estimand | Equal-task mean | Hierarchical 95% CI |
|---|---:|---:|
| Four-source retrieval (chance 0.25) | 1.000 | [1.000, 1.000] |
| Off-diagonal correct-donor retrieval | 1.000 | [1.000, 1.000] |
| Distance reduction toward donor | 0.763 | [0.733, 0.792] |
| Cosine alignment with donor direction | 0.967 | [0.957, 0.975] |
| Donor-axis projection | 0.966 | [0.956, 0.974] |
| Orthogonal residual / native separation | 0.227 | [0.200, 0.255] |
| Gaussian-target distance reduction | -0.068 | [-0.091, -0.046] |
| Gaussian-target projection | 0.017 | [0.010, 0.025] |

The conditional source-label permutation test gave `p=0.00009999`. Retrieval
was 1.000 at early, middle, and late states. Distance reduction was 0.765,
0.760, and 0.765 across those phases, respectively. Across the four
cohort-global native-separation quartiles, distance reduction rose from 0.641
in the smallest-separation quartile to 0.855 in the largest, with a positive
hierarchical confidence interval in every quartile. Leave-one-task-out distance
reduction ranged from 0.754 to 0.771.

An independent raw audit verified the exact 90-file inventory and recomputed
every reported aggregate from the read-only state outputs. It found zero
row-level or numerical discrepancies: all 1,080 donor axes were nondegenerate;
all 4,320 action-coordinate audits and 15,840 audits at each causal write site
passed; and all native, self, donor, and no-intervention replays were exact.

### Claim-safe language

> Across 90 fixed archival states spanning six tasks, five episodes per task,
> and three decision phases, transplanting one of four coherent Cosmos 3
> futures made the recomputed action closest to the action associated with that
> future in every tested source cell. Off-diagonal actions moved 76% of the
> native distance toward the donor on average, and the effect remained positive
> in every prespecified native-separation quartile.

This directly removes maximally separated donor-pair selection as a complete
explanation of the action-level result. Its scope remains predicted actions
from lossy reconstructed archival observations; it does not add a fresh
physical-endpoint or closed-loop task-success estimate.

## 5. Cosmos 3 single-call timing audit

### Frozen population and intervention

- Manifest: `cosmos3-timing-05f23896cd88b340`
- Manifest SHA-256:
  `a0ef31afe6f7996bfdb177957d2eabac6cc1d8263e6d04832602520e17e63759`
- Population: 30 fixed middle-phase states = six tasks × five archived
  episodes/states, with four native branches per state and all 12 ordered
  off-diagonal recipient→donor directions.
- Conditions: no intervention, exactly one of solver calls 0–3, or all four
  calls. Each condition included matched donor and self targets, giving exactly
  3,240 model calls.
- Task→state hierarchical bootstrap with 10,000 shared draws; the four
  call-local conjunctive tests used a frozen Holm correction.
- All `[32,8]` action-shape, finite-value, target, schedule, RNG, replay,
  intervention-site, and action-coordinate non-write controls passed exactly.

### Results

| Intervention timing | Correct off-diagonal donor retrieval | Raw distance reduction | Matched donor-specific distance gain |
|---|---:|---:|---:|
| Call 0 only | 0.728 [0.642, 0.803] | 0.375 [0.280, 0.445] | 0.462 [0.402, 0.515] |
| Call 1 only | 0.922 [0.856, 0.975] | 0.613 [0.546, 0.669] | 0.647 [0.596, 0.693] |
| Call 2 only | 1.000 [1.000, 1.000] | 0.761 [0.717, 0.798] | 0.781 [0.744, 0.812] |
| Call 3 only | 1.000 [1.000, 1.000] | 0.761 [0.715, 0.801] | 0.780 [0.741, 0.813] |
| All four calls | 1.000 [1.000, 1.000] | 0.760 [0.714, 0.801] | 0.781 [0.743, 0.814] |

Every call-local matched retrieval and distance-gain conjunction had raw
`p=0.00009999` and Holm-adjusted `p=0.00039996`. The average isolated-call
matched retrieval gain was 0.912 [0.876, 0.940], and the average isolated-call
matched distance gain was 0.668 [0.629, 0.704]. An independent audit reloaded
all 30 read-only state artifacts, verified all 3,240 recorded call outputs, and
recomputed 2,160 off-diagonal pair rows with zero discrepancies.

### Claim-safe language

> Across 30 fixed Cosmos 3 states, imposing the donor future at any single
> solver call produced a significant donor-specific action shift. Intervening
> only at call 2 or call 3 recovered the same correct-donor retrieval and
> essentially the same distance reduction as intervening at all four calls.

This rules out repeated clamping at every solver call as a requirement for the
observed directional action effect. No equivalence test was prespecified, so
the distance comparison is numerical rather than a formal equivalence claim.
It remains an imposed action-space timing and strength audit; it does not
establish natural mediation, an isolated local direct effect, physical task
success, or semantic planning.

## 6. Cosmos 3 future-strength dose response

### Frozen population and intervention

- Manifest: `cosmos3-dose-c00d81db8d910603`
- Manifest SHA-256:
  `1a37bdba796b580c905970c441aededccf4193c51fe370f0f3557eccd5a9ee7d`
- Population: 30 fixed middle-phase states = six tasks × five archived
  episodes/states, with four native branches and all 12 ordered
  recipient-to-donor pairs per state.
- At all four denoising calls, the imposed latent future was interpolated from
  recipient to donor at alpha in `{0, .25, .50, .75, 1}` while recipient action
  noise and all nonfuture inputs remained fixed.
- The complete design contained exactly 2,760 finite `[32,8]` action outputs.
  Exact source, RNG, target, replay, causal-site, and action-coordinate
  non-write controls passed.
- The 30 raw state files were inventoried, changed to mode 0444, and rehashed
  before the prespecified analyzer or independent auditor opened outcomes.

### Results

| Donor-future mixture alpha | Distance reduction | Donor-axis projection | Correct-donor retrieval |
|---:|---:|---:|---:|
| 0 | -0.022 [-0.040, -0.007] | -0.002 [-0.017, 0.011] | 0.000 [0.000, 0.000] |
| .25 | 0.033 [0.018, 0.046] | 0.056 [0.042, 0.073] | 0.003 [0.000, 0.011] |
| .50 | 0.232 [0.194, 0.264] | 0.303 [0.268, 0.334] | 0.142 [0.103, 0.183] |
| .75 | 0.712 [0.664, 0.753] | 0.877 [0.856, 0.895] | 0.992 [0.981, 1.000] |
| 1 | 0.763 [0.716, 0.800] | 0.969 [0.954, 0.983] | 1.000 [1.000, 1.000] |

The prespecified mean per-pair OLS slope for distance reduction was 0.899
[0.852, 0.936], and the endpoint contrast was 0.785 [0.746, 0.813]. All four
adjacent distance-reduction contrasts had hierarchical lower bounds above zero:
0.054 [0.042, 0.067], 0.199 [0.169, 0.226], 0.480 [0.427, 0.534], and 0.051
[0.039, 0.062]. Task-specific slopes ranged from 0.822 to 0.943, and
leave-one-task-out slopes ranged from 0.891 to 0.915. Descriptively, all 30
state slopes and all 360 within-state ordered-pair slopes were positive.

An independent raw audit recomputed every state and ordered-pair profile, the
shared task-to-state bootstrap, all primary and secondary contrasts, task and
leave-one-task-out estimates, and every control count. It found zero
discrepancies. At the individual-pair level, 257/360 profiles were
nondecreasing; the stronger supported statement is therefore about the graded
task-weighted mean response, not every individual pair.

### Claim-safe language

> Under prospectively fixed interpolation of the imposed Cosmos 3 future from
> recipient to donor, donor-directed action change increased with donor-future
> strength. The prespecified distance-reduction slope was 0.899 [0.852, 0.936],
> and the task-weighted mean increased at every adjacent mixture step.

This establishes a graded response to experimentally imposed latent-future
content under the all-call intervention. It remains an action-space
intervention rather than a physical-success or natural-mediation test.

## Artifact index

- Cosmos 3 K/V aggregate:
  `output/overnight_2026-09-03/cosmos3_kv_existing_v3/analysis/cosmos3_existing_kv_summary.json`
  (SHA-256 `7681fcb86bce473cd36497b518618f58869f096afb4ff29ba67784b720b998a6`)
- Cosmos 3 K/V independent audit:
  `docs/overnight_2026-09-03/cosmos3_existing_kv_factorial_adversarial_audit.md`
  (SHA-256 `7533626a25afe30dc227a8b809a4dbab5448454ec43d4142eb3e8a6f5e9ae2d3`)
- Cosmos 3 selection-free aggregate:
  `output/overnight_2026-09-03/cosmos3_archival_selection_v7_final/analysis/summary.json`
  (SHA-256 `cb914fead7e5fdd1303eb0f8086c8db53b7c0a8b0db1cf36439112e771c89889`)
- Cosmos 3 selection-free independent raw audit:
  `output/overnight_2026-09-03/cosmos3_archival_selection_v7_final/audit/independent_raw_audit.json`
  (SHA-256 `8c6a162a06e134e5bbf4becc7948c6c23e590d41df66a906a75be769cae2747e`)
- Cosmos 3 selection-free immutable-output inventory:
  `output/overnight_2026-09-03/cosmos3_archival_selection_v7_final/packaging/output_inventory.json`
  (SHA-256 `948f443fcd44a34a94a2a93f938a81f89c8affc37d6a7b90def985104c28914e`)
- FastWAM powered aggregate:
  `output/overnight_2026-09-03/fastwam_optional_idm_powered/analysis/fastwam_results.json`
  (SHA-256 `bbcc86f0398f92bf9f48dc6f1e47b20e28db0a4eb92ad229c6dc0a82c885ab0e`)
- FastWAM powered independent audit:
  `docs/overnight_2026-09-03/fastwam_optional_idm_powered_adversarial_audit.md`
  (SHA-256 `3fa38ed5ac36c88978845962dff69a779e9550d9ed9e6c15d2924190c390d936`)
- FastWAM powered presentation figure:
  `output/overnight_2026-09-03/fastwam_optional_idm_powered/presentation_v1/fastwam_optional_idm_summary.pdf`
- FastWAM future × cache factorial aggregate:
  `output/overnight_2026-09-03/fastwam_cache_factorial_powered_v3/analysis/fastwam_cache_factorial_results.json`
  (SHA-256 `483847ac549ea9b91919e4a88de5d88b98abd190709f0352d8585ef8583b34dd`)
- FastWAM future × cache factorial independent audit:
  `docs/overnight_2026-09-03/fastwam_cache_factorial_powered_v3_adversarial_audit.md`
  (SHA-256 `809483995ac39bea52395287133a4ec480fe7fe129bb19883859d689a721f441`)
- Compact cross-model four-cell pathway table (not inserted into the
  manuscript): `output/overnight_2026-09-03/combined_pathway_factorial_table.md`
  and `output/overnight_2026-09-03/combined_pathway_factorial_table.tex`. The
  table is an analogous-interface display, not a pooled cross-model estimand;
  its model-specific source hashes are recorded in the table note.
- Cosmos 3 single-call timing aggregate:
  `output/overnight_2026-09-03/cosmos3_single_call_timing_v5/evaluation/analysis/cosmos3_single_call_timing_results.json`
  (SHA-256 `2a988a805b7423efab65b3354043ef8dde5dd2c0043f00d5f99291bee6c00262`)
- Cosmos 3 single-call timing independent raw audit:
  `output/overnight_2026-09-03/cosmos3_single_call_timing_v5/evaluation/audit/independent_raw_audit.json`
  (SHA-256 `e710cffd44ed92b881eb480a2e7a08b1e404860bbb8e7e41adeac5ec2e569a89`)
- Cosmos 3 single-call timing manuscript-ready table and wording:
  `docs/overnight_2026-09-03/cosmos3_single_call_timing_v5_results.md`
  (SHA-256 `7cf91090577a7b7abe376c24812ee26b82e1bb3a89dc5052aa22771e791ed25e`)
- Cosmos 3 dose-response aggregate:
  `output/overnight_2026-09-03/cosmos3_future_strength_dose_v2/evaluation/analysis/summary.json`
  (SHA-256 `741307dff8ffac6458a65678db1f713aee3d74eb2b4ed11436d8345ad4c44a69`)
- Cosmos 3 dose-response immutable-output inventory:
  `output/overnight_2026-09-03/cosmos3_future_strength_dose_v2/packaging/output_inventory.json`
  (SHA-256 `8fa279868725119522bbcf1ecf5c3a3375833aadf3110b63d89a099720a600d8`)
- Cosmos 3 dose-response independent raw audit:
  `output/overnight_2026-09-03/cosmos3_future_strength_dose_v2/evaluation/audit/independent_raw_audit.json`
  (SHA-256 `c12cb8080df4af4f179a231cdd040936f9d693b8dd6c4b3bdb08be52f57c11ed`)
- FastWAM frozen analysis-v2 protocol and separate post-outcome record:
  `docs/fastwam-optional-idm-powered-analysis-v2-frozen.md` and
  `docs/fastwam-optional-idm-powered-analysis-v2-results.md`
- FastWAM smoke independent audit:
  `docs/overnight_2026-09-03/fastwam_optional_idm_adversarial_audit.md`
- Frozen decisions and amendments:
  `docs/overnight_2026-09-03/decision_ledger.md`
- Infrastructure and run provenance:
  `docs/overnight_2026-09-03/infrastructure_ledger.md` and
  `docs/overnight_2026-09-03/run_ledger.tsv`
