# Adversarial audit: Cosmos 3 existing-cohort future × K/V factorial

**Verdict: PASS, with mandatory scope and wording limits.** I found no
claim-breaking defect in the frozen v3 design, sequential execution, accepted
state reports, or aggregate analysis. The result supports a strong action-level
effect of future-token K/V identity in this existing selected-pair cohort. It
does not support a selection-free, behavioral-success, physical-endpoint, or
“visible future is irrelevant” claim.

## Frozen provenance and completeness

- Active manifest: `cosmos3-kv-existing-bb8591311eda8a59`, SHA-256
  `972bcab77e2b999c703250f9e7ab17f3d854c858d99712bb7873d0bbf589714f`.
  The copy in `run/manifest.json` is byte-identical.
- Runner SHA-256:
  `83285e99b993e7f996a40189332643338e33805fe03e00b931f0c214e32179de`.
  Sequential launcher SHA-256:
  `dd3cab800431ddf542bab574a528f3971d5df813750720b939c3a1247e2e68a4`.
  Analyzer SHA-256:
  `5a2fe27f850c72291530c99db363365a6ee43ed12f3760f4716c0409faa98f7f`.
- The source cohort contains exactly 22 canonical saved states. The evaluation
  set is exactly that set minus the prespecified development state
  `BananaInBowlTask_seed_103`: 21 unique states, with no other omission or
  addition.
- The 21 states span six tasks with counts 3, 3, 3, 4, 4, and 4. All 21 have
  exactly one report, receipt, and zero-exit log; there are no failed-attempt
  logs. All 63 input hashes and all 21 frozen branch-design records recomputed
  exactly.
- The manifest predates the first state run. State intervals run from
  2026-09-03 23:20:17 to 23:25:12 UTC, are in manifest order, and never
  overlap. The summary was emitted afterward at 23:25:43 UTC.
- Every receipt binds its report hash, manifest hash, runner hash, and the same
  server image, `sha256:95a8933deb69f2fc73697d846f7e17066a4d4e05dc9ef85718a7ee12cfea2a8c`.
  The 21 receipt-bound report hashes exactly match the analyzer provenance.

## Intervention and control audit

- The global-cache hazard is controlled. The runner is sequential, recording a
  recipient cache before all recipient-cache consumers and then recording the
  donor cache before its consumers. Record mode clears the process-global
  cache, and every patch names the just-recorded state-specific cache. I found
  42 unique cache IDs (recipient and donor for each state), with no cross-state
  reuse.
- All 126 instrumented response interfaces reported the intended action-query,
  future-video K/V intervention over layers 0–35. Every one of the 4,536
  layer-census entries reported exactly eight denoising calls. The 105 native
  or clean-clamp arms requested no attention intervention and had empty caches.
- All 105 exact repeat/record/replay errors are zero. All 168 intervention
  input/output action-coordinate error pairs are zero. Cache recording itself
  therefore behaves as a no-op, exact replay reproduces the record, and the
  patch changes action computation through attention rather than by writing
  action coordinates.
- Within every state, all 11 arms have one transformed-input state fingerprint
  and one parameter-probe fingerprint. The current observation, instruction,
  recorded proprioception, and first-frame input are consequently held fixed
  across arms.
- Recipient native/repeat arms use the same seed and have identical path-noise
  hashes; donor-native path noise is distinct. All 168 intervention metadata
  checks use the recipient seed and recipient record while selecting the
  correct recipient or donor future record. Research and cache IDs are
  state-specific routing keys; they do not vary the model-input fingerprint.
- Recipient-future output signatures are exactly identical across their four
  arms, donor-future signatures are exactly identical across their four arms,
  and recipient and donor target hashes differ in all 21 states. Thus realized
  future identity is held fixed within each K/V comparison.

One wording limitation is important: `target_future_max_error` is nonzero,
ranging from 0.0180719 to 0.0277936 in latent units. This does **not** invalidate
the factorial because the realized output-future hash and the error are exact
within each future-source set, but it rules out saying that the sampler output
is elementwise identical to the registered native target latent. Say “the
realized future was held fixed across K/V arms,” not “the output exactly equals
the native target future.”

The recipient-to-donor action axes are nondegenerate in every state
(`native_action_l2` 2.0526–6.8652).

## Independent recomputation

I parsed the four cells directly from all 21 raw reports, independently joined
them to the frozen manifest, and reimplemented the documented task→state
bootstrap without importing the analyzer. The 10,000-draw, seed-20260903
results match the JSON and aggregate CSV exactly (apart from a
1.4×10^-17 summation-order difference in one non-primary state-weighted mean).

| Visible future / future K/V | Equal-task mean donor projection | 95% hierarchical bootstrap CI |
|---|---:|---:|
| Recipient / recipient | -0.011222 | [-0.022435, 0.000349] |
| Donor / recipient | 0.118970 | [0.077599, 0.162466] |
| Donor / donor | 1.003768 | [0.987655, 1.025415] |
| Recipient / donor | 0.884242 | [0.848488, 0.918672] |

The corresponding state-weighted means are -0.011314, 0.120932, 1.003227,
and 0.881514. The analyzer correctly reports equal-task means as primary, so
the unequal 3/4-state task sizes do not silently reweight the population.

The prespecified contrasts also recompute exactly:

| Contrast | Equal-task mean | 95% CI |
|---|---:|---:|
| K/V effect at recipient future | 0.895464 | [0.859347, 0.928270] |
| K/V effect at donor future | 0.884798 | [0.845384, 0.921493] |
| Visible-future effect at recipient K/V | 0.130192 | [0.088945, 0.174043] |
| Visible-future effect at donor K/V | 0.119526 | [0.083893, 0.156677] |
| Future × K/V interaction | -0.010666 | [-0.031670, 0.006055] |

The raw-distance classification is 42/42 crossed arms following the K/V
source: all 21 donor-future/recipient-K/V actions are closer to recipient
native, and all 21 recipient-future/donor-K/V actions are closer to donor
native. There are no ties. This is not a threshold-edge result: the smallest
absolute donor-projection margin from 0.5 is 0.2199, and the smallest native
distance advantages are 0.9092 and 1.1207 action-L2 units in the two crossing
directions.

The direction is not driven by one task. Per-task K/V effects range
0.8345–0.9287 at recipient future and 0.8365–0.9227 at donor future; all six
task counts are perfect. Leave-one-task-out estimates remain 0.8888–0.9077 and
0.8772–0.8945, respectively. I also reproduced all 22 aggregate rows, all six
per-task rows, all six leave-one-task-out rows, every state-level contrast, and
the follow-count CSV. The LaTeX table and compact plot faithfully render those
values.

## Claim-safe interpretation

Supported wording:

> In the existing, previously selected 21-state Cosmos cohort across six
> tasks, replacing all future-video keys/values seen by action queries shifted
> the model’s action chunk strongly toward the K/V source. Both crossed arms
> were closer to the K/V source’s native action in all 42 within-state tests,
> while visible-future identity retained a smaller positive effect. The
> future×K/V interaction was not distinguishable from zero by the frozen
> hierarchical interval.

Required limitations:

- This is an action-only model intervention on an existing selected-pair
  cohort, not a selection-free population estimate and not the fresh held-out
  study.
- It measures predicted action chunks, not simulator endpoints, task success,
  or physical behavior.
- Do not say K/V “fully determines” action or that the visible future is
  irrelevant: both fixed-K/V visible-future contrasts are positive with CIs
  excluding zero.
- Do not interpret 42/42 as 42 independent experimental units. There are 21
  paired state units clustered within six tasks; the hierarchical bootstrap is
  the relevant uncertainty analysis.
- The all-layer intervention establishes a causal dependence of these action
  outputs on future-token K/V content under the audited model path. It does not
  localize particular layers, identify a unique semantic “plan,” or by itself
  establish behavioral mediation.

Subject to those limits, the v3 result and its frozen analysis pass audit.
