# Noon final results: DreamZero and LingBot-VA

**Target freeze:** 2026-09-04 noon PT  
**Status:** Computationally complete and locally verified at 11:50 PT.  
**Question:** With the present input and recipient action randomness fixed, does
replacing a native imagined-future representation directionally select the
action paired with that source future?

## Bottom line

Yes, for the predicted action chunk in both released systems. DreamZero gives
perfect correct-source retrieval in the frozen cohort under matched-noise future-latent
trajectory replay. LingBot-VA gives a smaller but clearly above-chance
source-retrieval result through its ordered future-to-cache-to-action route.
The two systems therefore support the same narrow causal statement:
model-native future-source identity can exert source-specific directional
control over predicted action under transplantation.

This result does **not** establish that either model naturally compares
alternative futures, that decoded branch differences are task-semantic, or
that transplanting a future improves closed-loop task success.

## Primary and secondary results

| Model | Frozen cohort | Correct source, off diagonal | Distance reduction | Projection | Cosine | Orthogonal residual |
|---|---:|---:|---:|---:|---:|---:|
| DreamZero | 30 states, 4 branches/state, complete 4x4 | 360/360 = 1.000 [1.000, 1.000] | 0.917 [0.897, 0.933] | 0.990 [0.984, 0.995] | 0.995 [0.991, 0.997] | 0.080 [0.065, 0.099] |
| LingBot-VA | 30 states, 4 branches/state, complete 4x4 | 269/360 = 0.747 [0.667, 0.819] | 0.473 [0.421, 0.524] | 0.675 [0.624, 0.723] | 0.814 [0.777, 0.847] | 0.388 [0.359, 0.418] |

Both within-state four-label permutation tests give one-sided Monte Carlo
`p=1/100001` against the 25% null. State is the independent unit; the 12
off-diagonal source directions are averaged within state before interval
estimation.

**Post-hoc secondary temporal audits.** DreamZero's full-chunk retrieval is 1.000 in every native-action-separation
quartile and every leave-one-verb-family-out analysis. Its literal first
low-level action remains strong: 348/360 retrieval, projection 0.950, and
distance reduction 0.775.

LingBot's result is temporally concentrated. Retrieval is 98/360 = 0.272 for
the first post-conditioning frame group and 48/360 = 0.133 for the literal
first low-level action; later predicted actions drive the three
post-conditioning frame groups of the predicted action chunk. These outputs
were not physically executed.

## Dose response

| Model | Recipient-to-donor alpha | Mean normalized path response | State-level monotonicity | Frozen slope test |
|---|---|---|---:|---|
| DreamZero | 0, .25, .50, .75, 1 | 0, .046, .465, .939, .985 | 30/30 interior triples nondecreasing | slope 1.786 [1.707, 1.853], one-sided state sign-flip `p=1/100001` |
| LingBot-VA | 0, .25, .50, .75, 1 | 0, .221, .487, .758, 1 | 30/30 interior triples nondecreasing | slope 1.073 [1.039, 1.109], one-sided within-state label-permutation `p=1/100001` |

These are pathwise latent-mixture responses, not semantic interpolations. The
DreamZero alpha 0/1 and LingBot alpha 0/1 endpoints are exact frozen-core
reuses and were excluded from the slope tests. The vertical axes are related
but not numerically identical estimands: DreamZero projects onto the native
recipient-action-to-native-donor-action axis, whereas LingBot projects onto
the observed alpha-0-to-alpha-1 action-endpoint segment, making its displayed
endpoints 0 and 1 by construction. The two slopes therefore should not be
compared as if they shared one denominator.

## Controls and mechanism status

- **DreamZero:** four distinct native future traces and actions per state;
  30/30 exact self cells; matched donor trace at every one of 16 solver steps;
  active recipient action-noise hashes match bit-for-bit; no action-coordinate
  writes; the excluded patched-server mode-off/record debug gate had maximum
  absolute error 0. On a separate excluded input, an untouched checkout of the
  pinned official commit and patched mode-off produced bitwise-identical 24x8
  action arrays (maximum absolute error 0).
- **LingBot-VA:** four distinct native futures and 30-layer predicted-future
  caches per state; 30/30 exact latent self replay and cache replay; testing on
  one excluded development input matched official upstream inference
  bit-for-bit; recipient action noise is fixed and no action coordinate is
  written.
- **Wrong-source coverage and permutation test:** the complete four-source
  grids contain every wrong native source. The chance test statistically
  permutes the four source labels within state; shuffled labels were not fed to
  either model as a separate intervention condition.
- **Gaussian controls:** DreamZero's per-step norm-matched Gaussian trace gives
  normalized displacement 2.070. LingBot's original four reference-matched
  Gaussian controls per state give normalized displacement 3.390. Both are
  strongly disruptive, so they diagnose generic interface sensitivity rather
  than serving as inert negative controls. In the completed 30-state LingBot
  four-source Gaussian grid, source retrieval was 42/360 = 0.117 [0.094,
  0.139] and native-donor alignment was 78/360 = 0.217 [0.192, 0.239], neither
  above the 0.25 chance rate (one-sided `p=1.0` and `p=0.99995`). Native
  recipient identity was retained in only 126/360 = 0.350 cells. Thus
  branch-statistic-matched arbitrary sources disrupted action but did not
  recover Gaussian-source identity or native-donor alignment above chance.
  This is an
  arbitrary-source routing audit, not an equal-geometry or semantic-content
  null.
- **K/V/cache crossing:** LingBot actions follow the installed future-derived
  cache exactly: donor raw future plus recipient cache returns recipient action
  geometry (projection 0), whereas recipient raw future plus donor cache
  returns donor-directed geometry (projection 0.675). This validates the
  released ordered routing interface; because the action stage consumes the
  cache rather than the raw latent, it is not an independently identifiable
  two-path mediation factorial. A new DreamZero attention-K/V factorial was
  not completed before the deadline; the paper's independently crossed
  visible-future x K/V mechanism result remains Cosmos 3.

## Cohort and generalization boundaries

- DreamZero states were deterministically frozen from 30 unique DROID episodes
  and instruction strings, with three states in each of ten predeclared verb
  families. No pixels, actions, predicted futures, outcome, success, or
  intervention result entered selection. The examples come from the released
  preprocessed training-data corpus, so they are analysis-held-out but not
  model-held-out.
- LingBot states were predetermined as initial-state indices 10, 20, and 30 for
  each of ten LIBERO-10 tasks; two state-0 inputs were used only for development
  gates. No native output or intervention result entered selection. The tested
  checkpoint was post-trained on these tasks, so this is not held-out-task
  generalization.

## Clean paper throughline

1. **Directionality:** changing only a model-native future source selects its
   paired action, rather than merely perturbing action.
2. **Cross-system recurrence:** the same source-specific effect occurs in
   Cosmos 3, FastWAM, DreamZero, and LingBot-VA, spanning simultaneous joint
   denoising and ordered video-first inference.
3. **Content profile:** Cosmos Policy points to camera-visible prospective
   robot motion, while Cosmos 3's complete-video effect cannot be reproduced by
   isolated robot or object pixels.
4. **Internal route:** Cosmos 3's independently crossed future x future-token
   K/V experiment suppresses, rescues, and redirects the action, identifying a
   major direct attention route for the effect.

Copy-ready claim:

> Across released world-action-model systems with joint and ordered
> video-to-action inference organizations, native predicted-future source
> identity exerts source-specific directional control over predicted action
> chunks under transplantation. In Cosmos 3, independently crossed
> future-token K/V interventions identify a major direct attention route for
> that effect.

## Authoritative local artifacts

- [Complete claim-safe synthesis](claim_safe_synthesis.md)
- [Independent adversarial audit](final_science_adversarial_audit.md)
- [DreamZero core summary](../../output/deadline_2026_09_04/dreamzero/core_analysis_final/summary.json)
- [DreamZero controls and dose](../../output/deadline_2026_09_04/dreamzero/control_analysis/summary.json)
- [DreamZero provenance reconciliation](dreamzero_provenance_reconciliation.md)
- [DreamZero clean-upstream parity audit](../../output/deadline_2026_09_04/dreamzero/upstream_native_parity/execution_receipt.json)
- [DreamZero exhaustive all-120 native media](../../output/deadline_2026_09_04/dreamzero/all_native_media/receipt.json)
- [DreamZero all-state overview and terminal-frame sheet](../../output/deadline_2026_09_04/dreamzero/all_native_media_derived/receipt.json)
- [LingBot frozen core summary](../../output/deadline_2026_09_04/lingbot/core_artifacts/summary.json)
- [LingBot frozen dose summary](../../output/deadline_2026_09_04/lingbot/dose_analysis/summary.json)
- [LingBot complete raw addendum](../../output/deadline_2026_09_04/lingbot_final_raw_addendum_v2/raw_addendum_receipt.json)
- [LingBot exhaustive all-120 decoded-future audit](../../output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1/provenance.json)
- [LingBot decoded-future execution-history addendum](../../output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1_execution_addendum_v2/execution_receipt.json)
- [LingBot all-120 Gaussian-latent provenance audit](../../output/deadline_2026_09_04/lingbot_gaussian_latent_provenance_addendum_v1/receipt.json)
- [LingBot complete four-source Gaussian routing audit](../../output/deadline_2026_09_04/lingbot_gaussian_grid_v2_final/summary.json)

The exhaustive LingBot visualization contains all 30 states x all four branches
in frozen manifest order. It is post-analysis descriptive media; nothing was
selected by appearance, semantics, or effect size.

Both exhaustive terminal-frame sheets were manually inspected after the
quantitative analyses and show valid decoded robot scenes without systematic
blank, NaN-like, or visibly corrupted outputs. This is a rendering-quality
check, not a semantic annotation of what differs between branches.

## Compute closure

The run used only the two dedicated 2xH100 Lambda instances
`if-overnight-external-wams` and `if-overnight-robolab-clients`. The four
`nla-*` instances were not accessed, and loaded Cosmos services were not
disrupted. At 11:50 PT, the LingBot workers had exited normally and the final
DreamZero parity server process tree was terminated after local hash verification.
Both dedicated nodes then showed zero GPU processes, 0 MiB allocated on all
four H100s, and 0% utilization. The instances remain running; they were not
stopped or deleted.
