# Claim-safe cross-model synthesis

**Status:** final deadline scientific handoff, separate from the manuscript.
Updated 2026-09-04 after the complete DreamZero core/control cohorts and
LingBot-VA core/dose/Gaussian-source cohorts passed their frozen analysis and
provenance audits.

## The defensible throughline

The paper asks a narrower and more falsifiable question than whether world
action models “plan”:

> If the present state, instruction, and action randomness are held fixed, does
> replacing one native predicted-future representation with another redirect
> the generated action toward the action associated with that source?

The experiment gives each predicted-future source a behavioral identity. Four
native continuations from the same state provide four representation/action
pairs. We reuse a recipient's action-noise path while installing each source,
then ask which native action is nearest to the recomputed action. Correct source
identification tests *which alternative* the intervention selects; projection,
distance reduction, cosine alignment, and orthogonal residual describe the
geometry of that change. Exact self/replay controls test implementation drift,
while native-source retrieval distinguishes directional identity from
displacement alone.

The strongest concise claim supported by the completed cohorts is:

> Across three released world-action-model systems spanning bidirectional joint
> denoising, chunk-autoregressive joint denoising, and an ordered video-first
> action-decoding route, transplanting a native predicted-future representation
> while holding the present and recipient action noise fixed redirects the
> predicted action toward the action paired with that future. In the
> video-first route, this effect cannot be explained solely by simultaneous
> video/action denoising during the action stage.

Use **three released systems with different inference organizations** rather
than “three architectures.” DreamZero is autoregressive across video chunks but
jointly denoises video and action within each chunk. LingBot-VA has a unified,
interleaved, shared-attention training architecture, while its released
per-chunk inference procedure first predicts the video latent and then decodes
the action conditioned on that prediction and cache. Always qualify the result
as occurring **under future transplantation**. The experiment establishes
counterfactual control of the model's action output; it does not show that the
model naturally samples, compares, scores, or chooses among several futures.

## What is already established

| Evidence | Frozen population | Audited result | Supported interpretation |
|---|---:|---|---|
| Cosmos 3 selection-free future transplantation | 90 archival states; 30 episodes; six RoboLab tasks; four futures/state | Four-source retrieval 100% in all 1,440 cells and all 1,080 off-diagonal cells; off-diagonal distance reduction 0.763 [0.733, 0.792], projection 0.966 [0.956, 0.974], orthogonal residual ratio 0.227 [0.200, 0.255] | Coherent future identity directionally controls the predicted action across unselected states, phases, and separation quartiles. This cohort is action-only and uses lossy archival observation reconstruction. |
| Cosmos 3 future x K/V factorial | 21 existing selected-pair states; six tasks | K/V effects 0.895 [0.859, 0.928] with recipient visible future and 0.885 [0.845, 0.921] with donor visible future; both crossed arms followed the K/V source in 42/42 comparisons | Future-token K/V makes a large causal contribution and can rescue/redirect donor-axis action steering. It is not complete or natural mediation: donor future with recipient K/V retains projection 0.119 [0.078, 0.162]. |
| Cosmos Policy future transplantation | 10 first-query LIBERO-10 states; one per task | Donor-minus-self action projection 0.499 [0.336, 0.679] and executed-endpoint projection 0.552 [0.387, 0.728]; both positive in 10/10 states | Earlier-generation support that future content can steer both action and physical execution at identifiable early states. The effect attenuates sharply later in episodes and is not a matched four-donor retrieval study. |
| Cosmos Policy content decomposition | 10 first-query states for modalities; separate content studies | Wrist future 0.435 [0.284, 0.605], primary camera 0.101 [0.057, 0.144], future proprioception 0.001 [-0.004, 0.008]; natural robot-only evidence is positive but based on four eligible units | The tested Cosmos Policy effect is most consistent with camera-visible prospective robot motion. Do not generalize that carrier to all models or claim task-object consequences are universally unused. |
| FastWAM Optional-IDM supporting replication | 120 states; all 40 LIBERO tasks; four futures/state | 1,919/1,920 all-cell and 1,439/1,440 off-diagonal correct-source retrieval; distance reduction 0.683 [0.638, 0.738], projection 0.869 [0.833, 0.912] | Existing evidence that source-specific future steering extends to an explicit staged future-to-action interface. Treat latent and cache access as one pathway, not two replications. |
| DreamZero deterministically selected future transplantation | 30 states from 30 unique DROID episodes and instruction strings in the released preprocessed training-data corpus; three states in each of ten predeclared verb families; four sources/state | Full 24-step action chunk: 360/360 off-diagonal source retrieval, permutation p=0.00001; distance reduction 0.917 [0.897, 0.933], projection 0.990 [0.984, 0.995], cosine 0.995 [0.991, 0.997], orthogonal residual 0.080 [0.065, 0.099]. The [1.000, 1.000] state bootstrap interval is degenerate because every state scored 1. Post-hoc first-step secondary: 348/360 retrieval (0.967), projection 0.950, distance reduction 0.775. | Native predicted-future source identity directionally controls DreamZero's predicted action under matched per-step latent replay. Full-chunk retrieval is 1.000 in every native-separation quartile and every leave-one-family-out analysis. Selection used no model outputs or outcome filtering, but these are analysis-held-out examples from released training data—not evidence of model-held-out generalization. This is action-output evidence, not natural temporal mediation or task success. |
| LingBot-VA predetermined future/cache transplantation | 30 LIBERO-10 states; three initial-state indices for each of ten tasks; four sources/state | Three post-conditioning frame groups of the predicted action chunk, after excluding the conditioned frame that LIBERO does not execute: 269/360 off-diagonal source retrieval (0.747 [0.667, 0.819]), permutation p=0.00001; distance reduction 0.473 [0.421, 0.524], projection 0.675 [0.624, 0.723], cosine 0.814 [0.777, 0.847], orthogonal residual 0.388 [0.359, 0.418]. Post-hoc retrieval for the first post-conditioning frame group (four low-level actions) is 98/360 (0.272); for the literal first low-level action it is 48/360 (0.133). | Future-derived cache source identity directionally controls LingBot-VA's full predicted action chunk in an ordered video-first/action-second inference route. Exact crossed controls show that actions follow installed cache identity: donor future plus recipient cache has projection 0, while recipient future plus donor cache has projection 0.675. Later action steps drive the chunk-level result. These action outputs were not physically executed. This is interface control, not discovery of two independent pathways or evidence of task success. |
| LingBot-VA latent-dose follow-up | Same 30 frozen states; one prespecified b0→b1 pair/state; recipient b0 action noise fixed | Mean normalized endpoint-axis response at alpha 0/.25/.5/.75/1: 0, 0.221, 0.487, 0.758, 1.000. The axis is the observed alpha-0-to-alpha-1 predicted-action endpoint segment, so endpoints are 0/1 by construction. Interior slope 1.073 [1.039, 1.109], one-sided within-state label-permutation p=0.00001; all 30 interior triples are nondecreasing. | The ordered interface responds gradually along the imposed latent segment. Endpoints are exact core-cell reuses and excluded from inference. This is pathwise sensitivity for one fixed latent interpolation—not semantic interpolation, naturalness, mediation, or a population intervention over future concepts. |
| LingBot-VA Gaussian-source audit | Same 30 frozen states; complete four recipient paths x four branch-statistic-matched Gaussian sources | Gaussian-source retrieval 42/360 (0.117 [0.094, 0.139]); native-donor alignment 78/360 (0.217 [0.192, 0.239]); native recipient retention 126/360 (0.350 [0.292, 0.417]). Gaussian-template projection 0.167 [0.156, 0.178] but distance reduction -0.309 [-0.329, -0.288]. | Post-analysis exploratory control: arbitrary Gaussian sources did not recover Gaussian-source identity or native-donor alignment above chance, and did not reliably preserve recipient identity. This is nonspecific disruption, not an inert or equal-geometry null. |

Numerical sources: [audited overnight handoff](../overnight_2026-09-03/manuscript_ready_results.md),
[Cosmos 3 population report](../cosmos3-population-results.md), [Cosmos Policy
confirmatory report](../confirmatory-results.md), and [content-factorization
report](../content-factorization-results.md). The immutable artifact hashes are
indexed in the [final artifact index](../overnight_2026-09-03/final_artifact_index.md).
DreamZero values come from the frozen [final core analysis](../../output/deadline_2026_09_04/dreamzero/core_analysis_final/summary.json)
and [control analysis](../../output/deadline_2026_09_04/dreamzero/control_analysis/summary.json).
Their successive-fix provenance chain is recorded in the
[DreamZero provenance reconciliation](dreamzero_provenance_reconciliation.md).
An additional excluded-state [clean-upstream parity audit](../../output/deadline_2026_09_04/dreamzero/upstream_native_parity/execution_receipt.json)
ran the untouched pinned official commit and patched mode-off on the same
input. Their 24x8 float32 action arrays were bitwise identical (maximum
absolute error 0). The immutable execution-receipt SHA-256 is
`f9f6294d582486d97c8a2def87c63d770985fa022016660b744df54489db9c11`,
and the artifact-index SHA-256 is
`5d2e3c71b85c9e2327337139062715aabbe9c6beae4b7d4fc2a5139c74da270f`.
The corrective [runtime provenance addendum](../../output/deadline_2026_09_04/dreamzero/provenance_addendum/runtime_provenance_addendum_v1.json),
SHA-256 `b7ce05309878376cc5f1fa1c091c4fa5007a7c9d705f7481cf902a9c54878078`,
supersedes only the stale server-log field in the raw receipt and explicitly
scopes its 258-package environment census as a postrun snapshot.
All 120 DreamZero native futures were regenerated as decoded MP4s after the
frozen analysis, without appearance or outcome selection. The exhaustive
[media receipt](../../output/deadline_2026_09_04/dreamzero/all_native_media/receipt.json),
SHA-256 `89c7f4742b42698d2a2cb122f63c5792d6e53da070280c019ad18a1fb3acacdb`,
records bit-exact action reproduction and byte-exact latent-trace reproduction
for every branch. These videos are descriptive media, not semantic labels.
LingBot-VA values come from the frozen packaged [core summary](../../output/deadline_2026_09_04/lingbot/core_artifacts/summary.json),
with exact native-inference parity documented in the excluded-state
[oracle receipt](../../output/deadline_2026_09_04/lingbot/upstream_parity/upstream_native_parity.json).
The separate LingBot dose values come from its frozen [dose summary](../../output/deadline_2026_09_04/lingbot/dose_analysis/summary.json)
and [artifact index](../../output/deadline_2026_09_04/lingbot/dose_analysis/artifact_index.json).
The complete read-only LingBot package is bound by its recursive
[artifact index](../../output/deadline_2026_09_04/lingbot_final_package_v1/artifact_index.json),
SHA-256 `698a818bce0f8dbeda22aee7df76752dce70f1d70089c6dd79cedf3e3faf273e`.
The complete core and dose raw trees omitted from that package are preserved in
the separate read-only [raw addendum](../../output/deadline_2026_09_04/lingbot_final_raw_addendum_v2/raw_addendum_receipt.json),
whose receipt SHA-256 is
`6208bd9109c29e00b1e5b8c6ce3b4f7c3c40fd8b5b87f4e2e14017370bfdb779`
and whose 275-entry recursive index SHA-256 is
`0e6ca53c3403434518a733fdc6d03532282be4941464a8d667b6ab1b3e9f62fb`.
All 120 native LingBot future latents were also decoded after the frozen
analysis, without visual or outcome selection. The exhaustive
[decode provenance](../../output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1/provenance.json)
binds the official decoder and all 30 states x four sources; its umbrella
artifact-index SHA-256 is
`f89a96b3b12c35e25cc121284c84dc83de1edb2c8cfc28b3e0faadcaa6c3b332`.
These images are a descriptive representation audit, not evidence that branch
differences are task-semantic.
The separate immutable [decode execution addendum](../../output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1_execution_addendum_v2/execution_receipt.json)
preserves the excluded failed smoke, successful smoke, both exhaustive launch
logs, finalizer log, and the fail-closed-only decoder delta; its artifact-index
SHA-256 is
`052f4c41f132ae1a42ed6eb5a715e8f744887e64c166a7febea235949a313ec3`.
The exact 30 x 4 Gaussian control tensors omitted from the original core tree
are preserved in a separate immutable
[latent-provenance addendum](../../output/deadline_2026_09_04/lingbot_gaussian_latent_provenance_addendum_v1/receipt.json),
with zero reconstruction discrepancies, 120/120 bitwise-exact present frames,
and artifact-index SHA-256
`77848b2450246dfa1b00d0ee3d0a8e182caab74c7c807da0c5ef0bc38c134983`.
The post-analysis complete Gaussian-source grid is summarized in its frozen
[control artifact](../../output/deadline_2026_09_04/lingbot_gaussian_grid_v2_final/summary.json).
Its analyzer/package artifact-index SHA-256 is
`6b53379df8c5a1f0030e7429df8527df33f68529482d144e60557ef8121af0ad`,
with tree aggregate
`9cb7d5b91ff1f19efc7bce2afbf9a5b0002682537a7c2e00455d151af2da2e1c`.

## Matched new evaluation and decision rules

The DreamZero and LingBot-VA cohorts each contain 30 frozen native-domain
states, four native predicted-future/action branches per state, and a complete 4 x 4
recipient-noise x future-source grid. DreamZero selection was deterministic and
outcome-independent, using predeclared balancing of three states across each of
ten verb families; copied success metadata was not used by eligibility,
ranking, quotas, or selection. This is a no-outcome-filter design, not a claim
of human blinding. State is the independent unit; directions and seeds are
within-state measurements. Analyze the frozen population only after all
completeness and control gates pass.

For DreamZero and LingBot-VA, the frozen audits establish four distinct native
latent tensors and source hashes, not decoded semantic differences among the
videos. Describe their result as **native predicted-future representation/source
identity**, not as a verified visual-content or semantic-future effect.
The [DreamZero decoded representative](../../output/deadline_2026_09_04/dreamzero/representative_media/)
was selected by the frozen median-effect rule and reproduces all four frozen
native actions bit-for-bit; LingBot's
[overview video](../../output/deadline_2026_09_04/lingbot_final_package_v1/core_artifacts/media/all_states_overview.mp4)
shows all 30 states in predetermined order. Both are descriptive media, not
additional inferential evidence.

DreamZero additionally has an exhaustive post-analysis decode of every one of
the 30 frozen states and four registered native branches. The immutable
[all-native-media receipt](../../output/deadline_2026_09_04/dreamzero/all_native_media/receipt.json),
SHA-256 `89c7f4742b42698d2a2cb122f63c5792d6e53da070280c019ad18a1fb3acacdb`,
binds 120 H.264 videos to the frozen core: every regenerated action is bitwise
identical and every matched-noise trace is byte-identical to its source. Its
artifact-index SHA-256 is
`8907b7f854f7ea5217cfdce842bb56d2cc86649fd2fa53eed11480966f5f5aa6`.
The separate selection-neutral
[manifest-order overview](../../output/deadline_2026_09_04/dreamzero/all_native_media_derived/all_30_states_4_branches_overview.mp4)
and [30 x 4 terminal contact sheet](../../output/deadline_2026_09_04/dreamzero/all_native_media_derived/all_30x4_terminal_contact_sheet.png)
include every branch without appearance or outcome selection. Their derived
receipt SHA-256 is
`80d0180c3df8fa88e715c7869c1345661c2fd4d85dbd7d062dd9d326cf6b533b`.
These decodes audit what was generated; they are not additional inferential
units and were not semantically annotated.

### Headline-positive pattern

A model supports **source-specific directional action selection** if:

1. all 30 registered states are accounted for, with every technical failure
   reported and no outcome-dependent replacement;
2. exact self-transplant and record/replay controls pass at their frozen
   tolerances; clean official-native parity passes on separate excluded inputs
   for both DreamZero and LingBot-VA; DreamZero additionally has exact
   patched-server mode-off/record parity plus exact self replay;
3. all four registered futures are distinct, the recipient action noise is
   hash-identical across each grid row, and the intervention never writes action
   coordinates;
4. **off-diagonal** four-way correct-source retrieval has a lower state-level
   confidence bound above 25% and a significant within-state donor-label
   permutation test;
5. distance reduction toward the correct donor has a confidence interval above
   zero;
6. projection and cosine alignment are positive, while orthogonal residual is
   reported rather than hidden; and
7. within-state label permutations yield the 25% null expectation, while the
   norm-matched Gaussian control is reported as a perturbation diagnostic;
   arbitrary Gaussian-source retrieval and alignment to native donors are kept
   separate, and neither is treated as coherent semantic content.

Treat off-diagonal retrieval as the claim-facing primary. Report the registered
all-cell 4 x 4 retrieval as a completeness/audit summary, not as the
substantive headline: its four diagonal cells are exact self-controls, so it
mixes off-diagonal source steering with deterministic identity replay.

### Metric-discordant patterns

| Observed pattern | Claim-safe conclusion |
|---|---|
| Retrieval above chance **and** distance reduction above zero | Strong source-specific directional steering. Projection/cosine/residual determine how nearly the recomputed action reaches the donor. |
| Projection above zero but retrieval at chance | The future perturbs action along pairwise donor axes, but does not reliably select the correct source. Do not claim donor identity selection. |
| Retrieval above chance but distance reduction interval includes zero | Some source identity is decodable from the action geometry, but the intervention does not reliably move closer to the paired donor action. Report as partial evidence. |
| Large displacement with large orthogonal residual and weak retrieval | Generic or off-manifold perturbation is more plausible than source-specific steering. |
| Gaussian perturbations strongly erase recipient identity but native transplants retain correct-donor retrieval | The model is sensitive to arbitrary off-manifold future trajectories, yet native predicted-future sources produce source-specific rather than merely large changes. Claim directional source specificity, while disclosing nonspecific instability; do not claim robustness to arbitrary future replacement. |
| The permutation null is not centered at chance | Stop: an analysis/control failure is viable. Do not interpret retrieval until resolved. |
| Exact self, the declared model-specific no-op/native-route parity check, action-noise identity, or action-coordinate non-write fails | Invalid intervention, not a scientific null. Fix plumbing only on excluded states and rerun a newly frozen evaluation if the change can affect outcomes. |
| Four futures are unique but native actions are nearly indistinguishable | The state is uninformative for source retrieval. Retain and disclose it under the no-filter rule; report separation strata and tie handling. |
| Complete powered cohort passes controls but retrieval remains at chance and distance reduction remains near zero | Interpretable model/domain/interface null. It limits generality; do not tune seeds or select states post hoc. |

### Completed cross-model result

Source-specific directional action steering now appears in Cosmos 3,
DreamZero's chunk-autoregressive system with joint within-chunk denoising, and
LingBot-VA's ordered video-first/action-second inference route. The LingBot
result rules out an explanation that requires simultaneous video/action
denoising during the action stage. It does not rule out learned cross-modal
consistency, and it does not establish that any system compares futures or
plans over their task consequences. FastWAM provides an additional supporting
replication of an explicitly staged future-to-action interface; Cosmos Policy
provides earlier action-and-execution evidence but not a matched four-source
retrieval cohort.

## Joint consistency versus ordered future conditioning

### Joint within-call denoising: Cosmos 3 and DreamZero

The installed donor future is part of a shared video/action computation. A
donor-directed action therefore establishes a causal counterfactual coupling:
when future coordinates are changed and action coordinates/noise are not, the
action changes in the corresponding direction. It does **not** establish that
the future is naturally upstream of the action. A jointly denoised sampler may
be reconciling two coupled variables with the imposed future, and ordinary
generation may allow action-to-future influence as well.

The official DreamZero paper is more specific than the shorthand “joint
model”: video is autoregressive across chunks, but video and action are jointly
denoised within each chunk. Its objective is described as video prediction plus
inverse-dynamics action prediction inside one end-to-end model. The experiment
therefore tests a second joint *within-chunk* computation, not a purely
bidirectional sequence model and not a separate video-then-action pipeline.

DreamZero's matched-noise trace design improves the intervention-strength
match: at each of 16 solver steps it replays the donor's future latent at the
same noise level while validating the active action noise against a separate
recipient-native trace. The completed result replicates directional joint
coupling outside Cosmos. It still does not, by itself, distinguish
future-to-action computation from bidirectional joint consistency.

DreamZero does not yet have a future x K/V factorial. In the released
attention implementation, the persistent cache contains prior context, while
the current future-video K/V is formed inside each joint block immediately
before attention with action/state tokens. Independently crossing current
future content and its direct K/V would require a separately audited
action-query attention path across every block and solver call. Swapping the
persistent cache would test the wrong representation, while globally replacing
current K/V would also alter video/state queries. Do not inherit Cosmos 3's K/V
pathway result for DreamZero.

The DreamZero dose response provides a second, graded check. Across 30 states,
mean donor-axis projection was 0.046 at alpha 0.25, 0.465 at alpha 0.50, and
0.939 at alpha 0.75. The three independently computed interior points were
monotone in 30/30 states; the state-level interior slope was 1.786 [1.707,
1.853], with a one-sided sign-flip p-value of 0.00001. Alpha 0 and 1 were exact
reused core cells and remain descriptive endpoints. This supports graded
control by the imposed latent trajectory, not linear semantic interpolation or
natural mediation. A per-step norm-matched Gaussian future caused a normalized
action displacement of 2.070 [1.702, 2.476] and left the recipient action
nearest in only 37/120 calls (0.308 [0.275, 0.350]). Thus the model is also
highly sensitive to incoherent futures. The Gaussian result is evidence of
nonspecific instability, not a recipient-preserving null; the native-source
result's distinctive evidence is correct donor identity, not displacement
alone.

Cosmos 3's future x K/V factorial adds a mechanistic statement: while visible
future identity is crossed independently, direct future-token K/V available to
action queries redirects the action toward the K/V source. This supports a
large contribution through that audited interface. The residual visible-future
effect and the intervention's imposed nature rule out “complete mediation,”
“the unique pathway,” or a classical natural indirect-effect interpretation.

### Ordered video-first/action-second inference: LingBot-VA

The released LingBot-VA system is not two independently trained stages. Its
paper describes a unified chunk-autoregressive model with interleaved
video/action tokens, dual streams, and shared attention. Its released per-chunk
inference procedure is nevertheless ordered: it first partially denoises the
future-video latent, then denoises the action conditioned on that prediction
and the existing cache. In the pinned runner, the installed donor future is
materialized as predicted-token K/V read during action denoising. The tested
route is therefore computationally ordered future -> cache -> action. The
completed 30-state evaluation shows that a donor cache redirects the action
with recipient action noise fixed: off-diagonal four-source retrieval is 0.747
[0.667, 0.819], versus a 0.25 label-permutation null, and mean donor-axis
projection is 0.675 [0.624, 0.723]. Simultaneous action-to-future co-adaptation
during the action stage therefore cannot explain this effect.

This is stronger evidence about direction of the *implemented interface*, not
evidence that the model evaluates task outcomes. In the current runner, once a
cache override is supplied, action generation reads the installed cache rather
than an independently active raw future latent. The exact crossed controls make
that routing explicit: donor future with recipient cache has projection 0,
whereas recipient future with donor cache has projection 0.675 [0.624, 0.723].
The action follows future-derived cache identity, not the nominal raw-future
label. The apparent raw-future x cache cross is consequently a cache-routing
control, not an identifiable two-path factorial and not a new circuit
discovery.

A separate post-analysis exploratory control crossed four recipient action
paths with four independently generated, branch-statistic-matched Gaussian
future sources in every frozen state. Gaussian-source retrieval was 42/360,
or 0.117 [0.094, 0.139], and native-donor alignment was 78/360, or 0.217
[0.192, 0.239]; neither exceeded its 0.25 permutation null. Projection toward
the Gaussian-source template was positive (0.167 [0.156, 0.178]), but distance
reduction was negative (-0.309 [-0.329, -0.288]), illustrating why projection
alone is insufficient. Relative to native actions, projection was -0.097
[-0.214, 0.009] and distance reduction was -0.014 [-0.019, -0.008]. The native
recipient remained nearest in only 126/360 cells, or 0.350 [0.292, 0.417]. Thus
arbitrary norm-matched Gaussian sources did not recover Gaussian-source identity
or native-donor alignment above chance and did not reliably preserve the
recipient; they caused nonspecific disruption in this cohort. Each
source inherits its paired branch's first and second moments and norm, so this
is an arbitrary-source routing audit, not an equal-geometry, semantic-content,
or natural-future experiment.

The separate dose follow-up tests whether this interface changes gradually
along one frozen b0-to-b1 latent segment. For each state, it linearly
interpolates the final normalized future-video latent in float32, casts back to
bfloat16, restores the identical present frame, recomputes the complete
official t=0 predicted-video K/V cache, and denoises action from the same b0
noise. Mean projected response along the A0-to-A1 intervention path at alpha 0,
0.25, 0.5, 0.75, and 1 is 0, 0.221, 0.487, 0.758, and 1.000. Using only the
three newly evaluated interior points,
the mean state-level slope is 1.073 [1.039, 1.109], with a one-sided
within-state label-permutation p-value of 0.00001; all 30 states are
nondecreasing across the interior points. Alpha 0 and 1 are exact core-cell
reuses and are excluded from the inferential test. This supports graded
pathwise sensitivity along that imposed latent/cache trajectory. It does not
show that alpha has a semantic meaning, that interpolated futures are natural,
or that the model naturally mediates actions through this trajectory.

This was a prespecified, frozen follow-up, not a public preregistration. Its
protocol was frozen while 16 of 30 core states had completed, before any core
effect metrics were inspected and before dose execution; the numerical
randomization clarification was frozen at 18 of 30 core states under the same
no-outcome-inspection condition.

## Compact results-table template

The matched core rows below are complete. The LingBot dose follow-up is kept
separate because it tests one fixed latent segment rather than the four-source
retrieval grid.

| System | Organization | Native domain | Frozen states | Off-diagonal retrieval (95% CI), claim-facing | All-cell retrieval (95% CI), audit | Distance reduction (95% CI) | Projection (95% CI) | Cosine | Orthogonal residual | Exact controls | Interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Cosmos 3 | Joint video/action | RoboLab/DROID | 90 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.763 [0.733, 0.792] | 0.966 [0.956, 0.974] | — | 0.227 [0.200, 0.255] | Passed | Audited positive; lossy archival inputs, action-only |
| DreamZero | Chunk-AR video; joint video/action denoising within each chunk | Released DROID training-data corpus | 30 | 360/360 (1.000; permutation p=0.00001) | 480/480 (1.000; deterministic-self audit included) | 0.917 [0.897, 0.933] | 0.990 [0.984, 0.995] | 0.995 [0.991, 0.997] | 0.080 [0.065, 0.099] | Passed | Full 24-step action chunk; first low-level step 348/360; action-only; not model-held-out |
| LingBot-VA | Chunk-AR; video-first, then cache-conditioned action at inference | LIBERO-Long / `libero_10` | 30 | 269/360 (0.747 [0.667, 0.819]) | 389/480 (0.810 [0.750, 0.865]) | 0.473 [0.421, 0.524] | 0.675 [0.624, 0.723] | 0.814 [0.777, 0.847] | 0.388 [0.359, 0.418] | Passed; one excluded input matched upstream exactly | Full predicted action chunk positive; post-hoc first post-conditioning frame group 98/360 and literal first low-level action 48/360; action follows cache source identity; action-only, not physically executed |

Report model-specific intervals and permutation tests; do not pool action
vectors or normalized effects across models into one numerical estimate. Add
per-task/family and leave-one-group-out rows in the appendix.

## Methods and provenance table

| System/study | Frozen source provenance | Cohort and branch design | Intervention and fixed variables | Audit status before interpretation |
|---|---|---|---|---|
| Cosmos Policy | Official `NVlabs/cosmos-policy` checkout pinned at `18a2accadf4e7a3531e56754102af5a24d2316da`; local results in `results/confirmatory_v1` | 30 LIBERO-10 states across ten tasks and three query phases; headline early-state n=10 | Natural future representation transplanted; current observation, instruction, recipient action noise, and schedule fixed; actions physically replayed | Completed saved-state analysis; not a matched four-source cohort |
| Cosmos 3 selection-free | `nvidia/Cosmos3-Nano-Policy-DROID`; checkpoint content manifest `b6cbee5522104145f71cfcc74f5049f61e7de3640b88f14831909c7f473c2660`; experiment manifest `cosmos3-archival-sf-507feb24297971eb`, SHA-256 `8ab294c7a581424cfa02faae1db85941e3f43c1e28100e68e21fa5f533703b9e` | 90 states, six tasks, 30 episodes, early/middle/late; four native sources and all 12 ordered donor pairs | Future coordinates clamped; action coordinates audited zero-write; fixed recipient path; exact native/self/donor/no-op replays | Complete; all prespecified criteria and independent raw audit passed |
| Cosmos 3 future x K/V | Manifest `cosmos3-kv-existing-bb8591311eda8a59`, SHA-256 `972bcab77e2b999c703250f9e7ab17f3d854c858d99712bb7873d0bbf589714f` | 21 existing selected-pair states across six tasks; four future/KV cells | Recipient/donor visible future crossed with recipient/donor K/V at all direct future-to-action attention interfaces | Complete; action-only, selected-pair cohort; exact replay and non-write audits passed |
| LingBot-VA evaluation | Official `Robbyant/lingbot-va` commit `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`; checkpoint `robbyant/lingbot-va-posttrain-libero-long` revision `0e89d1e753019988aba484e8da2dc0810e264d9f`; manifest SHA-256 `9c8d7dd547955e4622d98c7eb42fffed157f6db983f41f02929b28a434be47b4`; launched runner SHA-256 `902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2` | 30 predetermined states: ten tasks x init indices 10, 20, 30; two state-0 development inputs excluded; video seeds 101/211/307/401 and action seeds 1009/2017/3019/4021 | Fixed observation/prompt/initial latent; four pre-sampled recipient action-noise tensors; donor clean future reinstalled to form its 30-layer, 128-slot predicted K/V cache; no action-coordinate write | Complete. All exact controls passed. On one excluded development input, the audit against official upstream `_infer` was bitwise exact for encoder, future, and action with two controlled RNG calls; all 60 attention modules used torch SDPA and the import-only FlashAttention shim was called zero times. Oracle [receipt](../../output/deadline_2026_09_04/lingbot/upstream_parity/upstream_native_parity.json) SHA-256 `f54eebb2d4525ec8d8c57ebdc3b1294041194e5d1e13fd0033380a09453719aa`; packaged [artifact index](../../output/deadline_2026_09_04/lingbot/core_artifacts/artifact_index.json) SHA-256 `0cc2ab978a157496018f1f43514b190630b1074ddd03381299718295bb51bab9`; packaged summary SHA-256 `a6dea90f19892b73359f16f2e79e584ec46482d8ade2d5a835e4eb3aa58aafec`. The complete read-only package has recursive [artifact index](../../output/deadline_2026_09_04/lingbot_final_package_v1/artifact_index.json) SHA-256 `698a818bce0f8dbeda22aee7df76752dce70f1d70089c6dd79cedf3e3faf273e` and tree aggregate `3eef50acc94a348824daba4a7cc534ee2d6ac3d01c8b33629e75ef60b5879316`. The launch receipt did not capture `PYTHONPATH`; the postrun provenance and oracle bind the import-only shim by path/hash and verify zero calls. No checkpoint aggregate was captured at launch; the postrun checkpoint rehash matches prelaunch Hugging Face metadata/etags and file mtimes. |
| LingBot-VA latent-dose follow-up | Same core commit/checkpoint/manifest/oracle; frozen [protocol](../../output/deadline_2026_09_04/lingbot/canonical_protocols/lingbot_future_dose_v1.json) SHA-256 `8b6b4103b5c172f28c896b9834fda114aa52684c53f8c570c78c346fda9d3eba`; numerical [clarification](../../output/deadline_2026_09_04/lingbot/canonical_protocols/lingbot_future_dose_v1_analysis_clarification.json) SHA-256 `2f3ca2211b66100c6d99d44e879b632bee1434da1f8d371fb2ed27f981ee7f8e` | All 30 frozen states; fixed b0 recipient, b1 donor, and b0 action-noise source; alpha 0/.25/.5/.75/1; endpoints reused from core and excluded from inference | Interpolate final normalized future-video latent; restore present; recompute complete official t=0 cache independently at each interior alpha; no action-coordinate write | Complete. Runner SHA-256 `2d8b419be882eb979ed58091f7d0b0cd4322f2503aac9e4a854c558834f21b2e`; analyzer SHA-256 `03b899c5755f52023c094a1347760423f1f4c5d114757b3cc23b6e66ac367ac2`; [artifact index](../../output/deadline_2026_09_04/lingbot/dose_analysis/artifact_index.json) SHA-256 `52211b2f463ed907468f0749783c67500a7a20d6699687cf0749392659d1dd93`; summary SHA-256 `de891927550197f8d3275a14dce515644a09d1895bda7ef74bc244098fecba9d`. Independent recomputation from state metrics exactly reproduced the reported slope and alpha means to floating-point precision; all 11 indexed artifacts match their recorded hashes. |
| LingBot-VA Gaussian-source audit | Same pinned core commit/checkpoint/manifest/oracle; runner SHA-256 `10360ed7e1c166cb7cefd224ce3c936b0b45b57392e5504c4b51e0bdc9ee3e2f`; analyzer SHA-256 `8b24f096932b4a65ab0f6b585fd1053668eda298b7b3e47d15aabca21d9d0581` | All 30 frozen states; complete four recipient action paths x four Gaussian sources generated with fixed seeds 900000–900003 from each native branch's statistics | Install each saved Gaussian future/cache with fixed recipient action noise; no action-coordinate write; compare against both Gaussian diagonal-action templates and native actions | Complete post-analysis exploratory control. Both 15-state raw shards and the [final artifact](../../output/deadline_2026_09_04/lingbot_gaussian_grid_v2_final/summary.json) are immutable and independently rehashed. Final artifact-index SHA-256 `6b53379df8c5a1f0030e7429df8527df33f68529482d144e60557ef8121af0ad`; raw shard indexes `6a7265d6a474d6f584914c35d1fa27d4b51206b65d5b42f6979eeb55bc6887c1` and `6d697a62d344aa3063dbab0907c1f7ed9c84f54d53469d1e9d0ed3025ba7317d`. Two initial wrappers failed at distributed rendezvous before model construction or any result; their logs and dispositions are preserved. Both successful launches completed exactly 15 states with no traceback. |
| DreamZero evaluation | Official `dreamzero0/dreamzero` commit `ab790c198fbce33503358efbbd4187ce9a89adf3`; `GEAR-Dreams/DreamZero-DROID` revision `96ad344138c66e82536422432ad742f015784942`; DROID data revision `2abc197ca7f14f53a6bf464bf80018ce998f18cc`; manifest `dreamzero-droid-states-bef2b2e841db4dd3`, file SHA-256 `d1ffc3111a10bed9ac8fdd17c631dc3a5d8eb3128ac4fa250d9398bcede12cfc`; core runner SHA-256 `e627132e037679717512faac2f7bc46ddda8898f1e7bfe5637445a99e8163019` | 30 states from 30 unique DROID episodes and instruction strings in the official preprocessed training-data release; predeclared balancing of three states in each of ten verb families; middle frame with 48-frame margins; seeds 211/223/227/229; deterministic no-outcome-filter selection. Success metadata was copied into the manifest but was not used by eligibility, ranking, quotas, or selection; no pixels, parquet rows, model outputs, or actions were inspected for selection. These states are analysis-held-out, not model-held-out. | Replay donor video latent at each matched one of 16 joint solver steps; validate active action noise bitwise against recipient-native trace; trace hashes bind source and recipient; intervention writes future tensor only | Complete. Core and control analysis passed all frozen gates. Exact self replay error 0; the excluded patched-server mode-off/record debug gate had maximum absolute error 0. A separate excluded-state [clean-upstream parity audit](../../output/deadline_2026_09_04/dreamzero/upstream_native_parity/execution_receipt.json) ran an untouched checkout of the pinned official commit and patched mode-off on the same input. Their 24x8 float32 actions were bitwise identical (maximum absolute error 0), with execution-receipt SHA-256 `f9f6294d582486d97c8a2def87c63d770985fa022016660b744df54489db9c11` and artifact-index SHA-256 `5d2e3c71b85c9e2327337139062715aabbe9c6beae4b7d4fc2a5139c74da270f`. Gaussian runner `74e6c4d4ab76006aa48f8dcd7666fbf5b9ef786bd639daa684b9cc1e36b606a9`; dose runner `bc180328508c3d687c8003e245584ef7ee194529438a4ccf4dfacb11cb0c5d7d`; control analyzer `bcd3dbc5687f4e3bba941d99792f5b42b1f76b6075c16981b1d852fb7ff5ce57`. The raw [runtime receipt](../../output/deadline_2026_09_04/dreamzero/provenance/final_runtime_receipt.json), SHA-256 `57b89a7a98b3326812fa6652fff1f000f9bbe6940cd82a4e1a72b7983eaa06e5`, binds the exact checkpoint, patch, server launch, and 30-state result maps but contains a stale failed-launch server-log field. The immutable [runtime addendum](../../output/deadline_2026_09_04/dreamzero/provenance_addendum/runtime_provenance_addendum_v1.json), SHA-256 `b7ce05309878376cc5f1fa1c091c4fa5007a7c9d705f7481cf902a9c54878078`, binds the actual evaluated-server log through live process descriptors and records a postrun—not launch-time—258-package environment snapshot. The final core [artifact inventory](../../output/deadline_2026_09_04/dreamzero/core_analysis_final/artifact_inventory.json), SHA-256 `4474b1a7f2d9d2c7bf4682d4d94910782d55e0b26e3ec693b235902e1c500327`, binds final analyzer SHA-256 `7f17970a858927e37a664efbde0451655bac00e374b3e08446d9a7e9295efc30` to the final outputs and embeds the raw receipt. The control [artifact inventory](../../output/deadline_2026_09_04/dreamzero/control_analysis/artifact_inventory.json), SHA-256 `dbc56bd5e282b7c7cd30876206bf15b6a800d3f4e4912af1cf60a5f8b0f7c663`, binds the control analyzer to its outputs and exact source-result hashes. See the [reconciliation index](dreamzero_provenance_reconciliation.md). |

The LingBot and DreamZero revision hashes above were read from local pinned
checkouts/Hugging Face cache refs and frozen manifests. They are implementation
provenance, not bibliographic citations.

## Threat-to-validity checklist

### Before opening results

- [ ] Official repository commit, checkpoint revision/content, runner, patch,
  analyzer, and manifest hashes are frozen; environment evidence is either
  captured at launch or explicitly labeled as a postrun snapshot rather than a
  launch-time lockfile.
- [ ] Development states and every method change made after seeing them are
  excluded from evaluation.
- [ ] All 30 evaluation units are present or accounted for under a frozen
  fail-without-replacement rule.
- [ ] The declared model-specific native/no-op control equals the controlled
  record path at its frozen tolerance; clean upstream parity is established on
  a separately retained excluded-input oracle; self-transplant and
  record/replay are exact.
- [ ] Current observation, instruction, scheduler, recipient action noise, and
  all nonfuture coordinates are invariant; action-coordinate writes equal zero.
- [ ] Four future sources and their caches/traces are unique; native action
  separations and any ties are retained and disclosed.
- [ ] Gaussian construction is frozen before outcome inspection and matches the
  declared per-step norm/statistics without copying native donor structure.

### Analysis

- [ ] State is the independent unit; donors, directions, and RNG paths are
  averaged within state.
- [ ] Off-diagonal retrieval is the claim-facing primary and is tested against
  a prespecified Monte Carlo within-state four-label permutation null; all-cell retrieval
  is reported separately as an audit/completeness result.
- [ ] Retrieval is accompanied by distance reduction, projection, cosine, and
  normalized orthogonal residual; no conclusion rests on projection alone.
- [ ] Confidence intervals, permutation seeds, tie-breaking, degenerate axes,
  missing cells, and all exclusions are explicit.
- [ ] Task/family stratification and leave-one-group-out results are reported;
  models are shown side by side rather than pooled.
- [ ] Wrong native donors, within-state label-permutation tests, Gaussian controls, and exact
  identity controls are interpreted according to what they actually test.
- [ ] Representative videos are selected only after complete quantitative
  analysis under a disclosed rule; they are illustrations, not independent
  evidence.

### Interpretation

- [ ] Joint-model positives are described as directional counterfactual
  coupling or control, not proof that the future naturally precedes action.
- [ ] LingBot cache routing is described as an explicit architectural interface,
  not an independently discovered hidden circuit.
- [ ] “K/V carries most of the effect” is tied to the Cosmos 3 factorial and its
  residual visible-future effect; “complete mediation,” “necessary,” and
  “unique pathway” are avoided.
- [ ] New DreamZero/LingBot results are action-output results unless actions are
  separately executed. They do not inherit Cosmos Policy/Cosmos 3 endpoint
  evidence.
- [ ] No claim says the models compare futures, select the best future, reason
  about success, improve task success, or plan over consequences.
- [ ] No claim says the result is universal or SOTA. A controlled null is kept
  as a boundary condition.

## Suggested paper narrative

1. **Ambiguity.** Co-generation makes an imagined future visually plausible
   but does not show whether its predicted-future representation controls the
   action.
2. **Directional test.** Reachable donor transplantation assigns an expected
   behavioral direction to each intervention. Multi-donor retrieval asks
   whether future A selects action A rather than merely changing the output.
3. **Selection-free anchor.** Cosmos 3 recovers future-source identity across a
   broad frozen state grid, so the effect is not an artifact of maximally
   separated showcase pairs or projection alone.
4. **Cross-organization test.** DreamZero shows that the same directional
   coupling appears when generation is autoregressive across chunks but joint
   within each chunk. LingBot shows that source-specific steering also survives
   when predicted-video computation is explicitly ordered before action
   denoising in the released inference path. Simultaneous joint consistency
   during action denoising is therefore insufficient as a complete explanation
   of the cross-model phenomenon. Its separate dose follow-up shows graded
   pathwise sensitivity along a frozen latent/cache segment in every state.
5. **Pathway, not planning.** Cosmos 3 K/V rescue/redirection shows where much
   of the imposed donor signal reaches action queries. Cosmos Policy's physical
   replay and content studies show that the effect can reach behavior and can
   be state- and camera-dependent. Neither result shows comparison or
   optimization over outcomes.
6. **Conclusion.** Imagined futures can function as controllable prospective
   action representations. Their computational organization differs across
   models, which is precisely why matched interventions across different
   inference organizations matter.

### One-sentence version

> Across three released WAM systems with different inference organizations,
> transplanting a native predicted-future representation redirects the full
> predicted action chunk toward the action associated with that source; an
> ordered cache interface shows that this effect need not arise from
> simultaneous future/action denoising during the action stage.

### Phrases not supported by these experiments

- “The model imagines several outcomes and chooses the best one.”
- “The future naturally mediates the policy's action.”
- “The model plans with task consequences.”
- “K/V fully mediates the effect.”
- “Future transplantation improves task success.”
- “All world action models use futures this way.”
- “Three state-of-the-art architectures” without an independently verified
  benchmark/rank definition.

## Verified primary-source references

Verified 2026-09-04 against official publication records, repositories, and
model cards. These references establish identity and architecture, not a claim
that either system is “SOTA” or comparable to Cosmos under a universal ranking.

### DreamZero

- **Paper:** *World Action Models are Zero-shot Policies*. Seonghyeon Ye,
  Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel
  Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin
  Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang,
  Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong
  Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun
  Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi “Jim” Fan, and
  Joel Jang. arXiv:2602.15922 (2026). Official record:
  <https://arxiv.org/abs/2602.15922v1>. No archival conference/journal venue was
  verified.
- **Official release:** repository README title *NVIDIA DreamZero: World Action
  Models Are Zero-Shot Policies* at
  <https://github.com/dreamzero0/dreamzero>; model-card title
  *DreamZero-DROID: World Action Models are Zero-shot Policies* for DROID
  checkpoint `GEAR-Dreams/DreamZero-DROID` at
  <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>; training-data release
  `GEAR-Dreams/DreamZero-DROID-Data` at
  <https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data>; project
  page <https://dreamzero0.github.io/>.
- **Evaluation-boundary note:** the official repository describes
  `DreamZero-DROID-Data` as preprocessed training data. The frozen 30-state
  cohort was selected without model-output or outcome filtering and held out
  from this analysis workflow, but it is not a model-held-out generalization
  set.
- **Pinned evaluation revisions:** code commit
  <https://github.com/dreamzero0/dreamzero/commit/ab790c198fbce33503358efbbd4187ce9a89adf3>;
  checkpoint revision
  <https://huggingface.co/GEAR-Dreams/DreamZero-DROID/tree/96ad344138c66e82536422432ad742f015784942>;
  dataset revision
  <https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data/tree/2abc197ca7f14f53a6bf464bf80018ce998f18cc>.
- **Verified organization:** the paper describes one end-to-end model that
  autoregresses over video chunks and jointly denoises video and action within
  each chunk. The evaluated checkpoint is the released 14B DROID checkpoint.

### LingBot-VA

- **Final proceedings paper:** *Causal World Modeling for Robot Control*. Lin
  Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Luyao Zhang, Mingrui
  Yu, Zelin Gao, Nan Xue, Boyu Zhou, Xing Zhu, Mingyu Ding, Yujun Shen, and
  Yinghao Xu. *Robotics: Science and Systems XXII* (2026), DOI
  `10.15607/RSS.2026.XXII.016`. Official record:
  <https://www.roboticsproceedings.org/rss22/p016.html>; official PDF:
  <https://www.roboticsproceedings.org/rss22/p016.pdf>.
- **Preprint/release metadata discrepancy:** arXiv:2601.21998v2
  (<https://arxiv.org/abs/2601.21998v2>) and the official repository BibTeX list
  12 authors: Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei
  Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu.
  The final RSS record instead has the 14-author list above. Cite one record
  consistently; do not combine its author list with the other record's venue.
  The RSS landing-page abstract calls the framework “CauVA,” but the official
  RSS PDF, public code, and checkpoint call it “Lingbot-VA”/“LingBot-VA.”
  Describe the evaluated release as LingBot-VA; do not propagate “CauVA” from
  the landing-page inconsistency.
- **Official release:** repository README title *LingBot-VA: Causal World
  Modeling for Robot Control* at <https://github.com/Robbyant/lingbot-va>;
  model-card title *Causal World Modeling for Robot Control* for the evaluated
  LIBERO-Long checkpoint `robbyant/lingbot-va-posttrain-libero-long` at
  <https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long>; project
  page <https://technology.robbyant.com/lingbot-va>.
- **Pinned evaluation revisions:** code commit
  <https://github.com/Robbyant/lingbot-va/commit/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb>;
  checkpoint revision
  <https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long/tree/0e89d1e753019988aba484e8da2dc0810e264d9f>.
- **Verified organization:** the paper describes a unified causal
  chunk-autoregressive model with interleaved video/action tokens, dual-stream
  Mixture-of-Transformers processing, and shared attention. Its published
  final RSS paper specifies three video-denoising steps to flow time 0.6 before
  ten action-denoising steps, with the action conditioned on the predicted
  video transition and cached history. Thus “video-first/action-second at
  inference” is accurate; “two separately trained stages” is not.

The pinned commit and checkpoint revisions in the provenance table identify
the exact evaluated code and weights. The URLs above establish public release
metadata; neither type of evidence independently establishes benchmark caliber
or scientific outcome.
