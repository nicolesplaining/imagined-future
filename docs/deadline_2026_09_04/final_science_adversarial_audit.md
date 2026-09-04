# Final adversarial science audit

**Date:** 2026-09-04  
**Scope:** DreamZero core/controls/media; LingBot-VA core, dose, Gaussian-source
grid, exhaustive decode, and execution provenance; the claim-facing cross-model
handoff; and final compute closure. This audit was performed independently from
the result generators and does not modify the manuscript or live experiment
code.

## Decision

| Result | Decision | Maximum defensible interpretation |
|---|---|---|
| DreamZero four-source transplant | **GO** | Under matched per-step future-latent replay, source identity strongly controls the **generated 24-step action chunk**. |
| DreamZero dose response | **GO, qualified** | The action changes gradually along one imposed recipient-to-donor latent-trajectory segment. This is pathwise sensitivity, not semantic interpolation or natural mediation. |
| LingBot-VA four-source transplant | **GO** | In the released video-first/action-second inference path, future-derived cache identity controls the **full predicted action chunk**. |
| LingBot-VA cache cross | **GO as a routing control only** | Action follows the installed cache. It is not an independently identifiable raw-future-by-cache factorial and does not discover a second pathway. |
| LingBot-VA dose response | **GO, qualified** | The ordered interface responds gradually along one imposed latent segment. |
| LingBot-VA Gaussian-source 4x4 audit | **GO as a post-analysis exploratory control** | Branch-statistic-matched Gaussian sources produced nonspecific disruption: neither Gaussian-source retrieval nor native-donor alignment exceeded chance, and recipient identity was not reliably preserved. This is not an inert or equal-geometry null. |
| Exhaustive decoded-future media | **GO as descriptive QA only** | All 120 native futures/model were decoded without appearance/outcome selection. The sheets rule out gross rendering failures; they do not establish semantic branch differences or add inferential units. |
| Cross-model claim | **GO only in narrow form** | Model-native predicted-future representations can exert source-specific counterfactual control over generated action chunks under transplantation. |
| Planning, natural mediation, task success, universal WAM behavior, or model-held-out DreamZero generalization | **NO-GO** | None is tested by these external-model experiments. |
| Result and compute closure | **GO** | Frozen summaries/tables/hashes reconcile, experimental workers are closed, and all four dedicated H100s were idle at the recorded 11:50 PT check. The two instances remain running. |

## DreamZero: raw-result and statistical audit

Authoritative raw source:

`/lambda/nfs/imagined-future/results/deadline_2026_09_04/dreamzero_eval_v1/states/<state_id>/actions.npz`

The audit reloaded every raw array for all 30 states and recomputed every
off-diagonal cell. The saved arrays have shapes `native_actions = [4,24,8]`
and `replay_actions = [4,4,24,8]`. All arrays were finite; all 30 states were
present; exact diagonal replay error was zero; the action-noise source remained
the recipient in every grid row; and the intervention receipts recorded no
action-coordinate writes. Recomputed metrics agreed with the frozen analysis
to numerical precision (maximum discrepancy `2.22e-16`).

For the full action chunk:

- correct-source retrieval: **360/360** off-diagonal cells;
- distance reduction: **0.917** [0.897, 0.933];
- normalized projection: **0.990** [0.984, 0.995];
- cosine alignment: **0.995** [0.991, 0.997];
- normalized orthogonal residual: **0.080** [0.065, 0.099]; and
- within-state source-label permutation: **p = 1/100001**.

The `[1,1]` state-bootstrap interval for retrieval is mechanically degenerate
because every sampled state has retrieval 1. It must not be described as zero
population uncertainty. Prefer `360/360; permutation p = 1/100001`.

### Exact first-step calculation requested for the paper

This is a post hoc temporal robustness slice, not the frozen primary outcome.
For each state, the audit selected:

```text
native = native_actions[:, 0, :]          # [4 sources, 8 coordinates]
patched = replay_actions[:, :, 0, :]      # [4 recipient noises, 4 sources, 8 coordinates]
```

For each of the 12 ordered off-diagonal recipient/source cells, with recipient
action `a`, source action `b`, and patched action `x`:

```text
retrieval = argmin_j ||x - native[j]|| == source
projection = dot(x-a, b-a) / ||b-a||^2
distance reduction = 1 - ||x-b|| / ||a-b||
```

Cells were averaged within each state; states were then weighted equally.
This yields:

- retrieval **348/360 = 0.9667**; custom 100,000-draw state-bootstrap
  95% interval **[0.9306, 0.9944]**;
- projection **0.9499** [0.9175, 0.9796]; and
- distance reduction **0.7754** [0.7275, 0.8177].

Thus the full-chunk result is not created solely by late action steps. The
first-step result remains secondary because this slice was chosen during the
adversarial audit rather than in the frozen primary analysis.

### DreamZero boundaries and confounds

1. **Not model-held-out.** The 30 states came from
   `GEAR-Dreams/DreamZero-DROID-Data`, which the official release describes as
   DreamZero's preprocessed training data. They were held out from this
   analysis workflow, not from model training.
2. **Outcome-independent, not literally selection-free.** Selection was
   deterministic and did not inspect actions, model outputs, pixels, or
   intervention outcomes, but it balanced three examples from each of the ten
   most frequent first-token instruction families and imposed technical frame
   margins. Say “deterministically selected without outcome filtering.”
3. **Episodes are not tasks.** The cohort contains 30 unique episodes and 30
   unique instruction strings, grouped into ten first-token families. Do not
   write “30 episodes/tasks.”
4. **Decoded media are not semantic annotation.** All 30 states x four native
   branches were decoded after analysis in frozen manifest order. Their 120
   re-exported actions are bitwise equal and their matched-noise traces are
   bytewise equal to the frozen core; terminal-frame inspection found no
   systematic blank, NaN-like, or visibly corrupted outputs. This rendering QA
   still does not label or establish semantic differences among branches. Use
   “native predicted-future representation/source identity,” not “verified
   semantic future content.”
5. **No natural directionality.** DreamZero jointly denoises video and action
   within each chunk. Replaying a donor trace at all 16 matched solver steps
   establishes counterfactual coupling under repeated clamping, not that the
   future is naturally upstream of the action.
6. **No physical behavior outcome.** Actions were not executed in an
   environment. The result concerns generated action chunks, not endpoints,
   task progress, or success.
7. **Gaussian is a disruption diagnostic.** The norm-matched Gaussian future
   caused normalized displacement **2.070** [1.702, 2.476] and preserved the
   recipient as nearest in only **37/120** calls. It does not rule out generic
   sensitivity; source retrieval under native transplants is the specificity
   evidence.
8. **Dose is off-manifold and post hoc to the core.** The dose curve is graded,
   but it linearly interpolates latent trajectories. It does not establish a
   natural or semantic dose of imagined behavior.

DreamZero provenance is acceptable after reconciliation and the runtime-log
correction. The raw-run anchor is
`provenance/final_runtime_receipt.json` (SHA-256
`57b89a7a98b3326812fa6652fff1f000f9bbe6940cd82a4e1a72b7983eaa06e5`).
Its server-log field points to a stale failed-launch log. The immutable
`provenance_addendum/runtime_provenance_addendum_v1.json` (SHA-256
`b7ce05309878376cc5f1fa1c091c4fa5007a7c9d705f7481cf902a9c54878078`)
binds the actual evaluated-server log through the live process descriptors and
records a postrun, not launch-time, package snapshot. It also limits exact
native-route evidence in the original package to patched-server mode-off/record
parity plus exact self replay. A subsequent excluded-state
[clean-upstream parity audit](../../output/deadline_2026_09_04/dreamzero/upstream_native_parity/execution_receipt.json)
compared an untouched checkout of the pinned official commit with patched
mode-off on the same input. The 24x8 float32 action arrays were bitwise
identical (maximum absolute error 0).
Execution-receipt SHA-256 is
`f9f6294d582486d97c8a2def87c63d770985fa022016660b744df54489db9c11`;
artifact-index SHA-256 is
`5d2e3c71b85c9e2327337139062715aabbe9c6beae4b7d4fc2a5139c74da270f`.
This proves action-output parity for one excluded input; it is not a
cohort-wide comparison and does not compare internal future tensors or traces.
The final core inventory (SHA-256
`4474b1a7f2d9d2c7bf4682d4d94910782d55e0b26e3ec693b235902e1c500327`)
binds analyzer `7f17970a...` to the final outputs and embeds the raw receipt.
The control inventory (SHA-256
`dbc56bd5e282b7c7cd30876206bf15b6a800d3f4e4912af1cf60a5f8b0f7c663`)
separately binds the control analyzer. The successive analyzer snapshots do
not alter the frozen raw results.

## LingBot-VA: raw-result and statistical audit

Authoritative raw source:

`/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_eval_v1/<state_id>/actions.npz`

Authoritative packaged core source:

`/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_eval_v1_artifacts_v2`

Final immutable umbrella package:

`output/deadline_2026_09_04/lingbot_final_package_v1`

Its artifact index has SHA-256
`698a818bce0f8dbeda22aee7df76752dce70f1d70089c6dd79cedf3e3faf273e`
and records tree aggregate
`3eef50acc94a348824daba4a7cc534ee2d6ac3d01c8b33629e75ef60b5879316`.
The audit independently rehashed all 140 indexed files and found zero missing,
size-mismatched, or hash-mismatched entries; package files/directories are
read-only (`0444`/`0555`).

The complete core and dose raw trees omitted from that presentation package are
preserved in `output/deadline_2026_09_04/lingbot_final_raw_addendum_v2`. Its
receipt SHA-256 is
`6208bd9109c29e00b1e5b8c6ce3b4f7c3c40fd8b5b87f4e2e14017370bfdb779`,
and its 275-entry recursive artifact-index SHA-256 is
`0e6ca53c3403434518a733fdc6d03532282be4941464a8d667b6ab1b3e9f62fb`.
The indexed local and NFS files are read-only and bytewise reconciled.

All 30 predetermined states were present: ten LIBERO-Long tasks with initial
state indices 10, 20, and 30. The audit independently checked all raw NPZ and
result hashes, tensor shapes/dtypes/finiteness, exact self-latent replay, exact
cache replay, fixed row-wise recipient action noise, four unique future/cache
hashes, cache-routing equalities, and absence of action-coordinate writes. No
errors were found. Every file listed by the packaged artifact index also
matched its recorded size and SHA-256.

For the full post-conditioning action chunk:

- correct-source retrieval: **269/360 = 0.747** [0.667, 0.819];
- distance reduction: **0.473** [0.421, 0.524];
- projection: **0.675** [0.624, 0.723];
- cosine alignment: **0.814** [0.777, 0.847];
- normalized orthogonal residual: **0.388** [0.359, 0.418]; and
- within-state source-label permutation: **p = 1/100001**.

The published analysis resamples the 30 state units. Because there are three
states per task, a ten-task bootstrap is the appropriate sensitivity analysis
for claims extending beyond the fixed tasks. An independent task-bootstrap
gave retrieval **[0.667, 0.825]**, projection **[0.627, 0.730]**, and distance
reduction **[0.425, 0.529]**; the conclusion is unchanged.

### Exact earliest-action calculation requested for the paper

The saved arrays are:

```text
native_executed_actions = [4 sources, 7 action coordinates, 3 frame groups, 4 low-level actions]
latent_grid_executed_actions = [4 recipient noises, 4 sources, 7 coordinates, 3 frame groups, 4 low-level actions]
```

The reported `0.272` diagnostic selects the first **post-conditioning frame
group**, not one low-level action:

```text
native = native_executed_actions[:, :, 0, :]       # [4,7,4]
patched = latent_grid_executed_actions[:, :, :, 0, :]  # [4,4,7,4]
```

Applying the same 12-cell/state retrieval calculation as above gives
**98/360 = 0.2722**, with a custom 100,000-draw state-bootstrap interval
**[0.1722, 0.3778]**. Projection is **0.3507** [0.2618, 0.4414] and distance
reduction is **0.2314** [0.1621, 0.3019]. Retrieval therefore includes the
25% chance rate even though mean donor-axis movement is positive.

If “earliest action step” literally means the first low-level action
(`[..., 0, 0]`, seven coordinates), retrieval is only **48/360 = 0.1333**
[0.0833, 0.1861], projection **0.2535** [0.1724, 0.3378], and distance
reduction **0.1237** [0.0632, 0.1862]. The manuscript must not attach `0.272`
to the phrase “earliest action step.”

### LingBot cache interpretation

In the released inference path, the future is materialized into predicted-token
K/V, and the action stage reads that cache. Once a cache override is installed,
the nominal raw-future tensor is not independently read by action generation.
Accordingly:

- donor future + recipient cache exactly returns the recipient action; and
- recipient future + donor cache exactly returns the ordinary donor-cache
  transplant action.

This verifies cache routing and implementation identity. It is not evidence
for two independent raw-future and cache pathways, and it cannot estimate a
natural indirect effect. The meaningful LingBot finding is that an explicitly
ordered future-derived cache can source-specifically control a later action
chunk, so simultaneous video/action denoising during that action stage is not
required.

### LingBot dose audit

Raw dose source:

`/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_future_dose_v1/<state_id>/actions.npz`

Frozen analysis:

`/lambda/nfs/imagined-future/results/deadline_2026_09_04/lingbot_future_dose_v1_analysis`

The audit recomputed the response from all 30 raw arrays. All alpha-0 and
alpha-1 actions were bitwise equal to the corresponding frozen core cells;
the three interior points were new calls with identical recipient `b0` action
noise; no action coordinates were written; and the analysis CSV agreed to
`2.22e-16`. The LingBot response is specifically the projection of
`A_alpha - A_0` onto the observed predicted-action endpoint segment
`A_1 - A_0`, divided by `||A_1 - A_0||^2`. Its endpoint responses are therefore
0 and 1 by construction; this is not the native recipient-to-native donor axis
used for the DreamZero dose analysis. Mean normalized endpoint-axis response at
alpha `[0,.25,.5,.75,1]` was `[0, .220885, .487489, .757572, 1]`. The mean
interior slope was **1.07337**
[1.03941, 1.10862], all 30 state-level interior curves were nondecreasing,
and every task's mean slope was positive (range 0.969–1.176). The permutation
test was **p = 1/100001**. A ten-task bootstrap sensitivity interval for the
slope was **[1.0375, 1.1088]**.

This is a strong check against an all-or-nothing response, but it is also a
nearly direct test of continuity along a linearly interpolated latent segment.
It does not make that segment on-manifold or semantic. The protocol was frozen
while 16/30 states in the separate **core** cohort had completed, and the
numerical clarification while 18/30 core states had completed. Both preceded
every dose execution and state that no core or dose effect metrics had been
inspected. Call it an **outcome-blind, prespecified follow-up frozen before
result inspection**, not a public preregistration.

### LingBot Gaussian-source audit

The completed exploratory grid crossed all four frozen recipient action-noise
paths with four independently generated, branch-statistic-matched Gaussian
future sources in every state: **30 x 4 x 4 = 480 cells**, including 360
off-diagonal cells. All 120 Gaussian tensors were saved and uniquely hashed;
the exact replay, recipient-noise, cache-identity, and action-nonwrite controls
passed. Independent recomputation from `cells.csv` recovered:

- Gaussian-source retrieval: **42/360 = 0.1167** [0.0944, 0.1389], one-sided
  source-label permutation **p = 1.0**;
- alignment to the paired native donor action: **78/360 = 0.2167** [0.1917,
  0.2389], **p = 0.99995**;
- native-recipient retention: **126/360 = 0.3500** [0.2917, 0.4167];
- Gaussian-template projection **0.1671** [0.1564, 0.1780] but distance
  reduction **-0.3087** [-0.3294, -0.2883]; and
- native-template projection **-0.0971** [-0.2142, 0.0090] and distance
  reduction **-0.0135** [-0.0193, -0.0081].

The positive Gaussian-template projection paired with negative distance
reduction is an explicit example of why projection alone is insufficient. The
control supports native-source specificity only in the bounded sense that
arbitrary Gaussian sources did not recover Gaussian-source identity or native-
donor alignment above chance and did not reliably preserve the recipient. It is
nonspecific disruption, not a no-effect, equal-geometry, semantic-content, or
natural-future null.

The frozen final analysis is
`output/deadline_2026_09_04/lingbot_gaussian_grid_v2_final`: summary SHA-256
`0d4ff650d405c96097443b4954365d15b87b5a916e958378881332cc91d41e3e`,
artifact-index SHA-256
`6b53379df8c5a1f0030e7429df8527df33f68529482d144e60557ef8121af0ad`,
and tree aggregate
`9cb7d5b91ff1f19efc7bce2afbf9a5b0002682537a7c2e00455d151af2da2e1c`.
The two initially launched wrappers failed at distributed rendezvous before
model construction or any state result; the frozen execution receipt preserves
those failures and the two successful 15-state launches. The study itself was
requested after the core and dose results and must remain labeled
**post-analysis exploratory**, even though its analyzer was frozen before the
completed Gaussian outcomes were opened.

### LingBot exhaustive decode and visual-QA audit

The post-analysis decode includes every frozen native future: 30 states x four
branches x 13 frames, with no appearance, semantic, outcome, or effect-size
selection. All 120 core future-tensor hashes and all 120 decoded-array hashes
are unique. The official `VA_Server.decode_one_video` route produced finite
float32 arrays with shape `[13,128,256,3]`. The umbrella artifact-index SHA-256
is `f89a96b3b12c35e25cc121284c84dc83de1edb2c8cfc28b3e0faadcaa6c3b332`
and its tree aggregate is
`6eb59863a5cc7e4764e484cedc481974d140b9e5f2604aefc4dfb85769054217`.

The immutable execution-history addendum v2 records one failed engineering
smoke, one excluded successful smoke, both exhaustive 60-item launches, and the
finalizer. Its receipt SHA-256 is
`561b0db1f153f7cc139cd51d190502bb4e9fb032babf6aea79ed43bf5978238d`,
and its artifact-index SHA-256 is
`052f4c41f132ae1a42ed6eb5a715e8f744887e64c166a7febea235949a313ec3`.
The first smoke failed before installing a scientific output; the final decoder
delta added only fail-closed identity/schema/provenance gates and did not change
the decode computation. This is an execution history for **decoding**, not
evidence that predicted robot actions were physically executed.

I visually inspected the exhaustive 30 x 4 terminal-frame sheet. Every cell
contains a recognizable rendered robot scene, with no systematic blank,
NaN-like, or visibly corrupted outputs. That check establishes gross rendering
integrity only. It does not determine what semantically differs between the
branches, whether a branch is successful, or whether the future is natural.

### Other LingBot boundaries

1. The exhaustive addendum decodes all 120 native futures and passes gross
   rendering QA, but it does not semantically annotate their differences.
   Distinct latent and decoded-array hashes establish tensor/pixel distinctness,
   not task-semantic diversity.
2. The arrays named `executed_actions` are the slice that the upstream LIBERO
   client would execute after dropping a conditioned frame. This study did not
   physically execute those actions. Use “post-conditioning predicted action
   chunk,” not “executed behavior.”
3. The effect grows across the three retained frame groups: source retrieval
   is approximately 0.272, 0.494, and 0.750. The result is therefore primarily
   a later-chunk effect, not reliable immediate source identification.
4. The seventh output channel has near-zero source effect in the full chunk
   (retrieval 0, projection 0.015). Do not imply that every action coordinate
   is controlled. Confirm its semantic label from the released action schema
   before calling it the gripper channel.
5. The evaluated checkpoint is post-trained for LIBERO-Long. These are frozen
   evaluation initial states, not held-out tasks or out-of-domain evidence.
6. The original reference-matched Gaussian controls cause very large
   nonspecific displacement (3.390 [3.017, 3.775]). The later complete
   Gaussian-source grid likewise fails source retrieval and recipient
   preservation. Neither is a successful no-effect negative control.
7. The launch receipt omitted `PYTHONPATH` and did not contain a contemporaneous
   aggregate checkpoint digest. Postrun hashes, prelaunch file mtimes/Hugging
   Face etags, the import-only shim audit, and bitwise official `_infer` parity
   make an implementation mismatch unlikely, but this remains a provenance
   limitation rather than a mathematically impossible mutate-and-restore case.

The upstream `_infer` parity audit covers exactly **one excluded input and one
branch**, `dev_task00_state000` / `b0`; it is not a cohort-wide parity test.
For that singular input, encoder, future, and action matched bitwise under the
controlled two-random-draw injection. Its receipt SHA-256 is
`f54eebb2d4525ec8d8c57ebdc3b1294041194e5d1e13fd0033380a09453719aa`.

## Required claim wording

Safe cross-model wording:

> Across Cosmos 3, DreamZero, and LingBot-VA, replacing a model-native
> predicted-future representation while holding the present and recipient
> action randomness fixed redirected the generated action chunk toward the
> action associated with that source. This source-specific effect appeared in
> both joint-denoising systems and LingBot-VA's ordered
> future-cache-to-action inference path. These interventions establish
> counterfactual control of action generation; they do not show that the models
> naturally compare futures, optimize outcomes, or improve task success.

Safe DreamZero wording:

> Across 30 deterministically selected states from the released DreamZero
> DROID training domain, matched-noise future-latent replay produced 360/360
> correct off-diagonal source identifications and reduced distance to the
> source action chunk by 0.917 [0.897, 0.933].

Safe LingBot wording:

> Across 30 predetermined LIBERO-Long initial states, replacing the
> future-derived predicted-token cache produced 0.747 [0.667, 0.819]
> off-diagonal source identification and 0.473 [0.421, 0.524] distance
> reduction for the full predicted action chunk. Source identification in the
> earliest post-conditioning frame group was near chance, so the result is
> primarily a later-chunk effect.

## Post-audit resolution

Rechecked against `claim_safe_synthesis.md` after the audit edits:

1. **Resolved:** both DreamZero occurrences of “episodes/tasks” now say **“30
   states from 30 unique DROID episodes and instruction strings”** and retain
   the released-training-data boundary.
2. **Resolved:** the controls sentence now says **“Exact self/replay controls
   test implementation drift, while native-source retrieval distinguishes
   directional identity from displacement alone.”** It no longer claims that
   the disruptive Gaussian controls rule out generic perturbation.
3. **Resolved:** LingBot's `0.272` result is identified as the first
   post-conditioning frame group of four low-level actions, while the literal
   first low-level action is separately reported as `0.133`.
4. **Retained correctly:** the synthesis says tensor hashes do not establish
   decoded semantic differences and calls the LingBot dose study an
   outcome-blind prespecified/frozen follow-up rather than a public
   preregistration.
5. **Resolved:** every LingBot dose presentation now identifies the estimand as
   the normalized observed `A_0`-to-`A_1` predicted-action endpoint axis,
   states that 0/1 are fixed by construction, and warns against comparing its
   slope numerically with DreamZero's native recipient-to-native donor axis.
6. **Resolved:** upstream parity is consistently scoped to one excluded input,
   `dev_task00_state000`, and one branch, `b0`; the two development inputs used
   elsewhere are not misrepresented as two parity replications.
7. **Resolved:** the frozen LingBot Gaussian-source 4x4 result is reported as a
   post-analysis exploratory routing audit. Its below-chance retrieval and
   alignment plus weak recipient retention are described as nonspecific
   disruption, not as an inert, equal-geometry, semantic, or natural-future
   control.
8. **Resolved:** the exhaustive LingBot 120-future decode is explicitly
   post-analysis and selection-neutral. Execution-history addendum v2 preserves
   the failed smoke, excluded successful smoke, both exhaustive launches, and
   finalizer without conflating decode execution with physical action execution.
9. **Resolved:** the deadline handoff records final result and compute closure
   at 11:50 PT while stating that the two instances remain running.

No wording issue from the audit's correction list remains unresolved.

DreamZero now also has four post-analysis decoded native-future videos for the
state selected by the frozen median-effect rule
(`droid_episode_006474_frame_000068`; selection-rule SHA-256
`52ddbaf6c4de18fd46999490cf6b9d09763933243b056d4756cb8061938283aa`).
All four local video hashes and sizes match the media receipt, and every
re-exported action is recorded bit-for-bit equal to its frozen core action.
These videos are valid **descriptive media for one median-effect state**. They
were generated after analysis and do not establish semantic diversity over the
30-state inferential cohort; therefore they do not change the semantic-content,
training-domain, action-only, or natural-mediation boundaries above.

DreamZero's later exhaustive media package supersedes the representative-only
coverage concern: it contains all 30 x four native branches in manifest order.
Receipt SHA-256
`89c7f4742b42698d2a2cb122f63c5792d6e53da070280c019ad18a1fb3acacdb`
records 120/120 bit-exact actions and 120/120 byte-exact matched-noise traces;
artifact-index SHA-256 is
`8907b7f854f7ea5217cfdce842bb56d2cc86649fd2fa53eed11480966f5f5aa6`.
This expands rendering coverage, not the inferential sample or semantic claim.

## Final cross-file and compute reconciliation

The audit rechecked `noon_final_results.md`, `README.md`,
`goal_requirement_matrix.md`, `claim_safe_synthesis.md`, and the three
machine-readable CSVs against the frozen DreamZero/LingBot summaries and
receipts. Core and dose counts, estimates, confidence intervals, test labels,
cohort descriptions, scope boundaries, artifact hashes, and all local links
reconcile. The Gaussian CSV's one-sided test header now explicitly says
`one_sided_permutation_p_above_chance`; its two interpretation strings use
“did not recover ... above chance” rather than converting a nonsignificant
positive-direction test into a categorical population null. No numeric value
changed.

The reproducible final handoff checker passes all core-table, dose-definition,
Gaussian-table, local-link, frozen-hash, and compute-closure gates. The compute
receipt SHA-256 is
`7c1bbec4b1833a0c38b408d879616b1b648ea454c54b628efc0b68a742089835`.
At its 2026-09-04 11:50:33 PT check, both dedicated 2xH100 nodes reported zero
experimental processes, zero MiB allocated on every GPU, and 0% GPU
utilization. The LingBot workers had exited normally; the verified DreamZero
server process tree was terminated. The instances were deliberately left
running, not stopped or deleted.

**Final closure verdict:** **GO** for the narrow source-specific
counterfactual-control claim and its model-specific qualifications. The only
central requested scientific element still incomplete is a new independently
identifiable future x K/V factorial for an external model; LingBot's cache cross
must remain an architectural routing control, and DreamZero must not inherit
Cosmos 3's K/V mechanism.

## Final no-go list

Do not claim that these experiments show that a model:

- samples several futures and chooses the best one;
- naturally uses its generated future as a mediator;
- reasons about success or task consequences;
- improves task success or physical behavior under transplantation;
- uses one universal visual carrier or K/V pathway across architectures;
- generalizes on model-held-out DreamZero data or held-out LingBot tasks; or
- behaves this way at every action horizon or in every action coordinate.
