# FastWAM powered cache-factorial v3: independent adversarial audit

Verdict: **PASS, with mandatory interface-level wording.** I found no
claim-breaking completeness, metric, contrast, bootstrap, or gate defect. The
result validates the released Optional-IDM action interface under an explicit
K/V override. It must not be presented as evidence that the model learned to
prefer cache over latent content, as a natural-mediation result, or as an
independent replication of the donor-future effect.

## Frozen objects audited

- Manifest: `fastwam-kvfact-a09195568dd5a17f`; independently recomputed ID
  matches; file SHA-256
  `2cb76a6a4a6012caec72a976302f5dc91f75e15bb1dd9f3f4bc1699ae9a8f8fe`.
- Parent: `fastwam-813f0233b9a2c083`; independently recomputed ID and required
  parent-manifest SHA-256
  `d74edd650f32faf7a0907871ae43e7362b5be19e029bef0b17d055eb114d125a`
  both match.
- Frozen result JSON SHA-256:
  `483847ac549ea9b91919e4a88de5d88b98abd190709f0352d8585ef8583b34dd`.
  The locally synchronized analysis directory is byte-identical to the
  canonical external-node analysis directory.
- Protocol SHA-256 `dcb7f47c11f4070c80e64c295c8860f80e7ef73ae63fc157d28fa80af760e302`;
  runner SHA-256 `c3de7bae85d01411421fd5854fa0181d26621549774955d5c1b076923dcda357`.
  Both equal the hashes embedded in the manifest.
- Pre-outcome analysis record SHA-256
  `9ef76a75f5f2c83c8408702c3240905dbe942d80bac9f389b4c275fea9b08c75`;
  analyzer SHA-256
  `0009a896644c40624acff25f7151e3322fd54663f57737b610edaeb07319e615`;
  shared bootstrap SHA-256
  `ea0f7dde4278bd4904c0820f611f8b0902e4b1eebbf2b6d8a9c37b31958f65a3`.

I used a standalone parser and NumPy recomputation over the raw canonical
JSON/NPZ artifacts and parent actions. It did not import the frozen analyzer or
metric implementation.

## Completeness and raw recomputation

- The manifest contains exactly 120 unique states: four suites x ten tasks x
  three states. Every state has all twelve ordered recipient-donor pairs and
  four cells, giving exactly 5,760 JSON/NPZ pairs. There are no missing, extra,
  malformed, duplicate, or unexpected state/run files.
- The required parent contains exactly 8,640 run JSON/NPZ pairs. Its 120 state
  directories and every parent run record/array used here passed the same
  direct schema, ID, shape, and finite checks.
- All 5,760 factorial model and environment actions are finite float32 arrays
  of shape `(32, 7)`. All 1,440 native axes per cell are nondegenerate; the
  minimum native action-pair L2 distance is `0.259144305276944`.
- Recomputed candidate distances match stored values exactly. Across every raw
  directional metric, the maximum numeric discrepancy from JSON/run CSV is
  `4.44e-16`, with zero Boolean, nearest-label, or degeneracy mismatches.
  State-cell means match to `1.67e-16`.

The independently recomputed cells are:

| future / cache | donor retrieval | donor-distance reduction | donor projection |
|---|---:|---:|---:|
| recipient / recipient | `0 [0, 0]` | `0 [0, 0]` | `0 [0, 0]` |
| donor / recipient | `0 [0, 0]` | `0 [0, 0]` | `0 [0, 0]` |
| recipient / donor | `0.9993056 [0.9965278, 1]` | `0.6828367 [0.6369154, 0.7381214]` | `0.8689926 [0.8329163, 0.9122273]` |
| donor / donor | `0.9993056 [0.9965278, 1]` | `0.6828367 [0.6377824, 0.7375285]` | `0.8689926 [0.8319038, 0.9108990]` |

All intervals above are the frozen 10,000-draw suite-to-task-to-state
hierarchical intervals. The independently recomputed cache main effects are:

- retrieval `0.9993055556 [0.9965277778, 1]`;
- donor-distance reduction
  `0.6828366682 [0.6367925245, 0.7383576703]`;
- donor projection `0.8689926136 [0.8322872837, 0.9134698251]`.

The future-latent main effects and future-by-cache interactions are zero at the
state level for every metric; the reported `-2.60e-18` distance-to-donor future
effect is only floating-point evaluation order. Reimplementing all 28 cell and
21 contrast bootstraps, including the keyed RNG derivation, matches the result
JSON within `5.55e-17`. The two frozen cache-main lower bounds are positive, so
the complete evidence gate independently recomputes as PASS.

The hierarchy is implemented as prespecified: ordered directions are averaged
within state, then suites, tasks, and states are resampled. The four suites are
only four top-level clusters, so the intervals should be read as benchmark-grid
bootstrap summaries rather than asymptotic population guarantees. The result
is nevertheless insensitive to any single observed group: cache-main retrieval
is `0.999074-1.0` and distance reduction `0.655943-0.696102` across the four
leave-one-suite-out estimates; the corresponding leave-one-task-out ranges are
`0.999288-1.0` and `0.679220-0.686593` when the held task's three states are
removed and the remaining 117 state values are averaged.

## Exact causal controls

Direct array comparisons, rather than stored audit scalars, establish:

- recipient-cache actions are invariant to recipient versus donor latent in
  all 1,440 ordered pairs; global model-action and environment-action maximum
  errors are both `0.0`;
- donor-cache actions are invariant to recipient versus donor latent in all
  1,440 pairs; both global errors are `0.0`;
- all 1,440 recipient/recipient cells exactly match the parent `self_cache`
  action; all 1,440 recipient/donor cells exactly match parent `donor_cache`;
  all 1,440 donor/donor cells exactly match parent `donor_latent`;
- every per-run and per-state stored invariance/reference scalar equals the
  direct recomputation exactly.

The nearest native action follows the cache source in 5,758/5,760 arms:
1,440/1,440 in each recipient-cache cell and 1,439/1,440 in each donor-cache
cell. The two apparent misses are the same underlying ordered pair duplicated
by exact future-latent invariance:
`libero_goal_task04_state006_wait30`, recipient `b01`, donor `b00`, nearest
native `b03`. Thus donor-cache rescue is 1,439/1,440, not 1,440/1,440.

The parent native artifacts also confirm that the present is fixed: across all
720 within-state branch pairs, stored first-frame latent maximum difference is
`0.0`; all four futures are distinct in all 120 states, with minimum future
latent pair L2 `32.6618383508`. All 480 cache descriptors report 30 layers, and
no state has duplicate branch descriptors.

The runner-reported native-action and stored-FP16-latent regeneration maxima
are both `0.0`, but the regenerated full-precision latent/cache tensors were
not persisted. Those two checks therefore cannot be independently recomputed
from output artifacts alone; their evidence is the frozen runner plus its
atomic state summary. This is a provenance limitation, not a numerical gate
failure. The action invariances and three parent references above are fully
recomputable.

## Mechanistic interpretation

The exact equality has a sound and important code-level explanation. In the
archived patch, `video_latents_override` determines the candidate video latent,
but a non-null `video_cache_override` skips `prefill_video_cache`. Action
denoising then receives the supplied K/V list, recipient action latent/noise,
text context, and a shape-derived attention mask; it receives no video-latent
content. The separate action generator is seeded only with the frozen recipient
action seed. The patched checkout remains at upstream commit
`7faa71108368fbb3b6885649f112af607427a2d4`, and its two-file `git diff` exactly
matches archived patch SHA-256
`cb9291c112d6ac1c62e5d5e6664e577a62b45387909ecd2a2d3dadbd47188bf1`.

Consequently, same-cache future-swap equality is expected by construction once
the override is active. The empirical value is that the complete run validates
the intervention plumbing and shows that recipient versus donor cache is
sufficient to suppress/rescue the already-established directional action
effect over all 120 states. It does **not** show competition between two active
inputs, a learned preference for K/V, or that changing a latent while naturally
recomputing its cache would have no effect. Donor-latent and donor-cache routes
are two access points to one deterministic pathway, not independent evidence.

Claim-safe formulation:

> Under FastWAM Optional-IDM's explicit cache-override interface, the predicted
> action followed the supplied video-cache source in 5,758 of 5,760 factorial
> arms and was bitwise invariant to the paired latent source. Donor-cache rescue
> succeeded in 1,439 of 1,440 ordered donor comparisons. This validates the
> released architecture's future-to-action cache interface; it does not by
> itself establish natural mediation, closed-loop success, or semantic planning.

## Remaining provenance qualifications

The run manifest pins the parent manifest, runner, protocol, upstream commit,
checkpoint, and statistics, but it does not itself hash every imported local
module, analyzer, launcher/config, or the dirty upstream patch. The archived
closure supplies those hashes separately: launcher
`f7b7205ad791264278b0d38cf7e022a662fd38991a294af7c53b3e036b67dd3e`,
configuration
`38955b9e99c96e67f254dc6276fc96c02d73f00224a0dda7c4beba8aa03db193`,
factorial design module
`c0283c1acda01c37ed388ace0a0639802899778f0392c874f39be48ab7affcf3`,
base runner `2e396728973bc180dfbd129219aaaee7046b82d6ff5d4a2bfe1dff83e8d151d6`,
base utility module
`6092472d92b0c9d67c23bc5d1584fa54b0d82d1335767f0974c3d4322c71ff49`,
and the patch hash above. The completed log SHA-256 is
`7c5876357755e889da2d9a41bc507cf06a070798eaacbef28e71f224acc48a63`;
it has no traceback, fatal, OOM, or nonfinite report.

For archival reproducibility, these separately recorded hashes and the raw
run root must travel with the manifest. This packaging caveat limits claims
about a self-contained manifest, but the exact parent replays and raw
recomputation leave the reported interface-level result intact.
