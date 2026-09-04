# FastWAM future-latent x video-cache factorial (frozen before outcomes)

## Question

The powered eight-condition study contains three logical cells of the
recipient/donor future-latent x recipient/donor video-cache factorial. This
separate additive study supplies the missing donor-future/recipient-cache cell
and reruns all four cells through the explicit cache-override interface. It does
not modify the powered population manifest, its global condition enum, or its
running processes.

For recipient branch A and donor branch B, with A's action noise held fixed:

| Future latent | Video K/V cache | Condition |
|---|---|---|
| A | A | `future_recipient_cache_recipient` |
| B | A | `future_donor_cache_recipient` (missing suppression cell) |
| A | B | `future_recipient_cache_donor` (rescue) |
| B | B | `future_donor_cache_donor` |

The architecture-level prediction is that action depends on the supplied video
cache, not on the latent used only to establish the fixed token shape and mask:

- B-future/A-cache matches A-future/A-cache and remains recipient-like.
- A-future/B-cache matches B-future/B-cache and becomes donor-like.

## Frozen population and execution

- Parent powered manifest: `fastwam-813f0233b9a2c083`.
- Same 120 untouched states: all 40 LIBERO tasks x indices 4, 5, 6.
- Same four video/action seed pairs and all 12 ordered A-to-B pairs per state.
- Four factorial cells per ordered pair: 48 registered arms/state and 5,760
  atomic JSON/NPZ run pairs total.
- Independent unit: saved simulator state. Pairs are averaged within state.
- Execution begins only after all 8,640 parent powered arms complete. No partial
  parent or factorial outcome is inspected.
- Parent actions and stored FP16 future-latent artifacts are loaded by frozen
  run ID. The full-precision native future latent and video cache are
  deterministically regenerated from each branch's frozen video seed.
- Regeneration must reproduce the parent's native action within `1e-6`, and
  the regenerated latent cast through FP16 must reproduce the stored latent
  artifact within `1e-6`, before a state's factorial outputs are admitted.

## Exact controls and non-consumption audit

For every ordered pair, the extension checks:

- A-future/A-cache against the parent `self_cache` action.
- A-future/B-cache against the parent `donor_cache` action.
- B-future/B-cache against the parent `donor_latent` action.
- B-future/A-cache against A-future/A-cache.
- B-future/B-cache against A-future/B-cache.

The final two comparisons directly audit that changing the supplied video
latent does not change action once K/V is overridden. The patched FastWAM path
still validates the latent's same-present first frame and constructs fixed-size
attention masks, but it skips video-cache prefill and passes only the override
K/V tensors into every action-denoising call. All exact comparisons use maximum
absolute action error and the frozen `1e-6` tolerance.

## Outcomes and inference

Primary factorial outcomes are four-way correct-donor retrieval and donor
distance reduction in all four cells. Projection, cosine alignment, and
orthogonal residual ratio are secondary. Every degenerate native action axis is
retained and counted.

Population intervals use 10,000 deterministic suite-to-task-to-state bootstrap
draws with seed 20260903. The decisive interaction is cache dominance:

- With recipient cache, donor-future minus recipient-future action change is
  exactly zero within tolerance.
- With donor cache, donor-future minus recipient-future action change is exactly
  zero within tolerance.
- Averaged across future latents, donor-cache retrieval and distance reduction
  exceed recipient-cache values with hierarchical 95% lower bounds above zero.

## Frozen provenance

- FastWAM commit: `7faa71108368fbb3b6885649f112af607427a2d4`
- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Checkpoint SHA-256:
  `26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4`
- Dataset-statistics SHA-256:
  `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`
- Parent manifest SHA-256:
  `d74edd650f32faf7a0907871ae43e7362b5be19e029bef0b17d055eb114d125a`
- Factorial config SHA-256:
  `dbd9fb224daed4d261be86d093f5e6e4d57e306ec8672175640e5d646551cb12`
- Factorial manifest ID: `fastwam-kvfact-ae55b7ed720094d4`
- Factorial manifest SHA-256:
  `d6eb93b00af45ef7449247e7b038c8daa2907fdd25ed4ba4cb1e4275a95425b9`
- Factorial manifest path:
  `/lambda/nfs/imagined-future/results/overnight_2026_09_03/fastwam_cache_factorial_powered_v2/manifest.json`

Any runtime-only repair is logged with its code hash and is not permitted to
alter these states, seeds, cells, outcomes, comparisons, or thresholds.
