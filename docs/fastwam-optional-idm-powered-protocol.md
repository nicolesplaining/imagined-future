# FastWAM Optional-IDM powered replication (frozen before outcomes)

## Why this cohort is being run

The preregistered eight-state smoke matrix completed all 576 arms and passed
every frozen implementation and scale criterion before this powered manifest
was created. Correct future-latent and future-cache interventions both exceeded
four-way donor-retrieval chance and their respective controls; exact replay,
no-future invariance, finite-array, and native-future separation checks passed.
The smoke states are calibration data and are not part of this powered cohort.

## Frozen population

- Model: FastWAM Optional-IDM.
- LIBERO suites: Spatial, Object, Goal, and Long (`libero_10`).
- Tasks: all ten tasks in each suite (40 tasks total).
- States: initial-state indices 4, 5, and 6 for every task, after 30 settling
  steps (120 states total).
- Mechanical availability check: all 40 official initial-state files contain 50
  states, so no fallback was needed. The predeclared fallback was to abort the
  entire launch before inference if any requested index did not exist.
- No state, branch, donor pair, or output may be filtered for native success,
  action separation, future appearance, or intervention strength.
- Four branches per state use the smoke-frozen independent RNG pairs: video
  seeds 101, 211, 307, 401 and action seeds 1009, 2017, 3019, 4021.
- All 12 ordered recipient-to-donor pairs are evaluated within every state.
- The complete eight-condition matrix is retained: native, self-latent replay,
  self-cache replay, correct-donor latent, correct-donor cache, wrong latent,
  shuffled cache, and donor-labelled first-frame negative control.
- Expected matrix: 72 arms per state, 8,640 atomic JSON/NPZ run pairs total.

## Frozen provenance and hashes

- FastWAM repository: `https://github.com/yuantianyuan01/FastWAM.git`
- FastWAM commit: `7faa71108368fbb3b6885649f112af607427a2d4`
- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Optional-IDM checkpoint SHA-256:
  `26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4`
- Dataset-statistics SHA-256:
  `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`
- Config: `configs/fastwam_optional_idm_powered.toml`
- Config SHA-256:
  `dd4e1c9036dbbbbf2290665d4d9fd936de16b09229f583a5741ddd7458d6e47b`
- Manifest ID: `fastwam-813f0233b9a2c083`
- Manifest SHA-256:
  `d74edd650f32faf7a0907871ae43e7362b5be19e029bef0b17d055eb114d125a`
- Manifest path:
  `/lambda/nfs/imagined-future/results/overnight_2026_09_03/fastwam_optional_idm_powered_v1/manifest.json`

## Frozen outcomes and inference

The independent unit is the saved simulator state. The 12 ordered donor pairs
and four branches are within-state measurements and are averaged before
population inference.

Primary outcome:

- Four-way correct-donor action retrieval under the correct future-latent
  transplant.

Mechanistic co-primary outcome:

- Four-way correct-donor action retrieval under the correct future-cache
  transplant.

Secondary outcomes:

- Fractional reduction in distance to the correct donor action.
- Normalized donor-axis projection.
- Cosine alignment with the donor axis.
- Orthogonal residual normalized by native recipient--donor separation.

Controls and paired contrasts:

- Correct latent minus wrong latent, paired within state.
- Correct cache minus shuffled cache, paired within state.
- Self-latent and self-cache maximum replay error.
- First-frame maximum action variation across donor video seeds with the
  recipient action seed fixed.
- Native future-latent pairwise separation.

Population intervals use 10,000 deterministic hierarchical bootstrap draws
(seed 20260903): resample suites, tasks within selected suites, and states within
selected tasks; within-state branch/pair averages are never treated as
independent observations. State-bootstrap intervals are also reported as a
transparent sensitivity analysis, together with suite- and task-level tables.

The primary external-replication criterion is a hierarchical 95% lower bound
above 0.25 for correct-latent retrieval. The cache-route criterion is the same
for correct-cache retrieval. The paired treatment-minus-control retrieval and
distance-reduction contrasts must have hierarchical 95% lower bounds above
zero. Exact controls must remain within `1e-6`, all arrays must be finite, and
all four native future latents per state must be distinct above `1e-6`.

Degenerate native recipient--donor action axes are never removed. Retrieval and
distance remain reported for those rows; mathematically undefined directional
metrics are left null, and every such row is counted at run, state, task, suite,
and aggregate levels.

## Launch rule

The manifest, config, this protocol, and their hashes must be recorded in the
decision/run ledgers before either powered shard begins. Both H100s may be used
because the complete frozen smoke gate passed. Sharding follows manifest order
only; shard membership does not depend on any model output.
