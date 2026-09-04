# Cosmos 3 existing-cohort K/V factorial bridge

## Scope

This study extends the existing 22-state Cosmos 3 paper cohort with the full
predicted-future × future-token-K/V factorial. It is a fresh model-only rerun
from archived current observations, recorded proprioception, and the cohort's
already selected recipient/donor seeds. It is **not selection-free**, does not
execute physical endpoints, and does not replace the frozen fresh 48-state
selection-free study.

The excluded development state is `BananaInBowlTask_seed_103`. The remaining
21 saved states across six tasks are the independent units.

## Frozen intervention

For each state, the recipient observation, instruction, action noise, sampler,
and denoising schedule are fixed while crossing:

1. recipient future / recipient future-token K/V;
2. donor future / recipient future-token K/V (suppression);
3. donor future / donor future-token K/V;
4. recipient future / donor future-token K/V (rescue).

The K/V patch covers every architecture-defined direct future-to-action
connection (layers 0–35), preserves token count and order, and does not write
action coordinates.

Exact gates require native repeat identity, record/replay identity, no-op
recording relative to the corresponding uninstrumented clamp, stable visible
future identity across K/V crossings, distinct recipient/donor futures, one
model-state fingerprint, and zero action-coordinate overwrite error.

## Frozen analysis

Report all four cell means, K/V effects at each fixed visible future, visible-
future effects at each fixed K/V source, the factorial interaction, state-level
suppression/rescue counts, per-task results, leave-one-task-out estimates, and
10,000-sample task→state hierarchical bootstrap intervals. No row may be
silently excluded after launch.

## Immutable artifacts

- Active manifest:
  `/lambda/nfs/imagined-future/results/overnight_2026_09_03/cosmos3_kv_existing_v3/manifest.json`
- Manifest ID: `cosmos3-kv-existing-bb8591311eda8a59`
- Manifest SHA-256:
  `972bcab77e2b999c703250f9e7ab17f3d854c858d99712bb7873d0bbf589714f`
- Single-state runner SHA-256:
  `83285e99b993e7f996a40189332643338e33805fe03e00b931f0c214e32179de`
- Sequential launcher SHA-256:
  `dd3cab800431ddf542bab574a528f3971d5df813750720b939c3a1247e2e68a4`

The v1 manifest (`cosmos3-kv-existing-a554af44e7374424`) and v2 manifest
(`cosmos3-kv-existing-6b193ac193933863`) were frozen but superseded **before
any evaluation-state outcome was generated**. V2 added a one-shot worker exit
after its result is atomically written, flushed, and fsynced. V3 also requires
an explicit output-scope label so evaluation rows cannot inherit the excluded
development label. The scientific design and intervention never changed. All
manifests remain preserved.

An excluded-development exit test passed every exact gate and returned exit
code 0 in 15.1 seconds. Its result is retained under `smoke/` and is never
admitted to this study.
