# Adversarial audit: frozen FastWAM Optional-IDM smoke

**Verdict: PASS for the preregistered smoke implementation gate, with wording and
provenance qualifications below. No claim-breaking defect was found.** This audit
supports the narrow claim that, in these eight frozen states, changing only the
future-derived conditioning while holding recipient action noise fixed changes the
predicted action toward the action associated with the transplanted future. It does
not support treating latent replay and cache replay as two independent pathways or
generalizing beyond a smoke cohort.

## Scope and provenance

- Audited only manifest `fastwam-dd8450f367664a53` and its canonical completed
  output tree at
  `/lambda/nfs/imagined-future/results/overnight_2026_09_03/fastwam_optional_idm/fastwam-dd8450f367664a53`.
  No file or outcome under `fastwam_optional_idm_powered` was listed, opened, or
  analyzed.
- The frozen manifest recomputes to its recorded ID and has SHA-256
  `c872040226b259a8dd6d64b8966818a0232cdf4965dd3c24709df475469a92e7`.
- The execution node had FastWAM HEAD
  `7faa71108368fbb3b6885649f112af607427a2d4`; its two modified source files were
  byte-identical to applying the archived patch to a clean checkout. Both the live
  diff and archived patch hash to
  `cb9291c112d6ac1c62e5d5e6664e577a62b45387909ecd2a2d3dadbd47188bf1`.
- LIBERO was clean at `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
  Checkpoint and statistics hashes independently matched the frozen constants:
  `26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4`
  and `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`,
  respectively.
- The last run and state-summary completion markers predate the completeness and
  outcome analyses by about 280 seconds, consistent with outcome loading only after
  matrix completion.

## Artifact and numerical checks

Independent enumeration from the manifest produced exactly 576 unique run IDs:
32 each of `native`, `self_latent`, and `self_cache`, and 96 each of the other five
conditions. All 576 JSON/NPZ pairs and all eight valid state summaries were present;
there were no missing, unexpected, malformed, non-finite, or wrong-shaped arrays.
Every action was `(32, 7)` float32 and every native video latent was
`(1, 48, 3, 14, 28)` float16. Native action pair distances were 0.3325--0.9864 and
native video-latent distances were 37.4781--60.4310, so no directional axis or
future comparison was degenerate.

All 480 stored directional metric records were recomputed from NPZ arrays. Boolean
results agreed exactly; the largest floating disagreement was `7.11e-15`, and the
smallest nearest-candidate margin was `3.64e-4` (no tie). Recomputing state means,
the keyed 10,000-draw state bootstrap, and the two paired state contrasts directly
from raw actions agreed with `fastwam_results.json` to `3.55e-15`. Selected results
were:

| Quantity | Mean | Frozen 95% bootstrap interval |
| --- | ---: | ---: |
| Donor-latent retrieval | 1.000 | [1.000, 1.000] |
| Donor-cache retrieval | 1.000 | [1.000, 1.000] |
| Donor-latent distance reduction | 0.6075 | [0.5586, 0.6571] |
| Wrong-latent distance reduction | 0.0060 | [-0.0199, 0.0318] |
| Donor-cache minus shuffled-cache retrieval | 0.7708 | [0.7500, 0.8125] |

As a deterministic spot check, for
`libero_spatial_task00_state000_wait30`, recipient `b00`, donor `b01`, the native,
self-latent, and self-cache action byte hashes were all
`fc7d2a5882452bab4c5713b9a1bc5fd88d7ce0452ac1b4042e8769bc7c693627`.
Donor-latent and donor-cache hashes were both
`5485427811c102e5f2fb48142f4eaafbc43537ddc74074d6c7ca4c0d5f639226`;
independently recomputed donor retrieval was `true`, donor projection
`0.73637448`, and donor-distance reduction `0.52175261`.

## Causal/RNG audit

The patch creates separate local `torch.Generator` objects for video and action
noise. The runner always supplies the recipient's frozen action seed to every arm.
Donor identity is never an inference argument: it is used only to choose the latent
or cache before inference and to calculate metrics after inference. For override
runs, the supplied video seed can generate only a temporary video-noise tensor that
is overwritten; it cannot advance the separately seeded action generator.

The artifacts give strong negative evidence against metadata or seed leakage:

- native = self-latent = self-cache bit-for-bit in 32/32 branches;
- donor-latent = donor-cache bit-for-bit in 96/96 ordered pairs, despite their
  differing nominal video seeds and latent inputs;
- every wrong-latent action (96/96) is bit-for-bit equal to the donor-latent action
  with the same recipient and actual source but different donor/run metadata; and
- all three donor-labelled first-frame actions are bit-for-bit equal for every
  state/recipient group (32/32; global maximum error 0).

The first-frame control is valid as an architectural/implementation sanity check.
The patched Optional-IDM wrapper rejects latent/cache overrides and forwards only
the action seed into upstream `FastWAM.infer_action`; `video_seed` is explicitly
unused. It is therefore not a transplant-matched causal control and its three donor
rows per recipient are repeated labels, not independent samples. The analyzer
correctly averages them within state and gates only on exact invariance.

## Why latent and cache outcomes are identical

This equality is the expected mechanism, not suspicious duplication. In upstream
IDM inference, the denoised future is deterministically converted into one K/V pair
per MoT layer, and action denoising receives the future only through those K/V
tensors. With a cache override, recipient-latent values are used only to establish
the fixed token/mask shape; their would-be cache computation is bypassed before
action denoising. Thus donor-latent replay recomputes `KV_B`, donor-cache replay
supplies the recorded `KV_B`, and both start from the same recipient action noise.
Exact equality demonstrates cache sufficiency/bottleneck equivalence. It must not
be described as replication across two independent future-to-action routes.

## Qualifications that should constrain wording

1. `wrong_latent` is a specificity/relabeling control, not an independent treatment:
   its output is exactly the correct-latent output for its actual source, scored
   against a different registered donor. The implementation's `next(...)` choice is
   source-unbalanced (one source appears twice and another is absent per recipient).
   A balanced re-scoring using the already-computed source interventions gave mean
   wrong-donor distance reduction `0.00606` versus the registered `0.005998`, so this
   defect does not change the smoke conclusion.
2. The value-permutation cache control preserves tensor marginals but is strongly
   out of distribution: its mean distance reduction is `-19.62`. It establishes a
   need for coherent K/V correspondence, not a fine-grained semantic role for
   individual tokens.
3. Branches differ in stochastic future content. The result supports causal use of
   the realized future representation, but semantic/object-level interpretation
   requires separate semantic annotation or intervention evidence. The smoke tests
   predicted 32-step action chunks, not executed task success.
4. Recipient action noise is fixed within every intervention contrast, as required,
   but the four native reference actions use their own registered action seeds.
   Retrieval is therefore against single noisy branch exemplars, not
   seed-marginalized conditional action distributions.
5. The realized run's provenance passed, but the manifest/runner do not themselves
   hash-check the dirty FastWAM diff or validate the LIBERO HEAD, and the analyzer
   source is not content-addressed in the result tree. Also, the protocol says a
   restart reconstructs cache from a saved latent, while the runner actually
   regenerates native branches from frozen seeds. These are reproducibility and
   wording gaps, not failures of the audited outputs.
