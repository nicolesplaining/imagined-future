# FastWAM future-latent x video-cache factorial: powered v3 results

This is the labeled post-outcome record for the protocol frozen in
`docs/fastwam-cache-factorial-powered-v3-protocol.md` and the analyzer frozen in
`docs/fastwam-cache-factorial-powered-v3-analysis.md`.

## Completion and gate

- Manifest: `fastwam-kvfact-a09195568dd5a17f`.
- Population: 120 untouched states, all 40 LIBERO tasks, initial-state indices
  4, 5, and 6.
- Matrix: all twelve ordered recipient-to-donor pairs and all four future x
  cache cells per state.
- Completion: 5,760/5,760 valid arms and 120/120 state summaries; zero missing,
  malformed, unexpected, invalid, or degenerate-axis records.
- The analyzer opened causal payloads only after exact completion, clean worker
  exit, and a zero fatal-pattern count.
- Frozen decision: `cache_dominance_criteria_met` (PASS). Both hierarchical 95%
  lower bounds required by the gate exceeded zero, all outputs were finite,
  and every exact replay/invariance control passed.

## Four-cell results

Intervals are 95% suite-to-task-to-state hierarchical bootstrap intervals from
10,000 draws. Ordered pairs are repeated measurements averaged within state.

| Future latent | Video cache | Donor retrieval | Distance reduction | Projection | Cosine | Orthogonal ratio | Distance to donor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recipient | Recipient | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.776 [0.624, 0.934] |
| Donor | Recipient | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.776 [0.629, 0.936] |
| Recipient | Donor | 0.999 [0.997, 1.000] | 0.683 [0.637, 0.738] | 0.869 [0.833, 0.912] | 0.943 [0.927, 0.962] | 0.284 [0.244, 0.317] | 0.215 [0.179, 0.250] |
| Donor | Donor | 0.999 [0.997, 1.000] | 0.683 [0.638, 0.738] | 0.869 [0.832, 0.911] | 0.943 [0.927, 0.962] | 0.284 [0.244, 0.317] | 0.215 [0.180, 0.252] |

## Frozen factorial contrasts

The donor-cache main effect was:

- donor retrieval: 0.9993 [0.9965, 1.0000];
- donor-distance reduction: 0.6828 [0.6368, 0.7384];
- donor projection: 0.8690 [0.8323, 0.9135];
- cosine alignment: 0.9433 [0.9272, 0.9621];
- orthogonal residual ratio: 0.2839 [0.2439, 0.3178];
- distance to donor: -0.5610 [-0.7006, -0.4300].

The future-latent main effect was exactly zero for every directional metric
(distance-to-donor numerical mean -2.6e-18), and every future x cache
interaction was exactly zero.

## Exact controls

All five global maximum absolute errors were 0.0 (required <=1e-6): native
action regeneration, stored-native-latent regeneration, parent base-reference
replay, recipient-cache future-swap invariance, and donor-cache future-swap
invariance.

## Interpretation

For the released FastWAM Optional-IDM action path, the precomputed video K/V
cache is the complete future-conditioned interface to action denoising. Donor
cache is sufficient to reproduce donor-directed steering even when paired with
the recipient latent, while recipient cache completely suppresses steering
even when paired with the donor latent. The zero future-latent effect with cache
held fixed also verifies the implementation audit: once an explicit cache is
supplied, the raw video latent is not consumed downstream.

Combined with the separately frozen powered source-retrieval result, this
establishes coherent directional future conditioning and identifies the exact
released-model interface through which it reaches actions. Because Optional-IDM
materializes that cache before action denoising by design, this result is an
interface-level causal decomposition, not a localization claim about a hidden
subcircuit within the video generator.

## Artifacts and hashes

Local analysis directory:
`output/overnight_2026-09-03/fastwam_cache_factorial_powered_v3/analysis/`.

- Results JSON: SHA-256
  `483847ac549ea9b91919e4a88de5d88b98abd190709f0352d8585ef8583b34dd`.
- Aggregate CSV: SHA-256
  `7f419a8421436054461db452a2f054dcc33ab3c18fc943e6a8f8f23b85a69c6f`.
- State metrics CSV: SHA-256
  `f303c8e80ac57f9738338ae058e28fe4f105bbd75bc5bb5617614e497f0499c2`.
- Run metrics CSV: SHA-256
  `a5c2b3b051ca828b196feab2d69e3dabd921b2badb75655239c4bf4d6410e04a`.
- Completeness JSON: SHA-256
  `f85ba086042fd0c644a0cbe68a7fbb1bafc33e41ff4c2153ffa0f6e0bfdc75b1`.
- Missingness CSV: SHA-256
  `c079e6855b2de0cfd852643a300306218cc19bb4fe205c3d6df582243e2bc079`.
- LaTeX table: SHA-256
  `9ba6e11519400852b6c61611ed8fd1c9ea5aeb3b99785e5174652623613172f7`.
