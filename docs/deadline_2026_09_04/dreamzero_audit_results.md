# DreamZero future-latent intervention: audited results

## Confirmatory cohort

- 30 frozen states from 30 unique DROID episodes and instruction strings in
  DreamZero's released preprocessed training-data corpus: three states in each
  of ten predeclared verb families. Selection was deterministic and used no
  model outputs or outcome filtering.
- Four native branches per state (seeds 211, 223, 227, 229).
- Primary evaluation: all 12 off-diagonal recipient-noise × future-source cells per state; diagonal self-replays are implementation controls.
- The intervention replaces only the matched-noise future-video latent trajectory at all 16 solver steps. The recipient action-noise trace remains fixed and no action coordinate is written by the client.

## Primary result

| State-level estimand | Mean | 95% state-bootstrap CI |
|---|---:|---:|
| Four-way correct-source retrieval | 1.000 | [1.000, 1.000] |
| Distance reduction toward source action | 0.917 | [0.897, 0.933] |
| Normalized projection | 0.990 | [0.984, 0.995] |
| Cosine alignment | 0.995 | [0.991, 0.997] |
| Normalized orthogonal residual | 0.080 | [0.065, 0.099] |

The 100,000-draw within-state four-label Monte Carlo permutation test returned
`p = 1 / 100001`; its null mean was 0.2498. Retrieval remained 1.0 in every
native-separation quartile and every leave-one-task-family-out analysis.

## Dose response and controls

The preregistered 211→223 pair used three newly evaluated interior doses. The
0 and 1 endpoints reuse exact frozen core-grid cells and are descriptive only.

| Future-latent dose | Mean normalized projection | 95% state-bootstrap CI |
|---:|---:|---:|
| 0.00 (reused self anchor) | 0.000 | [0.000, 0.000] |
| 0.25 | 0.046 | [0.030, 0.063] |
| 0.50 | 0.465 | [0.411, 0.518] |
| 0.75 | 0.939 | [0.914, 0.960] |
| 1.00 (reused donor anchor) | 0.985 | [0.970, 0.998] |

All 30 states were monotonic across the three newly evaluated interior levels.
The interior-only state-level slope was 1.786 [1.707, 1.853], with a one-sided
100,000-draw state sign-flip `p = 1 / 100001`.

The per-step norm-matched Gaussian control caused large non-specific action
changes (normalized displacement 2.070 [1.702, 2.476]); the recipient remained
the nearest native action only 0.308 [0.275, 0.350] of the time. This control
shows that incoherent latents are disruptive. Coherence-specific evidence comes
from correct donor identity and direction, not raw action displacement.

## Audit and provenance

- Core runner SHA: `e627132e037679717512faac2f7bc46ddda8898f1e7bfe5637445a99e8163019`
- Core analyzer SHA: `7f17970a858927e37a664efbde0451655bac00e374b3e08446d9a7e9295efc30`
- Gaussian/control runner SHA: `74e6c4d4ab76006aa48f8dcd7666fbf5b9ef786bd639daa684b9cc1e36b606a9`
- Dose runner SHA: `bc180328508c3d687c8003e245584ef7ee194529438a4ccf4dfacb11cb0c5d7d`
- Control analyzer SHA: `bcd3dbc5687f4e3bba941d99792f5b42b1f76b6075c16981b1d852fb7ff5ce57`
- Intervention patch SHA: `7c601e25d335a348056ef674065e445279482e91098f28493b3f218882b3d25a`
- Frozen cohort manifest SHA: `d1ffc3111a10bed9ac8fdd17c631dc3a5d8eb3128ac4fa250d9398bcede12cfc`
- Immutable runtime receipt SHA: `57b89a7a98b3326812fa6652fff1f000f9bbe6940cd82a4e1a72b7983eaa06e5`
- Immutable runtime-log correction/addendum SHA: `b7ce05309878376cc5f1fa1c091c4fa5007a7c9d705f7481cf902a9c54878078`

The audit independently reconstructed every Gaussian and interpolated latent,
verified all action/noise/video hashes and server audits, checked the complete
core checkpoint and 4×4 grid, and bound all 30 core/Gaussian/dose result hashes
to the canonical manifest and runners. The exact patched-server mode-off versus
record control was bitwise identical. A separate excluded-state
[clean-upstream parity audit](../../output/deadline_2026_09_04/dreamzero/upstream_native_parity/execution_receipt.json)
also found bitwise-identical 24x8 action arrays between an untouched checkout
of the pinned official commit and patched mode-off on the same input (maximum
absolute error 0). Its execution-receipt SHA-256 is
`f9f6294d582486d97c8a2def87c63d770985fa022016660b744df54489db9c11` and
artifact-index SHA-256 is
`5d2e3c71b85c9e2327337139062715aabbe9c6beae4b7d4fc2a5139c74da270f`.
The raw receipt's server-log field pointed to a failed
launch. The corrective [provenance addendum](../../output/deadline_2026_09_04/dreamzero/provenance_addendum/runtime_provenance_addendum_v1.json)
binds the actual evaluated-server log and explicitly labels its environment
package census as postrun rather than a launch-time lockfile.

Local artifacts are under `output/deadline_2026_09_04/dreamzero/`. The
representative panel uses the state closest to the cohort-median primary
distance reduction, not the strongest or prettiest state. Four
[post-analysis decoded native-future videos](../../output/deadline_2026_09_04/dreamzero/representative_media/)
for that state reproduce the frozen native actions bit-for-bit; they are
descriptive media and were not used for selection or inference.

The exhaustive follow-up regenerated all 30 x 4 native futures, with no
appearance or outcome selection. Its
[receipt](../../output/deadline_2026_09_04/dreamzero/all_native_media/receipt.json)
(SHA-256
`89c7f4742b42698d2a2cb122f63c5792d6e53da070280c019ad18a1fb3acacdb`)
records 120/120 bit-exact actions, 120/120 byte-exact latent traces, valid H.264
videos for every branch, and an unchanged frozen core tree. These decoded
videos remain descriptive rather than semantic labels.

## Claim boundary

DreamZero supplies a strong external action-level replication and graded causal
control result. It does not add a physical-endpoint or closed-loop success
result. The inferential cohort is identified by native VAE-latent source, and
the post-analysis decoded representative does not establish semantic diversity
across all 30 states.
