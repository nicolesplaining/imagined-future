# Cosmos 3 single-call timing v5: independent raw-output audit

Verdict: **PASS**, with zero discrepancies. The independent audit opened the reports only after the completion inventory proved the exact 30-file, mode-`0444` cohort. It then recomputed all action metrics, state estimands, the shared task-to-state hierarchical bootstrap, call-local tests, Holm correction, task and leave-one-task-out summaries, pair rows, and global quartiles without importing the frozen runner, analyzer, or analysis helpers.

## Provenance and completeness

- Manifest: `cosmos3-timing-05f23896cd88b340`, SHA-256 `a0ef31afe6f7996bfdb177957d2eabac6cc1d8263e6d04832602520e17e63759`.
- Raw package inventory: SHA-256 `7dbb6f6b063f60679e31b5217ac3b95831b1e929ec9f9a820bb4bc83bfffcf9f`; exactly 30 states, 3,240 calls, and no missing or extra files.
- Frozen results JSON: SHA-256 `2a988a805b7423efab65b3354043ef8dde5dd2c0043f00d5f99291bee6c00262`. States, pairs, per-task, and leave-one-task-out CSVs also matched independently, with 30 state rows and 2,160 off-diagonal timing-pair rows.
- Independent audit JSON: [`independent_raw_audit.json`](../../output/overnight_2026-09-03/cosmos3_single_call_timing_v5/evaluation/audit/independent_raw_audit.json), SHA-256 `e710cffd44ed92b881eb480a2e7a08b1e404860bbb8e7e41adeac5ec2e569a89`. Auditor SHA-256: `8e4ccd2a4fe4ba44e792725488cbc09d7a8b8db8a848f01127f666c1fe42c040`.

## Recomputed primary results

Values are equal-task means with the prospectively frozen 10,000-draw task-to-five-state hierarchical percentile interval.

| timing condition | matched retrieval gain, mean [95% CI] | matched distance gain, mean [95% CI] |
|---|---:|---:|
| none | 0 [0, 0] | 0 [0, 0] |
| call 0 only | 0.725000 [0.633333, 0.802778] | 0.462276 [0.401560, 0.514932] |
| call 1 only | 0.922222 [0.855556, 0.975000] | 0.647270 [0.596124, 0.693175] |
| call 2 only | 1.000000 [1.000000, 1.000000] | 0.781475 [0.744115, 0.812456] |
| call 3 only | 1.000000 [1.000000, 1.000000] | 0.780340 [0.741320, 0.812619] |
| all calls | 1.000000 [1.000000, 1.000000] | 0.780864 [0.742590, 0.813574] |
| average single call | 0.911806 [0.876372, 0.939583] | 0.667840 [0.628916, 0.703921] |
| all calls minus average single | 0.088194 [0.060417, 0.123628] | 0.113024 [0.094483, 0.132131] |

For every call-local conjunctive test, both component centered-bootstrap p-values were `1/10001`; the conjunctive raw p-value was therefore `1/10001`, Holm-adjusted p was `0.000399960004`, and all four hypotheses rejected. The all-calls-minus-average-single interval was positive for both outcomes. Leave-one-task-out values remained positive: the sustained-minus-single retrieval contrast ranged `0.075833`–`0.097500`, and its distance contrast ranged `0.109393`–`0.118290`.

For the all-calls arm, raw off-diagonal donor retrieval and matched retrieval gain were both 1 [1, 1]. Secondary donor-directed metrics were: distance reduction `0.760111` [`0.713569`, `0.800739`], donor projection `0.968762` [`0.952445`, `0.982514`], cosine alignment `0.967031` [`0.952807`, `0.978266`], and normalized orthogonal residual `0.229377` [`0.190099`, `0.274094`]. The 360 native-separation axes formed four global 90-pair quartiles at boundaries `1.036259`, `1.420722`, and `1.928452`; all-calls distance-gain means were respectively `0.670915`, `0.774189`, `0.828698`, and `0.849655`. Every state is the prospectively fixed middle phase, so this experiment has no phase contrast.

## Controls and claim boundary

All 3,240 actions were finite and exactly `[32,8]`/256. The projection schema census was exactly 840 structural nulls, 2,160 finite off-diagonal values, and 240 native-field absences. Native replay, diagonal replay, none no-op, none source-invariance, target/source hashes, recipient RNG hashes, schedule/index telemetry, and signatures were exact. All action-input, action-output, active model-input clamp, returned-velocity, and inactive-write maxima were zero; all 360 native axes were nondegenerate (minimum separation `0.389016`).

The evidence supports donor-directed **action-space** effects from each imposed denoising-call intervention and a stronger all-calls effect than the average single-call effect. The ordering of the four single-call means is descriptive: the protocol did not license a monotone timing claim. In particular, the numerical closeness of calls 2/3 to all calls is not an equivalence result. Large descriptive final-target residuals after early-only clamps reflect subsequent un-clamped denoising (maximum absolute residuals reached `6.8073` for call 0, versus `0.0289` for all calls); the exact live site gates show this is not a failed intervention, but it limits interpretation to transient, call-local imposition. No result establishes natural mediation, necessity, an isolated local direct effect, semantic planning, or physical task success.

One reproduction detail is worth preserving: the server's optional donor-projection diagnostic subtracts serialized float32 actions before multiplying by a float64 direction. Reproducing that pinned operation order yields bit-exact agreement for all 2,160 applicable cells; casting both operands to float64 before subtraction can differ by a few parts in `1e-9`. The protocol estimands were independently recomputed in the specified flattened float64 action space and match the frozen analysis.
