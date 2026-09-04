# FastWAM Optional-IDM powered analysis v2: post-freeze result

This outcome record was written only after the complete powered matrix was
available. Its byte-exact frozen parent protocol is
`docs/fastwam-optional-idm-powered-analysis-v2-frozen.md`, SHA-256
`72b571d48494904c0f34d1334817cd59210736d0c97ff25e5cd255ad75e166c5`.

The matrix completed with 8,640/8,640 registered arms and 120/120 state
summaries. The v2 analyzer then opened the powered payloads for the first time.
All eleven frozen criteria passed.

For both latent and cache grids, correct-source retrieval was 1,919/1,920 =
0.999479, with suite-to-task-to-state hierarchical 95% interval
[0.997396, 1.000]. The single miss in each modality was the same logical cell:
state `libero_goal_task04_state006_wait30`, recipient `b01`, intended source
`b00`, nearest native action `b03`. The latent run ID was
`run-f722ad353aa074c2239e`; the cache run ID was
`run-50f0ec80ad16d2332d01`. Its distance to the intended donor was 2.014475,
distance to the recipient was 2.124016, and the native recipient-to-donor
distance was 2.937255.

The complete analysis bundle is mirrored at
`output/overnight_2026-09-03/fastwam_optional_idm_powered/analysis/`. Key
artifact hashes are:

- `fastwam_results.json`: `bbcc86f0398f92bf9f48dc6f1e47b20e28db0a4eb92ad229c6dc0a82c885ab0e`
- `fastwam_aggregate_metrics.csv`: `34c5254960756668493a0c12ef61abb59cefa3fba253fdea5226777e5e2aa640`
- `fastwam_run_metrics.csv`: `dd34a9ec407897631d6be6ea2a6cf70326f0d3980af621c3cba3f82370b911f6`
- `fastwam_source_grid_state_metrics.csv`: `a004eede2c904b3f9d587c935128b670463ae5f91756cf370836179386eb0df7`
- `fastwam_results.tex`: `7393bce52ae5ae07cd42b0e46417923d85ed96452848ab70b04b96cdc6846891`
- `fastwam_summary.png`: `cfccb6ea409db8d456f44abdf88ede8a2275dc58c46e2f94f39fa01550fe8cd1`
