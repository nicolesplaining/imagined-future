# Deadline-goal completion matrix

This matrix audits the 2026-09-04 noon experiment brief against frozen local
and NFS evidence. “Partial” means the requested implementation-level gate was
not fully satisfied; it is not silently replaced by a nearby result.

| Requirement | Status | Authoritative evidence |
|---|---|---|
| Use four dedicated H100s as 2 DreamZero + 2 LingBot without touching loaded Cosmos services | Achieved | [Compute closure receipt](../../output/deadline_2026_09_04/compute_closure.json); dedicated hosts were `if-overnight-external-wams` and `if-overnight-robolab-clients` |
| Pin official repositories, checkpoints, manifests, seeds, runners, and intervention sites | Achieved, with a DreamZero environment-provenance qualification | [Claim-safe synthesis methods table](claim_safe_synthesis.md); DreamZero has a postrun 258-package census rather than a launch-time lock/container |
| Excluded-development native/self/replay/fixed-noise/action-nonwrite gates | Achieved | DreamZero exact self, record/replay, fixed-noise, and nonwrite gates passed; an untouched checkout of the pinned official commit and patched mode-off also produced bitwise-identical 24x8 actions on one excluded input. One excluded LingBot input matched official upstream inference bit-for-bit. See the [DreamZero upstream parity audit](../../output/deadline_2026_09_04/dreamzero/upstream_native_parity/execution_receipt.json) and [LingBot oracle](../../output/deadline_2026_09_04/lingbot/upstream_parity/upstream_native_parity.json) |
| At least 30 outcome-unfiltered native-domain states/model, four branches/state, complete 4x4 | Achieved | [DreamZero summary](../../output/deadline_2026_09_04/dreamzero/core_analysis_final/summary.json) and [LingBot summary](../../output/deadline_2026_09_04/lingbot/core_artifacts/summary.json) |
| Four-way off-diagonal source identification plus distance reduction, projection, cosine, and orthogonal residual; state as unit | Achieved | [Machine-readable core table](../../output/deadline_2026_09_04/cross_model_results_table.csv) and frozen model summaries |
| Self/replay, wrong native sources, within-state label-permutation test, and norm-matched Gaussian controls | Achieved | Complete 4x4 native grids, model control summaries, and [LingBot Gaussian routing audit](../../output/deadline_2026_09_04/lingbot_gaussian_grid_v2_final/summary.json). “Shuffled label” is a statistical permutation test, not a model-input intervention |
| Independent 2x2 future x K/V/cache factorial after successful transplantation | Incomplete | LingBot's raw-future/cache cross verifies its explicit ordered interface but is not an independently active two-path factorial. A clean DreamZero current-future K/V cross requires a paired side stream and new per-block/per-solver action-query instrumentation; native donor K/V replay would leak donor-action feedback. See the [feasibility audit](dreamzero_kv_factorial_feasibility.md) and [implementation protocol](dreamzero_future_kv_factorial_protocol.md). The paper's independent K/V mechanism evidence remains Cosmos 3 |
| Five-level recipient-to-donor dose response | Achieved in both systems | [Dose table](../../output/deadline_2026_09_04/cross_model_dose_table.csv). The model-specific response axes differ and are recorded explicitly |
| Preserve raw actions, futures, trace/cache hashes, configs, failures, exclusions, and intervention audits | Achieved | DreamZero core/control inventories and provenance addendum; LingBot immutable package, raw addendum, Gaussian provenance, complete Gaussian execution receipt, and [decoded-future execution-history addendum](../../output/deadline_2026_09_04/lingbot_native_future_decode_all120_v1_execution_addendum_v2/execution_receipt.json) |
| Verified tables, plots, and post-analysis media | Achieved | [Deadline README](README.md) indexes all tables/plots and exhaustive 120-future media for each model |
| Claim-safe synthesis without editing or pushing the manuscript | Achieved | [Noon results](noon_final_results.md), [claim-safe synthesis](claim_safe_synthesis.md), and [adversarial audit](final_science_adversarial_audit.md); manuscript untouched by this deadline run |
| Wrap computation and synthesize by noon PT | Achieved at 11:50 PT | [Noon results](noon_final_results.md) and [compute closure receipt](../../output/deadline_2026_09_04/compute_closure.json) |

The only central scientific requirement not completed is the independent new
future x K/V/cache factorial. That absence is explicit throughout the handoff;
no claim inherits Cosmos 3's mechanism result for DreamZero or treats
LingBot's architectural cache-routing control as a discovered hidden pathway.
