# Adversarial audit: powered FastWAM Optional-IDM replication

**Verdict: PASS for the frozen powered, action-level replication.** I found no completeness, metric, bootstrap, or aggregation defect that changes the result. The evidence supports causal steering of predicted action chunks by the realized future representation in this fixed LIBERO cohort. It does not make latent and cache replay independent replications, and it does not establish executed task success or semantic understanding of the future.

## Frozen identity and provenance

- Manifest `fastwam-813f0233b9a2c083` independently regenerated its content ID and has SHA-256 `d74edd650f32faf7a0907871ae43e7362b5be19e029bef0b17d055eb114d125a`. The copy in the run root is byte-identical.
- The frozen config and original powered protocol independently match SHA-256 `dd4e1c9036dbbbbf2290665d4d9fd936de16b09229f583a5741ddd7458d6e47b` and `8822feb7384ab1e932d31c9605b8e58387b02457f79cc294a377ce0c0ca45bfe`.
- FastWAM is at commit `7faa71108368fbb3b6885649f112af607427a2d4`; its two intervention modifications exactly match the archived patch, SHA-256 `cb9291c112d6ac1c62e5d5e6664e577a62b45387909ecd2a2d3dadbd47188bf1`. LIBERO is clean at `8f1084e3132a39270c3a13ebe37270a43ece2a01`. Checkpoint and dataset-statistics hashes match `26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4` and `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`.
- Frozen module, runner, and analyzer hashes are respectively `6092472d92b0c9d67c23bc5d1584fa54b0d82d1335767f0974c3d4322c71ff49`, `2e396728973bc180dfbd129219aaaee7046b82d6ff5d4a2bfe1dff83e8d151d6`, and `ea0f7dde4278bd4904c0820f611f8b0902e4b1eebbf2b6d8a9c37b31958f65a3`.
- The recovered immutable analysis-v2 protocol is exactly 2,096 bytes and hashes to `72b571d48494904c0f34d1334817cd59210736d0c97ff25e5cd255ad75e166c5`. The original file later had a clearly labeled completion record appended; splitting that byte-exact prefix from the outcome record repairs the packaging ambiguity without changing the freeze. This was an outcome-blind analysis correction made while arms existed, not a wholly pre-generation preregistration.
- The final state summary was written at `2026-09-03T23:40:19.211117Z`; the first analysis artifact followed 53.29 seconds later. This is consistent with the recorded complete-matrix-before-analysis procedure, although filesystem evidence alone cannot prove that no person previously read a payload.

## Independent artifact and numerical recomputation

I expanded run identities independently from the manifest rather than importing the frozen analyzer. The Cartesian cohort is exactly 4 suites x 10 tasks x 3 untouched state indices = 120 states. Every state has the prescribed 72 arms: four each of native, self-latent, and self-cache, and twelve each of donor-latent, donor-cache, wrong-latent, shuffled-cache, and first-frame. All 8,640 JSON/NPZ pairs and 120 complete summaries exist, with no missing, extra, malformed, or mismatched run/state IDs.

All 8,640 model and environment action arrays are finite `(32, 7)` arrays. All 480 native video latents are finite `(1, 48, 3, 14, 28)` arrays. There are zero degenerate native action axes. The minimum native future-latent pair distance is `32.6618383508`; the maximum pairwise difference in the present latent frame is exactly zero. Self-latent and self-cache replay equal native actions bit-for-bit in 480/480 branches. First-frame actions are invariant across all three donor-labelled video seeds for every state/recipient, with global maximum error zero.

I recomputed all 7,200 stored directional-arm metrics from raw NPZ actions. Every Boolean classification agrees; the largest floating disagreement is `2.13e-14`. There are no nearest-action ties. For the confirmatory donor arms, the smallest first-versus-second nearest-action margin is `0.0470703`.

Independent state aggregation and a separately implemented 10,000-draw suite -> task -> state bootstrap reproduce the frozen report exactly:

| Estimand | Mean | Hierarchical 95% interval |
| --- | ---: | ---: |
| Latent 4x4 source retrieval | 1,919/1,920 = 0.999479 | [0.997396, 1.000] |
| Cache 4x4 source retrieval | 1,919/1,920 = 0.999479 | [0.997396, 1.000] |
| Donor-latent retrieval | 1,439/1,440 = 0.999306 | [0.996528, 1.000] |
| Donor-latent distance reduction | 0.682837 | [0.637480, 0.738026] |
| Latent minus wrong retrieval | 0.998611 | [0.993056, 1.000] |
| Latent minus wrong distance reduction | 0.738076 | [0.674795, 0.807201] |
| Cache minus shuffle retrieval | 0.763194 | [0.751389, 0.775694] |
| Cache minus shuffle distance reduction | 16.348044 | [13.313407, 19.971303] |

The full balanced source grid has an exact conditional label-permutation expectation of 0.25. The donor-only subset has three admissible donor labels and is correctly treated as secondary, not tested against 0.25. All eleven frozen v2 criteria recompute as true.

The only source-grid miss in each modality is the same cell: `libero_goal_task04_state006_wait30`, recipient `b01`, source `b00`, nearest `b03`. The latent and cache action arrays both hash to `7720b54815650181aafa2160f723ff69e57794fdd54735fb50e1befff5aed0e1`. All other 39 tasks are 48/48; that task is 47/48. Leave-one-task-out source-grid means range from `0.999466` to `1.000`; donor-latent retrieval ranges from `0.999288` to `1.000`, and donor-distance reduction from `0.679220` to `0.686593`. Direct recomputation of the task, suite, and 40 leave-one-task-out tables matches their CSVs within `7.11e-15` with no count mismatch.

The analyzed result file independently matches SHA-256 `bbcc86f0398f92bf9f48dc6f1e47b20e28db0a4eb92ad229c6dc0a82c885ab0e`.

## Causal and RNG interpretation

The patch constructs separate local video and action `torch.Generator` objects. Every intervention uses the recipient's frozen action seed. A donor-latent call may instantiate temporary video noise from the donor video seed, but the override replaces that tensor before video conditioning and cannot advance the separately seeded action generator. Donor IDs and run IDs are runner metadata used only to select stored tensors and score outputs; they are not model inputs.

The artifacts strongly reject trivial metadata/seed leakage:

- donor-latent and donor-cache actions are bit-identical in 1,440/1,440 matched pairs, despite their different nominal video-seed paths;
- every wrong-latent action is bit-identical in 1,440/1,440 cases to the donor-latent action with the same recipient and actual source but different registered donor/run metadata;
- all 480 native/self-latent/self-cache triplets are bit-identical; and
- all donor-labelled first-frame repeats are bit-identical.

Latent/cache equality is the expected mechanism. The IDM action stage receives future-video information only through the per-layer video K/V cache. Latent replay deterministically recomputes the donor cache; cache replay supplies that same recorded cache; both use identical recipient action noise. This demonstrates cache sufficiency at the implemented bottleneck. The two rows are not independent experimental confirmations, and their slightly different reported distance-reduction CI endpoints are only keyed Monte Carlo bootstrap noise over identical state values.

## Wording-limiting issues

1. `wrong_latent` is a specificity/relabeling control, not an independent treatment. Its deterministic `next(...)` construction is source-unbalanced: across the powered cohort the actual wrong sources occur `b00=720`, `b01=480`, `b02=240`, `b03=0`, versus 360 each for correct donors. Re-scoring both admissible wrong sources per pair gives distance reduction `-0.045953` instead of `-0.055240`; the balanced-minus-registered difference is `0.009287` with a descriptive 10,000-draw hierarchical sensitivity interval `[-0.010070, 0.033614]`. The conclusion is unchanged, but claims about this control must be limited to the frozen mapping.
2. The shuffled-cache control preserves per-layer tensor marginals but is strongly out of distribution (mean distance reduction `-15.6652`). It shows that coherent K/V correspondence matters; it does not isolate a semantic role for particular tokens.
3. The first-frame condition is a valid architecture/seed-isolation sanity check because the wrapper rejects future overrides and explicitly ignores `video_seed`. It is not a transplant-matched causal negative control, and its three donor labels per recipient are repeats rather than independent samples.
4. Retrieval uses single noisy native action exemplars in normalized model space. Branches and ordered pairs are within-state measurements, correctly averaged before inference. The result concerns predicted 32-step action chunks for fixed states, not executed trajectories, task success, physical endpoints, or seed-marginalized policies. Semantic/object-level interpretation of the stochastic futures needs separate annotation or intervention evidence.
5. The manifest itself pins the clean upstream commit but not the dirty intervention patch, runner, or analyzer hashes. Archived byte-exact code, patch, protocols, ledgers, and this raw-artifact recomputation make the realized result auditable, but future manifests should include that complete causal-code closure directly.
