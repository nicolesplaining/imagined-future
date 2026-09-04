# Cosmos 3 single-call timing v5: outcome-blind prelaunch audit

Verdict: **GO.** The frozen v5 population, implementation closure, excluded
smoke, and fresh post-smoke runtime satisfy the prospective timing-v2 protocol
and its structural-null and action-shape amendments. No timing-effect metric or
archival-v7 evaluation outcome was inspected.

Launcher authorization JSON:
`docs/overnight_2026-09-03/cosmos3_single_call_timing_v5_go_audit.json`,
SHA-256 `74e92fa567aa3437f5d100fc83af748d7269d35e54a8694a0d21614c58ca5c8b`.

## Frozen identity and population

- Evaluation manifest `cosmos3-timing-05f23896cd88b340`, SHA-256
  `a0ef31afe6f7996bfdb177957d2eabac6cc1d8263e6d04832602520e17e63759`;
  excluded-smoke manifest `cosmos3-timing-excluded-smoke-f04df7c864eb1422`,
  SHA-256
  `212f02aef35b991af679bac2ea49856d37e98401dabef714daae76ed9d4a191d`.
  Both content-addressed IDs were independently recomputed.
- Runtime closure SHA-256
  `e6d01150dfffd2d58fde50bf5f32f1bff0509636fba798d3c81c6f29e791e345`;
  21-file snapshot-list SHA-256
  `67455b7025e197e0f02a317d0026fd09e3f13a02257c51a96aa6a6e03b00e001`.
  Every listed file rehashed exactly and is mode 0444.
- Protocol SHA-256
  `526d126a1ffe6cc8216f8be1a0aaa93732faf637932f8155eea3d024fcb38c57`;
  checklist `288472c4f8ea914333916b6bff68c9777b4d07eb892646e70b0ebfa3749d7012`;
  structural-null amendment
  `23a5c922fb1abc7c3feaf409b764308060266ef171458bd8ba58fafaacab3f83`;
  action-shape amendment
  `70767fda042b3ba8dab888ea5c4325f34aa20d6a9e494b68dc47cbacf653e88f`.
- The 30 state objects are exactly all middle states from the archival-v7
  manifest in its frozen order: six tasks x five environment seeds. All 30 IDs
  are unique, branch order is 211/223/227/229, and the excluded Bagels state is
  disjoint. All 90 unique input assets independently rehashed with zero
  mismatch. Timing v2, v3, and v4 are explicitly non-admitted; v4 has zero
  evaluation calls.

## Design, implementation, and analysis

- The request matrix independently reconstructs to 108 unique labels/state:
  4 native + 4 native replay + six complete 4 x 4 timing grids + 4 all-calls
  diagonal replay, or exactly 3,240 admitted calls. Conditions are
  `none=[]`, four singleton chronological calls, and `all=[0,1,2,3]`; both
  sigma traces use the four exact frozen float32 values.
- The dedicated server SHA-256 is
  `1b67e0735c108f0c2209374e758851999e7b593b20b7e4997c344996a8bcbedf`,
  runner `7b9433be596ec00d8109d7b0d2d0a0d934be442315a3c597121a581a04f77d1d`,
  analyzer `4a15163676ab78fb6bec45f6edd0bcacd232e6f3b009f15968344683c37e8602`,
  and launcher
  `2dbed8f24ce082c41db2a16ea5b1fd0738a20da36fc6380716bcd265cf54d710`.
  The server emits genuine JSON null only for structurally undefined diagonal
  projection and explicitly emits applicability; the generic NaN server is not
  in the v5 closure.
- Every wire action is fail-closed at exact shape `[32,8]`, 256 finite
  coordinates. The analyzer independently rechecks all 4 stored native and 96
  timing-grid actions/state and requires the runner's 108-response census.
- The analyzer requires exact 30-file completeness and hashes, recomputes all
  action metrics from raw arrays, uses all 12 ordered off-diagonal pairs and a
  timing-matched self comparator, and rejects degenerate native axes. One shared
  10,000-draw `PCG64(20260903)` task-to-five-state table supplies every interval
  and contrast. The primary/sustained conjunctions, null-centered component
  p-values, four-test Holm procedure, task/leave-one-task-out results, complete
  4 x 4 retrieval with 0.25 permutation expectation, and 360-pair quartiles
  match the frozen protocol.

## Excluded smoke and fresh evaluation runtime

- Excluded-smoke state SHA-256
  `749df4692684ee7fcec69293e0c744106620a1817cae9b9fad939ae53a5cd65c`;
  control-only validator report SHA-256
  `b4595bebc91ad834959e05c23dc05e1e12f664d742e0360aad06100478203086`.
  Strict JSON parsing and the frozen analyzer validator independently pass.
- The smoke has 108/108 wire actions and all 100 retained actions at `[32,8]`;
  projection census 28 structural null, 72 finite off-diagonal, 8 absent native;
  all 12 off-diagonal `none` projections are exactly zero. Its only retained
  null paths are 24 diagonal projections and 96 inactive cache IDs. All required
  numeric fields are finite; all intervention-site/action-coordinate errors and
  inactive writes are zero; native, diagonal, no-op, and source-invariance
  replays are exact; input fingerprint and model probe are singleton; target
  and recipient RNG identities revalidate.
- Full checkpoint content is 87 files/32,937,437,706 bytes under manifest
  SHA-256 `b6cbee5522104145f71cfcc74f5049f61e7de3640b88f14831909c7f473c2660`.
  Pre-evaluation verification receipt SHA-256
  `1fa6b7b5f79178430c3db54261fcc15bac460edbb19eefcc807990b18e0c6c8b`.
- Before smoke, container
  `db3eb80f78f4f888035e05c26b30bab3b5f9d408e5b8419fa4291c7ba611bcd2`
  was freshly created and logged `registry_entries=0` before any inference.
  After smoke it was replaced by container
  `7a978f110f6ce885a13c4f4e74c6249e9146a2454d0efd0f6f454559018dd9b7`.
  The mode-0444 empty-registry receipt SHA-256 is
  `328a883d161692414ed87616dfcf8909670f0431ae2b5d980b6ab13e417a71bb`:
  exact image digest, dedicated v5 entrypoint, GPU 3/port 8004,
  `registry_entries=0`, and zero post-restart requests/evaluation calls.

## Claim boundary

Passing results may support donor-specific steering of the model's full,
unweighted 32 x 8 predicted action output under an imposed future clamp, and
the separately gated timing/strength statements in the protocol. They do not
establish natural mediation, an isolated local computation, semantic planning,
executed trajectories, physical success, or necessity. A failed positive gate
means only “not detected under this design”; no equivalence margin is frozen.
