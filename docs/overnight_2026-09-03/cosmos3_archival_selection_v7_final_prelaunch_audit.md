# Cosmos 3 archival selection-free v7 final: outcome-blind prelaunch audit

Verdict: **GO** for the frozen archival, action-only evaluation. This audit did
not open a v7 evaluation result or any v6 causal payload. It used only the
frozen manifests and code, source/provenance artifacts, and the excluded-task
smoke's controls and completeness fields.

## Frozen identity and closure

- Evaluation manifest: `cosmos3-archival-sf-507feb24297971eb`, SHA-256
  `8ab294c7a581424cfa02faae1db85941e3f43c1e28100e68e21fa5f533703b9e`.
- Excluded Bagels smoke manifest: `cosmos3-archival-excluded-smoke-aa5aa488fc691372`,
  SHA-256 `2cd0345566a334d2d6ff06638220237d62b73ba0b8c30d255467a3285eaeb5bd`.
- The snapshot checksum list has SHA-256
  `688b125658c66f083fb92a655749a7b37596572f180f3a8aae2cc478205853b0`.
  Its 17 entries are the complete snapshot file set; all 17 independently
  rehashed exactly. Directories are mode `0555` and files mode `0444`.
- Critical pinned hashes independently match: runner
  `430d04f9476d2c34099437ae0770bbfd5505dc5d42215dbe0e251a741d49232a`,
  launcher `01beb6a6da0aa80db60df36bf2bff2e54a7939f28cc03689f6170c24c87dff6f`,
  analyzer `70a86cb30b63da58afa917b3d5047ca2515eee3297e78ea85671c060fe425066`,
  server `64980c631d4bec71be3e41cb574c0b84c759e80ebbaf1e58eeb558f66be17073`,
  interventions `285e9218b1be148698d8f3cccce62551b8d93f64a7540e8c01e0c466c52e081b`,
  and upstream RoboLab service
  `86d0d0e70faefa88a8cee8594abb3baa541ff6f65145a80eae23b1d828500f91`.
- The full checkpoint content manifest has SHA-256
  `b6cbee5522104145f71cfcc74f5049f61e7de3640b88f14831909c7f473c2660`.
  The actual `/checkpoint` receipt, SHA-256
  `905f40239ae56db2f7d83de3e9002a18f3f1ec8ad454f7d254ef5ca509e8fabe`,
  records an exact set of 87 nonsymlink files, 32,937,437,706 bytes, with every
  size and SHA-256 matching. The frozen launcher reruns that full verifier
  before entering the 90-state loop. The per-response parameter probe is an
  additional identity check, not a substitute for this full-content proof.

## Design and source checks

- The manifest contains exactly 30 episodes and 90 unique states: six tasks by
  five environment seeds by three phases, with every cell present once. Counts
  are 15 states per task and 30 per phase; all 20 archived-success and 10
  archived-failure episodes are retained.
- All 90 phase assignments independently reproduce the frozen nearest valid
  `16 mod 32` rule for targets `q * (action_count - 1)`, use valid MP4/HDF5
  index `branch_step - 1`, and are distinct within episode. Their congruence
  makes them disjoint from the prior `0 mod 32` cohort.
- Every state has the canonical four branch seeds, all 16 recipient/source
  cells, and all 12 ordered off-diagonal pairs. Every wrong-donor label is a
  prespecified nonrecipient/nondonor label, and the four-label shuffle is a
  bijective cyclic derangement.
- I independently rehashed all 90 unique source files (30 each of MP4, HDF5,
  and environment configuration; 434,000,088 bytes). Every digest matches all
  270 state-level references in the manifest. The runner repeats these checks
  before making calls for each state.
- The pinned upstream `_build_sample` reads only prompt, image, joint position,
  and gripper position. Research mode, IDs, source labels, and seeds do not
  enter the transformed model batch. The server also rejects a recipient or
  donor record whose transformed-state fingerprint differs.

## Intervention, RNG, and smoke checks

- The donor path uses the donor future target with the recipient generation
  seed, recipient initial state, and recipient vision path noise. At each of
  four denoising calls the wrapper supplies
  `(1 - sigma) * target + sigma * recipient_noise` on future frames 1--8,
  retains the model's action velocity, and overwrites only the selected future
  velocity with `(sampler_future - target) / sigma`.
- The explicit `none` path supplies an empty active-call list and delegates all
  four velocity calls unchanged. Input and output action-coordinate comparisons
  are performed on every interventional response.
- Excluded smoke output SHA-256
  `f8d50b14c7d884449feeb27ecf31e9e7c2bf504b699a54684459d2a4509c7235`
  is complete and mode `0444`. Independent control-only reconstruction found
  exactly 56 requests, 48 site audits, 44 active responses, and 176 active
  sites. Requested, observed, and clamped calls and sigmas agree exactly.
- All 176 model-input clamp errors and all 176 returned-velocity overwrite
  errors are zero. All 48 per-response and 192 per-call input/output action
  nonwrite checks are zero, with zero inactive-wrapper writes.
- All four `none` arms exactly replay native action, future, x0 hashes, sigmas,
  and trace signature. Four native, four self-clamp, and twelve donor-clamp
  repeats are exact; all twelve Gaussian geometry checks satisfy `1e-5`.
- There is exactly one transformed-input fingerprint and the expected singleton
  parameter-probe hash. Across every audited arm, recipient path-noise and
  initial-state hashes match that recipient's native record, while donor target
  hashes match the requested donor native record. This directly checks fixed
  recipient RNG and source retrieval without relying on labels alone.
- Final sampler-state target residuals are required only to be finite and are
  summarized descriptively. Neither runner, launcher, analyzer, nor evidence
  gate uses `0.03` for stopping, admission, exclusion, or success.

## v6 exclusion and analysis safety

- The v7 manifest identifies v6 manifest SHA-256
  `d8394fb17900d2c9a0d032317fba2cf554aaaf1b3213908782995b4829c03211`
  as `permanently_failed_incomplete_not_admitted`, records only the previously
  disclosed gate scalar, and prespecifies a from-scratch rerun of all 90 states.
- The launcher requires the fresh v7 output root and assigns all 90 states to
  one sequential shard. Each result must carry the v7 manifest ID and hash.
  The analyzer refuses any missing or extra JSON file and revalidates identity,
  controls, site audits, and the current-manifest state before aggregation.
  Thus v6 files cannot be resumed, admitted, or silently pooled.
- The primary 4-way retrieval estimand has a true conditional chance expectation
  of `0.25`. The inferential procedure is correctly labeled a seeded 10,000-draw
  source-label permutation Monte Carlo procedure, not an exact enumerated test.
  State measurements are averaged before the task-to-episode-to-state bootstrap.

## Claim and packaging limits

This is selection-free with respect to the current model/intervention outcomes
within the fixed archival cohort; it is not an unselected population sample.
It supports only predicted-action evidence from lossy H.264 reconstruction and
recorded proprioception, not fresh-simulator success or physical endpoints.
The intervention-site zeros are telemetry from the pinned wrapper at the actual
write sites, strengthened by exact no-op and replay controls; they are not an
independent measurement of model internals. Finally, result files are initially
written mode `0644` even though overwrite is refused, so the completed package
must still be hash-inventoried and permission-frozen before being called
immutable.
