# Cosmos 3 archival selection-free v6: outcome-blind prelaunch audit

Verdict: **GO** for the frozen archival, action-only evaluation. No model evaluation output was opened or used in this audit.

## Frozen identity

- Manifest: `cosmos3-archival-sf-d2df8d9d1d9f0c19`
- Manifest SHA-256: `d8394fb17900d2c9a0d032317fba2cf554aaaf1b3213908782995b4829c03211`
- The manifest ID was independently regenerated from the canonical manifest body and matched.
- Frozen snapshot hashes independently matched the manifest: runner `3456e65fe0bb974b2cd808c45e238a5fe5cd693be5a82d05dd992227c5b6116b`, analyzer `1029cf3eb4b7d60fb9acf0f94ac3df1abe86580e38af92c4827feba08f20647e`, and launcher `950a068949a295bb6c60da02b6148adea8544fb825b35593d7cfe69e4c1c24dd`. Snapshot files were root-owned and mode `0444`.

## Independent checks

- The immutable cohort has exactly 30 episodes and 90 unique states: 6 tasks x 5 environment seeds x 3 phases. Counts are 15 states per task, 18 per environment seed, and 30 per phase. All 20 archived-success and 10 archived-failure episodes are retained.
- Every phase step independently reproduced the frozen nearest-step rule, is `16 mod 32`, is distinct within episode, does not overlap the prior `0 mod 32` cohort, and has valid MP4/HDF5 indices.
- Every state contains the canonical 12 ordered off-diagonal recipient/donor pairs and all 16 recipient/source cells. The wrong-donor map is balanced within recipient and always excludes recipient and true donor. The four-source shuffle is a bijective derangement.
- Donor interventions use the donor future target while retaining recipient sampling seed and recipient path noise. Only future visual-latent frames are clamped; action coordinates are audited rather than replaced.
- The pinned upstream `_build_sample` implementation (SHA-256 `86d0d0e70faefa88a8cee8594abb3baa541ff6f65145a80eae23b1d828500f91`) consumes prompt, image, joint position, and gripper position only. Research IDs, donor IDs, modes, and seeds therefore do not enter the transformed model batch. The runner additionally requires one transformed-input fingerprint per state.
- The analyzer refuses anything other than the exact 90-file cohort, averages repeated arms within state, and bootstraps task -> archived episode -> state. The primary grid has an exact conditional null expectation of 0.25; the reported test is correctly labeled a 10,000-draw Monte Carlo label-permutation test. Prespecified evidence gates require the hierarchical retrieval lower bound above 0.25, donor distance-reduction lower bound above zero, zero degenerate donor axes, and exact controls.
- Native-separation quartiles are assigned over all 1,080 off-diagonal arms before within-state aggregation, with arm/state/metric denominators reported.
- Fresh excluded-task smoke artifact SHA-256 `efbe5d3d47dad68e9bc2e9a1bc893fdcbfa9509c4707ca4cab7633652345dcf9` independently passed: 52 requests; 4/4 native, 4/4 self, and 12/12 donor deterministic replays; one input fingerprint and expected parameter probe; 44/44 zero action-coordinate audits; 44/44 finite target errors with maximum `0.027452468872070312 < 0.03`; and 12/12 finite Gaussian geometry controls below `1e-5`.

## Claim boundary

This design supports only selection-free donor-identity steering of model actions for the fixed lossy archival reconstructions. It is not a fresh simulator, task-success, physical-endpoint, or population-generalization study. The expected model hash is a deterministic sample-based parameter probe, not a cryptographic digest of every checkpoint byte; it should be described as a parameter probe.
