# Cross-model pathway-factorial evidence table

| Model | Recipient future / recipient path | Donor future / recipient path | Recipient future / donor path | Donor future / donor path |
|---|---:|---:|---:|---:|
| Cosmos 3 | -0.011 [-0.022, 0.000] | 0.119 [0.078, 0.162] | 0.884 [0.848, 0.919] | 1.004 [0.988, 1.025] |
| FastWAM Optional-IDM | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.869 [0.833, 0.912] | 0.869 [0.832, 0.911] |

Values are normalized donor-axis action projections with hierarchical 95%
bootstrap intervals. “Path” is an umbrella label for different, analogous
interfaces: future-token K/V at all audited action-query interfaces for Cosmos
3, and the architecture-defined precomputed video K/V cache for FastWAM
Optional-IDM.

The rows are not the same intervention or sampling estimand and must not be
pooled or differenced. Cosmos uses 21 previously selected single-pair states,
a realized future held fixed across K/V patches, and a task-to-state hierarchy.
FastWAM uses 120 untouched states with all 12 ordered pairs, a raw
future-latent candidate plus an explicit cache override, and a
suite-to-task-to-state hierarchy. With that FastWAM override active, action
denoising does not consume latent content; its same-cache zeros validate the
explicit interface rather than competition between two active inputs.

Sources: Cosmos summary SHA-256
`7681fcb86bce473cd36497b518618f58869f096afb4ff29ba67784b720b998a6`;
FastWAM summary SHA-256
`483847ac549ea9b91919e4a88de5d88b98abd190709f0352d8585ef8583b34dd`.
Numbers are rounded to three decimals from the model-specific equal-weight
point estimates and frozen hierarchical intervals.
