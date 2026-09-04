# Cosmos complete-cohort independent raw-output audit procedure

Status: prospective and outcome-blind. This procedure and its scripts were prepared
without parsing any partial archival-v7 or timing-v5 evaluation payload.

## Hard unblinding gate

Neither independent auditor may be invoked until its launcher is finished and a
completion packager has produced a mode-0444 hash inventory. The shared loader first
checks the frozen manifest hash, exact manifest-derived filename set, nonsymlink
regular-file status, mode `0444`, byte size, and SHA-256 of every output. It returns
paths for JSON parsing only after every check succeeds. A mere `90/90` or `30/30`
directory count is insufficient.

The frozen analysis is also read only after raw recomputation is complete and only
with an explicitly supplied SHA-256. Audit outputs refuse overwrite and record every
comparison discrepancy. The audit scripts import no runner, protocol helper, or
frozen analyzer.

## Archival v7: 90 states

Inputs:

- manifest ID `cosmos3-archival-sf-507feb24297971eb`;
- manifest SHA-256
  `8ab294c7a581424cfa02faae1db85941e3f43c1e28100e68e21fa5f533703b9e`;
- the inventory emitted by the already frozen archival completion packager;
- the immutable frozen-analysis summary JSON and its post-completion SHA-256.

[`audit_cosmos3_archival_raw_outputs.py`](../../scripts/audit_cosmos3_archival_raw_outputs.py)
recomputes, directly from native/self/donor/Gaussian action arrays:

- every nearest-native label, 4-by-4 source-retrieval indicator, shuffled-label and
  wrong-donor indicator;
- donor and Gaussian distance reduction, projection, cosine, orthogonal residual,
  target distance, and native separation;
- all state means and valid/null denominators;
- equal-task task-to-episode-to-state 10,000-draw PCG64(20260903) percentile
  intervals, per-task and leave-one-task-out values;
- early/middle/late estimates;
- global 1,080-arm separation quartiles assigned before state averaging, plus the
  explicitly secondary state-mean quartiles;
- the 4-by-4, 0.25-chance source-label Monte Carlo permutation audit;
- all replay, no-op, coordinate-nonwrite, intervention-site, RNG/source-hash,
  residual, and completeness counts.

Template command (fill only hashes/paths created after the complete freeze):

```bash
python /research/scripts/audit_cosmos3_archival_raw_outputs.py \
  --manifest /research/results/overnight_2026_09_03/cosmos3_archival_selection_v7_final/manifest.json \
  --expected-manifest-sha256 8ab294c7a581424cfa02faae1db85941e3f43c1e28100e68e21fa5f533703b9e \
  --run-root /research/results/overnight_2026_09_03/cosmos3_archival_selection_v7_final/run \
  --inventory ARCHIVAL_OUTPUT_INVENTORY.json \
  --expected-inventory-sha256 ARCHIVAL_INVENTORY_SHA256 \
  --summary-json FROZEN_ARCHIVAL_SUMMARY.json \
  --expected-summary-sha256 ARCHIVAL_SUMMARY_SHA256 \
  --state-csv FROZEN_ARCHIVAL_STATE_ROWS.csv \
  --expected-state-sha256 ARCHIVAL_STATE_ROWS_SHA256 \
  --output INDEPENDENT_ARCHIVAL_AUDIT.json
```

## Timing v5: 30 middle states

Inputs:

- manifest ID `cosmos3-timing-05f23896cd88b340`;
- manifest SHA-256
  `a0ef31afe6f7996bfdb177957d2eabac6cc1d8263e6d04832602520e17e63759`;
- an inventory emitted only after exact 30/30 completion by
  [`freeze_cosmos3_timing_output_package.py`](../../scripts/freeze_cosmos3_timing_output_package.py);
- frozen summary JSON, states CSV, pair CSV, per-task CSV, and LOTO CSV, each supplied
  with its post-completion SHA-256.

[`audit_cosmos3_timing_raw_outputs.py`](../../scripts/audit_cosmos3_timing_raw_outputs.py)
recomputes all six timing cells from the 3,240-call raw reports. It independently
checks `[32,8]`/256 action shape, 108-call order, source/RNG identities, exact no-op
and replay controls, schedule/site writes, and the 28-null/72-finite/8-absent
per-state projection census. It then reconstructs all native classifications and
action-space directional metrics, timing-matched self-versus-donor gains, the shared
10,000-draw task-to-five-state bootstrap, both prespecified positive contrasts,
one-sided centered bootstrap p-values, the four-test conjunctive Holm procedure,
per-task/LOTO results, 360-axis global quartiles and all 2,160 timing-pair rows, and
descriptive residual distributions.

Completion packaging must run before either analyzer:

```bash
python /research/scripts/freeze_cosmos3_timing_output_package.py \
  --manifest /research/results/overnight_2026_09_03/cosmos3_single_call_timing_v5/manifest.json \
  --expected-manifest-sha256 a0ef31afe6f7996bfdb177957d2eabac6cc1d8263e6d04832602520e17e63759 \
  --output-root /research/results/overnight_2026_09_03/cosmos3_single_call_timing_v5/run \
  --inventory-output TIMING_OUTPUT_INVENTORY.json
```

Then run the frozen analyzer, freeze/hash its complete output directory, and invoke:

```bash
python /research/scripts/audit_cosmos3_timing_raw_outputs.py \
  --manifest /research/results/overnight_2026_09_03/cosmos3_single_call_timing_v5/manifest.json \
  --expected-manifest-sha256 a0ef31afe6f7996bfdb177957d2eabac6cc1d8263e6d04832602520e17e63759 \
  --run-root /research/results/overnight_2026_09_03/cosmos3_single_call_timing_v5/run \
  --inventory TIMING_OUTPUT_INVENTORY.json \
  --expected-inventory-sha256 TIMING_INVENTORY_SHA256 \
  --summary-json ANALYSIS/cosmos3_single_call_timing_results.json \
  --expected-summary-sha256 SUMMARY_SHA256 \
  --states-csv ANALYSIS/cosmos3_single_call_timing_states.csv \
  --expected-states-sha256 STATES_SHA256 \
  --pairs-csv ANALYSIS/cosmos3_single_call_timing_pairs.csv \
  --expected-pairs-sha256 PAIRS_SHA256 \
  --per-task-csv ANALYSIS/cosmos3_single_call_timing_per_task.csv \
  --expected-per-task-sha256 PER_TASK_SHA256 \
  --loto-csv ANALYSIS/cosmos3_single_call_timing_leave_one_task_out.csv \
  --expected-loto-sha256 LOTO_SHA256 \
  --output INDEPENDENT_TIMING_AUDIT.json
```

Any discrepancy is a failed audit, not an invitation to select a different output,
state subset, tolerance, seed, or analysis version after viewing results.
