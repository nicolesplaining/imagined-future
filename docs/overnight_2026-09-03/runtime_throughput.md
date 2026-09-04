# Cosmos 3 overnight throughput estimate

## Measured anchors

- Warm Cosmos inference is approximately 1.15--1.2 seconds per ordinary
  request; the K/V audit requests are approximately 1.47 seconds.
- A prior complete six-task seeded recording took 693.6 seconds (11.56
  minutes) serially.
- Prior non-factorized Cosmos/RoboLab states with 17 physically executed
  action chunks took the following mean wall times, measured from the earliest
  to latest artifact timestamps in
  `results/cosmos3_population_confirmatory_v1`: Banana 6.55, Rubik's Cube 6.50,
  Mustard 8.67, Spoon 19.21, Marker 43.45, and Smartphone 12.34 minutes.
- These timings show why model-call latency and GPU-hours alone are misleading:
  Marker's branch point is 320 simulator steps, so exact fresh replay and
  physical execution dominate.

## Per-state estimate

The frozen selection-free runner physically executes 14 action chunks per
state (four native branches, one exact repeat, and nine intervention/control
arms). Its all-pair action grid adds approximately 35 model calls versus the
older 17-chunk jobs, or only about 42 seconds. Scaling the measured jobs gives:

| Task | Selection-free minutes/state | Minimal K/V minutes/state |
| --- | ---: | ---: |
| Banana | 6.09 | 3.92 |
| Rubik's Cube | 6.05 | 3.90 |
| Mustard | 7.84 | 5.17 |
| Spoon | 16.52 | 11.37 |
| Marker | 36.48 | 25.63 |
| Smartphone | 10.86 | 7.33 |
| **One six-task environment seed** | **83.85** | **57.33** |

The K/V estimate uses ten physical chunks per state (two native branches, one
repeat, and seven minimal-factorial endpoint arms) and approximately three
additional model calls relative to the older measured jobs.

## Total and balanced schedule

- Selection-free: 8 environment seeds × 83.85 min = 11.18 client-GPU hours.
- K/V factorial: 4 environment seeds × 57.33 min = 3.82 client-GPU hours.
- Recordings: 8 environment seeds × 11.56 min = 1.54 client-GPU hours.
- Total physical-client workload: approximately **16.54 GPU-hours**.

Two A10 clients can balance this to approximately **8.27 hours per lane**:

| Lane | Recording seeds | K/V seeds, serialized on port 8002 | Selection-free seeds |
| --- | --- | --- | --- |
| A10-0 | 3554, 5017, 6693, 8632 | 3554, 5017 | 3554, 5017, 6693, 8632 |
| A10-1 | 4828, 5428, 8281, 8901 | 4828, 5428 | 4828, 5428, 8281, 8901 |

After both recording queues finish, A10-0 starts its two K/V seeds while A10-1
runs selection-free states on the general server. When A10-0 releases the
dedicated K/V server, A10-1 runs its two K/V seeds while A10-0 runs its
selection-free states. This keeps exactly one client on the stateful K/V
service and one client on the general service.

Allow approximately one hour for cross-region image/code transfer, tunnel and
renderer gates, and one hour for result synchronization and analysis. The
measured-rate forecast is therefore about **10.3 hours end to end**. Applying a
conservative 1.5× slowdown to all A10 experiment work gives approximately
**14.4 hours**, inside the 16-hour window. A 2× slowdown would take roughly
18.5 hours, so the first complete six-task seed should be used as the runtime
checkpoint; do not infer speed from the 1.2-second Cosmos calls alone.

The FastWAM work is isolated on its two-H100 node and is not included in these
physical-client GPU-hour totals.
