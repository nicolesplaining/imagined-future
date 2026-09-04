# FastWAM cache-factorial powered v3 analysis freeze

Frozen while the powered factorial was incomplete and before any powered
factorial action or causal payload was opened.

- Factorial analyzer:
  `scripts/summarize_fastwam_cache_factorial.py`, SHA-256
  `0009a896644c40624acff25f7151e3322fd54663f57737b610edaeb07319e615`.
- Shared hierarchical bootstrap implementation:
  `src/imagined_future/fastwam_analysis.py`, SHA-256
  `ea0f7dde4278bd4904c0820f611f8b0902e4b1eebbf2b6d8a9c37b31958f65a3`.
- Required population: 120 states and 5,760 valid registered arms, with the
  complete 8,640-arm powered parent also required.
- Independent unit: saved state. The twelve ordered recipient-to-donor pairs
  are averaged within state.
- Interval: suite-to-task-to-state hierarchical bootstrap, 10,000 draws, seed
  20260903.
- Primary contrasts: cache main effect on donor retrieval and donor-distance
  reduction. The frozen evidence gate requires each hierarchical 95% lower
  bound to exceed zero.
- Mandatory secondary outputs: all four cells for all directional metrics,
  future main effect, future-by-cache interaction, and every exact control.
- Exact tolerance: 1e-6. Degenerate axes are retained and counted.

The donor-retrieval outcome here is used only inside a paired four-cell cache
main-effect contrast. It is not compared with a 0.25 chance rate. The analyzer
must audit completeness before loading action arrays and must not report a gate
from a partial matrix.

