# DreamZero future-latent transplant — evaluation

Complete states: 30/30

| Estimand | Mean | State-bootstrap 95% CI |
|---|---:|---:|
| retrieval accuracy off diagonal | 1.0000 | [1.0000, 1.0000] |
| distance reduction | 0.9172 | [0.8968, 0.9333] |
| normalized projection | 0.9902 | [0.9840, 0.9950] |
| cosine alignment | 0.9947 | [0.9913, 0.9971] |
| orthogonal residual | 0.0799 | [0.0648, 0.0986] |

Off-diagonal within-state future-label permutation p: 9.9999e-06
Permutation-null mean: 0.2498
Chance rate: 0.25
Maximum self-replay error: 0

The saved DROID state is the independent unit. The primary retrieval estimand excludes the four diagonal self-replays and averages the twelve donor cells within each state.
