# Confirmatory results

Status: action-level LIBERO analyses complete; physical endpoint replay, continuation labels, and RoboCasa replication in progress. This document is updated only at prespecified milestones and keeps incomplete outcomes visibly incomplete.

## Semantic future intervention

The independent unit is a saved state. Each cell first averages both reciprocal directions and all three future-noise seeds within state, then reports the mean over ten LIBERO-10 tasks. Confidence intervals use the registered 10,000-resample state bootstrap.

| timing stratum | states | all-future donor − self | 95% CI | positive states | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| first query | 10 | +0.4986 | [+0.3360, +0.6785] | 10/10 | +0.4128 |
| after 3 chunks | 10 | +0.0310 | [+0.0057, +0.0659] | 8/10 | +0.0163 |
| after 6 chunks | 10 | +0.0272 | [−0.0099, +0.0929] | 6/10 | +0.0009 |
| all registered states | 30 | +0.1856 | [+0.0953, +0.2905] | 24/30 | +0.0217 |

At the first query, all 60 reciprocal-direction/noise-seed repetitions are positive. The exact two-sided state sign-test p-value is 0.00195 (Holm-adjusted across the three timing strata: 0.00586). The registered 0.10 smallest effect of interest is exceeded by the first-query interval, but not by the mid, late, or pooled lower confidence limits. Early-minus-mid and early-minus-late task-paired differences are +0.4676, 95% CI [+0.3186, +0.6361], and +0.4714, 95% CI [+0.3298, +0.6453], respectively; all ten task-wise differences have the registered direction in both comparisons.

The early effect survives the main controls:

| first-query contrast | mean | 95% CI | positive states |
| --- | ---: | ---: | ---: |
| donor − exact norm/distance-matched Gaussian | +0.4973 | [+0.3353, +0.6792] | 10/10 |
| donor − preselected natural control | +0.1031 | [+0.0492, +0.1595] | 8/10 |
| donor − within-task shuffled future | +0.2550 | [+0.1820, +0.3363] | 10/10 |

The modality decomposition localizes most of the first-query effect to the wrist-camera future (+0.4347, 95% CI [+0.2872, +0.5996]). The primary-camera future contributes +0.1006, 95% CI [+0.0584, +0.1448]. Future proprioception alone is null (+0.0013, 95% CI [−0.0045, +0.0080]).

These results reject a universal observation-to-action shortcut at early LIBERO decision points: replacing a model-consistent future representation while holding the current observation, instruction, recipient action noise, and denoising schedule fixed changes the action toward the transplanted natural future. They do not support the stronger claim that future mediation is uniform over an episode. The registered mid/late strata reveal a sharp state dependence, including a late mean dominated by one task.

## Future-to-action attention

The block-27 future-key removal produces a monotonic first-query action-disruption curve in normalized action L2: 0.0217, 0.0352, 0.0535, and 0.0683 at gates 0.25, 0.5, 0.75, and 1.0. The full-removal 95% state-bootstrap CI is [0.0538, 0.0842]. The all-key recomputation control is exactly zero in every state.

Specificity controls temper the necessity claim. At the first query, full future-key removal is more disruptive than equal-count random-key removal by +0.0156, 95% CI [+0.0032, +0.0293], and more disruptive than block-0 future-key removal by +0.0483, 95% CI [+0.0385, +0.0602]. Equal-count current-key removal is more disruptive than future-key removal by +0.0243, 95% CI [+0.0144, +0.0337]. Thus the late-layer future-to-action edge is causally active, but it is not the only or largest information route through block 27.

## Incomplete registered outcomes

- Physical/proprioceptive endpoint replay is running for every semantic and attention condition. No action-only result will be relabeled as behavioral endpoint evidence.
- Natural continuation seed 353 is screening the primary pairs and preselected natural controls. Seeds 359 and 367 will be used only for pairs entering the registered second stage.
- The separate six-unit RoboCasa replication remains unobserved at intervention time. Its environment, checkpoint revision, compatibility overrides, exact-replay criteria, endpoint schema, and pair-selection rule were frozen before collection.
