# Confirmatory results

Status: all registered LIBERO action and physical endpoint analyses, continuation labels, and the exploratory RoboCasa replication are complete.

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

## Executed physical endpoints

Every saved action condition was replayed after verifying the exact saved simulator-state digest. The independent-unit and within-state averaging rules match the action analysis. Donor-minus-self steering along the natural recipient-to-donor physical endpoint axis is:

| timing stratum | states | physical donor − self | 95% CI | positive states | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| first query | 10 | +0.5520 | [+0.3875, +0.7279] | 10/10 | +0.4913 |
| after 3 chunks | 10 | +0.0036 | [−0.0433, +0.0378] | 8/10 | +0.0132 |
| after 6 chunks | 10 | +0.0249 | [+0.0050, +0.0437] | 8/10 | +0.0274 |
| all registered states | 30 | +0.1935 | [+0.0914, +0.3052] | 26/30 | +0.0422 |

At the first query, the physical effect beats exact norm/distance-matched Gaussian corruption by +0.5491, 95% CI [+0.3830, +0.7282], a preselected natural-future control by +0.1103, 95% CI [+0.0541, +0.1665], and within-task shuffled futures by +0.2773, 95% CI [+0.1986, +0.3569]. All ten task-level effects are positive for donor-minus-self, donor-minus-Gaussian, and donor-minus-shuffled. Early-minus-mid and early-minus-late task-paired physical differences are +0.5484, 95% CI [+0.3611, +0.7403], and +0.5271, 95% CI [+0.3710, +0.6982]; all ten paired differences are positive. Thus the early action effect survives actual simulator execution and is not merely a change in an action-vector metric.

## Future-to-action attention

The block-27 future-key removal produces a monotonic first-query action-disruption curve in normalized action L2: 0.0217, 0.0352, 0.0535, and 0.0683 at gates 0.25, 0.5, 0.75, and 1.0. The full-removal 95% state-bootstrap CI is [0.0536, 0.0849]. The all-key recomputation control is exactly zero in every state.

Specificity controls temper the necessity claim. At the first query, full future-key removal is more disruptive than equal-count random-key removal by +0.0156, 95% CI [+0.0032, +0.0296], and more disruptive than block-0 future-key removal by +0.0483, 95% CI [+0.0385, +0.0602]. Equal-count current-key removal is more disruptive than future-key removal by +0.0243, 95% CI [+0.0142, +0.0340]. Thus the late-layer future-to-action edge is causally active, but it is not the only or largest information route through block 27.

Executed endpoints support the early necessity result. First-query future-key removal changes physical endpoint steering by an absolute mean 0.2808, 95% CI [0.1639, 0.4298], positive in 10/10 states. It is more disruptive than equal-count random-key removal by +0.1687, 95% CI [+0.0579, +0.3222], in 9/10 states. Its difference from equal-count current-key removal is −0.0354 with a confidence interval crossing zero [−0.2200, +0.1111]. The all-key endpoint control is exactly zero in all 30 states. At mid and late states, future-key removal still changes endpoints, but it is not more disruptive than the random/current controls; specificity is therefore concentrated at the first query.

## Natural continuation outcomes

Seed 353 screened every frozen primary pair and its preselected distance-matched natural controls. Three of 30 primary pairs had different binary success outcomes and therefore entered the registered second stage. Each apparent failure changed to success under at least one of seeds 359 and 367. The frozen labeled manifest consequently contains zero robust success/failure contrasts. This is a negative result for binary success mediation and an identifiability limitation, not evidence against the continuous action and physical-state mediation effects.

## RoboCasa exploratory replication

The separately frozen replication uses six exact-replay states, two each from OpenDrawer, PnPCounterToCab, and TurnOffMicrowave. It is never pooled with LIBERO. Semantic donor-minus-self action steering is small but positive on average (+0.0144, state-bootstrap 95% CI [+0.0034, +0.0276], 4/6 states and 3/3 task means positive). The action contrast does not reliably beat the Gaussian control (+0.0025, interval crosses zero), so the action-only replication is suggestive rather than decisive.

Executed physical endpoints are more consistent: donor-minus-self steering is +0.0287, 95% CI [+0.0101, +0.0492], positive in 6/6 states, and donor-minus-Gaussian is +0.0187, 95% CI [+0.0063, +0.0329], also positive in 6/6. Both contrasts are positive after averaging the two states within each of the three task families. The unadjusted exact state sign-test p-value for each 6/6 contrast is 0.03125; because RoboCasa is a small exploratory replication, task-clustered estimates and all individual states are reported alongside it.

Future-key removal at block 27 changes RoboCasa actions in 6/6 states (mean L2 0.0339, 95% CI [0.0260, 0.0432]) and executed physical endpoints in 6/6 (mean absolute steering change 0.0468, 95% CI [0.0126, 0.1000]). The all-key control is exactly zero. Equal-count current-key removal is slightly larger in action space and substantially more variable at physical endpoints, so RoboCasa supports an active but non-exclusive future-to-action route.

## Overall interpretation

The combined sufficiency and necessity results rule out a universal behaviorally epiphenomenal future head. Cosmos Policy causally uses imagined-future representations at early LIBERO decision points, and the effect transfers to executed simulator state and a small second checkpoint/environment study. The mechanism is strongly state dependent, shares influence with current-observation routes, and is not identified here as a mediator of robust binary success.

Machine-readable state repetitions and bootstrap summaries are in [`results/confirmatory_v1`](../results/confirmatory_v1) and [`results/robocasa_replication_v1`](../results/robocasa_replication_v1). Both PDF and PNG versions of the action and endpoint figures are stored with the confirmatory results.
