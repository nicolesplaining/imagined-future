# Cosmos 3 confirmatory population results

## Status

The frozen confirmatory population study is complete at its prespecified
minimum: 22 independent saved-state clusters across all six eligible RoboLab
tasks were analyzed. The manifest selected 24 clusters before intervention
outcomes were available. Two native branches did not reproduce the full
32-action, 33-frame videos required to serve as registered intervention
sources, so those frozen units remain missing and were not replaced. The
minimum was 20 clusters across at least five tasks.

The separate robot/object factor study analyzed 10 of 12 selected clusters
across the same six tasks, meeting its frozen minimum of 10 clusters across at
least five tasks. No included intervention terminated early. The manifest,
per-unit summaries, aggregate, and failure report are stored under
`results/cosmos3_population*`.

## Primary directional result

All effects are normalized recipient-to-donor projections, so zero denotes the
recipient and one denotes the donor. Intervals are 10,000-resample cluster
bootstraps over saved states; task-balanced intervals average task means before
resampling tasks.

| registered contrast | mean | state-bootstrap 95% CI | positive clusters | task-balanced mean [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| predicted donor minus self, action | 0.992 | [0.972, 1.010] | 22/22 | 0.991 [0.967, 1.010] |
| predicted donor minus natural control, action | 0.433 | [0.332, 0.538] | 21/22 | 0.428 [0.391, 0.459] |
| executed donor minus executed self, action | 0.983 | [0.890, 1.064] | 22/22 | 0.978 [0.871, 1.073] |
| executed donor minus executed self, physical endpoint | 1.003 | [0.950, 1.050] | 22/22 | 1.001 [0.959, 1.041] |
| executed donor minus Gaussian, action | 0.941 | [0.837, 1.032] | 22/22 | 0.937 [0.843, 1.024] |
| executed donor minus natural control, action | 0.391 | [0.246, 0.533] | 18/22 | 0.385 [0.284, 0.473] |

Every task mean is positive for every primary contrast. All registered primary
donor and K/V sign tests remain positive after Holm correction. These results
establish population-level, cross-task directional sufficiency: a particular
coherent alternative future selects its associated alternative action, and the
effect survives exact simulator execution.

## Future-token pathway mediation

Replacing donor-run future-video keys and values with self-future keys and
values preserves token count, token order, current vision, language, action
tokens, and action noise. The reduction in donor projection is:

| future source and outcome | mediation loss | state-bootstrap 95% CI | positive clusters | task-balanced mean [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| predicted future, action | 0.877 | [0.842, 0.910] | 22/22 | 0.877 [0.854, 0.902] |
| executed future, action | 0.814 | [0.723, 0.900] | 22/22 | 0.812 [0.747, 0.871] |
| executed future, physical endpoint | 0.851 | [0.766, 0.922] | 22/22 | 0.853 [0.801, 0.900] |

This identifies a content-carrying future-token K/V pathway. It does not claim
a population natural indirect effect or a complete circuit decomposition.

## Robot-motion versus object-state separation

The primary factor study creates native-recipient videos with disjoint,
object-priority pixel edits: donor object pixels, donor robot pixels outside
the object mask, both, or neither. Its primary endpoint uses the full registered
physical state rather than small-denominator component projections.

| isolated factor | action effect | state-bootstrap 95% CI | full physical effect | state-bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| robot pixels | 0.0022 | [-0.0016, 0.0060] | 0.0102 | [-0.0067, 0.0275] |
| object pixels | 0.0061 | [0.0009, 0.0119] | -0.0001 | [-0.0254, 0.0239] |

All four intervals lie inside the preregistered +/-0.10 equivalence region,
and task-balanced estimates agree. The small positive action effect of object
pixels is statistically detectable but only 0.6% of the native donor
displacement; it is practically negligible under the registered smallest
effect of interest. Robot-specific endpoint projection gives a similarly small
robot effect, 0.0118 [0.0026, 0.0225]. Object-specific endpoint ratios are
unstable because their native denominators are near zero and are not used for
equivalence claims.

The rigorous interpretation is therefore compositional: whole coherent Cosmos
3 futures steer strongly, while neither isolated robot-motion pixels nor
isolated object-state pixels are sufficient for a meaningful fraction of that
effect. This does not prove that both factors are jointly necessary; context,
distributed visual changes, tokenizer interactions, or other correlated future
content may carry the whole-future signal.

## Missingness and claim boundary

`MarkerInMugTask-seed-103` and `MustardInLeftBinTask-seed-101` are missing
because a registered native intervention-source branch terminated after 21 and
29 actions, respectively, instead of producing the required 32 actions. The
Mustard failure was reproduced in an independent process. Neither unit was
replaced, and no favorable intervention outcome entered selection.

The study supports cross-generation, population-level future-content steering
and K/V mediation. It does not establish search over alternative futures,
selection by expected task value, mediation of binary success, or general use
of predicted object consequences. The Cosmos Policy content results favor a
visible inverse-dynamics account at early LIBERO states; the Cosmos 3 factor
result shows that this carrier is not universal as an isolated pixel factor.
