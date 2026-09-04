# Manuscript-ready experimental update

This handoff supplies concise, audited language and compact tables. It does not
modify the manuscript.

## Recommended scientific throughline

> Coherent predicted futures directionally select their associated robot
> actions across unselected states and across two released world action models.
> In Cosmos 3, future-token attention K/V can rescue and redirect this effect.
> The task-weighted response grows with imposed donor-future strength, and a
> single late solver-call intervention numerically recovers the headline
> steering observed with persistent intervention.

This statement is supported directly by the frozen experiments below. Keep
physical endpoint claims attached to the paper's existing endpoint cohort; the
new selection-free, timing, dose, K/V-factorial, and FastWAM studies measure
predicted action chunks.

## Compact headline table

| Question | Frozen evaluation | Headline result |
|---|---|---|
| Does steering survive selection-free evaluation? | Cosmos 3; 90 states, six tasks, four futures/state | Correct future-source action in 100% of 1,440 balanced source cells; off-diagonal distance reduction 0.763 [0.733, 0.792] |
| Does it generalize beyond Cosmos? | FastWAM Optional-IDM; 120 states, all 40 LIBERO tasks | Correct source in 1,919/1,920 cells (99.95%); distance reduction 0.683 [0.638, 0.738] |
| Can future-token K/V redirect the action? | Cosmos 3; future × K/V factorial, 21 states | All 42 crossed arms followed the K/V source; K/V effects 0.895 [0.859, 0.928] and 0.885 [0.845, 0.921] |
| Does the effect require clamping every solver call? | Cosmos 3; 30 states, six timing conditions | Calls 2 and 3 alone each gave 100% donor retrieval and approximately 0.761 distance reduction, numerically similar to all calls (0.760) |
| Is the response graded with future strength? | Cosmos 3; 30 states, five mixture levels | Distance-reduction slope 0.899 [0.852, 0.936]; the task-weighted mean increased at every adjacent alpha step |

Intervals use each experiment's frozen hierarchy: task→episode→state for the
archival Cosmos evaluation, suite→task→state for FastWAM, and task→state for
the other Cosmos studies. Branches and ordered donor directions are averaged
within state.

## Insert-ready result paragraphs

### Selection-free Cosmos 3 evaluation

We next tested whether directional steering survives without selecting
maximally separated or visually compelling branches. We froze 90 archival
states spanning six tasks, five episodes per task, and early, middle, and late
decision phases, then evaluated all four native futures and all 12 ordered
recipient-to-donor pairs at every state. The recomputed action was closest to
the action associated with the transplanted future in every balanced source
cell (100%, hierarchical 95% CI [100%, 100%]; exact conditional
label-permutation expectation 25%). Across off-diagonal
pairs, transplantation reduced distance to the donor action by 0.763 [0.733,
0.792], with donor-axis projection 0.966 [0.956, 0.974] and normalized
orthogonal residual 0.227 [0.200, 0.255]. The effect held at every decision
phase and in every prespecified native-separation quartile.

### Cross-model replication

To test whether directional future steering extends beyond Cosmos, we applied
the same source-identification logic to the released FastWAM Optional-IDM
architecture, which generates a future-video representation before denoising
its action. The frozen evaluation covered 120 untouched states across all 40
LIBERO tasks. With recipient action noise held fixed, the recomputed action
identified the transplanted coherent future in 1,919 of 1,920 balanced source
cells (99.95%, hierarchical 95% CI [99.74%, 100%]; exact conditional
label-permutation expectation 25%). Off-diagonal
interventions reduced distance to the donor action by 0.683 [0.638, 0.738],
with projection 0.869 [0.833, 0.912] and cosine alignment 0.943 [0.927, 0.962].
The matched `first_frame` route was exactly invariant to donor-video seed,
showing that donor-video identity cannot steer the action through a route that
does not consume a predicted future.

### Cosmos 3 future × K/V factorial

We crossed the identity of the realized future with the identity of the
future-token keys and values available to action queries. With a recipient
future, replacing recipient K/V by donor K/V increased donor projection by
0.895 [0.859, 0.928]; with a donor future, the corresponding K/V effect was
0.885 [0.845, 0.921]. The two crossed conditions followed the K/V source in all
42 state-level comparisons. Visible-future identity retained a smaller effect
of 0.120–0.130. Under this factorial intervention, changing the K/V source
produced most, but not all, of the donor-axis shift.

### Single-call timing audit

Finally, we tested whether the Cosmos 3 result depends on repeatedly clamping a
clean target at every solver call. Across 30 fixed states, imposing the donor
future at any single call produced a significant donor-specific action shift
after Holm correction. Correct off-diagonal donor retrieval was 0.728 [0.642,
0.803] at call 0, 0.922 [0.856, 0.975] at call 1, and 1.000 [1.000, 1.000] at
calls 2 and 3. Calls 2 and 3 alone produced distance reductions of 0.761
[0.717, 0.798] and 0.761 [0.715, 0.801], respectively, essentially identical
to the all-call intervention, 0.760 [0.714, 0.801]. The directional result
therefore does not require persistent clamping throughout the solver. No
equivalence test was prespecified, so this is numerical agreement rather than
a formal equivalence claim.

### Future-strength dose response

To test whether steering appears only under complete replacement, we
prospectively interpolated the imposed Cosmos 3 latent future from recipient to
donor at alpha in `{0, .25, .50, .75, 1}` across 30 fixed states. Mean
donor-directed distance reduction increased from -0.022 [-0.040, -0.007] at
alpha 0 to 0.763 [0.716, 0.800] at alpha 1, with a prespecified per-pair OLS
slope of 0.899 [0.852, 0.936]. The task-weighted mean increased at every
adjacent alpha step, with all four hierarchical lower bounds above zero.
Correct-donor retrieval rose from 0.000 at alpha 0 to 0.992 [0.981, 1.000] at
alpha .75 and 1.000 [1.000, 1.000] at alpha 1. All six task-specific slopes and
all six leave-one-task-out slopes were positive; descriptively, every one of
the 360 within-state pairwise slopes was positive.

## Compact mechanism table

| Visible future | Future-token K/V | Donor projection [95% CI] |
|---|---|---:|
| Recipient | Recipient | -0.011 [-0.022, 0.000] |
| Donor | Recipient | 0.119 [0.078, 0.162] |
| Recipient | Donor | 0.884 [0.848, 0.919] |
| Donor | Donor | 1.004 [0.988, 1.025] |

Suggested caption: **Future-token K/V redirects Cosmos 3 actions.** Crossing
visible-future identity with future-token K/V identity shows that the action
primarily follows the K/V source. Each estimate first averages donor directions
within state and then weights the six tasks equally; brackets show hierarchical
95% confidence intervals.

## Internal scope notes

- The new Cosmos 3 selection-free study uses lossy reconstructions of fixed
  archival observations and adds action evidence, not new executed endpoints.
- The timing intervention isolates one solver call, not an individual layer,
  head, or naturally occurring mediation event.
- The dose result supports a graded task-weighted mean response under imposed
  all-call latent interpolation. It does not establish a monotone response for
  every ordered pair; 257/360 individual pair profiles were nondecreasing.
- FastWAM latent transplantation and cache transplantation access the same
  deterministic future-to-action bottleneck; do not present them as two
  independent replications.
- The FastWAM cache factorial validates the explicit cache interface. Under a
  supplied cache override, action denoising bypasses latent prefill, so the
  exactly zero same-cache latent effect is an architecture control rather than
  a learned cache-preference result.
- The Cosmos 3 K/V factorial supports a large causal contribution and
  conditional rescue/redirection. Because donor-future/recipient-K/V retains a
  positive projection of 0.119 [0.078, 0.162], reserve “strict necessity” or
  “complete mediation” for a stronger experiment.

## Primary artifacts

- Full evidence package: `docs/overnight_2026-09-03/submission_update.md`
- Selection-free summary figure:
  `output/overnight_2026-09-03/cosmos3_archival_selection_v7_final/presentation_v1/cosmos3_archival_selection_free_summary.pdf`
- Selection-free table:
  `output/overnight_2026-09-03/cosmos3_archival_selection_v7_final/presentation_v1/cosmos3_archival_selection_free_table.tex`
- Timing summary figure:
  `output/overnight_2026-09-03/cosmos3_single_call_timing_v5/evaluation/analysis/cosmos3_single_call_timing_summary.png`
- Timing table:
  `output/overnight_2026-09-03/cosmos3_single_call_timing_v5/evaluation/analysis/cosmos3_single_call_timing_results.tex`
- Cross-model pathway table:
  `output/overnight_2026-09-03/combined_pathway_factorial_table.tex`
- Dose-response summary figure:
  `output/overnight_2026-09-03/cosmos3_future_strength_dose_v2/presentation_v1/cosmos3_future_strength_dose_response.pdf`
- Dose-response table:
  `output/overnight_2026-09-03/cosmos3_future_strength_dose_v2/presentation_v1/cosmos3_future_strength_dose_response_table.tex`
- FastWAM presentation figure:
  `output/overnight_2026-09-03/fastwam_optional_idm_powered/presentation_v1/fastwam_optional_idm_summary.pdf`
