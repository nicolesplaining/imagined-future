# Skeptical reviewer readout

## Bottom line

The overnight results fix the paper's largest internal-validity weakness and
strengthen its mechanism claim. For a five-page paper, the main
text should contain exactly two new results: the selection-free Cosmos 3 study
and the Cosmos 3 future x K/V factorial. The other completed experiments are
valuable robustness or external-validity evidence, but presenting them at equal
weight would dilute the discovery narrative.

## Ranking by reviewer value

1. **Selection-free Cosmos 3 multi-donor evaluation.** This is the most
   important addition because it directly answers the strongest cherry-picking
   objection. The frozen cohort contains 90 states across six tasks, five
   episodes per task, and early/middle/late phases, with all four native futures
   and all 12 ordered donor pairs. Future-source retrieval was 1.000 in all
   1,440 balanced cells, versus 0.25 chance; off-diagonal distance reduction was
   0.763 [0.733, 0.792], projection 0.966 [0.956, 0.974], and normalized
   orthogonal residual 0.227 [0.200, 0.255]. The effect remained positive in
   every separation quartile and stable under leave-one-task-out analysis.
   Because the result is not supported by projection alone, it substantially
   weakens the argument that the metric merely rewards motion along a
   pair-selected axis.

2. **Cosmos 3 future x K/V factorial.** This is the strongest mechanistic
   result. Crossing visible-future and future-token K/V identity in 21 states
   produced K/V effects of 0.895 [0.859, 0.928] and 0.885 [0.845, 0.921]. Both
   crossed conditions followed the K/V source in all 42 state-level
   comparisons. Crucially, this includes rescue/redirection: donor K/V moved an
   action toward the donor even while the realized visible future remained the
   recipient future. It is more informative than suppression alone.

3. **Cosmos 3 future-strength dose response.** This is unusually clean causal
   support for a graded rather than merely binary intervention effect. Across 30
   states and 360 ordered pairs, the distance-reduction slope was 0.899 [0.852,
   0.936], and the task-weighted mean increased at every adjacent alpha step,
   with every contrast's hierarchical lower bound above zero. All 30 state
   slopes and all 360 pairwise OLS slopes were positive. The nonlinear jump from
   alpha 0.50 to 0.75 also usefully shows that the response is graded but not
   proportional.

4. **FastWAM Optional-IDM replication.** This provides strong external validity:
   1,919/1,920 correct-source cells across 120 untouched states and all 40
   LIBERO tasks, with distance reduction 0.683 [0.638, 0.738]. It shows that
   directional future-conditioned action steering is not unique to Cosmos-style
   joint denoising, although it remains action-only and FastWAM has an explicit
   future-to-action architecture.

5. **Cosmos 3 single-call timing audit.** This rebuts the claim that steering
   exists only because a clean future is clamped at every solver call.
   Calls 2 and 3 alone each achieved 1.000 retrieval and approximately 0.761
   distance reduction, versus 0.760 for all calls. It does not localize a
   computation because the intervention alters state inherited by later calls.

6. **FastWAM cache factorial.** This is an implementation and architecture
   control, not a headline discovery. Donor-cache rescue succeeded in
   1,439/1,440 comparisons, but a supplied cache explicitly bypasses latent
   prefill, making zero same-cache latent effects expected by control flow.

## Main text versus appendix

**Main text:** Give the selection-free Cosmos 3 result one compact table or
panel containing retrieval, distance reduction, projection, and orthogonal
residual, plus one sentence covering phases/separation quartiles. Give the
Cosmos future x K/V factorial its 2x2 table and the two K/V contrasts. Together
these establish the clean evidence chain: the directional effect survives
unselected states, then a controlled internal intervention rescues and redirects
it.

**Appendix:** Put the full timing table, FastWAM powered replication, and
FastWAM cache factorial there, together with the full dose-response profile. In
the main text, one sentence may report the positive dose slope and one may
mention FastWAM replication, but neither should displace the two core tables.
Keep all control matrices,
per-task/phase/quartile analyses, exact replay checks, and provenance details in
the appendix.

## Remaining claim-language vulnerabilities

- Do not call the archival Cosmos cohort "fresh held-out data." It is a frozen,
  selection-free archival cohort using lossy reconstructed observations.
- Do not infer natural mediation from imposed transplantation. The results show
  causal dependence under intervention, not that the model naturally consults
  or compares alternative futures during ordinary generation.
- "K/V carries most of the effect" is acceptable only when tied to this audited
  all-layer factorial. Do not say K/V is necessary, fully mediates the effect, or
  uniquely determines action: donor future with recipient K/V retains projection
  0.119 [0.078, 0.162], and both visible-future contrasts are positive.
- Do not call the timing result formal equivalence or localize computation to a
  solver call. The late-call and all-call estimates are numerically similar, but
  no equivalence test was prespecified.
- Describe dose response at the aggregate or slope level, not as universal
  stepwise monotonicity: only 257/360 pair profiles were nondecreasing at every
  step. It is all-call latent interpolation, potentially off-manifold, not a
  policy-native measure of "future strength" or evidence of natural mediation.
- Say the result "extends to FastWAM," not that it is universal across WAMs.
  Do not call FastWAM SOTA without a benchmark claim, and do not count latent and
  cache replay as two replications.
- The new studies measure predicted action chunks. Claims about physical robot
  behavior or endpoint mediation must remain attached to the paper's existing
  executed-endpoint evidence, not these new cohorts.
- Perfect nearest-native retrieval does not establish semantic planning,
  outcome evaluation, or success/failure comparison. The four futures are
  distinct stochastic continuations, but their task semantics were not
  independently annotated in these studies.

## Revised evidence hierarchy

1. **Core phenomenon:** selection-free Cosmos 3 transplantation identifies the
   paired action across states, phases, tasks, and separation strata.
2. **Mechanistic bridge:** Cosmos future-token K/V rescue/redirection shows that
   the action largely follows the K/V source while preserving a smaller visible-
   future contribution.
3. **Graded causal control:** Cosmos donor-directed action displacement rises
   with imposed recipient-to-donor latent interpolation.
4. **Cross-architecture scope:** FastWAM reproduces directional source-specific
   steering across its full 40-task benchmark grid.
5. **Intervention robustness:** a single late Cosmos solver-call intervention
   recovers the numerical all-call effect.
6. **Interface validation:** the FastWAM cache factorial verifies its explicit
   future-to-action bottleneck but should not be sold as mechanistic discovery.
