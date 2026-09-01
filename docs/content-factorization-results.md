# Robot-versus-object future-content results

## Question

Earlier reachable-future transplantation showed that Cosmos Policy actions and
executed endpoints move toward a donor future. Those early endpoints mostly
encoded visible robot motion, not goal-object motion. The follow-up asks whether
the action pathway uses predicted task-world consequences independently of the
future robot pose.

Two prospectively frozen studies address complementary parts of this question.
The natural-pair study searches policy-sampled endpoints from the exact same
saved state for (i) object-divergent, robot-matched pairs and (ii)
robot-divergent, object-matched pairs with a common recipient. The secondary
2x2 study re-renders natural current and future endpoints as O0R0, O1R0, O0R1,
and O1R1 cells, allowing object content to vary while robot pose and camera
viewpoint are exactly fixed. The hybrid cells are counterfactual and therefore
are not treated as on-distribution natural futures.

## 2x2 hybrid study

All 20 frozen LIBERO-10 units passed the registered target validation. The
largest factor-preservation error across object and robot checks was exactly
zero. Natural endpoint changes were substantial: median task-world goal L2 was
0.187 and median end-effector displacement was 0.161 m. Fresh simulator renders
are not interchangeable with the live current observation: median live-versus-
rerender L1 was 0.466 primary-camera pixels and 0.552 wrist-camera pixels. The
v2 pipeline therefore preserves the live observation as policy input and uses
fresh renders only as future targets; exact native-action reproduction is
mandatory.

The manipulation succeeded. Decoded primary-camera futures identified their
four-cell target with mean top-1 0.925 (95% cluster-bootstrap interval 0.85 to
1.00); wrist futures reached 0.9875 (0.9625 to 1.00). Both own-versus-nearest-
alternative margins were positive. Executed endpoints, however, identified the
target cell at exactly 0.25, the balanced four-cell chance rate.

The preregistered object main effect on executed task-world donor steering was
0.00018 (95% CI -0.00040 to 0.00102), far below the 0.10 smallest meaningful
effect. The robot main effect was detectable but small: 0.00207 on goal steering
and 0.00589 on robot-endpoint steering (robot endpoint CI 0.00043 to 0.01640).
One task-9 state produced the largest robot-endpoint effect (0.1028); the median
unit effect was 0.00047. The object effect never exceeded 0.0067 at any unit.

Thus the intervention can impose specific object/robot future content without
inducing a correspondingly specific action or endpoint. This is a content-use
null under rendered counterfactual targets, not evidence that the cache is
unused and not an on-distribution natural-future null. Hybrid contact counts can
differ from natural cells (maximum increase 15 contacts), which remains an
explicit distribution-shift limitation.

## Natural reachable pairs

The complete frozen screen evaluated 20 trajectories and 56 sequential
candidate pools. Ten trajectories contained at least one object-divergent,
robot-matched pair, five contained at least one robot-divergent,
object-matched pair, and four had both classes at a common recipient. The four
eligible units were task 1/state 17, task 2/state 17, and task 9/states 17 and
35. Their selected object donors changed task-world goal features by 0.00349,
0.00405, 0.00387, and 0.01546 while matching end-effector position within
0.00048 to 0.00150 m. Their robot donors matched object goal features within
0.000058 to 0.000498 while changing a registered robot component.

Because this is below the registered minimum of ten units, inference is
exploratory. Reciprocal object-donor transplantation produced mean action
steering 0.0207 and executed-goal steering 0.0206 (95% CI -0.0908 to 0.1320),
or 0.46 mm signed projection (-0.34 to 1.46 mm). The four unit goal effects
were -0.0404, +0.1350, -0.1411, and +0.1290. Compared with the
distance-matched natural control, object action steering was -0.0242 and goal
steering was -0.0448 (-0.2025 to 0.0500). Correct-donor top-1 minus each unit's
own chance rate was exactly zero.

Robot-only transplantation was much stronger: reciprocal action steering was
0.291 (0.060 to 0.521) and executed robot-endpoint steering was 0.315 (0.059 to
0.571), a 1.85 mm projection (0.25 to 3.45 mm). Wrist imagery was the largest
single-modality effect (0.154), whereas numeric proprioception was null (0.0007)
and primary-camera content was small (0.0140); modality effects were
heterogeneous across units. This pattern is consistent with visually encoded
prospective robot motion and nonlinear multimodal coupling.

## Interpretation

Together with the prior reachable-donor study, the most precise supported claim
is: Cosmos Policy's future interface supports directional control through
prospective visible robot motion, but the present experiments do not establish
meaningful use of task-object consequences. RIFT-style destructive interventions
show that future-cache values and positions matter; these content interventions
show that necessity does not imply semantic task-state use.

| hypothesis | destructive cache tests | robot-only content | object-only content | observed fit |
| --- | --- | --- | --- | --- |
| arbitrary workspace or normalization | can be positive | no directional prediction | no directional prediction | incomplete |
| prospective visual inverse dynamics | positive | strong directional effect | weak or null | best fit |
| task-state consequence reasoning | positive | may be positive | meaningful directional effect | not supported |

The evidence does not establish planning, search, goal reasoning, or causal
mediation of success. The natural object estimand is structurally rare and
underpowered, while the 2x2 design achieves broad task coverage at the cost of
counterfactual-state distribution shift. Replication in a second model family
with a future interface and naturally diverse object outcomes is the strongest
next generalization test.
