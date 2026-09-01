# Prospective robot-versus-object future-content protocol

Status: frozen before any intervention outcome for this study.

## Question and claim boundary

The prior study showed that reachable future transplantation directionally controls Cosmos Policy actions and executed robot endpoints. Its strongest early LIBERO endpoints contained robot motion but essentially no goal-object motion. This study asks whether future visual content carries task-world consequences beyond prospective robot pose.

The primary question is: when two natural futures end at closely matched robot poses but different task-object states, does transplanting the object-divergent future steer the action and executed goal state toward its associated natural branch? A positive result supports use of predicted task-state consequences. A null result with a positive robot-only contrast supports prospective visual inverse dynamics instead.

The study does not call either result planning, search, or success mediation. All targets are naturally realized endpoints from the exact saved state. Simulator-rendered hybrid images and generative inpainting are excluded from the primary study.

## Independence and calibration

The 30 prior confirmatory states are a calibration set only. Their natural branches were audited to establish physically interpretable matching thresholds; all prior intervention outcomes are prohibited from selection. Those states are excluded from this study.

The candidate grid is all ten LIBERO-10 tasks crossed with initial-state indices 17 and 35. The deterministic screening trajectory uses model seed 503 and at most 32 action chunks. For each trajectory, the three chunk starts whose next natural chunk produces the largest goal-feature change are candidate branch points. Ties favor earlier chunks. Selection uses only natural actions and exact simulator endpoints.

Candidates are evaluated in the frozen order. At the first candidate, 32 fixed branch seeds are collected. An additional 32 prespecified seeds are collected only if no common recipient has both registered pair classes. If the candidate remains ineligible after 64 branches, the same two-stage procedure advances to the next candidate. The first eligible candidate is selected. A trajectory with no eligible candidate after all three 64-branch pools is structurally unavailable and excluded. It is not replaced after any intervention is observed. At least ten eligible saved states are required for the confirmatory aggregate.

## Robot and task-world variables

Cosmos Policy's official nine-dimensional LIBERO proprioception is decomposed according to the public evaluator: two gripper joint coordinates, three end-effector position coordinates in meters, and a four-dimensional quaternion. Quaternion difference is the sign-invariant geodesic angle.

Task-world features follow LIBERO's parsed goal predicates in goal order. They include goal-argument positions and low-dimensional articulated joint coordinates; quaternion components are included only for orientation predicates. Fixed receptacle coordinates can appear in the vector but cancel in pairwise differences.

An **object-divergent, robot-matched** pair must satisfy:

- normalized action L2 at least 0.01;
- end-effector position difference at most 0.003 m;
- end-effector orientation difference at most 0.03 radians;
- gripper-coordinate L2 at most 0.005; and
- task-world goal-feature L2 at least 0.003.

A **robot-divergent, object-matched** pair must satisfy:

- normalized action L2 at least 0.01;
- task-world goal-feature L2 at most 0.0005; and
- at least one of end-effector position difference at least 0.003 m, orientation difference at least 0.03 radians, or gripper-coordinate L2 at least 0.003.

Pair selection requires a common recipient with at least one donor of each class. Eligible common-recipient triples are scored using within-state ranks fixed in code: object donors favor larger action and object differences and smaller robot differences; robot donors favor larger action and robot differences and smaller object differences. Ties are lexicographic. A joint donor and a distance-matched natural control are selected without intervention outcomes for the multi-donor analysis.

## Interventions

For each anchor recipient, the public Cosmos tokenizer encodes the recipient, object donor, robot donor, joint donor, and natural-control endpoints into future proprioception, wrist-image, and primary-image latent slots. Each target is clamped through all denoiser evaluations while current observation, instruction, recipient action noise, value noise, schedule, and solver remain fixed. Future-noise seeds are 1201, 1213, and 1217. Both pair directions are tested when the reverse direction independently satisfies the registered class thresholds.

All-future clamps are primary. Wrist, primary-camera, and proprioception-only clamps are secondary. A clean-latent Gaussian target exactly matched in norm and recipient distance is the nonsemantic control. Decoded-future distances are mandatory manipulation checks.

The multi-donor identification analysis evaluates whether each transplanted future aligns most strongly with its own associated natural action rather than merely moving away from the recipient. It reports the diagonal alignment margin and top-1 donor identification rate over the prespecified donor set.

## Outcomes and inference

The primary outcome is donor steering of the executed LIBERO goal-feature vector for object-divergent, robot-matched pairs after the 16-step action chunk. The principal secondary outcome is normalized action donor steering for the same pair. Robot-only action and executed robot-endpoint steering, modality effects, multi-donor identification, decoded-future similarity, and Boolean predicate changes are secondary.

Saved state is the independent unit. Reciprocal directions, donors, and noise seeds are repeated measurements. Means receive 10,000-resample state-cluster bootstrap intervals. Complete unit effects and task identities are reported. The registered smallest effect of interest is 0.10 donor-steering units. No binary success claim is made without stable continuation labels under a separately frozen protocol.

## Interpretation

- Positive object-pair action and executed goal-state effects above controls support causal use of predicted task-world consequences beyond endpoint robot pose.
- Positive robot-pair effects with null object-pair effects support prospective robot-motion conditioning or visual inverse dynamics.
- Positive effects that fail to beat Gaussian or alternative natural controls are interpreted as generic coupling.
- Fewer than ten structurally eligible units leaves the confirmatory object-content estimand underidentified; available units remain an explicitly exploratory report.
