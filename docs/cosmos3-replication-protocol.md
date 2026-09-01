# Cosmos 3 cross-generation replication protocol

## Central question

Does the released Cosmos3-Nano-Policy-DROID action pathway use the *content* of
its jointly generated future video? The confirmatory claim requires more than
sensitivity to a cache. From the exact same current observation, robot state,
instruction, and action-noise trajectory, replacing only future-video content
with a coherent alternative must move the generated action toward the action
associated with that alternative future.

This is a cross-generation replication, not a pooled reanalysis of the earlier
Predict2-based Cosmos Policy result. The released DROID checkpoint and RoboLab
are the primary model/environment pair. The public Cosmos 3 LIBERO post-training
recipe is a separately labeled extension because it changes both training data
and checkpoint.

## Frozen implementation contract

The implementation follows the public Cosmos Framework state order
`[vision | lidar | action | sound]` and derives every boundary from prepared
tensor shapes; it does not reuse Predict2 block 27 or any old frame offsets.
Cosmos 3 uses rectified flow, so a donor target `z*` with common path noise `e`
is presented at noise level `s` as `(1-s) z* + s e`. The model's final guided
velocity—not merely its conditional pre-guidance velocity—is intercepted at the
sampler boundary. Selected future-vision coordinates receive a velocity whose
clean estimate is `z*`; the current vision and every action coordinate are
copied exactly from the native call.

The following checks must pass before outcomes are inspected:

1. Token census records raw frames, VAE latent frames, spatial token geometry,
   action rows, packed indices, and flat modality slices from a real DROID
   request.
2. A recorder exposes conditional and final guided clean-video estimates at all
   four denoising steps and labels the distinction explicitly.
3. Identity layout capture and identity sampler wrapping reproduce native
   action and vision latents bit-for-bit for a fixed request and seed.
4. Native self recomputation reproduces the native output under the registered
   tolerance. The mechanics-matched clean-future self clamp is not treated as
   an identity operation. Donor clamping changes only unconditioned
   future-vision coordinates; programmatic sentinels verify zero
   action-coordinate mutation.
5. Donor encode/decode round trips are audited before behavioral results are
   interpreted. An excluded engineering run showed that absolute raw-RGB
   nearest-cell identity is dominated by common VAE reconstruction error when
   factorial cells differ in only a small pixel region. The confirmatory gate
   therefore tests whether each of the four paired robot/object difference
   directions is preserved after encode/decode; at least 90% must have positive
   raw-versus-decoded difference cosine. Absolute four-cell RGB identity remains
   a reported diagnostic. The public Cosmos
   [`CropToMultiple`](https://github.com/NVIDIA/cosmos-framework/blob/d4599e2e43fbd06168e9884205b9b66c3902d8f6/cosmos_framework/data/generator/augmentors/cropping.py)
   transform center-crops the
   540-pixel input height to 528 pixels (the nearest lower multiple of 16), so
   raw-target comparisons apply that same deterministic six-pixel crop at the
   top and bottom rather than requiring an incorrect 540-pixel decoded height.
6. Gaussian controls match both donor latent norm and recipient-to-donor
   distance within relative tolerance `1e-5`. Natural shuffled controls use a
   valid endpoint from the same task and never the selected donor.
7. RoboLab recreates a fresh environment for every condition, restores the
   recorded initial state, and replays the recorded prefix with zero HDF5 state
   error. Independent reconstructions must have identical branch-state digests,
   and one repeated native continuation must have an identical endpoint digest,
   before reciprocal branches are collected.

Any failed check blocks the associated causal claim; it is not converted into
a negative behavioral result.

## Outcome-independent screen

The eight pilot tasks and environment seeds are fixed in
`configs/cosmos3_replication.toml`. The original three seeds (101, 103, and
107) screen task feasibility. Before any population intervention was run, seed
109 was added because the six calibration-feasible tasks yield at most 18
states under three seeds, below the frozen 20-state minimum. The four-seed
native census still yielded fewer than 20 eligible units after the unchanged
task-success and full-video rules were applied. Before any population
intervention, seed 113 was therefore frozen as a balanced final expansion on
all six calibration-feasible tasks. No further seed may be added after any
population intervention begins. Sixteen native diffusion
branches are drawn from each saved state. All 16 action chunks are generated,
the maximum native action-L2 pair is selected with a fixed lower-seed tie
break, and only that pair plus an exact repeated continuation is physically
executed. A state whose fixed pair fails the endpoint-displacement threshold is
ineligible without testing a replacement pair. Pilot units are used only to establish
feasibility, calibrate representation interfaces, and identify tasks with both
native success and distinct physically reachable endpoints. Intervention
outcomes may not enter selection. Pilot states and attention-localization states
are excluded from confirmatory inference.

Branch times are fixed at the task level from the excluded seed-0 native
trajectories: step 64 for banana, Rubik's cube, and smartphone; step 96 for
mustard; step 128 for spoon; and step 320 for marker. These points precede the
first substantive target-object displacement in the calibration trajectory and
are not retuned on population states.

Before any population intervention, the native census found that all four
smartphone rollouts failed the frozen task-success criterion and that one
marker unit could not supply a full 32-action donor video. Rather than relax
either rule or add favorable environment seeds, both reserve tasks already
named in the pilot-task list were screened. Their branch points were frozen
from the excluded seed-0 trajectories before reserve screening: step 64 for
bagels (first registered grab at step 70) and step 192 for yogurt (first
registered grab at step 215). Every reserve task with at least one successful
native rollout is admitted and then subjected to the unchanged 16-seed donor,
exact-replay, endpoint, and task-cap rules. This is a transparent
pre-intervention power amendment; reserve-task intervention outcomes cannot
enter admission.

Confirmatory tasks need at least one successful native rollout and a fixed
action-divergent pair whose two policy-generated endpoints are also different
and reachable from the exact same state. We
retain at least five tasks and 20 independent saved-state clusters, capped at
six states per task. If that frozen minimum is unavailable, the result is
reported as an underpowered feasibility study rather than broadened after
seeing transplantation effects.

## Reachable donor experiment

For a recipient branch A and donor branch B from state S:

- Native A: current(S), noise(A) -> future(A), action(A).
- Native B: current(S), noise(B) -> future(B), action(B).
- Exact self control: recompute native A with the identical request and seed.
- Mechanics-matched self clamp: current(S), noise(A), clamp the executed A future.
- Donor transplant: current(S), noise(A), clamp future(B).
- Controls: distance-matched Gaussian, a preselected non-donor natural future,
  and shuffled natural future content.

The primary action outcome is the signed projection of the transplanted action
change onto `action(B)-action(A)`, normalized so A is 0 and B is 1. Every pair
is run reciprocally. The corresponding action chunks are executed after
restoring S, and the same normalized projection is computed in registered
object and robot endpoint spaces. Correct-donor top-1 identification tests
specificity among several natural donors rather than merely movement away from
the recipient.

The exact self recomputation must reproduce native A bit-for-bit. The
mechanics-matched self clamp is reported separately because forcing a clean
future along a constructed rectified-flow path is not an identity operation,
even when its endpoint came from A. K/V record-and-replay identity is audited
against the same clamped-self trajectory before mediation is interpreted.

## Robot versus object content

Natural branches are screened for two pair classes with a common recipient:

- object-divergent, robot-pose-matched futures;
- robot-divergent, object-state-matched futures.

Thresholds are frozen in the config. If natural object-only pairs are rare, a
simulator-rendered 2x2 object/robot factorization may be run as secondary
evidence, with live current pixels preserved and hybrid targets explicitly
labeled counterfactual. Robot-pixel masks, object-pixel masks, and view-specific
clamps distinguish visible inverse dynamics from use of task consequences.

The population factorization is a separate confirmatory family rather than a
post hoc interpretation of the primary mediation effect. Before any factorial
intervention, at least ten primary-population units spanning at least five tasks
are selected round-robin by native target-object position separation, with
environment seed as the tie break. For each unit, four exact simulator-state
videos are first rendered from the same saved recipient/donor trajectories:
recipient object/recipient robot, recipient object/donor robot, donor
object/recipient robot, and donor object/donor robot. An excluded pilot found
that a full non-Fabric rerender introduces a large common camera-domain shift.
The confirmatory primary design therefore uses those paired rerenders only to
derive frozen RGB-difference masks (threshold 8, two-pixel dilation), then
copies donor pixels within the robot and/or object mask onto the native
recipient video. The native current frame is copied exactly. Full state
rerenders remain a secondary design. Only future frames change; the live
current observation, instruction, proprioception, and action noise are
identical. The robot and object main effects on action and physical execution
are the registered contrasts. Hybrid targets are labeled counterfactual
because their mixed sequence need not be a dynamically realizable rollout.
Decoded factor-edge direction preservation must clear the registered 90% gate
before behavioral contrasts are interpreted. Absolute four-cell raw-RGB
nearest-target accuracy remains a diagnostic because an excluded run showed
that common VAE reconstruction error can dominate the small factor edits.
Both metrics use the same public center-crop geometry as the tokenizer; no
behavioral outcome enters either audit.

## Timing and attention interface

All four denoising steps are tested separately and jointly. Layer localization
scans all 36 Cosmos 3 transformer layers on excluded calibration units. The
direct causal interface is action-query attention receiving future-vision
keys/values. Because full attention also permits an indirect
future-video-to-current-video-to-action path across layers, a total-mediation
condition excludes future-video keys/values from both current-video and action
queries at every layer while leaving future queries intact. The direct edge and
total barrier are reported separately. Equal-count current-vision, text,
action, and random-layer controls are required. The smallest best contiguous
layer window is frozen before the confirmatory units are evaluated; no Predict2
layer number is carried over.

Destructive K/V exclusion is treated only as localization. It changes softmax
normalization and may induce non-monotonic action changes even when the clamped
future remains exact. A content-mediation claim additionally requires
token-count-preserving replacement of future-video K/V with a matched self or
alternative-future cache at the frozen interface. The replacement must retain
native current-video, text, and action K/V, preserve key order and count, and
pass an exact self-cache no-op check.

## Statistical interpretation

Saved simulator state is the cluster unit. Paired reciprocal effects are
summarized with 10,000 cluster bootstrap resamples and familywise correction
across the registered primary family. The smallest effect of interest is 0.10
normalized donor steering. Results are reported per task, per state, per camera
view, and by native success/failure stratum.

A positive content-specific result requires donor steering above self and all
registered controls, decoded-target fidelity, and a corresponding physical
endpoint effect. A decoded-video change without action change supports
behavioral epiphenomenality. Robot-only steering with object-only nulls supports
prospective visual inverse dynamics, not task-state planning. Evidence from
either model does not by itself establish search, goal reasoning, or mediation
of task success.
