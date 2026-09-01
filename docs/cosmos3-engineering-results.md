# Cosmos 3 intervention validation

## Status

These are engineering and excluded-pilot results from the released
Cosmos3-Nano-Policy-DROID checkpoint. The first sections validate the port. The
RoboLab sections are positive same-state physical pilots, but they are excluded
from confirmatory inference because they contain one outcome-independent donor
pair each from one task. The initial-state pilot isolates robot motion; the
pre-grasp pilot includes coupled robot and object motion.

## Exact no-op and token census

The pinned 16B checkpoint was run through the official DROID request transform
with guidance 3, four UniPC steps, shift 5, a 33-frame 540x640 video, and 33
action rows (one current-state row plus 32 generated rows). The prepared flat
state contained:

- vision `[1,48,9,33,40]`, flat coordinates `[0,570240)`;
- action `[33,64]`, flat coordinates `[570240,572352)`.

The 33 RGB frames therefore become nine VAE latent frames. Post-guidance clean
estimates were exposed at sampler sigmas 0.999, 0.937, 0.833, and 0.624. An
identity layout-capture builder plus an identity post-guidance sampler wrapper
reproduced both final action and vision latents bit-for-bit: maximum absolute
error was exactly zero for both modalities.

This test also resolved an architecture-specific trap. The public velocity
postprocess hook runs on the conditional branch before text classifier-free
guidance; treating it as the final clean-video interface would not implement an
exact clamp at guidance 3. The port instead intervenes on the combined velocity
at the sampler boundary.

## Synthetic natural-future pilot

Two native generations from the same bundled banana observation and instruction
used diffusion seeds 0 (recipient) and 1 (donor). Their 32x8 external action
chunks differed by L2 8.080. Future latent frames 1 through 8 were clamped while
the conditioned current frame and every action coordinate were preserved.
Programmatic sentinels measured exactly zero direct action-input and
action-output mutation on every denoising call.

| target | action projection toward donor | action L2 from recipient | decoded L1 to donor |
| --- | ---: | ---: | ---: |
| self future | -0.195 | 2.050 | 0.0998 |
| donor future | **0.325** | 6.358 | **0.0000** |
| matched Gaussian | -0.059 | 1.884 | 0.2000 |

The donor-minus-self contrast is +0.520 and donor-minus-Gaussian is +0.384.
The Gaussian target matched donor norm with relative error `5.42e-7` and
recipient distance with relative error `2.33e-6`. The decoded transplanted
future was identical to the decoded donor under the measured L1 metric.

This is the first positive Cosmos 3 content-specific action signal: a coherent
natural future steered action toward its associated donor more than self or a
geometry-matched nonsemantic target. Its limitations are decisive: the current
observation is checkpoint-bundled, the two futures were not validated as
physically reachable from a saved simulator state, there is one pair, and no
action was executed. It supports proceeding to RoboLab; it does not yet support
a cross-task or physical causal claim.

Machine-readable reports are in `results/cosmos3_noop_v1/`.

## Released-policy RoboLab validation

The unmodified released checkpoint completed public RoboLab
`BananaInBowlTask`: object grasp occurred at step 88 and the banana was dropped
into the bowl successfully at step 137. The run stored all 137 actions, its
initial state, and per-step robot and object states. Replaying those actions in
the pinned RoboLab image reproduced success and all three events at the same
steps. Every saved robot, banana, bowl, and table state tensor matched the
original at every step with maximum and mean absolute error exactly zero.

RoboLab's recorded-config loader could not be used because it deserializes
conditional callables as strings. The replay instead regenerated the config
from the identical pinned source and container. The resulting state equality
shows that this workaround did not change the episode. The RoboLab container
digest is recorded in the replication config.

## Research-server audit

The research service uses explicit request seeds and IDs, fingerprints the
officially transformed current image, current action state, and instruction,
and rejects donor/recipient fingerprint mismatches. A fixed public-observation
audit found:

- two native calls with the same seed had bit-identical actions, final future
  latent, and all four clean-video and clean-action estimate hashes;
- all future clamps had exactly zero direct action-input and action-output
  coordinate mutation;
- the post-guidance sigmas were 0.999, 0.937, 0.833, and 0.624;
- tokenizer-encoded executed videos required the official uint8-to-`[-1,1]`
  normalization and removal of reflection-padded latent sites;
- after that correction, predicted, self, Gaussian, and encoded-video targets
  had final maximum latent errors between 0.0178 and 0.0217.

RoboLab state restoration was bit-exact, but independently re-rendering the
same state differed by up to 8, 11, and 16 uint8 levels in three checks. The
pilot therefore caches and reuses one current observation for every model
request and donor frame zero, while exact state replay determines physical
execution. This enforces the intended intervention rather than treating
headless renderer noise as part of the treatment.

## Same-state reachable-donor pilot

Four native diffusion branches (211, 223, 227, 229) were generated from the
same recorded state. The donor pair was selected solely by maximum native
endpoint separation before any intervention was evaluated: seed 223 was the
recipient and seed 229 the donor. Each complete 32-action chunk was executed
from the exact state, its 33-frame RGB trajectory was encoded by Cosmos 3's
released Wan2.2 VAE, and only future latent frames were clamped on the
recipient noise path.

| target | action donor projection | physical robot endpoint projection |
| --- | ---: | ---: |
| self | -0.0333 | -0.0447 |
| predicted donor future | **0.9832** | **0.9784** |
| executed reachable donor future | **0.9988** | **0.9978** |
| norm/distance-matched Gaussian | 0.0319 | 0.0397 |

The native recipient/donor action L2 was 4.586. The executed-donor target's
maximum final latent error was 0.0216, and direct action-coordinate mutation
remained exactly zero. Thus a coherent reachable future moved both generated
action and physical execution almost exactly to the associated donor, whereas
a geometry-matched nonsemantic target did not.

This is a strong positive content-specific sufficiency result for Cosmos 3,
but its interpretation is deliberately narrow. The object-state projection is
undefined because no object moved during this initial chunk. The result
therefore establishes prospective visible robot-motion/inverse-dynamics use,
not object-consequence reasoning, task planning, or mediation of success. The
pilot unit and its selected seeds are excluded from confirmatory inference.
The compact machine-readable report is in
`results/cosmos3_robolab_pilot/summary.json`; full videos, simulator states, and
per-denoising hashes remain in the external run directory.

## Pre-grasp reachable-donor pilot

A second excluded pilot branched at recorded step 64, before contact with the
banana, and evaluated the next 32 actions through grasp formation. Every
condition recreated the RoboLab environment, restored the public recorded
initial state, and replayed the 64-action prefix with zero HDF5 state error.
Three independent reconstructions produced the same bitwise branch-state
digest. Repeating one identical native continuation in another fresh
reconstruction also produced the same bitwise endpoint digest. This stricter
protocol was necessary because repeated reset or snapshot restoration within
one physics environment exposed hidden continuation state.

Four native branches (307, 311, 313, 317) were collected. Maximum native
endpoint separation selected seed 311 as recipient and 313 as donor before any
intervention outcome was inspected.

| target | action donor projection | all-state endpoint | robot endpoint | object endpoint |
| --- | ---: | ---: | ---: | ---: |
| self | -0.0229 | -0.4964 | -0.5184 | -0.0426 |
| predicted donor future | **0.9751** | **0.8743** | **0.8624** | **1.1207** |
| executed reachable donor future | **0.8190** | **0.8737** | **0.8319** | **1.7379** |
| norm/distance-matched Gaussian | -0.0128 | -0.5946 | -0.6046 | -0.3871 |

The predicted and reachable executed donor targets therefore steered both the
action and the physically executed grasp/object outcome toward the selected
donor, while self and geometry-matched Gaussian targets did not. Direct action
coordinate mutation remained exactly zero, and final target latent errors were
0.0194 and 0.0210 for predicted and executed donors respectively.

This is evidence beyond the initial robot-only chunk, but it is not yet an
object-only content result: the banana moves as part of a coupled grasp and the
robot endpoint moves in the same donor direction. Robot-pose-matched,
object-divergent donors and explicit robot-pixel interventions remain necessary
to distinguish task-state consequence use from visible inverse dynamics. The
machine-readable report is in
`results/cosmos3_robolab_step64_v3/summary.json`.

## Denoising-time localization

The same excluded step-64 state and frozen recipient/donor seeds were rerun in
a new server-process block. In addition to the all-step clamp, the future was
clamped during exactly one of the four UniPC denoiser calls. Every single-step
intervention strongly steered the action toward the donor:

| active clamp calls | action donor projection | robot endpoint projection |
| --- | ---: | ---: |
| step 0 only | **1.0616** | **0.5209** |
| step 1 only | **0.9785** | **0.5420** |
| step 2 only | **1.0139** | **0.5293** |
| step 3 only | **0.9917** | **0.4844** |
| all steps | **0.9990** | **0.5054** |
| self | 0.0107 | -0.3012 |
| matched Gaussian | 0.0018 | -0.2706 |

Thus the Cosmos 3 effect is not confined to one privileged denoising call:
each call is individually sufficient for strong directional action steering on
this state. The native donor object-position displacement was only 9.85 mm,
whereas interventions moved the object by 31--41 mm. Normalized object-position
projections of 3--4 are therefore overshoots along a small denominator and are
not interpreted as isolated object-semantic use. The compact report is in
`results/cosmos3_robolab_timing_v1/summary.json`.

## Server-restart reproducibility audit

Within one loaded server process, identical requests and seeds recompute
bit-for-bit. Across fresh server processes, however, the released stack is not
bitwise deterministic. Four controlled restart pairs used the same transformed
observation hash, full initial sampler-state hash, and vision path-noise hash.
Action differences ranged from 0.493 to 0.722 in L2 and 0.100 to 0.133 maximum
absolute error. Clean-action and clean-video hashes diverged from the first
forward pass.

The final pair also had an identical deterministic head/tail fingerprint over
every loaded model parameter
(`21b79382b84b4bdebb943a2659c0272c99267ef433d83818f9a44b742c1170cc`).
Forcing the public cuDNN attention backend did not remove the difference. These
checks localize the issue after weight loading and sampler-noise construction,
to process-dependent numerical behavior in the forward stack. They do not
invalidate paired interventions performed within one process, but a server
process must be treated as a blocking factor. Confirmatory estimates will use
at least three fresh-process blocks rather than claiming cross-process bitwise
reproducibility. The compact audit is in
`results/cosmos3_restart_probe/summary.json`.
