# Manuscript framing draft

## Working title

**Mechanistic Interventions Reveal How Imagined Futures Steer Robot Actions in World Action Models**

## The distinction the paper should own

RIFT establishes that world action model action experts causally use the
future-read interface. Masking, corrupting, or reassigning future-cache values
changes executed trajectories and usually reduces task success. Its spatial and
temporal reassignments further show that future values matter at their assigned
positions. It would therefore be inaccurate to say that RIFT measures only
performance, or that it provides no evidence that futures influence motion.

The unresolved question is more specific:

> When a world action model reads an imagined future, what actionable content
> does it use, and how does that content determine the action it takes?

RIFT's interventions predominantly remove or disrupt information. They measure
whether execution changes and whether the task still succeeds, but they do not
associate the edited cache with a coherent alternative behavior and test
whether the robot moves toward that alternative. A large trajectory deviation
under corruption establishes causal dependence, not a directional mapping from
future content to action.

Our reachable-donor intervention supplies that missing target. From the same
physical state, two native branches provide a recipient and a physically
reachable donor, including their future observations, actions, and executed
endpoints. We transplant the donor's future representation into the recipient
computation while holding the current observation, instruction, recipient
action noise, and denoising schedule fixed. We then measure a signed quantity:
whether the resulting action and executed endpoint move specifically along the
recipient-to-donor direction. Multiple donors test whether the intervention
selects the correct alternative rather than merely perturbing behavior.

The conceptual progression is:

1. **RIFT: causal dependence.** Does removing or corrupting the future read
   change execution or success?
2. **Our transplantation experiments: directional semantic sufficiency.** Does
   inserting a coherent alternative future move action and execution toward the
   behavior associated with that particular future?
3. **Our content interventions: functional content.** Is the steering carried
   by visual robot-motion content, task-object consequences, or future
   proprioception?
4. **Our K/V patching: pathway mediation.** Does replacing future-token K/V
   content remove donor-induced steering while preserving token count,
   positions, and the rest of the input structure?

A compact statement of the gap is:

> **Sensitivity is not semantics.** A corrupted future can make a policy move
> differently or fail. A coherent donor future tests whether future content
> controls *which way* it moves.

## Central claim

At identifiable decision states in two generations of Cosmos policies,
imagined visual futures are not merely correlated auxiliary predictions. Their
coherent content can directionally control generated actions and executed
robot motion, and future-token K/V content mediates the Cosmos 3 population
effect. The semantic carrier is not universal: Cosmos Policy interventions
favor prospective visible robot motion, whereas neither isolated robot pixels
nor isolated object pixels reproduce a meaningful fraction of Cosmos 3's
whole-future effect. Future use is state dependent and should not be described
as general planning over task consequences.

## What is genuinely new relative to RIFT

| Question | RIFT | This paper |
| --- | --- | --- |
| Does the action expert use the future interface? | Yes: masking and value corruption alter trajectories and success. | Replicated with complementary controls. |
| Do positions and organization matter? | Yes: spatial and temporal reassignments are disruptive. | Not the principal novelty. |
| Is a final clean cache sufficient during action denoising? | Yes, approximately, for supported architectures. | Not the principal novelty. |
| Can coherent alternative future content select a corresponding action? | Not tested directionally. | Yes: natural donor futures steer actions toward donor actions. |
| Does the effect survive physical execution in the predicted direction? | Execution divergence is measured relative to the original trajectory. | Yes: signed physical endpoints move toward donor endpoints. |
| Can the policy distinguish among several coherent alternatives? | Not tested. | Yes in the Cosmos 3 multi-donor experiments. |
| What semantic content is used? | Not isolated beyond positional cache organization. | Cosmos Policy favors visible robot motion; Cosmos 3 requires a compositional or distributed account because both isolated pixel factors are negligible. |
| Does future K/V content mediate semantic steering without deleting tokens? | Not tested for a donor-directed effect. | Yes: token-count-preserving self-K/V patching suppresses donor steering across 22 Cosmos 3 clusters and six tasks. |

## Recommended introduction arc

### Paragraph 1: Promise and ambiguity

World action models jointly predict how a scene may evolve and how a robot
should act. This architecture invites a planning-like interpretation: the model
imagines a future and selects actions using the anticipated consequences. But
joint generation alone does not establish this interpretation. A future may be
behaviorally irrelevant, may serve as an undifferentiated latent workspace, or
may encode little more than the robot motion already implied by the action.

### Paragraph 2: Begin from RIFT, not before it

Recent intervention evidence substantially narrows this ambiguity. RIFT edits
the future-position key/value cache of several world action models and shows
that masking or reassigning future values changes physical trajectories and
reduces success. World action model policies therefore do read meaningful,
position-bound future representations. Yet destructive sensitivity leaves a
different question open: what does this representation tell the policy to do?
If a coherent alternative future is substituted, will the policy take the
corresponding alternative action, and which visual content produces that
change?

### Paragraph 3: Methodological gap

Answering this question requires an intervention with a semantic direction.
Noise, masking, and token shuffling can reveal necessity, but they do not define
a behavior the intervention should induce. We introduce reachable-donor future
transplantation. Two native continuations from the same saved state define
alternative, model-compatible futures and their associated action and physical
endpoint directions. We transplant one future into the other's action
computation and score whether action and execution move specifically toward the
donor. Geometry-matched Gaussian targets, natural controls, reciprocal
directions, multiple donors, exact no-op tests, and physical replay distinguish
semantic steering from generic disruption.

### Paragraph 4: Findings

Across all ten LIBERO-10 tasks at the first decision state, donor futures steer
Cosmos Policy actions and executed endpoints toward the donor branch, with mean
normalized effects of 0.499 and 0.552. The effect is largest for wrist-camera
futures, smaller for the primary camera, and absent for future proprioception.
A separate 20-state factorial study changes decoded robot and object content
with high target specificity but finds essentially no isolated object-content
effect. Natural robot-only pairs show strong steering, whereas object-only
pairs are weak and inconsistent. The effect also attenuates sharply at later
registered states.

### Paragraph 5: Cross-generation mechanism and conclusion

The released Cosmos 3 policy provides cross-generation evidence. Reachable
predicted and executed futures directionally steer actions and physical robot
endpoints across 22 saved-state clusters and six RoboLab tasks.
Token-count-preserving activation patching then replaces donor-run future-token
keys and values with their self-future counterparts and removes most
donor-directed action and physical steering in every cluster. A separate
10-cluster factor study finds that neither isolated robot pixels nor isolated
object pixels produce a meaningful fraction of the coherent-future effect.
Together, the models support directional future use while cautioning against a
single universal semantic carrier or an interpretation as task-consequence
planning.

## Abstract draft

World action models jointly generate future observations and robot actions,
but the functional role of their imagined futures remains unclear. Recent
destructive interventions show that world action model policies depend on
future-position representations; they do not establish whether coherent future
content directionally selects corresponding behavior or what content the
policy uses. We introduce reachable-donor future transplantation, which swaps
natural future representations between alternative continuations of the same
physical state while holding the current observation, instruction, action
noise, and denoising schedule fixed. We then measure whether generated actions
and executed endpoints move toward the donor continuation. In Cosmos Policy,
donor futures steer both action and physical execution at early states across
all ten LIBERO-10 tasks. Camera-specific and factorial interventions indicate
that Cosmos Policy's effective content is predominantly prospective visible
robot motion: wrist and primary-camera futures steer action, future
proprioception does not, and isolated object-state effects are negligible in
the tested states. The effect attenuates sharply later in episodes. In Cosmos
3, natural transplants reproduce directional action and physical steering
across 22 clusters and six RoboLab tasks. Token-count-preserving activation
patching of future-token keys and values largely suppresses donor steering. Yet
a masked factor study finds practically negligible effects from isolated robot
and object pixels, implying a more distributed or contextual carrier. These
results go beyond future sensitivity to show how coherent imagined content
controls action, while providing no evidence that the studied policies
generally plan over task-object consequences.

## Claim language to use consistently

Prefer:

- "RIFT establishes causal use, dependence, or sensitivity of the future-read
  interface."
- "We establish directional, content-specific, or semantic steering toward a
  coherent donor alternative."
- "Future-token K/V content mediates donor-induced action steering."
- "Cosmos Policy's content evidence is most consistent with prospective visible
  robot motion or an inverse-dynamics-like role."
- "Cosmos 3's coherent whole-future effect is not reproduced by either isolated
  robot-motion or isolated object-state pixels."
- "The effect is state dependent."

Avoid:

- "RIFT shows only a performance effect."
- "RIFT does not show steering" without qualifying *directional semantic*
  steering.
- "World action models plan with imagined consequences."
- "The model imagines success" or "chooses the successful future."
- "Object consequences are unused" as a universal claim.
- "We identify the complete circuit." The present mechanism is localized at
  the representation and pathway level.
