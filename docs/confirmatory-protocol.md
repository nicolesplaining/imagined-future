# Confirmatory multi-state protocol

Status: frozen before any intervention outcome is generated for the units below.

## Scope and claim

The study tests whether future-state content generated during Cosmos Policy's direct, parallel policy inference has a state-dependent causal effect on the action. It does not assume that future use is universal, and it does not estimate a classical natural indirect effect. Cosmos Policy jointly denoises action and future frames with bidirectional self-attention, so the primary estimands are controlled semantic interventions and fixed-site future-to-action edge interventions.

The exploratory task-4/state-2 positive, task-8/state-3 null, and every state previously screened in tasks 4 and 8 are excluded from confirmatory estimation. They are used only to select the fixed attention sites and design the controls.

## Experimental units

The independent unit is an exactly restored LIBERO simulator state. The fixed grid contains 30 units: all ten LIBERO-10 tasks crossed with three previously unseen initial-state/timing strata.

| Stratum | Initial-state index | Deterministic prefix |
|---|---:|---:|
| Early | 10 | 0 chunks |
| Middle | 27 | 3 chunks |
| Late | 43 | 6 chunks |

Prefixes use model seed 307. If a prefix completes the task before the registered branch point, that unit is excluded as structurally unavailable; it is not replaced after intervention outcomes are viewed. Every unit uses branch seeds 311, 313, 317, 319, 331, 337, 347, and 349. Tokenizer determinism, exact replay, the upstream commit, checkpoint revision, container digest, and GPU UUID are mandatory run-validity checks.

## Pair selection

Donor pairs are selected using only natural branch actions and executed 16-step endpoints. No patched action, decoded patched future, attention ablation, or other intervention outcome may enter selection.

For each state, pairwise normalized-action and physical-endpoint distances are converted to within-state ranks. The physical endpoint concatenates LIBERO goal-object positions and articulated joints with the official nine-dimensional endpoint proprioception. The primary pair maximizes the mean action/endpoint rank subject to normalized action L2 at least 0.01 and nonzero physical-endpoint L2 above `1e-8`. Both directions are tested, so each branch serves once as recipient and once as donor. Ties are broken lexicographically by branch index.

Goal-relevant endpoint features are constructed in parsed goal order from LIBERO's own predicate arguments: object positions and articulated-joint coordinates are included; quaternion components are included only for orientation predicates. Endpoint proprioception is always appended because early action chunks can produce distinct task-directed robot motion before contact moves a goal object.

Continuation outcomes are secondary labels evaluated in two registered stages. Seed 353 screens the primary pair and its preselected distance-matched natural controls at every unit. Seeds 359 and 367 are run only when the primary pair has different outcomes under seed 353. A branch is called a robust success or failure only when all three tested continuations agree; units not entering stage two receive no robust label. Success-specific contrasts are reported only for robust failure-to-success pairs and are not substituted for the primary endpoint-divergence analysis.

## Semantic interventions and controls

All paired runs use common random numbers and preserve current observation, instruction, action-frame noise, value-frame noise, schedule, and solver. Future-noise seeds are 401, 409, and 419.

The primary semantic contrast is an all-future clamp to the donor endpoint minus an all-future clamp to the recipient endpoint. Secondary modality clamps test wrist image, primary image, and proprioception separately.

Required controls are:

- untouched generation and a recipient/self clamp;
- a same-outcome donor matched on action and endpoint distance when a robust label is available;
- a within-task shuffled donor fixed without intervention outcomes;
- a Gaussian latent target exactly matched to the semantic donor in clean-latent norm and distance from the recipient target;
- direct action transplantation as an implementation and simulator-response positive control.

Decoded-future distances and latent intervention distances are manipulation checks. Controls with failed exactness checks are exclusions, not zeros.

## Necessity interventions

Block 27 is the fixed primary attention site; block 0 is secondary. Future-key removal from action queries is evaluated with gates 0.25, 0.5, 0.75, and 1.0. The gate interpolates between the public all-key attention output and the output recomputed without selected keys. This preserves the native attention implementation and permits a dose-response check.

Each run includes a bitwise all-key recomputation control, an equal-count current-key removal control, and an equal-count random-key-frame control chosen before outcomes. Layer selection is not repeated on the confirmatory units.

## Outcomes

The primary outcome is donor steering of the executed physical endpoint—goal-relevant object coordinates plus official proprioception—after the 16-step action chunk. Secondary outcomes separate object-only steering, normalized action donor steering, endpoint proprioception steering, primary-image donor preference, individual LIBERO goal predicates, and full-episode success under shared continuation seeds.

Flattened MuJoCo state distance and pixel distance remain broad diagnostics. They are not substitutes for task-relevant physical outcomes.

## Estimation

Saved state is the clustering unit. The main result is the mean within-state semantic donor-minus-recipient contrast with a 10,000-resample state-clustered bootstrap confidence interval. Task-stratified effects and the complete state-level distribution are reported before aggregation. Donor directions and noise draws are repeated measurements, not independent samples.

The all-future semantic clamp is the sole primary intervention family. Wrist, primary-camera, proprioception, block-27 necessity, block-0 necessity, and success-specific analyses are secondary and use Holm correction within their stated families. Effect sizes and confidence intervals are reported regardless of thresholded significance.

## Interpretation

- General semantic use requires a positive primary interval and superiority to matched nonsemantic controls.
- State-dependent use is supported if effects are reproducible but heterogeneous, with pre-intervention state features explaining held-out variation.
- Coupling without semantic use is concluded if interventions move actions but semantic targets do not outperform Gaussian and shuffled controls.
- A runtime-null region is supported when the manipulation check succeeds and the confidence interval excludes the frozen smallest effect of interest.

The frozen smallest effect of interest is `0.10` donor-steering units: ten percent of the natural recipient-to-donor physical displacement. Bitwise no-op controls have zero variability and therefore cannot provide a useful nonzero threshold. A positive-use claim requires uncertainty above zero and is interpreted as practically meaningful only when it also excludes `0.10`.

## Pre-intervention amendment: physical endpoint

Added 2026-08-31 after the first natural-only held-out branch collection and before any held-out intervention. Goal-object coordinates were exactly tied across all branches at an early state because the first 16 actions moved the robot toward the objects without contacting them. The registered primary endpoint was therefore clarified to concatenate goal-relevant coordinates with official endpoint proprioception for every unit, rather than using proprioception only as a missing-feature fallback. The object-only endpoint remains secondary. No patched action, decoded patched future, or attention-ablation outcome had been generated when this amendment was made.
