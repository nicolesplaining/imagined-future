# FastWAM Optional-IDM powered analysis v2: balanced source retrieval

Frozen before any powered action or causal output was opened. This analysis
supersedes only the retrieval-chance interpretation in the powered analyzer; it
does not change the frozen manifest, states, seeds, run matrix, interventions,
controls, or generated outputs.

## Correction

For a fixed recipient, each off-diagonal donor condition has three admissible
donor labels. Its donor-only retrieval rate therefore must not be tested against
0.25. Donor-only retrieval remains a secondary descriptive metric. Its neutral
reference is conservatively 1/3, and its confirmatory evidence comes from the
predeclared paired contrasts against wrong-latent and shuffled-cache controls.

## Primary balanced source-retrieval estimand

For each state and modality, form the complete 4 by 4 future-source grid:

- latent: four `self_latent` diagonal cells plus twelve `donor_latent`
  off-diagonal cells;
- cache: four `self_cache` diagonal cells plus twelve `donor_cache`
  off-diagonal cells.

In each cell, identify the nearest of the four native branch actions and score
whether its branch label equals the transplanted future source. Every source
label appears in exactly four cells, so exact label-permutation chance is 0.25.
Average the sixteen cell scores within state. Compute the population interval
with the already frozen suite-to-task-to-state hierarchical bootstrap (10,000
draws, seed 20260903).

## Powered gate v2

Replace the two invalid donor-only-versus-0.25 criteria with:

- latent 4 by 4 correct-source retrieval hierarchical 95% lower bound > 0.25;
- cache 4 by 4 correct-source retrieval hierarchical 95% lower bound > 0.25.

Retain all other frozen criteria: complete registered matrix, finite arrays,
exact self replay, distinct native future latents, exact first-frame invariance,
and positive hierarchical lower bounds for both retrieval and distance-reduction
contrasts (`donor_latent - wrong_latent` and
`donor_cache - shuffled_cache`). No gate or metric may be changed after powered
outputs are opened.
