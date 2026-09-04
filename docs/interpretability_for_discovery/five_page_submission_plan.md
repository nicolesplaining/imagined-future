# Interpretability for Discovery: five-page submission plan

## Hard constraints from the CFP

- Maximum: five pages of main text. References and appendices are excluded.
- The main text must be self-contained; reviewers are not required to read the appendix.
- Double-blind, private OpenReview submission using the NeurIPS 2026 workshop template.
- A short responsible-use statement is mandatory; omission is grounds for desk rejection.
- The venue explicitly welcomes methods for knowledge discovery across unfamiliar architectures and modalities, empirical interpretability-driven discoveries, and careful negative results or failed interpretations.

## Correct baseline

The current compiled paper has nine counted main-text pages. References begin on page 10 and the appendix follows the references. The target is therefore to move or compress four pages of main-text material, not twelve pages.

## Recommended workshop framing

Frame the paper as an interpretability method that converts a visually plausible model output into falsifiable knowledge about model computation. The non-obvious discoveries are:

1. A particular coherent predicted future directionally controls the associated action; activity at future-token positions is not merely necessary in the generic sense.
2. In Cosmos 3, predicted-future K/V carries 83--88% of the donor-directed behavioral effect.
3. The content profile differs across architectures: Cosmos Policy is dominated by camera-visible robot motion, whereas isolated robot or object pixels are insufficient to recreate Cosmos 3's full-future effect.
4. The negative results constrain interpretation: neither model is shown to compare outcomes or plan, and apparently interpretable isolated pixel factors can fail to explain a distributed causal representation.

This fits the CFP most directly under “methods and models for knowledge discovery” and “failure cases and negative results.” Do not claim discovery of new facts about the physical world. Claim a reusable causal-interpretability framework that discovers testable facts about unfamiliar multimodal action-model mechanisms.

## Recommended title

Primary choice:

> Interpreting Imagined Futures: Discovering How World Action Models Steer Robot Behavior

More conservative alternative:

> From Prediction to Mechanistic Discovery: How World Action Models Use Their Futures

## Five-page main-text architecture

### Page 1 — Question, gap, and discoveries

- Title and abstract.
- Compress the introduction to approximately four paragraphs.
- Retain the distinction from RIFT: destructive dependence tests cannot predict the direction of behavioral change.
- State the two-stage contribution and the three principal discoveries.
- Fold essential related work into one paragraph; move the standalone Related Work section to the appendix.

### Page 2 — Unified intervention method

- Use one two-panel methodology figure: Stage 1 future transplant and Stage 2 nested K/V pathway patch.
- Define recipient, donor, held-fixed variables, donor projection, and replacement loss in compact prose plus at most one displayed equation.
- Give models, state counts, tasks, and controls in one paragraph.
- Keep enough detail to establish that donor actions are never shown to the model and action coordinates are never overwritten.
- Move enumerated procedures, exact sampler equations, tensor layouts, schedules, cache alignment, selection rules, and full statistical details to the appendix.

### Page 3 — Stage 1 discoveries

- Lead with the population directionality result for Cosmos 3 and the cross-model Cosmos Policy result.
- Retain action and executed-endpoint evidence and the most important self, natural, and Gaussian comparisons.
- Summarize content interventions in one paragraph: wrist-camera/visible-robot-motion concentration for Cosmos Policy; isolated robot/object pixel insufficiency for Cosmos 3.
- Replace the current large directionality table and two content tables with one compact discovery-results panel or small headline table. Move complete tables and confidence intervals to the appendix.

### Page 4 — Stage 2 pathway discovery

- Explain K/V in two sentences, not a full tutorial.
- Show the nested intervention and the Cosmos 3 83--88% replacement-loss result.
- Mention Cosmos Policy's active late edge in one sentence as weaker convergent evidence.
- Keep one compact pathway plot/table. Move the five-step procedure, gating curve, control details, layer calibration, and full results table to the appendix.

### Page 5 — Interpretation, limits, and responsible use

- Center the interpretation on what the interventions let us discover about model representations across unfamiliar multimodal architectures.
- Retain the planning boundary: causal conditioning is not candidate comparison or consequence-sensitive planning.
- Retain concise limitations: simulation, two checkpoints, early-state concentration, off-distribution hybrids, incomplete circuit.
- Include a clearly labeled short responsible-use statement in the main text.
- End with one short conclusion paragraph. Move extended future-work discussion to the appendix or remove it.

## Exact move-to-appendix list

Move without deleting scientific content:

- Most of the standalone Related Work section.
- Stage 1's four-step numbered procedure.
- The second projection equation and detailed self-reference explanation.
- Exact future-insertion mechanics.
- Detailed population selection, restoration, controls, and statistics.
- Full directionality table and donor-minus-control breakdowns.
- Full Cosmos Policy content table and full Cosmos 3 factor table.
- The separate Cosmos Policy scope subsection and later-state details.
- Stage 2's five-step numbered procedure.
- Detailed Cosmos Policy key-gating method and full gating curve.
- Cosmos 3 cache alignment, single-layer calibration, and implementation controls.
- Full pathway table if the values appear in a compact main figure.
- Extended limitations and the final future-work paragraph.

Do not move these essentials out of the main text:

- Recipient/donor definition and same-state design.
- What is held fixed and the fact that the donor action is never an input.
- Meaning of donor projection and K/V replacement loss.
- Independent-state counts and task coverage.
- Headline uncertainty or state-consistency evidence.
- Planning caveat and responsible-use statement.

## Figure strategy

Use no more than two main-text figures:

1. **Two-stage method:** actual future-video frames, future insertion, recomputed action, then the nested K/V patch. This replaces the two current methodology figures.
2. **Three-part discovery summary:** directionality, content profile/negative factor result, and K/V replacement loss. Complete per-condition figures remain in the appendix.

Avoid full-width tables in the five-page version. A reviewer should understand the causal chain from the two figures and headline numbers alone.

## Emergency execution order

1. Duplicate the latest source into the separate Interpretability for Discovery variant; never edit Nicole's main manuscript.
2. Change the workshop title and paper title in the variant.
3. Move the sections listed above below `\\appendix` with labels preserved.
4. Merge the two methodology figures or temporarily use one compact Stage-1 figure and explain Stage 2 in prose if the merged figure is not ready.
5. Replace three large results tables with compact prose and one summary visual.
6. Compile and measure where References begin; the goal is References on page 6.
7. Tighten paragraphs only after structural moves. Do not spend deadline time polishing text that will move to the appendix.
8. Run the desk-rejection checklist: five main pages, anonymized, correct workshop name/template, responsible-use statement, references/appendix in the same PDF, no identifying repository links.

## Go/no-go principle

The fastest defensible submission is not a new scientific paper. It is a self-contained five-page interpretability narrative over the existing evidence, with technical completeness preserved in the uncounted appendix. If space remains tight, preserve the Stage-1 directional test and Cosmos 3 K/V pathway result in the main text; move secondary scope analyses before weakening those two links.
