# Manuscript-grounded figure notes

These captions use the estimands and qualifications in `paper/main.tex` after commit
`98e06c0`.

Plot elements remain vector, and the PDFs embed CID TrueType fonts rather than
Type 3 fonts.

## Figure 2 — Future conditions on one scale

Future interventions and controls on one common scale. The horizontal axis is the
normalized donor projection $\pi_x$, with self fixed at 0 and the paired donor at
1 defining the native recipient-to-donor direction. Rows are nested by
model and future source; controls progress from natural to shuffled to matched
Gaussian where available, followed by the paired transplant. Cosmos Policy
control projections are reconstructed within task from the registered pairwise
contrasts before bootstrapping. Light shading identifies the paired-transplant
rows, and thin gray segments connect paired action and endpoint means. Large marks
and bars are means and 95% independent-unit bootstrap intervals across restored
simulator states for Cosmos 3 and task-level first-query units for Cosmos Policy.

## Figure 3 — Cosmos 3 pixel factors

Cosmos 3 disjoint pixel-factor study across ten restored states. (a,b) The four
cells of the 2x2 hybrid design, expressed by the source of the robot and object
pixels. (c) Whole-future donor-minus-self steering compared with the robot- and
object-pixel main effects derived from those four cells. The shaded region is the
prespecified [-0.10, 0.10] equivalence region. The design tests whether either
masked pixel factor recreates whole-future steering; it does not test semantic
irrelevance or joint necessity.

## Figure 4 — Cosmos 3 future-token pathway

Cosmos 3 future-token K/V replacement across 22 restored states. (a-c) Each line
connects unpatched donor-minus-self steering to steering after donor-future keys
and values are replaced with self-future keys and values. (d) The corresponding
replacement loss, defined as unpatched minus replaced steering. Black diamonds
and bars are state means and 95% state-bootstrap intervals; colors denote tasks.

## Figure 5 — Cosmos Policy scope

Cosmos Policy donor-directed effects. (a) Donor-minus-self action and endpoint
steering in the registered first-query, after-three-chunk, and after-six-chunk
state cohorts. These cohorts contain different physical states and do not estimate
causal temporal decay. (b) First-query action steering when the whole future or a
single future component is transplanted. (c) First-query donor advantage over
self, Gaussian, natural, and shuffled comparison futures. Individual marks denote
tasks; large marks and bars are means and 95% task-bootstrap intervals.
