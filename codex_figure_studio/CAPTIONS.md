# Caption drafts

These captions describe the generated figures without adding interpretive
annotations inside the panels.

## Figure 1 — Reachable-future transplantation and steering

From one restored state, recipient and donor branches provide alternative
predicted futures, executed futures, actions, and physical endpoints. The
recipient computation is repeated with a fixed current observation,
instruction, recipient action-noise trajectory, and denoising schedule while
the future target varies. Projections are normalized so the native recipient is
0 and native donor is 1. Panels b–d show all 22 restored states across six
RoboLab tasks; faint lines join conditions within state. Black diamonds and
bars denote state means and 95% percentile-bootstrap intervals. The natural
control is a prospectively fixed third predicted future from the same state,
including where it is displayed beside the executed-donor conditions. The
dotted line is 0.10 projection units.

## Figure 2 — Self-future K/V replacement

Donor-future projection before and after replacing donor-run future-video keys
and values with recipient self-future keys and values. Each row is one of 22
restored states, grouped by task. Filled circles are donor-future runs; open
squares are the corresponding self-future K/V replacements. The displayed
mean losses and intervals are the registered state-bootstrap estimates. Zero
and one denote the native recipient and donor coordinates.

## Figure 3 — Robot × object pixel factorization

Cosmos 3 action and physical-endpoint projections under the four disjoint
robot × object pixel conditions. `O0/R0` denotes recipient object/robot pixels
and `O1/R1` denotes donor object/robot pixels. The coherent executed-donor future
is shown as a reference condition. Points are the ten restored factor-study
states across six tasks; faint lines join conditions within state. The lower
panels show robot and object main effects and their interaction. Black diamonds
and bars are state means and 95% percentile-bootstrap intervals. Gray regions
mark the registered `[-0.10, 0.10]` equivalence range.

## Figure 4 — Cosmos Policy state/timing and modality scope

Donor-minus-recipient action and executed-endpoint projection at the first
query and after three and six open-loop action chunks, followed by first-query
effects for all future components, wrist video, primary video, and future
proprioception. Reciprocal directions and future-noise repetitions are averaged
within each restored state before plotting. Faint trajectories join the ten
task-state means across registered state indices or modalities; diamonds and
bars are means and 95% state-bootstrap intervals. The later indices are
different registered states, so panels a–b show a state/timing association, not
causal temporal decay within a fixed state. The dashed line is the registered
0.10 threshold.

## Supplement 1 — Metric sanity checks

Raw transplanted displacement versus the corresponding native recipient–donor
separation. Dashed identity lines indicate equal displacement. Each point is
one of the 22 Cosmos 3 restored states; colors denote tasks.

## Supplement 2 — Cosmos Policy controls

First-query donor-minus-control action and executed-endpoint projections for
self, norm-and-distance-matched Gaussian, natural-future, and within-task
shuffled controls, followed by the state-level relation between action and
endpoint steering. Points are restored-state means; diamonds and bars are
means and 95% state-bootstrap intervals.
