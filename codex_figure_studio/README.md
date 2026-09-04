# Independent figure studio

This directory contains a from-scratch figure set made directly from the
repository's state-level result artifacts. It does not import, edit, or write
to the existing manuscript figure pipeline.

Generate everything with:

```bash
MPLCONFIGDIR=/tmp/codex_figure_matplotlib \
  /tmp/imagined_future_plot_env/bin/python codex_figure_studio/generate.py
```

The generator needs NumPy and Matplotlib only. Outputs are written to
`rendered/` as vector PDF and 360-dpi PNG. Machine-readable plotted estimates
and condition definitions are written to `tables/`.

## Main figures

1. `fig1_transplantation_and_steering`: minimal same-state transplant diagram
   plus raw Cosmos 3 source-condition projections. The natural control is
   labeled as predicted wherever it appears beside executed-future conditions.
2. `fig2_kv_pathway`: one dumbbell row per restored state, grouped by task,
   comparing donor-future steering before and after self-future K/V replacement.
3. `fig3_pixel_factorization`: the four robot × object pixel cells, coherent
   whole-donor reference, and registered factorial effects.
4. `fig4_cosmos_policy_scope`: separate task-level action and endpoint
   trajectories over registered state indices, followed by first-query modality
   effects.

## Supplementary figures

- `supp1_metric_sanity`: native separation versus transplanted raw displacement.
- `supp2_policy_controls`: Cosmos Policy control comparisons and the first-query
  action–endpoint relationship.

## Statistical unit and visual conventions

- Cosmos 3 population: 22 restored states across six tasks.
- Cosmos 3 factor study: 10 restored states across six tasks.
- Cosmos Policy: reciprocal directions and future-noise repetitions are first
  averaged within each restored state; the plotted first-query/timing points
  are the resulting ten task-state means.
- Black diamonds and bars are state means with 10,000-resample percentile
  bootstrap intervals.
- Solid zero and dashed one lines denote recipient and donor coordinates.
- The dotted `0.10` line or shaded `[-0.10, 0.10]` region marks the registered
  smallest effect of interest where applicable.

The timing panels compare different registered states at different indices;
their trajectories show a state/timing association, not causal decay within a
fixed state.
