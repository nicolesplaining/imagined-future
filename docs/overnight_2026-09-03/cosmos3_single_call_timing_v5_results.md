# Cosmos 3 single-call timing result

Status: manuscript-ready wording based on the complete, independently audited
timing-v5 cohort. This note does not modify the manuscript.

## Proposed subsection

### A single denoising call can recover sustained future steering

Does donor steering require clamping the transplanted future throughout
denoising? We repeated the intervention on 30 frozen, selection-free middle
states spanning six tasks, activating the donor-future clamp at only one of the
four chronological denoising calls or at all four calls. For each state, we ran
the complete $4\times4$ recipient--future-source grid while holding the
recipient observation, instruction, action noise, initial sampler state, and
path noise fixed. The matched analysis compares each donor future with the
self-future clamp at the same call.

Every individual call produced a donor-specific action effect under the
prespecified tests (Holm-adjusted $p=4.0\times10^{-4}$ for each call). The raw
correct-donor retrieval rate increased from 72.8% at call 0 to 92.2%
at call 1 and 100% at calls 2 and 3. Strikingly, calls 2 and 3 alone
numerically matched the all-calls intervention: all three yielded 100%
donor retrieval, while their mean raw distances to the donor action were
0.316, 0.318, and 0.321, respectively. No equivalence test was
prespecified, so we interpret this as numerical agreement rather than formal
statistical equivalence. Averaged across all four single-call conditions, the
matched retrieval gain was 0.912 (95% CI 0.876--0.940) and the matched
distance gain was 0.668 (95% CI 0.629--0.704). Sustained clamping was
stronger than the average single-call intervention because the first two calls
were weaker.

The complete 3,240-call run passed all frozen manipulation and replay checks:
the no-intervention arms were exact no-ops, native and all-calls replays were
exact, the intervention wrote no action coordinates, and the active future
clamp and returned-velocity audits had zero error. An independent audit
recomputed the action metrics directly from all 30 immutable state files and
found no discrepancy with the frozen analyzer. Thus, the steering effect is
not contingent on clamping the future at every denoising call; a single late
intervention can produce the same headline action-level behavior.

## Proposed compact table

```latex
\begin{table}[t]
  \centering
  \small
  \setlength{\tabcolsep}{4.2pt}
  \caption{\textbf{A single late future intervention recovers sustained
  action steering.} We activate the donor-future clamp at one chronological
  denoising call or at all four calls. Correct-donor retrieval is nearest-native
  identification among four candidate actions for the 12 off-diagonal
  recipient--donor pairs per state. Distance is the raw $\ell_2$ distance in
  the flattened $32\times8$ action chunk; reduction is relative to the native
  recipient--donor distance. Values are equal-task means with 95\% hierarchical
  task-to-state bootstrap intervals over 30 states from six tasks.}
  \label{tab:single_call_timing}
  \begin{tabular}{lccc}
    \toprule
    Future clamp & Donor retrieval $\uparrow$ & Distance to donor $\downarrow$ & Rel. reduction $\uparrow$ \\
    \midrule
    None & $0.000\,[0.000,0.000]$ & $1.528\,[1.268,1.794]$ & $0.000\,[0.000,0.000]$ \\
    Call 0 only & $0.728\,[0.642,0.803]$ & $0.916\,[0.690,1.207]$ & $0.375\,[0.280,0.445]$ \\
    Call 1 only & $0.922\,[0.856,0.975]$ & $0.554\,[0.412,0.753]$ & $0.613\,[0.546,0.669]$ \\
    \textbf{Call 2 only} & $\mathbf{1.000\,[1.000,1.000]}$ & $\mathbf{0.316\,[0.265,0.386]}$ & $\mathbf{0.761\,[0.717,0.798]}$ \\
    \textbf{Call 3 only} & $\mathbf{1.000\,[1.000,1.000]}$ & $\mathbf{0.318\,[0.264,0.401]}$ & $\mathbf{0.761\,[0.715,0.801]}$ \\
    All calls & $1.000\,[1.000,1.000]$ & $0.321\,[0.265,0.408]$ & $0.760\,[0.714,0.801]$ \\
    \bottomrule
  \end{tabular}
\end{table}
```

## Short caption alternative

**Single-call timing of future steering.** Donor steering strengthens across
the four denoising calls. Activating the intervention only at call 2 or call 3
already gives 100% donor identification and the same numerical donor distance
as clamping at all calls. Intervals are hierarchical task-to-state bootstrap
95% CIs over 30 frozen states from six tasks.

## Exact claim boundary

This is an imposed action-space timing and strength intervention. A clamp at one
call changes the sampler state inherited by later calls, so the result shows
that an intervention initiated at that call can steer the final action; it does
not isolate a local computation at that call. It also does not establish natural
future mediation, necessity, physical task success, or semantic planning.

## Audited artifacts

- Independent raw audit: `output/overnight_2026-09-03/cosmos3_single_call_timing_v5/evaluation/audit/independent_raw_audit.json`
  (SHA-256 `e710cffd44ed92b881eb480a2e7a08b1e404860bbb8e7e41adeac5ec2e569a89`).
- Frozen analyzer result: `output/overnight_2026-09-03/cosmos3_single_call_timing_v5/evaluation/analysis/cosmos3_single_call_timing_results.json`
  (SHA-256 `2a988a805b7423efab65b3354043ef8dde5dd2c0043f00d5f99291bee6c00262`).
- Raw-output inventory: `output/overnight_2026-09-03/cosmos3_single_call_timing_v5/evaluation/packaging/output_inventory.json`
  (SHA-256 `7dbb6f6b063f60679e31b5217ac3b95831b1e929ec9f9a820bb4bc83bfffcf9f`).
