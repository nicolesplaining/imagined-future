# DreamZero independent future × current-future K/V factorial

**Status:** implementation protocol only; this experiment has not been run.

## Scientific question

Stage 1 showed that replaying a donor's matched-noise future-latent trajectory
redirects DreamZero's predicted action toward the action paired with that
future. This experiment asks whether the donor effect reaches action tokens
through the **current generated future's self-attention keys and values**.

Do not call these tensors DreamZero's persistent `kv_cache`. In the released
implementation, that object contains past observed/reference context. The
causal variable here is the current-future K/V made inside each transformer
block during each joint video/action denoising call.

## Why native donor-K/V replay is invalid

Do not record K/V during an ordinary donor run and replay it into a recipient
run. DreamZero jointly updates video, action, and state tokens. After the first
block, a native donor's video hidden states have already incorporated donor
action/state information. Replaying those tensors could copy donor-action
information rather than isolate a future-to-action route.

Instead, construct the K/V source online with a paired future side stream whose
action and state hidden rows are synchronized to the experimental recipient
stream at every block. The only source-specific information in that side stream
is its recipient or donor future trajectory.

## Frozen inputs

Use the already frozen DreamZero identities and cohort:

- Official repository: `dreamzero0/dreamzero`, commit
  `ab790c198fbce33503358efbbd4187ce9a89adf3`.
- Checkpoint: `GEAR-Dreams/DreamZero-DROID`, revision
  `96ad344138c66e82536422432ad742f015784942`.
- Existing 30-state manifest and its four native branches with seeds
  `211, 223, 227, 229`.
- Primary recipient `R = branch 0` (seed `211`) and donor `D = branch 1`
  (seed `223`) in every state. This pair is fixed by manifest order; do not
  select it from observed effects. Because the existing cohort has already
  been analyzed, label this a post-analysis mechanistic follow-up. For a
  confirmatory claim, freeze a fresh outcome-uninspected cohort first.
- Recipient observation, instruction, past-context cache, state input, action
  noise, sampler schedule, and solver-step order remain fixed in all cells.
- Use the existing matched-noise recipient and donor future traces at all 16
  solver steps. Do not substitute a final clean future at every step.

The minimal evaluation design is 30 states × four factorial cells. On the
existing cohort it is a post-analysis mechanistic follow-up; repeat it on a
fresh sealed cohort for confirmatory wording. Reciprocal `branch 1 → branch 0`
and third-source redirection with branch 2 are useful secondary extensions,
but they must not replace the fixed primary design.

## The four cells

Here `F_X` is the raw matched-noise future trajectory installed in the main
stream. `KV_Y` is current-future K/V constructed online from future `Y` while
using the main stream's evolving action/state hidden rows.

| Cell | Main-stream future | K/V visible to main action queries | Role |
|---|---|---|---|
| `RR` | `F_R` | `KV_R` | Recipient baseline and identity gate |
| `DD` | `F_D` | `KV_D` | Full Stage-1 donor-future effect under recipient action noise |
| `DR` | `F_D` | `KV_R` | Necessity/suppression |
| `RD` | `F_R` | `KV_D` | Rescue/sufficiency |

The cell names list **raw future first, action-query K/V source second**.

## Exact implementation site

Patch only the ordinary PyTorch path; disable TensorRT. The relevant released
code is:

- `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`,
  `CausalWanSelfAttention.forward`, currently around lines 1041–1080.
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`, the
  16-step loop around lines 1229–1305.

At inference, `_forward_blocks` lays tokens out as:

```text
[ current-video tokens | action tokens | state tokens ]
```

`action_register_length` includes both action and state tokens. Derive all
slices at runtime and assert them:

```text
video = [0 : seq_len]
action = [seq_len : seq_len + action_length]
state = [seq_len + action_length : total_length]
```

Never hard-code the observed `24` action-token count without asserting it
against `action_length`, `num_action_per_block`, and the actual tensor shape.

## Paired-stream block algorithm

For each factorial cell and each of 16 solver steps, maintain:

1. a **main stream** containing raw future `F_X` and the recipient's evolving
   action/state stream; and
2. a **source stream** containing future `F_Y` but no independently evolving
   donor action/state stream.

At the entrance to every one of the checkpoint's 40 transformer blocks:

1. Copy the main stream's action and state hidden rows into the source stream.
   Assert equality before computing Q/K/V.
2. Compute normalized, RoPE-applied Q/K/V separately for the main and source
   streams using the same block weights, timestep, conditional/unconditional
   context for that rank, and unchanged recipient past-context cache.
3. Run the native main attention once for all main queries.
4. Run a second **same-shape** attention call for all main queries, replacing
   only the current-video K/V slice with the source-stream current-video K/V:

   ```text
   Q = all main Q
   K,V = [recipient past-context K/V,
          source-stream current-video K/V,
          main action K/V,
          main state K/V]
   ```

5. Construct the main output by concatenating the **native-call** main video
   rows, the **hybrid-call** action rows, and the **native-call** main state
   rows. A same-shape hybrid call is required so identical-source replay can be
   tested bitwise. Do not write into the noisy action latent, scheduler state,
   action decoder output, or action-noise buffer.
6. Advance the source stream's **video rows** through the same block using its
   own current-video rows and the synchronized main action/state rows. Discard
   its action/state outputs. After the block, again replace its action/state
   rows with the main stream's post-block rows before entering the next block.

This construction allows the source future to develop through depth while
preventing a native donor action trajectory from entering its K/V. Main video
and state query rows receive no direct patch, but they may differ across cells
at later blocks through the model's ordinary action-to-video/state feedback.
Indirect main-future routes therefore remain intact; the intervention
specifically tests the direct current-future-K/V-to-action-query route.

On two-GPU classifier-free-guidance inference, apply the same code path on both
ranks with the rank's own prompt embedding and cache. Record rank identity in
the trace. DreamZero ultimately uses the conditional action prediction, but
patching both ranks keeps video evolution and distributed execution faithful.

## Required implementation modes

Expose explicit, mutually exclusive modes:

- `off`: byte-identical released inference.
- `paired_identity`: paired stream enabled with `X = Y`; must reproduce `off`.
- `factorial`: independent `X` and `Y` selected from the frozen trace map.
- `record_audit`: same computation as `off`, plus tensor metadata/hashes only.

Every result must record repository/checkpoint identity, patch hash, manifest
hash, cell, state, branch identities, action-noise hash, future-trace hashes,
past-cache hashes, tensor slices, solver step, block, CFG rank, dtype, shape,
and device.

Do not persist the complete internal K/V tensors: one source is approximately
21.5 GiB for one CFG branch and approximately 43 GiB for both branches. Stream
each current-future K and V tensor through a cryptographic hash and discard the
bytes, retaining the digest plus shape/dtype/norm. If full per-block CPU
hashing is too slow, that is a runtime limitation to report—not a reason to
replace cryptographic provenance with an undocumented checksum.

## Fail-fast gates on excluded states

Do not inspect confirmatory outcomes until every gate passes:

1. Clean upstream versus patched `off`: bitwise-identical action output.
2. `paired_identity` versus `off`: bitwise-identical action and video outputs.
3. `RR` versus the frozen recipient/self replay: bitwise-identical action.
4. `DD` versus the frozen Stage-1 donor-future transplant: bitwise-identical
   action.
5. Same-run K/V recomputation/replay: bitwise exact.
6. Same-source side-stream current-video K/V equals main current-video K/V at
   every block, solver step, and CFG rank; cross-source K/V differs.
7. Main and source action/state rows equal at every block entrance.
8. Recipient action-noise, observation, instruction, state, scheduler, present
   frame, and past-context-cache hashes identical across all four cells.
9. Within each block, the emitted main video and state attention rows equal the
   corresponding rows from that cell's native-main attention call exactly;
   neither row range is taken from the hybrid call.
10. Write audit confirms that the intervention modifies only the K/V supplied
   to action queries; no action coordinate or decoded action is overwritten.
11. Exact trace counts are 40 blocks × 16 solver steps × both CFG ranks; runtime
    assertions expect 1,760 current-video tokens, 24 action tokens, and one
    state token for this checkpoint, while still deriving slices from shapes.
12. The runner never loads or passes a donor action/noise/hidden tensor. Add a
    forbidden donor-action sentinel that raises on any attempted access.
13. All outputs finite; no skipped/missing block, solver step, or CFG-rank
    trace records.

Any failure is a hard stop. Do not weaken equality tolerances or silently drop
the affected state.

## Execution

After freezing the passing patch, protocol, excluded-state receipts, and output
schema:

1. Run `RR`, `DD`, `DR`, and `RD` on all 30 states.
2. Keep the recipient action-noise path fixed within state and cell quartet.
3. Write each state atomically and support resume by verified cell hash.
4. Preserve failures and exclusions. The only admissible exclusions are frozen
   mechanical criteria defined before outcomes; report the full funnel.
5. Analyze only after all 120 cells and integrity checks complete.

Measured ordinary throughput was 2.866 s/call. With the paired side stream and
native/hybrid attention calls, budget roughly 12–20 minutes wall time on the
loaded 2×H100 server for the 120-cell core (about 0.4–0.7 aggregate GPU-hours),
plus hashing and I/O. Budget 3–6 engineering hours for the patch, static audit,
identity gates, and smoke run; compute is not the bottleneck.

## Estimands

Compute action geometry against the existing native recipient action `a_R` and
native donor action `a_D`:

- normalized donor projection;
- distance reduction toward `a_D`;
- cosine alignment with `a_D - a_R`;
- orthogonal residual;
- donor-versus-recipient identification.

Average repeated measurements within state. State is the independent unit;
use equal verb-family weighting with a family→state bootstrap, or report equal
family means plus leave-one-family-out sensitivity. Directions and seeds are
within-state measurements.

Report the four cell means and these prespecified contrasts:

```text
Stage-1 effect       = DD - RR
suppression residual = DR - RR
K/V rescue           = RD - RR
K/V effect at donor  = DD - DR
factorial interaction = (DD - DR) - (RD - RR)
```

For projection, an interpretable suppression pattern is
`RR ≈ DR < RD ≈ DD`. Also test whether each crossed arm is closer to the native
action paired with its K/V source. Report uncertainty for every cell and
contrast. Do not report “percent mediated” solely from `DD → DR`; rescue must
also succeed and the paired-stream identity gates must be exact. If an
aggregate suppression fraction is shown, name it suppression, bootstrap the
ratio, and show the raw contrast beside it.

## Interpretation matrix

- **Suppression and rescue both succeed:** current-future K/V available to
  action queries is a necessary and interventionally sufficient major direct
  route for the imposed future's action effect.
- **Suppression succeeds, rescue fails:** evidence for necessity or interface
  disruption only; do not claim sufficiency or mediation.
- **Rescue succeeds, suppression fails:** evidence for sufficiency, not that the
  route normally carries most of Stage 1.
- **Neither succeeds with all gates passing:** DreamZero's Stage-1 effect likely
  reaches action through other direct or indirect routes.

Even the strongest outcome would not show that DreamZero naturally compares
alternative futures, plans over consequences, or improves closed-loop task
success. Those require separate experiments.

## Recommended secondary extensions

Only after the frozen 2×2 is complete:

1. reciprocal `branch 1 → branch 0`;
2. recipient future plus branch-2 K/V to test third-source redirection;
3. K-only and V-only crosses;
4. six predeclared layer bands and four solver-time bands;
5. physical execution or closed-loop task evaluation.

Do not start with a layer/head sweep. The architecture-defined all-block,
all-solver-step direct route is the confirmatory intervention.
