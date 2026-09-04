# Cosmos 3 single-call timing v4: outcome-blind prelaunch audit

Verdict: **NO-GO.** Timing v4 must remain smoke-only and non-admitted. Its
excluded smoke passed every runtime manipulation/control gate, but the action
estimand implemented by the runner/analyzer does not match the frozen protocol
wording: the protocol specifies a 32 x 7 action chunk, while the released
policy returns and the code flattens a 32 x 8 chunk.

No timing-v4 evaluation outcome was opened or generated for this audit. Only
manifests, frozen code, input hashes, layout/checkpoint receipts, and control or
schema fields from the excluded smoke were inspected.

## Frozen artifacts

- Evaluation manifest: `cosmos3-timing-89823a363d824c4a`, SHA-256
  `5c4e4d50c8d0132b0ac8d57fe509784b6badc28441c004f0a10311f61deef073`.
- Excluded-smoke manifest: `cosmos3-timing-excluded-smoke-ac67e566390c2204`,
  SHA-256
  `5c924e46fd403fcb32f71367531dfc3f63436099ada6d927e7462ca117c5e7f6`.
- Excluded-smoke state artifact SHA-256
  `17ce67034159bb2d98892e64ece8f3e88e5a6785e9652cf4dea4c07df75e4e66`.
- Runtime closure SHA-256
  `ab5305797cefcb5a001365d11ddbb5cd7eb1fdaf49b746f4a9a1de231e67aa39`;
  30-file snapshot checksum-list SHA-256
  `d764bf5b949cfc56f33d16f55db256c8fd63271400b20111eabf6e0274514513`.
- Dedicated server SHA-256 `1b67e0735c108f0c2209374e758851999e7b593b20b7e4997c344996a8bcbedf`;
  runner `541375d66b1db32af64b8dc139dde3851b6eeda0d1b56d4ba428068f7cceef24`;
  analyzer `1ac68d3bd1a07bd480b51d95748f4c15d5a4f0547b64d70f6b8c2609c1c5bc5e`;
  launcher `b06ebedaf72888b003489cf7b2f34132ff5b7656ab45fed7c5c8130907111b71`.

## Checks that passed

- Both content-addressed manifest IDs were independently recomputed. The
  evaluation manifest contains exactly the 30 middle states copied byte-for-
  byte as JSON objects from archival v7: six tasks x five environment seeds,
  with branch order 211, 223, 227, 229 and no state filtering or replacement.
- All 90 unique input assets were independently rehashed with zero mismatch.
  The 20-file executable closure and its closure digest were independently
  reproduced; the generic NaN-emitting server is absent from v4.
- The request matrix contains exactly 108 unique labels: 4 native, 4 native
  replay, six complete 4 x 4 timing grids, and 4 all-calls diagonal replays.
  Timing indices, four float32 sigmas, source cells, and 12 off-diagonal pairs
  match the prospective protocol.
- Full-checkpoint receipts bind 87 files and 32,937,437,706 bytes to content
  manifest SHA-256
  `b6cbee5522104145f71cfcc74f5049f61e7de3640b88f14831909c7f473c2660`.
- The excluded smoke is exact and complete: 108 requests; structural projection
  census 28 null, 72 finite, 8 absent; one input fingerprint and parameter
  probe; all active input/returned-velocity and action-coordinate errors zero;
  inactive writes zero; native, diagonal, no-op, and source-invariance replay
  errors zero; every runtime gate true; no IEEE NaN or infinity.
- The frozen analyzer's complete single-report validator passed the smoke.
  Independent schema inspection found 4 stored native plus 96 timing actions,
  all and only shape `[32, 8]`, with 256 finite coordinates each.

## Sole blocker and remedy

The prospective protocol says “Flatten each 32 x 7 action chunk in float64.”
The server returns `[32, 8]` (seven joint coordinates plus gripper), and the
runner/analyzer flatten all 256 coordinates without an exact shape gate. This
is a prespecification mismatch even though the runtime behavior is otherwise
sound; its effect cannot be judged from outcomes after launch.

The prospective action-shape amendment at
`docs/overnight_2026-09-03/cosmos3_single_call_timing_protocol_v2_action_shape_amendment.md`,
SHA-256 `70767fda042b3ba8dab888ea5c4325f34aa20d6a9e494b68dc47cbacf653e88f`,
defines the full `[32, 8]`/256-coordinate policy action and exact runner,
manifest, analyzer, test, and census gates. Activation requires a new version,
fresh empty-registry server, new content-addressed manifests, fresh 108-call
excluded smoke, and a new independent outcome-blind audit. Passing v4 controls
must not be used to admit or resume v4.
