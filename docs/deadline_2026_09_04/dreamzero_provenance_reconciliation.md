# DreamZero provenance reconciliation

**Purpose:** record the successive-fix provenance chain for the completed
DreamZero evaluation without implying that an earlier runtime receipt certifies
a later analyzer. This is a provenance index, not a new analysis and not a
scientific result.

The authoritative packaged machine-readable index is
[`provenance_chain_v2.json`](../../output/deadline_2026_09_04/dreamzero/provenance_addendum/provenance_chain_v2.json),
SHA-256 `cd6adce62ff09be5c8092212c50d1b49aeb2fef08be7490da848e3db261e1c36`.
It preserves the earlier
[`provenance_chain.json`](../../output/deadline_2026_09_04/dreamzero/provenance_chain.json),
SHA-256 `34f52efc5f960a271ae291fc94c3527622e14edbd0de3d7e992cce5b0bc1aa63`,
and supersedes only its runtime-log binding.

## Authoritative chain

| Object | Local path | SHA-256 | What it binds |
|---|---|---|---|
| Raw runtime receipt | [`final_runtime_receipt.json`](../../output/deadline_2026_09_04/dreamzero/provenance/final_runtime_receipt.json) | `57b89a7a98b3326812fa6652fff1f000f9bbe6940cd82a4e1a72b7983eaa06e5` | Canonical repository/checkpoint/patch; manifest; server launch fields; immutable core, Gaussian, and dose result maps; and the control analysis inventory. Its post-call server-log field is stale and is superseded by the next row. |
| Runtime provenance addendum | [`runtime_provenance_addendum_v1.json`](../../output/deadline_2026_09_04/dreamzero/provenance_addendum/runtime_provenance_addendum_v1.json) | `b7ce05309878376cc5f1fa1c091c4fa5007a7c9d705f7481cf902a9c54878078` | Correct evaluated-server log through live PID/cwd/command/stdout/stderr bindings; actual log SHA; postrun 258-package census; and explicit boundaries on launch-time environment and clean-server parity claims. |
| Clean-upstream parity addendum | [`execution_receipt.json`](../../output/deadline_2026_09_04/dreamzero/upstream_native_parity/execution_receipt.json) | `f9f6294d582486d97c8a2def87c63d770985fa022016660b744df54489db9c11` | On one excluded debug input, an untouched checkout of the pinned official commit and patched mode-off returned bitwise-identical float32 24x8 action arrays (maximum absolute error 0). This is action-output implementation parity for one input, not future/trace or cohort-wide parity. Artifact-index SHA-256: `5d2e3c71b85c9e2327337139062715aabbe9c6beae4b7d4fc2a5139c74da270f`. |
| Final core analyzer | [`summarize_dreamzero_future_transplants.py`](../../scripts/summarize_dreamzero_future_transplants.py) | `7f17970a858927e37a664efbde0451655bac00e374b3e08446d9a7e9295efc30` | Code that produced the authoritative final core analysis. |
| Final core analysis inventory | [`artifact_inventory.json`](../../output/deadline_2026_09_04/dreamzero/core_analysis_final/artifact_inventory.json) | `4474b1a7f2d9d2c7bf4682d4d94910782d55e0b26e3ec693b235902e1c500327` | Final analyzer SHA to every final core output hash and sidecar, including [`summary.json`](../../output/deadline_2026_09_04/dreamzero/core_analysis_final/summary.json), SHA-256 `83844de65e5b23e6f6cdb3991b3ddfacb8ba23ef65aeab5fe2bb97cad9439d61`, and an embedded byte-identical copy of the raw runtime receipt. |
| Control analyzer | [`summarize_dreamzero_controls.py`](../../scripts/summarize_dreamzero_controls.py) | `bcd3dbc5687f4e3bba941d99792f5b42b1f76b6075c16981b1d852fb7ff5ce57` | Code that produced the authoritative Gaussian and dose-response analysis. |
| Control analysis inventory | [`artifact_inventory.json`](../../output/deadline_2026_09_04/dreamzero/control_analysis/artifact_inventory.json) | `dbc56bd5e282b7c7cd30876206bf15b6a800d3f4e4912af1cf60a5f8b0f7c663` | Control analyzer SHA to every control output hash and the exact per-state core, Gaussian, and dose source hashes, including [`summary.json`](../../output/deadline_2026_09_04/dreamzero/control_analysis/summary.json), SHA-256 `e00616ebc2c910c14a71716cc75a4ad0b29da479ba11bf836cf77ba2dff8f7b1`. |
| Exhaustive native-media receipt | [`receipt.json`](../../output/deadline_2026_09_04/dreamzero/all_native_media/receipt.json) | `89c7f4742b42698d2a2cb122f63c5792d6e53da070280c019ad18a1fb3acacdb` | Post-analysis export of all 30 frozen states x four native seeds in manifest order. It binds 120 videos to the frozen core: every regenerated action is bitwise exact and every replay trace is bytewise exact. Its 392-entry artifact-index SHA-256 is `8907b7f854f7ea5217cfdce842bb56d2cc86649fd2fa53eed11480966f5f5aa6`. |
| Selection-neutral derived-media receipt | [`receipt.json`](../../output/deadline_2026_09_04/dreamzero/all_native_media_derived/receipt.json) | `80d0180c3df8fa88e715c7869c1345661c2fd4d85dbd7d062dd9d326cf6b533b` | A 30 x 4 terminal contact sheet and 54-second manifest-order overview built from every exhaustive decode without outcome or appearance selection. The receipt rehashes every source before and after derivation; its artifact-index SHA-256 is `85b551a45f5561b3bc71370cf4d24c9fd07992eb6528651673872854cb106a0e`. |

## Reconciliation note

The raw runtime receipt lists an earlier core analyzer,
`f992cec00a13402f8974ade28f8966ec5d5810fae840b0a3366ec95cc0a070d2`,
because the receipt was frozen before the final analyzer-only correction. It
therefore **does not directly certify** the final analyzer. The authoritative
binding is instead:

```text
raw runtime/results receipt 57b89a7a...
                 ^
                 | embedded as a hashed final artifact
final core inventory 4474b1a7... -- final analyzer 7f17970a...
                 |
                 +-- final core output hashes and sidecars
```

The control chain is independently bound by control inventory
`dbc56bd5...`, which names analyzer `bcd3dbc5...` and inventories the exact
source and output hashes. This is successive-fix provenance, not a mismatch in
the underlying run results. For scientific reporting, use only the final core
summary under `core_analysis_final/` and the frozen control summary named above.

The receipt's named `dreamzero_server.log` is a failed 08:33 UTC launch log,
not the evaluated server's output. The v2 chain binds
`dreamzero_intervention_server_20260904T090417Z.log` instead (SHA-256
`a01f759f9ef18022df7aeb4d543eac5626b08a15ffc80389533538ed24f4c07e`),
whose path matches file descriptors 1 and 2 for the live torchrun parent and
both workers. This correction changes no raw action, trace, manifest, analysis,
or reported estimate. The accompanying distribution census was captured
postrun and is not represented as a launch-time environment lockfile.
