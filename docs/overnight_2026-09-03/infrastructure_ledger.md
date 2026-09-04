# Overnight Lambda infrastructure ledger

Provisioned for the 2026-09-03 imagined-future overnight experimental run.
The launch was explicitly limited to eight additional GPUs, and no pre-existing
instance was modified.

## Capacity

- Account capacity before launch: 24 of 32 GPUs allocated.
- New allocation: 8 H100 GPUs.
- Account capacity after launch: 32 of 32 GPUs allocated.
- Region: `us-southeast-1` (Georgia, USA).
- SSH key: `harshvardhan-imagined-future`
  (`78c74aa0bc9b43fcb49c62617fb97a53`).
- Shared filesystem: `imagined-future`
  (`3899f468f32048b798157a0e004b0b1f`).
- Mount point: `/lambda/nfs/imagined-future`.

## Instances

| Purpose | Name / hostname | Instance ID | Public IP | Type | GPUs | Status at verification |
| --- | --- | --- | --- | --- | ---: | --- |
| Cosmos servers | `if-overnight-cosmos-servers` | `22ffe24aaac84edfa3f1ea34004dcc1f` | `68.209.73.251` | `gpu_4x_h100_sxm5` | 4 | active |
| RoboLab clients | `if-overnight-robolab-clients` | `d51abf0e889541b2ad5ceacd7e405836` | `68.209.75.174` | `gpu_2x_h100_sxm5` | 2 | active |
| External WAMs | `if-overnight-external-wams` | `a7098d9c97204bb7a353df309867fb1f` | `68.209.72.187` | `gpu_2x_h100_sxm5` | 2 | active |

## Verification

Verified at 2026-09-03 14:57 PDT (21:57 UTC):

- SSH succeeded on all three nodes using the locally held private key associated
  with `harshvardhan-imagined-future`.
- Hostnames exactly matched the requested names.
- `nvidia-smi` reported 4, 2, and 2 NVIDIA H100 80 GB HBM3 GPUs, respectively.
- All eight GPUs showed 0 MiB memory use and 0% utilization at verification.
- `/lambda/nfs/imagined-future` was present and mounted as NFSv4 on every node.

The API credential is intentionally not recorded in this ledger.

## Frozen runtime images

Recorded before any frozen intervention output was generated:

| Runtime | Pinned source | Image digest | Base image digest |
| --- | --- | --- | --- |
| Cosmos framework, clean base | `d4599e2e43fbd06168e9884205b9b66c3902d8f6` | `sha256:2fd099d606fc83eb4e0a3516b18987ea1dee3d68bf3778409e8992dc6340b175` | `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:e071e85c52ad91fc9ea24158ff5330876b2d1a5c4ac83ccc6066976835873c01` |
| Cosmos framework, policy-server runtime | same clean source, derived only by locked policy-server wheels | `sha256:95a8933deb69f2fc73697d846f7e17066a4d4e05dc9ef85718a7ee12cfea2a8c` | clean Cosmos base above |
| RoboLab, client node | `9db0aaf09d9fe5d4f37b168320788258c7012463` | `sha256:ea2b318e344149b2acdd2a1c8b51f30e549a366166d23e9ee8c6f4975a2b2a67` | `nvcr.io/nvidia/isaac-lab:2.2.0@sha256:b4d8e96cbfb9a6c40067bec6cc5ee180e36d4c0164b25f7215c5f47e31897b94` |
| RoboLab, server node | `9db0aaf09d9fe5d4f37b168320788258c7012463` | `sha256:f78dad4bc92ca4e571f3a4f6cfe1c771f2bbaf18c892277fa5a0f8f759887971` | `nvcr.io/nvidia/isaac-lab:2.2.0@sha256:b4d8e96cbfb9a6c40067bec6cc5ee180e36d4c0164b25f7215c5f47e31897b94` |

The two RoboLab images are clean independent builds from the same source and
base. Their node-local image IDs differ because build metadata is not
reproducible byte-for-byte; the code and base pins are identical. The OpenPI
client is pinned to `aa6420561529593114160d05e5ad155792b272f3`.

The policy-server runtime adds only the three wheels required by the upstream
`policy-server` extra: `openpi-server==0.1.0` (wheel SHA-256
`e4db08c1f718cb704eb1c995270b46ffd013bf1e2cf95300d5b303f2040551af`),
`openpi-client==0.1.2` (`b8aa6d2ea7172d3b7c1fcf867d43930052b5ff19ea7b777966781105aa989897`),
and `filelock==3.29.0` (`96f5f6344709aa1572bbf631c640e4ebeeb519e08da902c39a001882f30ac258`).

## Isaac Sim compatibility incident

At 2026-09-03 15:40 PDT, three RoboLab recording queues and one excluded-state
K/V integration smoke were rejected as invalid infrastructure attempts. Each
failed inside Isaac Sim's `AppLauncher`, before the scientific runner began,
with `ERROR_INCOMPATIBLE_DRIVER` / `Failed to create any GPU devices`. NVIDIA
H100 GPUs do not provide the RT-core graphics support required by Isaac Sim.

- The four Isaac processes and their queue wrappers were stopped. The Cosmos
  servers and FastWAM processes were not modified.
- No recording, branch, action, endpoint, or K/V outcome artifact was created.
- The diagnostic logs are preserved under
  `/lambda/nfs/imagined-future/results/overnight_2026_09_03/logs/` with the
  suffix `.failed_h100.log`.
- These attempts are excluded from every scientific analysis. Fresh runs must
  use an Isaac-compatible RTX GPU and the final runner SHA-256
  `a231d03b38e23e5e278e926f2b8941fe8de57e156e64d7642b3de7748ed5cbe7`.

Before any authorized client-instance replacement, the verified client-node
RoboLab image was exported once to
`/lambda/nfs/imagined-future/runtime/images/robolab_overnight-2026-09-03_ea2b318e.tar.zst`.
The 18,304,716,347-byte archive has SHA-256
`a47a9e97620c0ba46746f867d68f5788ec4388a78f1f6f16adbd738c47722b5c`;
its portable checksum and Docker inspection sidecars are stored beside it.
Recovery commands and the exact bind/frozen-input inventory are in
`docs/overnight_2026-09-03/a10_recovery_kit.md`.
The corresponding immutable input bundle is
`runtime/images/cosmos3_a10_inputs_a231d03b.tar.gz` (24,637 bytes, SHA-256
`ccd3c699511d48b88b34dff951884d5962a3f5d5e4eaa3731496f04b5de6fd12`).

The Cosmos services on public IP `68.209.73.251` ports 8001 and 8002 timed out
from both the workstation and another Lambda node. Both ports were reachable
on the same-region private IP `172.26.133.114`. No firewall was changed.
The available A10 recovery capacity is cross-region (`us-east-1` or
`us-west-1`), so a replacement A10 cannot mount the Georgia NFS or use its
private address. The recovery kit therefore uses verified image/code transfer,
local result staging, and restricted SSH forwards to the existing public SSH
endpoint. No tunnel, key authorization, instance termination, or replacement
has been performed yet.

## Excluded model-only K/V engineering audit

Five excluded-development attempts were preserved while hardening the
action-only K/V audit. Attempts 1--4 failed, respectively, on a missing optional
video import before inference, an overstrict clean-clamp identity assumption, a
future-target signature check, and strict JSON serialization of a diagnostic
NaN. None was admitted as science. Attempt 5 passed all intended engineering
gates and atomically wrote
`results/overnight_2026_09_03/smoke/kv_factorial_action_only_dev.json`
(SHA-256 `0722ced8938d90fbf9c7707b24bb4cab08346f60d92c0ef2a6bc9415a88c3cee`,
mode 0644). The final audit script SHA-256 is
`4a3997fe5c5cdf6d6ab291091fff11922479b2e8339ca1539a9bcc7f6123d815`.
Exact record/replay and native-repeat errors are zero; recipient and donor
future signatures are internally stable and mutually distinct; action-input
and action-output overwrite errors are zero; exactly one state hash appears.
The container exited after the atomic write. This audit is action-only and
excluded from held-out RoboLab estimates.

The active existing-cohort bridge is frozen at
`results/overnight_2026_09_03/cosmos3_kv_existing_v3/manifest.json`, manifest
ID `cosmos3-kv-existing-bb8591311eda8a59`, SHA-256
`972bcab77e2b999c703250f9e7ab17f3d854c858d99712bb7873d0bbf589714f`.
It contains 21 evaluation states from the existing 22-state selected-pair
cohort, with `BananaInBowlTask_seed_103` reserved for development. V1 and V2
were superseded before any evaluation outcomes; V2 added clean process exit,
and V3 adds explicit scope metadata. The active runner SHA-256 is
`83285e99b993e7f996a40189332643338e33805fe03e00b931f0c214e32179de`;
the prose protocol SHA-256 is
`c64c13b3f8732405a91650f862e83e34ce5e243caff7cd4e81de711bbc338bb2`.
The excluded scope/exit test passed all exact gates and wrote
`smoke/kv_factorial_action_only_dev_scope_exit_test.json` (SHA-256
`a946d9e5c716b255f57877002791ebaf7f3f5772fe44133153ddc87cb24120ae`,
mode 0644). This bridge is model-only and selected-pair; it is not the fresh
selection-free study and has no physical endpoint claims.

The final same-container topology gate ran inside `if-cosmos-kv-8002` with
`/workspace/.venv/bin/python` against `localhost:8002`. The excluded
development process exited zero in 15.0 seconds and passed every exact replay,
signature, scope, and coordinate-write gate. Its immutable artifact is
`results/overnight_2026_09_03/smoke/kv_factorial_action_only_dev_server_container_test.json`
(SHA-256 `8bbfec3b846137808555a519756b496e733299242fc078414f3b22b3104f3ece`).
This is the approved batch topology; the gate remains excluded engineering
data and was not admitted to a scientific estimate.

At approximately 2026-09-03 23:19 UTC, the frozen existing-cohort bridge batch
was launched inside the dedicated server container. The launcher PID is 42129,
the frozen launcher SHA-256 is
`dd3cab800431ddf542bab574a528f3971d5df813750720b939c3a1247e2e68a4`,
and the output root is
`results/overnight_2026_09_03/cosmos3_kv_existing_v3/run`. Validate-only passed
before launch. The first state completed atomically and the second state began;
no scientific outcomes were opened or inspected while recording this status.

The batch subsequently completed 21/21 accepted states with no retries or
exclusions. The frozen analyzer is
`scripts/summarize_cosmos3_existing_kv_factorial.py` (SHA-256 prefix
`5a2fe27f`), the analysis directory is
`results/overnight_2026_09_03/cosmos3_kv_existing_v3/analysis`, the summary
JSON SHA-256 prefix is `7681fcb8`, and the plot SHA-256 prefix is `64e19022`.
All 42 crossed arms followed K/V identity. The equal-task projections were
R/R -0.0112 [−0.0224, 0.00035], D/R 0.1190 [0.0776, 0.1625], R/D 0.8842
[0.8485, 0.9187], and D/D 1.0038 [0.9877, 1.0254]; the two fixed-future K/V
effects were 0.8955 [0.8593, 0.9283] and 0.8848 [0.8454, 0.9215]. The updated
prose protocol SHA-256 prefix is `efaff8ac`.

An independent adversarial audit reproduced every K/V cell, confidence
interval, and the 42/42 crossed-arm count without finding a selection,
exclusion, or aggregation defect. The audit is
`docs/overnight_2026-09-03/cosmos3_existing_kv_factorial_adversarial_audit.md`
(SHA-256 `7533626a25afe30dc227a8b809a4dbab5448454ec43d4142eb3e8a6f5e9ae2d3`).
It constrains interpretation: realized-future hashes are fixed within K/V
crossings, target residuals span 0.01807--0.02779 latent units, and the visible
future retains a +0.12--0.13 residual contribution. It does not support saying
that the visible future is irrelevant or that K/V fully determines the action.

## Frozen archival action-only fallback

The input reconstruction audit is
`results/overnight_2026_09_03/smoke/archival_input_reconstruction_audit.json`
(SHA-256 `f6400974cc740f8eb3ac3f1932123e332bbb840c5cdec7d6cadbf23aca3dcd91`).
For all 22 prior branch states, MP4 frame `branch_step-1` with panel order
head/left/right/wrist and the Cosmos composition wrist + bilinear half-scale
left/right was the best mapping across all tested offsets/permutations. Mean
MAE was 2.6696/255 and mean PSNR was 36.66 dB. The source is H.264 lossy, so
this fallback is explicitly archival, lossy-input, and action-only.

Manifest v1 (`8228a2fc...`) and v2 (`e359b7ff...`) were preserved and
superseded before any evaluation model call. V1 predated the corrected full
4x4 future-source retrieval contract and v2 did not bind the host-mounted
server dependency closure. The active held v3 manifest is
`results/overnight_2026_09_03/cosmos3_archival_selection_v3/manifest.json`, ID
`cosmos3-archival-sf-371ff46b7b33fb38`, SHA-256
`e3ea2beda662bce662c8835f8240ffc855ef30aff250725692a1bdf1274cf89f`.
It freezes 30 episodes and 90 states, four branch seeds, the full 16-cell
recipient x future-source retrieval grid, and the 12 off-diagonal directional
arms. The runner SHA-256 is
`0da5d4ab7015c0ac042eb7f1d1445459b2180c0a52ccad4e816ffa7875146465`,
launcher `2735a23110acce6bed6aedca376012311f3926eeeea536da6f13c3f4d7bc4dcd`,
and distinct analyzer
`60fe76f33b0fc2dd3f6cc941d24c29313b18088dce41423c09436e148784162a`.
The host-mounted causal server closure is pinned as server script
`b43454826fc0a8d3a93eb198e1046b962a54d0cbbaed40b6cfb4ca43bfde9da8`,
future interventions
`8d11f8ba8ff84a914fa00ebcc072849594b9b77a833085c14f2ba777758f950f`,
and attention module
`5f6359e0ec14da1480e18a4f2e1f82231a559175f61256d7b7a021eda7af19d1`.

The exact runner passed a separate excluded Bagels seed-101 middle-phase smoke
on general port 8001: 52 requests, clean exit, four exact native repeats, four
exact self repeats, 12 exact donor replays, 16 retrieval cells, and 12 Gaussian
controls. All 44 full-clamp target errors were below the frozen 0.03 threshold
(maximum 0.0274525), Gaussian matching errors were at most 1.3e-10, action
coordinate writes were zero, and the state-hash count was one. The artifact is
`results/overnight_2026_09_03/cosmos3_archival_excluded_smoke/run/BagelsOnPlateTask_seed_101_phase_middle_step_464.json`
(SHA-256 `010d11ad0b134988b3391f9d3ed23bb7dfb0f88450ebc2c003ba98550def0d63`).
Its admission is `excluded_development_smoke`; it is not one of the 90 states.
The untouched evaluation remains held pending adversarial approval and, if
approved, will use only port 8001 sequentially to avoid a port-8002
attention-runtime lane confound.

Herschel's adversarial pre-outcome audit issued NO-GO on v3 because it did not
pin the checkpoint parameter-probe fingerprint, called the transformed-input
fingerprint a model-state hash, lacked explicit evidence gates and complete
metric denominators, and did not bind every causal dependency at launch. V3
remains preserved and no evaluation call was made from it. V4 and v5 also
remain preserved as excluded engineering iterations: their Bagels smokes
failed before producing an admitted artifact because the expanded metadata
signature first included wall-clock timing and then rejected the server's
expected non-finite self-projection diagnostic. Neither iteration made an
evaluation call.

The audit-ready replacement is immutable v6:
`results/overnight_2026_09_03/cosmos3_archival_selection_v6/manifest.json`, ID
`cosmos3-archival-sf-d2df8d9d1d9f0c19`, SHA-256
`d8394fb17900d2c9a0d032317fba2cf554aaaf1b3213908782995b4829c03211`.
Its read-only source snapshot is under the same directory. The runner SHA-256
is `3456e65fe0bb974b2cd808c45e238a5fe5cd693be5a82d05dd992227c5b6116b`,
analyzer `1029cf3eb4b7d60fb9acf0f94ac3df1abe86580e38af92c4827feba08f20647e`,
launcher `950a068949a295bb6c60da02b6148adea8544fb825b35593d7cfe69e4c1c24dd`,
and manifest builder
`2ab9f5a302b12797a9f39ea357eb1c65a4c72d9746a9f7e8d36dd160b8d03153`.
Fourteen focused tests pass in the pinned Cosmos container. The manifest pins
checkpoint parameter probe
`21b79382b84b4bdebb943a2659c0272c99267ef433d83818f9a44b742c1170cc`,
the upstream RoboLab policy service SHA-256
`86d0d0e70faefa88a8cee8594abb3baa541ff6f65145a80eae23b1d828500f91`,
and all client- and server-side causal modules. Inspection of the pinned
`RobolabPolicyService._build_sample` confirms that only prompt, image, joint
position, and gripper position enter the model sample; `research_*` metadata
is ignored during sample construction.

V6 defines native-separation quartiles on all 1,080 off-diagonal donor arms
before averaging arms within state inside each quartile. The analyzer reports
the deterministic cohort-global boundaries, arm/state/task/episode counts,
and valid/null metric denominators. It also freezes claim gates for the
four-way retrieval lower confidence bound, donor-distance-reduction lower
confidence bound, exact controls, complete cohort, and zero degenerate axes.

The exact v6 runner passed a new immutable excluded Bagels smoke. The smoke
manifest SHA-256 is
`8c060d85ca66f6ef4a71b2091085899b7af8ec9b4e23979587422ee8313f7cec`;
the artifact is
`results/overnight_2026_09_03/cosmos3_archival_excluded_smoke_v6/run/BagelsOnPlateTask_seed_101_phase_middle_step_464.json`
(SHA-256
`efbe5d3d47dad68e9bc2e9a1bc893fdcbfa9509c4707ca4cab7633652345dcf9`).
All 52 requests completed. Full deterministic native/self/donor metadata and
actions replayed exactly after excluding request IDs and timings; the expected
parameter probe and one transformed-input fingerprint were present; all 44
action-coordinate audits were zero; all 44 target errors were finite and at
most 0.0274525 (below 0.03); and all 12 Gaussian geometry checks were finite
and passed. This state is excluded from evaluation. The 90-state launch
remains held for outcome-blind adversarial GO and will use only general port
8001 sequentially.

Herschel independently returned GO on archival v6 after checking the manifest
self-ID/SHA, root-owned mode-0444 snapshot, unchanged 30-episode x three-phase
cohort, balanced 4x4 retrieval and 12-pair mappings, complete-cohort analysis,
metadata isolation, and fresh excluded smoke. The allowed claim remains
lossy-archival, action-only, with no physical-endpoint inference; the frozen
parameter probe is a deterministic sampled-parameter fingerprint rather than
a full checkpoint-file digest. The audit report is
`docs/overnight_2026-09-03/cosmos3_archival_selection_v6_prelaunch_audit.md`
(SHA-256
`859fc005ff0e3771c0f6d8ca5d96a96f4ed3ee4dde1b5e6b48e61be74aafea82`).

The untouched 90-state v6 evaluation launched at 2026-09-04T00:12:27Z inside
`if-cosmos-general-8001` on the sole audited general server `localhost:8001`.
The launcher PID inside the container is `5413`; log is
`results/overnight_2026_09_03/cosmos3_archival_selection_v6/logs/launcher.log`,
and output root is the sibling `run/` directory. Prelaunch checks confirmed
the exact image digest, zero pre-existing evaluation JSONs, and the launcher
began with manifest unit 1/90. Monitoring is restricted to process state,
completed-file count, and error lines until 90/90 immutable outputs exist; no
causal payload or partial analysis will be opened.

At 2026-09-04T00:28:29Z, the v6 launcher stopped before atomically writing
unit 19 (`RubiksCubeTask_seed_103_phase_early_step_16`). One donor-clamp
manipulation audit reported target maximum error 0.0334150791, above the
frozen 0.03 threshold. Exactly 18 prior immutable state files exist. Launcher
PID 5413 is defunct and no archival worker remains. No causal payload from any
state and no partial state-19 output was opened; only the gate scalar in the
traceback was inspected. The cohort is held as a failed frozen manipulation
gate. The threshold, state roster, and outputs will not be changed, excluded,
or resumed without a separately documented adversarial decision.

V6 is permanently failed/incomplete and its 18 outputs remain preserved and
unopened. The corrective v7 implementation audits the actual model-input
future clamp and returned future-velocity overwrite at every active denoising
call, while treating final sampler-state residual as finite descriptive
telemetry only. V7 retains all 90 original states and reruns them from a fresh
root; no v6 state is resumed or admitted.

The checkpoint-bound final v7 root is
`results/overnight_2026_09_03/cosmos3_archival_selection_v7_final`. Its
read-only snapshot checksum list has SHA-256
`688b125658c66f083fb92a655749a7b37596572f180f3a8aae2cc478205853b0`.
The exact runner SHA-256 is `430d04f9476d2c34099437ae0770bbfd5505dc5d42215dbe0e251a741d49232a`,
server `64980c631d4bec71be3e41cb574c0b84c759e80ebbaf1e58eeb558f66be17073`,
intervention module `285e9218b1be148698d8f3cccce62551b8d93f64a7540e8c01e0c466c52e081b`,
analyzer `70a86cb30b63da58afa917b3d5047ca2515eee3297e78ea85671c060fe425066`,
launcher `01beb6a6da0aa80db60df36bf2bff2e54a7939f28cc03689f6170c24c87dff6f`,
and checkpoint verifier `e28cac4836a7e4cf90498b2133fb8879a06c1a1e4725288519ceb0efe8295435`.
Thirty-two focused tests passed before the final snapshot was made read-only.

The full checkpoint content manifest contains 87 files totaling
32,937,437,706 bytes and has SHA-256
`b6cbee5522104145f71cfcc74f5049f61e7de3640b88f14831909c7f473c2660`.
A file-by-file verification against the actual read-only `/checkpoint` mount
on the Cosmos server passed exact file-set, size, and SHA-256 checks; receipt
SHA-256 is `905f40239ae56db2f7d83de3e9002a18f3f1ec8ad454f7d254ef5ca509e8fabe`.
The sampled parameter probe remains separately and accurately labeled.

Final v7 evaluation manifest
`cosmos3-archival-sf-507feb24297971eb` has SHA-256
`8ab294c7a581424cfa02faae1db85941e3f43c1e28100e68e21fa5f533703b9e`.
Its isolated server is container `if-cosmos-archival-v7-final-8003` on GPU 2,
port 8003, using the pinned image, a read-only checkpoint, the immutable v7
snapshot, an initially empty registry, and `PYTHONDONTWRITEBYTECODE=1`.
The fresh manifest-bound excluded Bagels smoke has manifest SHA-256
`2cd0345566a334d2d6ff06638220237d62b73ba0b8c30d255467a3285eaeb5bd`
and artifact SHA-256
`f8d50b14c7d884449feeb27ecf31e9e7c2bf504b699a54684459d2a4509c7235`.
All 56 calls completed: 48 site audits, 44 active responses, and 176 active
sites; both live-tensor audit maxima were exactly zero, all action-coordinate
writes were zero, all native/self/donor replays and four inactive no-op arms
were exact, and the expected input/probe singleton gates passed. Only excluded
control fields were inspected. The 90-state evaluation remains held pending
explicit adversarial GO.

Herschel's independent outcome-blind review returned explicit GO for the final
v7 manifest, checkpoint receipt, snapshot closure, and excluded smoke. After
that GO, the GPU-2 server was restarted once more from the identical image and
immutable snapshot, and its startup log confirmed `registry_entries=0`. The
full 90-state launcher started at 2026-09-04T01:24:47Z as host PID 80690,
single sequential shard on `localhost:8003`. Its first operation reverified
all 87 checkpoint files in 22.26 seconds and passed; unit 1/90 then started.
The launcher log is
`results/overnight_2026_09_03/cosmos3_archival_selection_v7_final/logs/launcher.log`
and outputs are atomically written to the sibling `run/` directory. Until all
90 exact files exist, monitoring is restricted to process status, file counts,
timing, and errors; no partial scientific payload is opened or analyzed.

The final adversarial GO report is
`docs/overnight_2026-09-03/cosmos3_archival_selection_v7_final_prelaunch_audit.md`
(SHA-256
`fa2a2db8264c1f14eefb10e24a05cfeb3a6de068563bf171847ffd40bfbdfbab`).
Its completion requirement is frozen: after exactly 90/90 accepted outputs and
before running the analyzer, generate a 90-file SHA-256 inventory and change
all runner outputs from their initial mode 0644 to mode 0444. Only then may
the pinned analyzer read the complete immutable package.

The v7 launcher subsequently completed the exact 90/90 manifest cohort with
`resume_skipped=0` and emitted `shard_complete`. No partial scientific payload
was opened. The frozen completion packager (SHA-256
`8dab849f9a66891a49ed2decc3f196220cbb6bb971e43de5d7c07d99e090ecc0`)
verified the manifest-derived file set and admitted headers, hashed all 90
files, changed them from mode 0644 to mode 0444, and rehashed every file after
the mode change. The resulting mode-frozen/read-only inventory is
`results/overnight_2026_09_03/cosmos3_archival_selection_v7_final/packaging/output_inventory.json`
(SHA-256
`948f443fcd44a34a94a2a93f938a81f89c8affc37d6a7b90def985104c28914e`).
The first non-privileged packager invocation stopped at its first chmod with a
permission error because the runner outputs are root-owned; it had already
validated and hashed the set but wrote no inventory and changed no modes. The
identical frozen packager then completed under `sudo`. Only after this exact
packaging PASS was the pinned analyzer (SHA-256
`70a86cb30b63da58afa917b3d5047ca2515eee3297e78ea85671c060fe425066`)
started against the packaged 90-file input set.

The pinned analyzer exited successfully after the packaging step and wrote
`results/overnight_2026_09_03/cosmos3_archival_selection_v7_final/analysis/summary.json`
(SHA-256
`cb914fead7e5fdd1303eb0f8086c8db53b7c0a8b0db1cf36439112e771c89889`,
77,654 bytes). The summary was then changed to mode 0444 and rehashed with no
byte change. Scientific interpretation remains deferred to an independent raw
output/analysis audit; this entry records only process completion and artifact
identity.

An optional Cosmos-3 future-strength dose-response v2 package was then frozen
without making a dose model request or opening the v7 scientific payload. Its
pre-smoke root is
`results/overnight_2026_09_03/cosmos3_future_strength_dose_v2_pre_smoke`.
The 18-file mode-0444 snapshot checksum list has SHA-256
`126d387ccd39dc0c6e23beefac462a80670266d647463c77622da7ec5cb6a100`.
It pins the original prospective protocol (`7b559902...`) and v2 amendment
(`a02e3a74...`), including the full `[32, 8]` / 256-coordinate action schema,
pairwise OLS definitions, shared seeded hierarchical bootstrap, and restricted
claim language. The 30-state/2,760-call pre-smoke manifest is
`cosmos3-dose-ecae4c40bd3437ca` (SHA-256
`f256829232223376016e433d427795246e263e175059ec2c1ed992e9a72db394`).
The nonoverlapping Bagels 92-call excluded-smoke manifest is
`cosmos3-dose-excluded-smoke-832b41a3b67fc689` (SHA-256
`8b0bfd7968cf8df4275aaf835ccf7362e65bd797008de5802de928b5d85cd82a`).
Ten focused tests passed both locally and in the exact Cosmos image; static
compilation, Pyflakes, frozen entry-point imports, request counts, and snapshot
closure checks also passed. The audit handoff has SHA-256
`12fdd1bdb2f6a994e167f770fa423b96d3bbc8c2fe566a6adc92e0558bc5f13a`.
No dose server or model call is authorized until independent pre-smoke GO.

Herschel's independent archival-v7 raw audit subsequently returned PASS with
zero discrepancies. The primary raw-audit artifact is
`results/overnight_2026_09_03/cosmos3_archival_selection_v7_final/audit/independent_raw_audit.json`
(SHA-256
`8c6a162a06e134e5bbf4becc7948c6c23e590d41df66a906a75be769cae2747e`).
The independent state-row reproduction artifact is the sibling
`independent_state_rows_audit.json` (SHA-256
`408da6d9360d88d3506b9d39937dfb081547a5ef2d2d33df19a564fa73fe8eaf`).
Interpretive estimates were sent separately to the parent; this ledger records
the independent integrity result and artifact identities.

Herschel issued `GO_SMOKE` for exactly one excluded Bagels dose state and 92
calls. The signed authorization JSON has SHA-256
`f3f9c695219ad6922642e4dd1c42a52c692551234950ebaf4f74483514ef568c`.
After preserving and stopping only the completed v7 server container, a fresh
`if-cosmos-dose-v2-presmoke-8003` container (ID prefix `1ab028f2e8ea`) was
started on GPU 2 / port 8003 with the exact snapshot server SHA-256
`a6b02c876f617ef87eaa8508e2a8bcb95c48e926c5f5b58b5e3662740930771e`.
Its startup log showed `registry_entries=0`, and an 87-file checkpoint
verification passed (runtime receipt SHA-256
`13192812d77b333b6e27c3bf5c169f9e01f0a05a3ed1823da46a2686f60b4418`).

The authorized excluded smoke completed and its atomic artifact is
`cosmos3_future_strength_dose_v2_pre_smoke/smoke/excluded_state.json`
(SHA-256
`933569c399deed28b3b8b3e2505376769bd5c3fbb73883f5bc9eb4a84afa9ea1`,
mode 0444). The frozen control-only validator returned PASS in
`smoke/controls.json` (SHA-256
`797a1d4cef08c330ba6536a45a342d57be721ffbf36eab8bf93cb00476815050`,
mode 0444): exactly 92 finite `[32,8]` actions, 80 active responses and
320 active sites, zero live input-clamp error, zero returned-velocity error,
84 action-nonwrite passes, exact replay/no-op/alpha-zero-routing counts, and
one input fingerprint and parameter probe. It reports no scientific outcome.

The presmoke container was then stopped and preserved. A second fresh server,
`if-cosmos-dose-v2-eval-8003` (ID prefix `cba98ed0dc87`), started from the
same image and byte-identical snapshot on GPU 2 / port 8003 with an empty
registry and zero websocket connections. A new full checkpoint verification
receipt has SHA-256
`f3615a06511d2f8259d22887c7752aff8545a0618b3c213810a82a689cf4a8e1`;
the post-smoke empty-registry receipt has SHA-256
`5d07e1faa1d8a026ce0335f31412298d3b76797fa311e0f8a7f01ac6a9a0dd8f`.
The evaluation-ready manifest is `cosmos3-dose-c00d81db8d910603`, SHA-256
`1a37bdba796b580c905970c441aededccf4193c51fe370f0f3557eccd5a9ee7d`,
under `results/overnight_2026_09_03/cosmos3_future_strength_dose_v2`.
There are zero powered output files and zero powered requests; launch remains
held for independent final GO.

At 2026-09-04T03:16:32Z, after signed independent GO audit SHA-256
`742665605e2a3ceb9ee45b8ef3bd09a3fa6501c482abddd3d8d0078aae9792d8`,
the unchanged 30-state / 2,760-call powered cohort was launched on the fresh
GPU 2 server. The sequential launcher host PID is `169798`; its log is
`cosmos3_future_strength_dose_v2/logs/launcher.log`, and atomic state outputs
are written under `cosmos3_future_strength_dose_v2/evaluation/states`.
Partial payloads remain unopened: monitoring is restricted to process state,
completed-file count, and fatal/error-pattern count until exact 30/30
completion and packaging.

The powered launcher exited cleanly at 2026-09-04T03:56:48Z with exact
`assigned=30`, `completed=30`, and `resume_skipped=0`; its log contained zero
fatal/error patterns. The frozen completion packager (SHA-256
`b9a11108958ceb2a79578cc9b07afe5692679ce87683066285da6820752c3cc3`)
then admitted the exact 30-file manifest-derived set, changed every raw output
to mode 0444, and verified each post-mode hash. The resulting inventory is
`cosmos3_future_strength_dose_v2/packaging/output_inventory.json`, SHA-256
`8fa279868725119522bbcf1ecf5c3a3375833aadf3110b63d89a099720a600d8`.

Only after that packaging gate passed, the frozen analyzer (SHA-256
`6b7a642a7687f2176c17b4cf2050da6be3de01361540b58179c25c0b83d68f86`)
processed all 30 states with 10,000 hierarchical bootstrap draws and seed
20260903. Its mode-0444 summary is
`cosmos3_future_strength_dose_v2/evaluation/analysis/summary.json`, SHA-256
`741307dff8ffac6458a65678db1f713aee3d74eb2b4ed11436d8345ad4c44a69`.
The sealed package was handed to the independent raw auditor before any claim
interpretation or presentation rendering.

The independent raw recomputation then passed with zero discrepancies. Its
mode-0444 JSON is
`cosmos3_future_strength_dose_v2/evaluation/audit/independent_raw_audit.json`,
SHA-256
`c12cb8080df4af4f179a231cdd040936f9d693b8dd6c4b3bdb08be52f57c11ed`;
the claim-safe audit note has SHA-256
`c047cbc20f6daed6587f6bb66d10601cbd0cc3115358b28637029125572299c0`.

After that PASS, presentation-only renderer SHA-256
`440ca76a6ade51117f46b403dfd92802a71d01ce69ff18e0b1278cede3adaf37`
generated the read-only `presentation_v1` bundle from the audited summary.
The PNG was visually inspected for legibility, complete axes and intervals,
and clipping. The complete 76-file NFS root was then merged into the existing
local `output/overnight_2026-09-03/cosmos3_future_strength_dose_v2` directory:
local and remote file counts match, checksum comparison found no content
differences, and all 30 raw files independently matched the frozen inventory.
