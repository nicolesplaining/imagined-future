# Cosmos 3 future-strength dose v2: pre-smoke audit handoff

Status: **frozen, no model call, hold for independent pre-smoke GO**.

This package implements the prospective dose-response protocol without opening
the archival-v7 scientific payload. It uses all and only the 30 frozen
middle-phase archival states (six tasks x five episodes), four branch seeds,
all 12 ordered off-diagonal pairs, and five alphas. The exact matrix is 92
calls per state (2,760 powered calls) and the excluded Bagels smoke uses the
same 92-call matrix.

## Frozen design inputs

- Original protocol SHA-256:
  `7b559902b640610d88f54b1953151249118ad6d162aa6389ac3cbcaca01cb0cc`
- Prospective v2 amendment SHA-256:
  `a02e3a74d0d5b7f4a9d72401c8f869519acf7a4f808067ee2a2180e68775158f`
- Parent archival-v7 manifest SHA-256:
  `8ab294c7a581424cfa02faae1db85941e3f43c1e28100e68e21fa5f533703b9e`
- Checkpoint content-manifest SHA-256:
  `b6cbee5522104145f71cfcc74f5049f61e7de3640b88f14831909c7f473c2660`
- Checkpoint verification receipt SHA-256:
  `905f40239ae56db2f7d83de3e9002a18f3f1ec8ad454f7d254ef5ca509e8fabe`
- Cosmos image digest:
  `sha256:95a8933deb69f2fc73697d846f7e17066a4d4e05dc9ef85718a7ee12cfea2a8c`
- Expected sampled-parameter probe:
  `21b79382b84b4bdebb943a2659c0272c99267ef433d83818f9a44b742c1170cc`

## Immutable pre-smoke package

- NFS root:
  `/lambda/nfs/imagined-future/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2_pre_smoke`
- Container-visible root:
  `/research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2_pre_smoke`
- Snapshot: 18 exact files, every file mode 0444 and directory mode 0555.
- Snapshot checksum-list SHA-256:
  `126d387ccd39dc0c6e23beefac462a80670266d647463c77622da7ec5cb6a100`
- Powered pre-smoke manifest ID: `cosmos3-dose-ecae4c40bd3437ca`
- Powered pre-smoke manifest SHA-256:
  `f256829232223376016e433d427795246e263e175059ec2c1ed992e9a72db394`
- Excluded smoke manifest ID:
  `cosmos3-dose-excluded-smoke-832b41a3b67fc689`
- Excluded smoke manifest SHA-256:
  `8b0bfd7968cf8df4275aaf835ccf7362e65bd797008de5802de928b5d85cd82a`

## Static evidence

- `py_compile` passed for all dose modules, builders, runner, launcher,
  analyzer, validator, and tests.
- Pyflakes passed with no findings.
- Ten focused tests passed locally and again in the exact Cosmos image Python
  environment from the frozen snapshot.
- All five executable entry points import and render `--help` in the exact
  Cosmos image environment.
- Frozen manifest checks passed: 30 middle-phase states, 92 ordered requests
  per state, 2,760 powered requests, exact `[32, 8]` / 256-coordinate action
  schema, both protocol hashes, and one nonoverlapping excluded Bagels state
  with 92 requests.
- Snapshot closure has no symlink, bytecode, or unlisted file.
- No dose server was launched and no dose model request was made.

## Intended excluded-smoke topology after GO

After archival-v7 raw audit completes, stop only its now-idle GPU-2 server and
create a fresh isolated container named `if-cosmos-dose-v2-presmoke-8003` on
GPU 2 / host port 8003. Mount the pinned Cosmos checkout and checkpoint
read-only, mount NFS at `/research`, set
`PYTHONDONTWRITEBYTECODE=1`, and set `PYTHONPATH` to the frozen dose snapshot
followed by `/source`. The entry point is the frozen
`run_cosmos3_future_strength_dose_server.py` with checkpoint `/checkpoint`,
port 8003, seed 0, and registry limit 4096. Require the startup receipt to show
the exact image/entrypoint and `registry_entries=0`; reverify the complete
checkpoint before the excluded call. The smoke runner must receive an
independently signed audit artifact authorizing exactly the excluded manifest
and 92 calls. No powered call is authorized by this pre-smoke package.

