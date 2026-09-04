# Lambda filesystem handoff

Heavy generated outputs are stored on the shared Lambda filesystem rather than
in Git history. They are available from either dedicated experiment node.

## Canonical paths

| Contents | Shared path |
|---|---|
| DreamZero and LingBot-VA deadline raw outputs, traces, decoded media, audits, and immutable packages | `/lambda/nfs/imagined-future/results/deadline_2026_09_04/` |
| Prior Cosmos 3 and FastWAM overnight outputs | `/lambda/nfs/imagined-future/results/overnight_2026_09_03/` |
| Final paper-ready documents and deadline receipt | `/lambda/nfs/imagined-future/results/deadline_2026_09_04/final_handoff/` |
| Full local figure-candidate archive | `/lambda/nfs/imagined-future/assets/cover_figure_candidates/` |

At the deadline freeze, the first directory occupied approximately 1.7 GiB
and the prior overnight directory approximately 442 MiB. The figure archive is
copied separately because it is generated media rather than experimental raw
evidence.

## Storage split

Git is canonical for source, tests, protocols, paper assets, compact summaries,
tables, plots, receipts, and content hashes. Lambda is canonical for raw tensor
arrays, exhaustive decoded media, working shards, and the full figure-option
archive. Local copies may remain present even when ignored by Git.

No checkpoint weights, credentials, private keys, or environment secrets are
part of the Git handoff.

## Verification

From a connected experiment node:

```bash
du -sh /lambda/nfs/imagined-future/results/deadline_2026_09_04
du -sh /lambda/nfs/imagined-future/results/overnight_2026_09_03
du -sh /lambda/nfs/imagined-future/assets/cover_figure_candidates
find /lambda/nfs/imagined-future/results/deadline_2026_09_04/final_handoff \
  -maxdepth 1 -type f -print
```

Use the artifact indices and SHA-256 receipts referenced by
[README.md](README.md) to validate individual frozen result packages.
