# Prospective Cosmos 3 single-call timing protocol v2: structural-null amendment

Status: **prospective and inactive.** This clarification was written after the
first excluded-development smoke failed closed, but before any admitted timing
evaluation call or outcome. The failed smoke produced no state artifact. Its
snapshot and manifests remain permanently failed and non-admitted.

This amendment changes no population, timing condition, request count,
intervention, estimand, bootstrap, multiplicity procedure, evidence criterion,
or claim boundary in the v2 protocol. It narrows only what “all arrays and audit
scalars are finite” means for one optional server diagnostic whose denominator
is structurally zero on diagonal recipient/source calls.

## Exact structural-null rule

The server diagnostic `research_action_donor_projection` is

```text
dot(action - recipient_action, donor_action - recipient_action)
----------------------------------------------------------------
       ||donor_action - recipient_action|| squared
```

It is not a protocol estimand or model input. When the registered recipient and
donor IDs are the same, its denominator is identically zero and the diagnostic
is undefined. The superseding implementation must encode this case as JSON
`null`, never IEEE NaN or infinity. It must accept `null` for this field if and
only if all of the following hold:

1. `research_recipient_id == research_donor_id`;
2. the row is one of the frozen diagonal recipient/source calls or its frozen
   all-calls diagonal replay;
3. the stored recipient and donor future hashes are equal; and
4. the recipient and donor native-action identities are the same frozen branch.

The exact per-state census is:

| response class | field state | count |
|---|---:|---:|
| Six timing grids, diagonal cells | `null` | 24 |
| Extra all-calls diagonal replays | `null` | 4 |
| Six timing grids, off-diagonal cells | finite | 72 |
| Native plus native-replay responses | field absent | 8 |
| **Total** | 28 null, 72 finite, 8 absent | **108** |

Every off-diagonal occurrence must be present and finite. The twelve
off-diagonal `none` calls must additionally equal exactly zero because the full
no-op action equals the recipient action. An off-diagonal zero denominator,
missing diagnostic, null at any other numeric diagnostic path, NaN, or infinity
is a gate failure; it must not be converted to a dropped row.

The unrelated optional metadata path
`research_attention_interface.cache_id` may remain JSON `null` when no
attention-cache request is active. This is nonnumeric routing metadata. No
other null exemption is created by this amendment.

## Fields that remain strictly finite

The following remain finite and schema-required on every applicable response:

- all action arrays and all native-action separations;
- the full sigma/x0 schedule, requested and observed timing indices, and active
  and inactive call telemetry;
- live model-input clamp and returned-future-velocity errors;
- action-input/action-output nonwrite errors and inactive-write counts;
- final-sampler target residuals, which remain descriptive and non-admissive;
- all off-diagonal matched retrieval/distance gains, projections, cosine and
  residual metrics, bootstrap draws, intervals, contrasts, and p-values; and
- every numeric field used by any runtime or evidence gate.

The runner and analyzer must implement a path-aware schema validator rather
than recursively skipping arbitrary null/nonfinite values. Their output gate
must report the exact absent/null/finite census above and explicitly distinguish
`required_numeric_fields_finite` from `structural_null_census_exact`.

## Required supersession procedure

The superseding content-addressed manifest must hash this amendment and record:

- failed evaluation manifest `cosmos3-timing-1a1e2733084791f0`, SHA-256
  `aeaebdf3e8ebb2acd81dfae3d083ebe366045fa62fd0dca943bb7d056b95fa77`;
- failed smoke manifest `cosmos3-timing-excluded-smoke-8247bc17a53b432a`,
  SHA-256 `86a9f91f1939776264ac4166ebe12d1a400c2ddc58215d88ffa1167cc34d7318`;
- failed snapshot checksum-list SHA-256
  `87888ce836c87a515df92943e0a588aef60bd2efb8619d170cfec7dbcab495f2`;
- failure before atomic state output and permanent non-admission of the entire
  failed root; and
- a fresh versioned root, final code hashes, new empty-registry server, and new
  excluded-state 108-call smoke before evaluation launch.

The fresh smoke must exercise and verify all 28 structural-null, 72 finite, and
8 absent cases in addition to every original v2 checklist item. Any different
null path or count requires another prospective amendment and fresh smoke.
