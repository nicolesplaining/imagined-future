# Cosmos 3 single-call timing v3: outcome-blind checklist amendment

This checklist incorporates the v2 checklist with SHA-256
`288472c4f8ea914333916b6bff68c9777b4d07eb892646e70b0ebfa3749d7012`.
Every v2 item remains mandatory. Before GO, additionally verify:

- [ ] The v2 root, snapshot, manifest, and failed excluded smoke are preserved
  and explicitly non-admitted; v3 uses a new root, snapshot, manifest ID, empty
  registry, and complete excluded smoke.
- [ ] Raw `research_action_donor_projection` is NaN only when the named
  recipient/source IDs are identical and the native action axis is zero; the
  runner converts only that exact field to JSON null and adds
  `research_action_donor_projection_applicable=false`.
- [ ] Every off-diagonal projection is finite and has applicability true; all
  other nulls, NaNs, and infinities fail.
- [ ] The complete 108-call smoke has exactly 28 structural-null intervention
  diagnostics, 72 finite off-diagonal diagnostics, and eight native/replay
  responses without the field.
- [ ] All scientific actions and metrics, residuals, timing/site captures,
  masks, RNG/target hashes, and action-coordinate nonwrite audits remain
  finite-required with no skipped path.

The v3 launch remains **NO-GO** until an independent auditor signs this
amendment and every incorporated v2 checklist item against the final v3
snapshot, manifest, and excluded smoke.
