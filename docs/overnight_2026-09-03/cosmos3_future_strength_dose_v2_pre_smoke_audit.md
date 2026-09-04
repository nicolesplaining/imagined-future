# Cosmos future-strength dose v2: outcome-blind pre-smoke audit

Verdict: **GO_SMOKE**, conditional on the five runtime conditions recorded in the
companion JSON. This authorizes exactly one excluded Bagels state and 92 calls. It
does not authorize a powered call.

The independently verified closure is the immutable 18-file snapshot list at
SHA-256
`126d387ccd39dc0c6e23beefac462a80670266d647463c77622da7ec5cb6a100`.
Every listed file matches its digest and mode `0444`, all snapshot directories are
mode `0555`, and the snapshot has no missing, extra, symlink, or bytecode file. The
powered pre-smoke manifest is `cosmos3-dose-ecae4c40bd3437ca`, SHA-256
`f256829232223376016e433d427795246e263e175059ec2c1ed992e9a72db394`.
The only manifest authorized here is the excluded smoke manifest
`cosmos3-dose-excluded-smoke-832b41a3b67fc689`, SHA-256
`8b0bfd7968cf8df4275aaf835ccf7362e65bd797008de5802de928b5d85cd82a`.

The manifest contains exactly 30 selection-free archival-v7 middle states (six
tasks by five episodes), exactly matching the parent manifest, while the smoke is a
single nonadmitted Bagels state. The frozen per-state matrix is exactly
`4 native + 4 native replay + 4 none + 4 self + 4 self replay + 12x5 dose +
12 midpoint replay = 92`; labels are unique and ordered, and the powered total is
2,760. The action schema is the full `[32,8]` chunk (256 finite coordinates).

The original protocol and prospective amendment match SHA-256 values
`7b559902b640610d88f54b1953151249118ad6d162aa6389ac3cbcaca01cb0cc`
and
`a02e3a74d0d5b7f4a9d72401c8f869519acf7a4f808067ee2a2180e68775158f`.
The implementation uses recipient seed, initial state, and path noise at every
alpha; only the eight future latent frames are actively clamped. Alpha 0 and 1 use
direct registered recipient/donor endpoints, and interior alphas use the frozen
native-dtype `F_A + alpha*(F_B-F_A)` operation. Actual model-input and returned-
velocity sites, nonfuture coordinates, endpoint hashes, source IDs, schedule,
action nonwrites, and formula error all fail closed. Alpha-zero actions and behavior
signatures must equal explicit self and remain invariant across donor labels;
12 midpoint repeats must be exact.

The analyzer independently rebuilds all action metrics from the raw `[32,8]`
arrays. Its primary is the equal mean of 12 pairwise OLS-with-intercept slopes per
state, with the exact denominator `0.625`. It freezes one shared 10,000-draw
PCG64(20260903) task-to-five-state table for all estimands, linear percentile
intervals, six task means, and LOTO sensitivities. Only a primary lower bound above
zero licenses “positive linear dose trend.” A strict adjacent-step statement
requires all four hierarchical lower bounds above zero; no null permits equivalence,
necessity, or no-effect language.

Independent compilation and formula checks passed for all 16 Python files. The
frozen ten-test suite was reported passing in the exact image; its assertions were
also manually re-executed for the request census, endpoint interpolation, action
shape, OLS constants, deterministic seed order, and the first shared bootstrap draw.

The smoke must run only after a fresh manifest-pinned image/container starts the
dedicated dose server on GPU 2/port 8003 with `registry_entries=0`, followed by the
full checkpoint verifier. A failed smoke or any code/schema change requires a new
version. A new content-addressed final manifest, fresh empty-registry restart, and a
separate outcome-blind GO are mandatory before powered evaluation.
