# DreamZero future x K/V factorial feasibility audit

**Deadline disposition:** not run. A deadline patch would not isolate future
information cleanly and is therefore excluded from the scientific results.

DreamZero's object named `kv_cache` is not a cache of the currently generated
future. It is populated from past observed/reference frames with `action=None`
in `wan_flow_matching_action_tf.py:1120-1194`. During each of the 16 current
denoising steps, the future and action latents enter together with cache updates
disabled (`wan_flow_matching_action_tf.py:1539-1559`).

The relevant current-future interface is
`wan_video_dit_action_casual_chunk.py:1041-1080`. Current video K/V is joined
with past-context K/V, and one shared attention call serves video, action, and
state queries. The existing intervention patch changes only the raw future
latent trajectory; the attention module remains byte-identical to the pinned
official commit.

A valid action-query-only experiment must:

1. split the 24 true action queries from video queries and the one state query;
2. avoid directly patching video and state query rows, while allowing them to
   evolve natively within the factorial cell after ordinary action-to-video
   feedback;
3. give only action queries donor current-future K/V while preserving past
   observation K/V and current recipient action/state K/V; and
4. repeat this at every one of 40 blocks and 16 solver steps with exact same-run
   replay and no-write controls.

Native donor K/V cannot simply be replayed from a donor run: after the first
block, donor video hidden states already contain feedback from donor action and
state tokens. Such replay would leak the donor action trajectory. A clean design
requires a paired donor-video side stream conditioned on the same evolving
recipient/hybrid action and state stream.

Measured core throughput was 2.866 seconds per ordinary call. A 30-state binary
2x2 contains 120 calls; paired-side-stream and native/hybrid attention overhead
make roughly 12–20 minutes on the loaded 2xH100 server a more realistic budget
than a single fixed estimate. The implementation, static audit, excluded-state
identity gates, and smoke validation are estimated at 3-6 engineering hours.
Full raw K/V persistence would be impractical (about 21.5 GiB per source for
one CFG branch, or about 43 GiB for both), so a future run should hash tensors
online and discard their bytes.

This is why the deadline handoff retains Cosmos 3 as the independently crossed
future x K/V mechanism result and treats DreamZero as action-level replication.
