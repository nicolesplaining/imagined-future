# SOTA external-WAM feasibility check

## Decision

Do not replace the completed FastWAM Optional-IDM replication with Faster-WAM
for the current paper. Faster-WAM is newer and its authors report
state-of-the-art benchmark performance, but its primary released inference path
does not expose the same scientific object: an explicitly generated future
followed by an action conditioned on that future. It is a strong candidate for
a later **future-representation** replication, not a cleaner deadline-time test
of causal influence from a generated future.

## Source audit

- Official paper: [Faster-WAM: Efficient Inference-Time Future Conditioning for
  Robust World Action Models](https://arxiv.org/abs/2608.04404).
- Official code: [hustvl/FasterWAM](https://github.com/hustvl/FasterWAM), audited
  at commit `83667817df0d4f823f39d90700e61ea2f432ac45`.
- Official checkpoint repository:
  [hustvl/FasterWAM](https://huggingface.co/hustvl/FasterWAM), audited at
  revision `6bf9471ced6919a15ab8fded89f7772f5060c44b`. The released LIBERO
  checkpoint is 11,117,679,929 bytes.
- The repository was cloned for read-only inspection at
  `/home/ubuntu/FasterWAM_2608` on the existing external-WAM host. No checkpoint
  or model outcome was generated.
- Relevant pinned source hashes are: LIBERO task configuration
  `526f3ed6194b394a1896e7d32600bae9086f20754159cd5815a8933283695de6`,
  `jointwam.py` `553c7ae58b8e8a8d5f75b10ded3e28d86f69870b9af5a152abd721d0e52b3a30`,
  `sparse_mot.py` `225ebf706cef1bb7f33ada6ca5f6f9c84fc43529c0cccbf24816d4f842e8e89c`,
  and `fasterwam.py`
  `cf85bb59275bc741cec7fd726a755a731be7a1c10620a45b47db5b8d373cc532`.

## Interface finding

The released LIBERO configuration sets
`default_action_infer_mode: one_pass_future_cache`. In
`JointWAM.infer_action_one_pass_future_cache`, the model:

1. initializes a future-video latent from noise and replaces its first latent
   frame with the encoded observation;
2. runs the video expert once at the first video timestep;
3. forms Interval-KV-Fusion caches at eight conditioning layers; and
4. reuses those caches while denoising the action.

That primary path neither completes iterative future-video denoising nor
returns a decoded generated future. The architecture still contains a video
model and future-position representations, but a cache transplant there would
test whether the action follows a one-pass latent representation—not whether a
particular generated future video steers the action.

By contrast, the completed FastWAM Optional-IDM test uses the same released
checkpoint in an explicit imagine-then-act mode: future video is generated
first and its deterministic cache conditions action denoising. Its matched
`first_frame` route also supplies a direct no-future interface control. This is
the closer cross-architecture replication of the paper's stated causal
question, even though it should be described as a recent high-performing
released WAM rather than as the current unqualified SOTA model.

## High-value later experiment

If the scope expands, test Faster-WAM separately and label it as a
future-**representation** intervention:

- freeze four video seeds and four independent action seeds per state;
- record and transplant the fused K/V produced by the one-pass future latent;
- use exact self-cache replay, wrong-source cache, norm-matched/shuffled cache,
  and current-frame-only controls;
- report source retrieval, donor distance reduction, projection, and
  orthogonal residual across untouched LIBERO states;
- do not call the source a generated or imagined video unless a separate
  iterative video-generation path is actually run and decoded.
