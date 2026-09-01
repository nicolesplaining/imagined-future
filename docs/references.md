# Public foundations

The implementation extends public code rather than reconstructing Cosmos Policy or LIBERO.

- [Cosmos Policy paper](https://arxiv.org/abs/2601.16163) and [official code](https://github.com/NVlabs/cosmos-policy), pinned to commit `18a2accadf4e7a3531e56754102af5a24d2316da` for the pilot.
- [Cosmos 3 technical report](https://arxiv.org/abs/2606.02800),
  [official Cosmos code](https://github.com/NVIDIA/cosmos), and
  [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework). The DROID
  replication uses the released
  [Cosmos3-Nano-Policy-DROID checkpoint](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)
  and the official [DROID policy server](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_server.md).
- [RoboLab](https://github.com/NVlabs/RoboLab), used for DROID embodiment
  evaluation, exact initial-state restoration, per-step scene-state validation,
  and physical execution of reciprocal branches.
- [LIBERO paper](https://arxiv.org/abs/2306.03310) and [official benchmark code](https://github.com/Lifelong-Robot-Learning/LIBERO).
- [Keep the Future, Drop the Rollout (RIFT)](https://arxiv.org/abs/2608.11521), whose cache masking and reassignment experiments establish future-interface necessity and positional sensitivity. Our transplantation studies ask the distinct content-specific sufficiency question.
- [robosuite's official state-playback implementation](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py), which restores flattened MuJoCo states and calls `sim.forward`; the secondary factorial study reuses this state interface and robosuite's joint-address APIs.
- [RoboCasa paper](https://arxiv.org/abs/2406.02523) and the [Cosmos Policy evaluation fork](https://github.com/moojink/robocasa-cosmos-policy), pinned to commit `edd9a328b3ec98050f42d194c1419307a79c4d87` for the cross-domain replication.
- [Causal mediation analysis for neural networks](https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html), which motivates distinguishing encoded information from information the model uses.
- [Localizing Model Behavior with Path Patching](https://arxiv.org/abs/2304.05969), used to frame future-to-action edge tests.
- [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042), used for metric, corruption, and held-out validation choices.
- [The Curse of Multiple Mediators](https://arxiv.org/abs/2606.27510), used to limit mediation claims and motivate explicit interaction analyses.
- [pyvene](https://arxiv.org/abs/2403.07809), a public intervention library. We use a smaller adapter initially because Cosmos's denoising solver requires interventions across repeated forward passes, but will compare semantics before adopting custom hooks more broadly.

Public upstream code remains under its original license and is installed separately. This repository contains only the intervention and analysis layer specific to the research question.
