"""Frozen public Cosmos Policy configuration used by LIBERO experiments."""

from __future__ import annotations


def libero_policy_config(task_suite_name: str = "libero_10", *, unnormalize_actions: bool = True):
    """Return the shared deterministic five-step LIBERO inference config."""

    from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

    return PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
        t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
        task_suite_name=task_suite_name,
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=unnormalize_actions,
        chunk_size=16,
        num_open_loop_steps=16,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        deterministic=True,
        randomize_seed=False,
        use_variance_scale=False,
    )
