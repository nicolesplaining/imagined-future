#!/usr/bin/env python3
"""Excluded development-state parity audit against official ``VA_Server._infer``.

This is not an intervention or an evaluation-state analysis.  It reruns one
predetermined development branch through the unmodified upstream inference
loop while injecting the exact frozen initial video and action noise tensors at
the two ``torch.randn`` call sites.  The receipt distinguishes this controlled
RNG injection from an uninstrumented upstream call.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
EXPECTED_CORE_RUNNER = "902d749772eadef213988c1b30ddb1ef2182da7e2f3852dc6ba596d05bf09fc2"
EXPECTED_CHECKPOINT_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"
EXPECTED_GATE_SHA256 = "6437f774088084b67e6aea001376304dfe01ee622358358dd40642d49d0a67d5"
EXPECTED_ENVIRONMENT_ADDENDUM_SHA256 = "339b1c50892445ea2a8869bb5e96c3ac51d7e11f6187682bd2266cfc118b8bda"
DEFAULT_STATE_ID = "dev_task00_state000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--controlled-result-root", type=Path, required=True)
    parser.add_argument("--parity-gate", type=Path, required=True)
    parser.add_argument("--environment-addendum", type=Path, required=True)
    parser.add_argument("--core-provenance-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-id", default=DEFAULT_STATE_ID)
    parser.add_argument("--branch-index", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_current_checkpoint(
    checkpoint: Path, inventory: dict[str, Any]
) -> str:
    expected_paths = [str(item["path"]) for item in inventory.get("files", [])]
    actual_paths = sorted(
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts
    )
    if actual_paths != expected_paths:
        raise RuntimeError("current checkpoint payload set differs from frozen inventory")
    aggregate = hashlib.sha256()
    for item in inventory["files"]:
        path = checkpoint / item["path"]
        if path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"current checkpoint size mismatch: {item['path']}")
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"current checkpoint hash mismatch: {item['path']}")
        aggregate.update(str(item["path"]).encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(item["bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    observed = aggregate.hexdigest()
    if observed != inventory.get("aggregate_sha256"):
        raise RuntimeError("current checkpoint aggregate differs from frozen inventory")
    return observed


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def max_abs(left: torch.Tensor | np.ndarray, right: torch.Tensor | np.ndarray) -> float:
    left_array = (
        left.detach().cpu().float().numpy()
        if isinstance(left, torch.Tensor)
        else np.asarray(left, dtype=np.float64)
    )
    right_array = (
        right.detach().cpu().float().numpy()
        if isinstance(right, torch.Tensor)
        else np.asarray(right, dtype=np.float64)
    )
    return float(np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))))


def main() -> None:
    args = parse_args()
    if not 0 <= args.branch_index < 4:
        raise ValueError("branch-index must be in [0, 3]")
    lingbot_root = args.lingbot_root.resolve()
    checkpoint = args.checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    controlled_root = args.controlled_result_root.resolve()
    gate_path = args.parity_gate.resolve()
    environment_addendum_path = args.environment_addendum.resolve()
    core_provenance_path = args.core_provenance_receipt.resolve()
    output_root = args.output_root.resolve()
    if sha256_file(gate_path) != EXPECTED_GATE_SHA256:
        raise RuntimeError("upstream parity gate differs from the outcome-blind freeze")
    if sha256_file(environment_addendum_path) != EXPECTED_ENVIRONMENT_ADDENDUM_SHA256:
        raise RuntimeError("parity environment addendum differs from the frozen version")
    environment_addendum = json.loads(environment_addendum_path.read_text())
    required_environment = environment_addendum["required_environment"]
    if os.environ.get("PYTHONPATH") != required_environment["PYTHONPATH"]:
        raise RuntimeError("oracle PYTHONPATH differs from the frozen core compatibility path")
    compatibility_shim = Path(required_environment["shim_path"]).resolve()
    if sha256_file(compatibility_shim) != required_environment["shim_sha256"]:
        raise RuntimeError("FlashAttention import-only compatibility shim changed")
    gate = json.loads(gate_path.read_text())
    identity = gate["canonical_identity"]
    for label, actual in (
        ("manifest_path", manifest_path),
        ("core_result_root", controlled_root),
        ("lingbot_repo_path", lingbot_root),
        ("checkpoint_path", checkpoint),
        ("core_environment_provenance_path", core_provenance_path),
    ):
        if Path(identity[label]).resolve() != actual:
            raise RuntimeError(f"runtime {label} differs from frozen parity gate")
    if args.state_id != identity["development_state_id"]:
        raise RuntimeError("runtime development state differs from frozen parity gate")
    if args.branch_index != identity["branch_index"]:
        raise RuntimeError("runtime branch differs from frozen parity gate")
    if sha256_file(manifest_path) != identity["manifest_sha256"]:
        raise RuntimeError("manifest hash differs from frozen parity gate")
    commit = subprocess.run(
        ["git", "-C", str(lingbot_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"LingBot checkout {commit} != {EXPECTED_COMMIT}")
    official_source = lingbot_root / "wan_va/wan_va_server.py"
    if sha256_file(official_source) != identity["official_source_sha256"]:
        raise RuntimeError("official VA_Server source differs from frozen parity gate")
    core_provenance = json.loads(core_provenance_path.read_text())
    if core_provenance["repository"]["commit"] != EXPECTED_COMMIT:
        raise RuntimeError("core provenance repository commit mismatch")
    if Path(core_provenance["checkpoint"]["path"]).resolve() != checkpoint:
        raise RuntimeError("core provenance checkpoint path mismatch")
    if core_provenance["checkpoint"]["huggingface_revision"] != EXPECTED_CHECKPOINT_REVISION:
        raise RuntimeError("core provenance checkpoint revision mismatch")
    if core_provenance["frozen_design"]["manifest_sha256"] != identity["manifest_sha256"]:
        raise RuntimeError("core provenance manifest mismatch")
    if core_provenance["intervention"]["live_runner_sha256"] != EXPECTED_CORE_RUNNER:
        raise RuntimeError("core provenance runner mismatch")
    core_runner_path = Path(identity["core_runner_path"]).resolve()
    if (
        Path(core_provenance["intervention"]["live_runner_path"]).resolve()
        != core_runner_path
        or sha256_file(core_runner_path) != EXPECTED_CORE_RUNNER
    ):
        raise RuntimeError("live core-runner path/content differs from frozen gate")
    checkpoint_inventory_path = (
        core_provenance_path.parent
        / core_provenance["checkpoint"]["content_inventory_file"]
    )
    checkpoint_inventory = json.loads(checkpoint_inventory_path.read_text())
    if (
        checkpoint_inventory["aggregate_sha256"]
        != core_provenance["checkpoint"]["aggregate_sha256"]
        or checkpoint_inventory["huggingface_revisions"]
        != [EXPECTED_CHECKPOINT_REVISION]
        or Path(checkpoint_inventory["checkpoint_root"]).resolve() != checkpoint
    ):
        raise RuntimeError("checkpoint content inventory differs from core provenance")
    current_checkpoint_aggregate = verify_current_checkpoint(
        checkpoint, checkpoint_inventory
    )

    manifest = json.loads(manifest_path.read_text())
    records = {
        str(record["state_id"]): record
        for record in manifest["states"]
        if record["admission"] == "development"
    }
    if args.state_id != DEFAULT_STATE_ID:
        raise RuntimeError(
            f"parity audit is preregistered to {DEFAULT_STATE_ID}, not {args.state_id}"
        )
    record = records.get(args.state_id)
    if record is None:
        raise RuntimeError(f"development state absent from manifest: {args.state_id}")
    if args.branch_index != 0:
        raise RuntimeError("parity audit is preregistered to branch index 0")
    state_root = controlled_root / args.state_id
    controlled_metadata = json.loads((state_root / "result.json").read_text())
    if controlled_metadata.get("status") != "complete":
        raise RuntimeError("controlled development state is not complete")
    if controlled_metadata.get("admission") != "development":
        raise RuntimeError("parity audit source is not a development state")
    if controlled_metadata.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("controlled state and frozen manifest differ")
    controlled_actions_path = state_root / "actions.npz"
    expected_controlled_identity = {
        "runner_sha256": EXPECTED_CORE_RUNNER,
        "upstream_commit": EXPECTED_COMMIT,
        "checkpoint_revision": EXPECTED_CHECKPOINT_REVISION,
        "actions_sha256": sha256_file(controlled_actions_path),
        "prompt": record["prompt"],
    }
    identity_mismatches = {
        key: (controlled_metadata.get(key), expected)
        for key, expected in expected_controlled_identity.items()
        if controlled_metadata.get(key) != expected
    }
    if identity_mismatches:
        raise RuntimeError(
            f"controlled development-state identity mismatch: {identity_mismatches}"
        )

    frozen = torch.load(state_root / "frozen_inputs.pt", map_location="cpu")
    init_latent_cpu = frozen["init_latent"].detach().cpu()
    action_noise_cpu = frozen["action_noises"][args.branch_index].detach().cpu()
    if tensor_hash(action_noise_cpu) != controlled_metadata["action_noise_hashes"][
        args.branch_index
    ]:
        raise RuntimeError("controlled action-noise tensor hash mismatch")
    future_payload = torch.load(
        state_root / f"future_b{args.branch_index}.pt", map_location="cpu"
    )
    controlled_future = future_payload["future"].detach().cpu()
    if future_payload.get("video_seed") != manifest["video_seeds"][args.branch_index]:
        raise RuntimeError("controlled future video seed mismatch")
    if tensor_hash(controlled_future) != controlled_metadata["future_hashes"][
        args.branch_index
    ]:
        raise RuntimeError("controlled future tensor hash mismatch")
    with np.load(controlled_actions_path, allow_pickle=False) as archive:
        controlled_action = np.asarray(
            archive["native_actions"][args.branch_index], dtype=np.float32
        )
    observation_path = Path(record["observation_path"])
    if sha256_file(observation_path) != record["input_sha256"]:
        raise RuntimeError("frozen development observation hash mismatch")
    with np.load(observation_path, allow_pickle=False) as observation:
        obs = {
            "obs": [
                {
                    "observation.images.agentview_rgb": observation[
                        "agentview"
                    ].copy(),
                    "observation.images.eye_in_hand_rgb": observation["wrist"].copy(),
                }
            ]
        }

    sys.path.insert(0, str(lingbot_root))
    sys.path.insert(0, str(lingbot_root / "wan_va"))
    import flash_attn as compatibility_module

    if Path(compatibility_module.__file__).resolve() != compatibility_shim:
        raise RuntimeError("imported flash_attn module is not the frozen compatibility shim")
    shim_call_count = [0]
    original_shim_function = compatibility_module.flash_attn_func

    def audited_shim(*shim_args: Any, **shim_kwargs: Any):
        shim_call_count[0] += 1
        return original_shim_function(*shim_args, **shim_kwargs)

    compatibility_module.flash_attn_func = audited_shim
    # wan_va_server deliberately adds ``wan_va/`` to sys.path and imports its
    # collaborators through the top-level ``modules``/``configs`` names.  Use
    # those exact module identities here too: importing ``wan_va.modules``
    # would execute the same file a second time under a different module name,
    # making class/function identity checks fail even on the torch-SDPA path.
    from configs import VA_CONFIGS
    from distributed.util import init_distributed
    from modules.model import WanAttention, custom_sdpa
    import wan_va.wan_va_server as upstream_module

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    init_distributed(world_size, local_rank, rank)
    config = copy.deepcopy(VA_CONFIGS["libero"])
    if config.video_exec_step != -1:
        raise RuntimeError("parity audit requires the full official video schedule")
    config.wan22_pretrained_model_name_or_path = str(checkpoint)
    config.local_rank = local_rank
    config.rank = rank
    config.world_size = world_size
    config.enable_offload = False
    config.save_root = str(output_root / "upstream_debug")
    server = upstream_module.VA_Server(config)
    attention_modules = [
        module for module in server.transformer.modules() if isinstance(module, WanAttention)
    ]
    if not attention_modules or any(
        module.attn_op is not custom_sdpa for module in attention_modules
    ):
        raise RuntimeError("released oracle did not instantiate only torch SDPA attention")

    # First verify that the official observation encoder reproduces the frozen
    # controlled input after a clean reset.  The generation audit then injects
    # that already-frozen latent so encoder state cannot confound loop parity.
    server._reset(prompt=str(record["prompt"]))
    freshly_encoded = server._encode_obs(obs).detach().cpu()
    encoder_bitwise_equal = torch.equal(freshly_encoded, init_latent_cpu)
    encoder_max_abs_error = max_abs(freshly_encoded, init_latent_cpu)

    server._reset(prompt=str(record["prompt"]))
    video_generator = torch.Generator(device=server.device).manual_seed(
        int(manifest["video_seeds"][args.branch_index])
    )
    video_noise = torch.randn(
        1,
        48,
        server.job_config.frame_chunk_size,
        server.latent_height,
        server.latent_width,
        device=server.device,
        dtype=server.dtype,
        generator=video_generator,
    )
    action_noise = action_noise_cpu.to(device=server.device, dtype=server.dtype)
    queued = [video_noise, action_noise]
    original_randn = upstream_module.torch.randn
    original_encode_obs = server._encode_obs
    call_shapes: list[list[int]] = []
    randn_call_count = 0

    def injected_randn(*shape: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal randn_call_count
        randn_call_count += 1
        if not queued:
            raise RuntimeError(
                f"official _infer made unexpected torch.randn call {randn_call_count}"
            )
        requested_shape = tuple(int(value) for value in shape)
        supplied = queued.pop(0)
        if requested_shape != tuple(supplied.shape):
            raise RuntimeError(
                f"upstream RNG call shape {requested_shape} != frozen {tuple(supplied.shape)}"
            )
        requested_device = torch.device(kwargs.get("device", supplied.device))
        requested_dtype = kwargs.get("dtype", supplied.dtype)
        if requested_device != supplied.device or requested_dtype != supplied.dtype:
            raise RuntimeError("upstream RNG device/dtype differs from frozen tensor")
        call_shapes.append(list(requested_shape))
        return supplied.clone()

    def frozen_encode_obs(_: object) -> torch.Tensor:
        return init_latent_cpu.to(device=server.device, dtype=server.dtype).clone()

    started = time.time()
    try:
        upstream_module.torch.randn = injected_randn
        server._encode_obs = frozen_encode_obs
        upstream_action, upstream_future = server._infer(obs, frame_st_id=0)
    finally:
        upstream_module.torch.randn = original_randn
        server._encode_obs = original_encode_obs
    duration_seconds = time.time() - started
    if queued or randn_call_count != 2 or len(call_shapes) != 2:
        raise RuntimeError(
            "official _infer RNG contract failed: "
            f"calls={randn_call_count}, recorded_shapes={call_shapes}, queued={len(queued)}"
        )

    upstream_action = np.asarray(upstream_action, dtype=np.float32)
    upstream_future = upstream_future.detach().cpu()
    if shim_call_count[0] != 0:
        raise RuntimeError("FlashAttention compatibility shim was called")
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "scientific_role": "excluded development-state implementation parity audit",
        "included_in_evaluation": False,
        "state_id": args.state_id,
        "branch_id": manifest["branch_ids"][args.branch_index],
        "branch_index": args.branch_index,
        "official_entrypoint": "wan_va.wan_va_server.VA_Server._infer",
        "official_entrypoint_source": str(lingbot_root / "wan_va/wan_va_server.py"),
        "official_entrypoint_source_sha256": sha256_file(
            lingbot_root / "wan_va/wan_va_server.py"
        ),
        "upstream_commit": commit,
        "checkpoint_revision": EXPECTED_CHECKPOINT_REVISION,
        "parity_gate": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
        },
        "environment_addendum": {
            "path": str(environment_addendum_path),
            "sha256": sha256_file(environment_addendum_path),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "compatibility_shim_path": str(compatibility_shim),
            "compatibility_shim_sha256": sha256_file(compatibility_shim),
            "shim_call_count": shim_call_count[0],
            "attention_module_count": len(attention_modules),
            "all_attention_modules_use_custom_torch_sdpa": True,
        },
        "core_environment_provenance": {
            "path": str(core_provenance_path),
            "sha256": sha256_file(core_provenance_path),
            "checkpoint_aggregate_sha256": checkpoint_inventory[
                "aggregate_sha256"
            ],
            "current_checkpoint_rehash_sha256": current_checkpoint_aggregate,
            "current_checkpoint_rehash_matches_inventory": True,
            "checkpoint_content_manifest_path": str(checkpoint_inventory_path),
            "checkpoint_content_manifest_sha256": sha256_file(
                checkpoint_inventory_path
            ),
        },
        "controlled_rng_injection": {
            "used": True,
            "reason": "official _infer samples video and action tensors from one implicit global RNG; the controlled run preregisters independent video/action seeds",
            "scope": "only the two torch.randn return tensors at _infer entry",
            "upstream_loop_body_modified": False,
            "torch_randn_call_count": randn_call_count,
            "rng_call_shapes": call_shapes,
            "video_seed": int(manifest["video_seeds"][args.branch_index]),
            "action_seed": int(manifest["action_seeds"][args.branch_index]),
            "video_initial_noise_sha256": tensor_hash(video_noise),
            "action_initial_noise_sha256": tensor_hash(action_noise),
        },
        "frozen_input": {
            "observation_path": str(observation_path.resolve()),
            "observation_sha256": sha256_file(observation_path),
            "init_latent_sha256": tensor_hash(init_latent_cpu),
            "official_encoder_bitwise_equal": encoder_bitwise_equal,
            "official_encoder_max_abs_error": encoder_max_abs_error,
        },
        "comparison": {
            "future_bitwise_equal": torch.equal(upstream_future, controlled_future),
            "future_max_abs_error": max_abs(upstream_future, controlled_future),
            "action_bitwise_equal": bool(np.array_equal(upstream_action, controlled_action)),
            "action_max_abs_error": max_abs(upstream_action, controlled_action),
            "controlled_future_sha256": tensor_hash(controlled_future),
            "upstream_future_sha256": tensor_hash(upstream_future),
            "controlled_action_sha256": hashlib.sha256(
                controlled_action.tobytes()
            ).hexdigest(),
            "upstream_action_sha256": hashlib.sha256(upstream_action.tobytes()).hexdigest(),
        },
        "paths": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "controlled_result_root": str(controlled_root),
            "controlled_runner_sha256": controlled_metadata.get("runner_sha256"),
            "controlled_result_json_sha256": sha256_file(
                state_root / "result.json"
            ),
            "controlled_actions_sha256": sha256_file(controlled_actions_path),
            "controlled_frozen_inputs_sha256": sha256_file(
                state_root / "frozen_inputs.pt"
            ),
            "controlled_future_file_sha256": sha256_file(
                state_root / f"future_b{args.branch_index}.pt"
            ),
            "checkpoint": str(checkpoint),
            "output_root": str(output_root),
        },
        "invocation": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "LOCAL_RANK",
                    "RANK",
                    "WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                )
            },
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(server.device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                server.device
            ).total_memory,
        },
        "duration_seconds": duration_seconds,
        "audit_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    receipt["parity_gate_passed"] = bool(
        receipt["frozen_input"]["official_encoder_bitwise_equal"]
        and receipt["frozen_input"]["official_encoder_max_abs_error"] <= 0.0
        and receipt["comparison"]["future_bitwise_equal"]
        and receipt["comparison"]["future_max_abs_error"] <= 0.0
        and receipt["comparison"]["action_bitwise_equal"]
        and receipt["comparison"]["action_max_abs_error"] <= 0.0
    )
    if not receipt["parity_gate_passed"]:
        receipt["status"] = "failed"
    atomic_json(output_root / "upstream_native_parity.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["parity_gate_passed"]:
        raise RuntimeError("exact upstream native-parity gate failed; dose is blocked")


if __name__ == "__main__":
    main()
