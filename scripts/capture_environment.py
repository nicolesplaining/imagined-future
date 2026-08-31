"""Write a machine-readable environment record for every experiment run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _command(*args: str) -> str | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _torch_record() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    record: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    try:
        record["cuda_available"] = torch.cuda.is_available()
        record["device_count"] = torch.cuda.device_count()
        record["devices"] = (
            [torch.cuda.get_device_name(index) for index in range(record["device_count"])]
            if record["cuda_available"]
            else []
        )
        if record["cuda_available"]:
            torch.cuda.init()
            record["cuda_initialized"] = True
    except RuntimeError as exc:
        record["cuda_initialized"] = False
        record["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cosmos-dir", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "project_commit": _command("git", "-C", str(project_root), "rev-parse", "HEAD"),
        "project_dirty": bool(_command("git", "-C", str(project_root), "status", "--porcelain")),
        "cosmos_commit": (
            _command("git", "-C", str(args.cosmos_dir), "rev-parse", "HEAD") if args.cosmos_dir else None
        ),
        "container_image": os.environ.get("IMAGINED_FUTURE_CONTAINER_IMAGE"),
        "container_digest": os.environ.get("IMAGINED_FUTURE_CONTAINER_DIGEST"),
        "nvidia_smi": _command("nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader"),
        "torch": _torch_record(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
