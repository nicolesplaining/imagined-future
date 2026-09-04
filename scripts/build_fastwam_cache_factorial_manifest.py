#!/usr/bin/env python3
"""Freeze the additive FastWAM future x video-cache factorial manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from imagined_future.fastwam_cache_factorial import (
    build_cache_factorial_body,
    file_sha256,
    freeze_cache_factorial_manifest,
    write_cache_factorial_manifest,
)
from imagined_future.fastwam_optional_idm import load_frozen_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    base = load_frozen_manifest(args.base_manifest)
    base_sha = file_sha256(args.base_manifest)
    if base["manifest_id"] != config["base_manifest_id"]:
        raise ValueError("config base manifest ID does not match the supplied manifest")
    if base_sha != config["base_manifest_sha256"]:
        raise ValueError("config base manifest SHA-256 does not match")
    body = build_cache_factorial_body(
        study_name=str(config["study_name"]),
        base_manifest=base,
        base_manifest_sha256=base_sha,
        design=config["design"],
    )
    manifest = freeze_cache_factorial_manifest(body)
    write_cache_factorial_manifest(args.output, manifest)
    print(f"{manifest['manifest_id']} {args.output}")


if __name__ == "__main__":
    main()

