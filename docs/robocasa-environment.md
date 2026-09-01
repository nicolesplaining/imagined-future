# RoboCasa environment reconstruction

This replication keeps the official LIBERO container immutable and uses a separate Cosmos Policy checkout and virtual environment for RoboCasa. The commands below describe the recorded environment without machine-specific paths, credentials, or host addresses.

## Pinned sources

```bash
git clone https://github.com/NVlabs/cosmos-policy.git cosmos-policy-robocasa
git -C cosmos-policy-robocasa checkout --detach 18a2accadf4e7a3531e56754102af5a24d2316da

git clone https://github.com/moojink/robocasa-cosmos-policy.git robocasa-cosmos-policy
git -C robocasa-cosmos-policy checkout --detach edd9a328b3ec98050f42d194c1419307a79c4d87
```

The Cosmos virtual environment uses Python 3.10.18 and the `robocasa` dependency group with the CUDA 12.8 extra. The RoboCasa fork is installed editable so its assets remain outside the virtual environment.

```bash
uv sync --extra cu128 --group robocasa --no-group dev
uv pip install --python .venv/bin/python -e /path/to/robocasa-cosmos-policy
```

## Compatibility override

The fork pins NumPy 1.23.3 and Numba 0.56.4, which predate the `numpy.dtypes` API imported by the released Cosmos Policy Megatron runtime. The following compatibility-only override is applied after installing both public projects:

```bash
.venv/bin/python -m pip install --force-reinstall \
  numpy==1.26.4 \
  numba==0.61.2 \
  llvmlite==0.44.0 \
  protobuf==6.33.5 \
  opencv-python==4.11.0.86 \
  opencv-python-headless==4.11.0.86
```

These versions satisfy Megatron Core's NumPy upper bound, retain a Numba version supporting NumPy 1.26, and match Cosmos Policy's OpenCV floor. No source file or checkpoint is patched.

## Assets and model

Run the fork's public setup scripts once, then cache the released checkpoint:

```bash
printf 'y\n' | .venv/bin/python robocasa/scripts/download_kitchen_assets.py
.venv/bin/python robocasa/scripts/setup_macros.py
.venv/bin/python .venv/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
```

The checkpoint is `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`, resolved to Hugging Face revision `4b2a04c80d97202f86127ebec80461e8016ec1dc`. The experiment image digest is `sha256:2f5ff2badf82657c82e046bff84cc754acf4a2b3973828f12ab802c751f439e0`.

## Validation gate

Before collecting branches, the environment must pass both checks:

1. create and render the registered OpenDrawer scene after ten zero-action stabilization steps;
2. load the released checkpoint and reproduce simulator state plus all three rendered observations exactly across three fresh environment reconstructions.

The first frozen unit passed both gates. Its simulator state and observation SHA-256 digests were identical in all three reconstructions.
