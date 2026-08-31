"""Narrow adapters around the public Cosmos Policy inference API."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch

from imagined_future.interventions import X0Function


@contextmanager
def transform_model_x0_factory(model: Any, transform: Callable[[X0Function], X0Function]) -> Iterator[None]:
    """Temporarily transform every denoiser callable created by a Cosmos model.

    This avoids vendoring or editing NVIDIA's implementation. It is deliberately
    not thread-safe; experiments must use one inference worker per model process.
    """

    original = model.get_x0_fn_from_batch

    def patched(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if isinstance(result, tuple):
            x0_fn, *rest = result
            return (transform(x0_fn), *rest)
        return transform(result)

    model.get_x0_fn_from_batch = patched
    try:
        yield
    finally:
        model.get_x0_fn_from_batch = original


def cosmos_initial_noise(
    model: Any,
    data_batch: dict[str, Any],
    *,
    seed: int,
    n_sample: int | None = None,
    state_shape: tuple[int, ...] | None = None,
    sigma_max: float | None = None,
) -> torch.Tensor:
    """Reproduce the public sampler's initial noise so frames can be paired.

    Call this before ``generate_samples_from_batch`` mutates/normalizes the batch,
    then pass the result through its public ``x_sigma_max`` argument.
    """

    from cosmos_policy._src.imaginaire.utils import misc

    is_image_batch = model.is_image_batch(data_batch)
    input_key = model.input_image_key if is_image_batch else model.input_data_key
    video = data_batch[input_key]
    if n_sample is None:
        n_sample = int(video.shape[0])
    if state_shape is None:
        frames, height, width = video.shape[-3:]
        state_shape = (
            int(model.config.state_ch),
            int(model.tokenizer.get_latent_num_frames(frames)),
            int(height // model.tokenizer.spatial_compression_factor),
            int(width // model.tokenizer.spatial_compression_factor),
        )
    maximum = float(model.sde.sigma_max if sigma_max is None else sigma_max)
    return misc.arch_invariant_rand(
        (n_sample,) + tuple(state_shape),
        torch.float32,
        model.tensor_kwargs["device"],
        seed,
    ) * maximum


@contextmanager
def transform_model_initial_noise(
    model: Any,
    transform: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
) -> Iterator[None]:
    """Temporarily transform Cosmos's initial sample while preserving its API.

    The wrapper uses the public ``x_sigma_max`` argument. It refuses variance
    scaling because the experimental design requires common random numbers with
    a fixed, known scale.
    """

    original = model.generate_samples_from_batch

    def patched(data_batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError("paired experiments must call generate_samples_from_batch with keyword arguments")
        if kwargs.get("use_variance_scale", False):
            raise ValueError("initial-noise interventions require use_variance_scale=False")
        if kwargs.get("x_sigma_max") is not None:
            raise ValueError("x_sigma_max was already supplied; refusing to overwrite it")
        initial = cosmos_initial_noise(
            model,
            data_batch,
            seed=int(kwargs.get("seed", 1)),
            n_sample=kwargs.get("n_sample"),
            state_shape=kwargs.get("state_shape"),
            sigma_max=kwargs.get("sigma_max"),
        )
        transformed = transform(initial, data_batch)
        if transformed.shape != initial.shape:
            raise ValueError("initial-noise transform changed the latent shape")
        kwargs["x_sigma_max"] = transformed
        return original(data_batch, **kwargs)

    model.generate_samples_from_batch = patched
    try:
        yield
    finally:
        model.generate_samples_from_batch = original
