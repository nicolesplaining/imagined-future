from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from imagined_future.model_patch import (
    defer_hf_config_checkpoint_resolution,
    transform_model_initial_noise,
    transform_model_x0_factory,
)


class FakeModel:
    def __init__(self) -> None:
        self.seen_x_sigma_max = None

    def get_x0_fn_from_batch(self, _batch, *, return_orig_clean_latent_frames=False):
        fn = lambda x, _sigma: x + 1
        return (fn, "frames") if return_orig_clean_latent_frames else fn

    def generate_samples_from_batch(self, _batch, **kwargs):
        self.seen_x_sigma_max = kwargs["x_sigma_max"]
        return self.seen_x_sigma_max


def test_x0_factory_transform_preserves_auxiliary_return() -> None:
    model = FakeModel()

    with transform_model_x0_factory(model, lambda fn: lambda x, sigma: fn(x, sigma) * 2):
        fn, frames = model.get_x0_fn_from_batch({}, return_orig_clean_latent_frames=True)
        result = fn(torch.tensor(2.0), torch.tensor(1.0))

    assert result.item() == 6.0
    assert frames == "frames"
    assert model.get_x0_fn_from_batch({}, return_orig_clean_latent_frames=False)(torch.tensor(2.0), None).item() == 3.0


def test_initial_noise_transform_uses_public_argument(monkeypatch) -> None:
    model = FakeModel()
    monkeypatch.setattr(
        "imagined_future.model_patch.cosmos_initial_noise",
        lambda *_args, **_kwargs: torch.zeros(1, 1, 2, 1, 1),
    )

    with transform_model_initial_noise(model, lambda noise, _batch: noise + 4):
        result = model.generate_samples_from_batch({}, seed=3, use_variance_scale=False)

    assert torch.all(result == 4)


def test_initial_noise_transform_rejects_variance_scaling(monkeypatch) -> None:
    model = FakeModel()
    monkeypatch.setattr(
        "imagined_future.model_patch.cosmos_initial_noise",
        lambda *_args, **_kwargs: torch.zeros(1, 1, 2, 1, 1),
    )
    with transform_model_initial_noise(model, lambda noise, _batch: noise):
        with pytest.raises(ValueError, match="variance_scale"):
            model.generate_samples_from_batch({}, use_variance_scale=True)


def test_defer_hf_config_checkpoint_resolution_is_narrow_and_restored() -> None:
    calls: list[str] = []

    def resolve(path: str) -> str:
        calls.append(path)
        return f"resolved:{path}"

    checkpoint_db = SimpleNamespace(get_checkpoint_path=resolve)
    with defer_hf_config_checkpoint_resolution(checkpoint_db):
        assert checkpoint_db.get_checkpoint_path("hf://org/model/base.pt/") == "hf://org/model/base.pt"
        assert checkpoint_db.get_checkpoint_path("/local/model") == "resolved:/local/model"

    assert checkpoint_db.get_checkpoint_path is resolve
    assert calls == ["/local/model"]
