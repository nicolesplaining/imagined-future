"""Architecture-specific interventions for public Cosmos 3 policy inference.

Cosmos 3 samples a flattened rectified-flow state in the order
``[vision | lidar | action | sound]``.  These helpers recover that layout from
the prepared public inference objects and intervene on future vision
coordinates at the sampler boundary, after text classifier-free guidance has
been combined.  The action coordinates passed to and returned by the wrapped
velocity function are never replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

Tensor = torch.Tensor
VelocityFunction = Callable[[list[Tensor], Tensor], list[Tensor]]


@dataclass(frozen=True)
class FlatModalitySlice:
    """Named half-open range in one sample's flattened diffusion state."""

    name: str
    start: int
    stop: int
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class Cosmos3FlatLayout:
    """Per-sample slices matching ``OmniMoTModel._get_velocity`` exactly."""

    samples: tuple[tuple[FlatModalitySlice, ...], ...]

    @classmethod
    def from_prepared(cls, sequence_plans: Sequence[Any], gen_data_clean: Any) -> "Cosmos3FlatLayout":
        vision_items = gen_data_clean.num_vision_items_per_sample
        lidar_items = gen_data_clean.num_lidar_items_per_sample
        has_lidar = gen_data_clean.x0_tokens_lidar is not None
        has_actions = gen_data_clean.x0_tokens_action is not None
        has_sound = gen_data_clean.x0_tokens_sound is not None
        vision_offset = lidar_offset = action_offset = sound_offset = 0
        sample_layouts: list[tuple[FlatModalitySlice, ...]] = []

        for sample_index, plan in enumerate(sequence_plans):
            offset = 0
            slices: list[FlatModalitySlice] = []
            number_vision = int(vision_items[sample_index]) if vision_items is not None else 1
            for item_index in range(number_vision):
                shape = tuple(int(x) for x in gen_data_clean.x0_tokens_vision[vision_offset + item_index].shape)
                size = int(torch.tensor(shape).prod().item())
                slices.append(FlatModalitySlice(f"vision_{item_index}", offset, offset + size, shape))
                offset += size
            vision_offset += number_vision

            if has_lidar and plan.has_lidar:
                number_lidar = int(lidar_items[sample_index]) if lidar_items is not None else 1
                for item_index in range(number_lidar):
                    shape = tuple(int(x) for x in gen_data_clean.x0_tokens_lidar[lidar_offset + item_index].shape)
                    size = int(torch.tensor(shape).prod().item())
                    slices.append(FlatModalitySlice(f"lidar_{item_index}", offset, offset + size, shape))
                    offset += size
                lidar_offset += number_lidar

            if has_actions and plan.has_action:
                shape = tuple(int(x) for x in gen_data_clean.x0_tokens_action[action_offset].shape)
                size = int(torch.tensor(shape).prod().item())
                slices.append(FlatModalitySlice("action", offset, offset + size, shape))
                offset += size
                action_offset += 1

            if has_sound and plan.has_sound:
                shape = tuple(int(x) for x in gen_data_clean.x0_tokens_sound[sound_offset].shape)
                size = int(torch.tensor(shape).prod().item())
                slices.append(FlatModalitySlice("sound", offset, offset + size, shape))
                offset += size
                sound_offset += 1

            sample_layouts.append(tuple(slices))

        return cls(tuple(sample_layouts))

    def modality(self, sample_index: int, name: str) -> FlatModalitySlice:
        matches = [item for item in self.samples[sample_index] if item.name == name]
        if len(matches) != 1:
            raise KeyError(f"expected one {name!r} slice for sample {sample_index}, found {len(matches)}")
        return matches[0]


@dataclass
class PreparedLayoutCapture:
    """Capture prepared token geometry through the public builder interface."""

    layout: Cosmos3FlatLayout | None = field(default=None, init=False)
    generation_data: Any | None = field(default=None, init=False)
    sequence_plans: Sequence[Any] | None = field(default=None, init=False)

    def __call__(
        self,
        *,
        model: Any,
        net: Any,
        cond_tokens: list[list[int]],
        sequence_plans: Sequence[Any],
        gen_data_clean: Any,
    ) -> None:
        del model, net, cond_tokens
        self.sequence_plans = sequence_plans
        self.generation_data = gen_data_clean
        self.layout = Cosmos3FlatLayout.from_prepared(sequence_plans, gen_data_clean)
        # Returning None is intentional: the public generator keeps its native
        # CFG path, making this capture eligible for exact no-op equivalence.
        return None


def temporal_mask(shape: Sequence[int], frame_indices: Sequence[int], *, device: torch.device) -> Tensor:
    """Return a flat mask for selected T coordinates of ``[C,T,H,W]`` or ``[B,C,T,H,W]``."""

    if len(shape) not in (4, 5):
        raise ValueError(f"vision latent must have shape [C,T,H,W] or [B,C,T,H,W], got {tuple(shape)}")
    temporal_axis = len(shape) - 3
    temporal_size = int(shape[temporal_axis])
    indices = tuple(int(index) for index in frame_indices)
    if not indices:
        raise ValueError("at least one future latent frame is required")
    if min(indices) < 0 or max(indices) >= temporal_size:
        raise IndexError(f"frame indices {indices} are invalid for latent T={temporal_size}")
    mask = torch.zeros(tuple(shape), dtype=torch.bool, device=device)
    index = [slice(None)] * len(shape)
    index[temporal_axis] = indices
    mask[tuple(index)] = True
    return mask.reshape(-1)


def gaussian_target_on_mask(recipient: Tensor, donor: Tensor, mask: Tensor, *, seed: int) -> Tensor:
    """Match donor norm and recipient distance within selected flat coordinates."""

    if recipient.ndim != 1 or donor.shape != recipient.shape or mask.shape != recipient.shape:
        raise ValueError("recipient, donor, and mask must be equal-length flat tensors")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("mask must be boolean and select at least one coordinate")
    reference = recipient[mask].double()
    donor_selected = donor[mask].double()
    reference_norm_sq = torch.dot(reference, reference)
    if reference_norm_sq == 0:
        raise ValueError("selected recipient target must have nonzero norm")
    donor_norm_sq = torch.dot(donor_selected, donor_selected)
    distance_sq = torch.dot(donor_selected - reference, donor_selected - reference)
    alpha = (donor_norm_sq + reference_norm_sq - distance_sq) / (2.0 * reference_norm_sq)
    generator = torch.Generator(device=recipient.device)
    generator.manual_seed(seed)
    random = torch.randn(reference.shape, generator=generator, device=reference.device, dtype=torch.float64)
    orthogonal = random - torch.dot(random, reference) / reference_norm_sq * reference
    orthogonal = orthogonal / torch.linalg.vector_norm(orthogonal)
    beta_sq = torch.clamp(donor_norm_sq - alpha.square() * reference_norm_sq, min=0.0)
    matched = alpha * reference + torch.sqrt(beta_sq) * orthogonal
    output = recipient.clone()
    output[mask] = matched.to(recipient.dtype)
    return output


@dataclass
class GuidedFutureClamp:
    """Clamp future vision along an RF path at the post-guidance interface.

    ``target`` and ``path_noise`` contain only the flattened vision item.  The
    model sees ``(1-sigma)*target + sigma*path_noise`` at selected future
    coordinates.  Its action velocity is retained, while its vision velocity
    is replaced so the sampler state's clean estimate is exactly ``target``.
    """

    layout_capture: PreparedLayoutCapture
    target: Tensor
    path_noise: Tensor
    future_frame_indices: tuple[int, ...]
    active_call_indices: tuple[int, ...] | None = None
    num_train_timesteps: float = 1000.0
    sample_index: int = 0
    vision_name: str = "vision_0"
    calls: list[float] = field(default_factory=list, init=False)
    clamped_call_indices: list[int] = field(default_factory=list, init=False)
    maximum_action_input_error: float = field(default=0.0, init=False)
    maximum_action_output_error: float = field(default=0.0, init=False)

    def _validate(self, flat: Tensor, vision: FlatModalitySlice) -> Tensor:
        expected = (vision.size,)
        if tuple(self.target.shape) != expected or tuple(self.path_noise.shape) != expected:
            raise ValueError(
                f"target and path_noise must match flattened {self.vision_name} shape {expected}, "
                f"got {tuple(self.target.shape)} and {tuple(self.path_noise.shape)}"
            )
        if flat.numel() < vision.stop:
            raise ValueError("flat sampler state is shorter than the captured vision slice")
        return temporal_mask(vision.shape, self.future_frame_indices, device=flat.device)

    def wrap_velocity(self, velocity_fn: VelocityFunction) -> VelocityFunction:
        def interventional(noisy_state: list[Tensor], timestep: Tensor) -> list[Tensor]:
            if self.layout_capture.layout is None:
                raise RuntimeError("prepared layout has not been captured")
            if len(noisy_state) != len(self.layout_capture.layout.samples):
                raise ValueError("sampler batch size differs from captured layout")

            model_state = [item.clone() for item in noisy_state]
            vision = self.layout_capture.layout.modality(self.sample_index, self.vision_name)
            selected = self._validate(model_state[self.sample_index], vision)
            sigma = float(timestep.reshape(-1)[0].detach().double().cpu()) / self.num_train_timesteps
            if not 0.0 <= sigma <= 1.0 + 1e-6:
                raise ValueError(f"rectified-flow sigma is outside [0,1]: {sigma}")
            sigma = min(max(sigma, 0.0), 1.0)
            call_index = len(self.calls)
            self.calls.append(sigma)

            if self.active_call_indices is not None and call_index not in self.active_call_indices:
                return velocity_fn(noisy_state, timestep)
            self.clamped_call_indices.append(call_index)

            target = self.target.to(device=model_state[self.sample_index].device, dtype=model_state[self.sample_index].dtype)
            path_noise = self.path_noise.to(device=target.device, dtype=target.dtype)
            model_vision = model_state[self.sample_index][vision.start : vision.stop]
            donor_path = (1.0 - sigma) * target + sigma * path_noise
            model_vision[selected] = donor_path[selected]

            action = None
            native_action_input = None
            try:
                action = self.layout_capture.layout.modality(self.sample_index, "action")
                native_action_input = noisy_state[self.sample_index][action.start : action.stop].clone()
                changed_action_input = model_state[self.sample_index][action.start : action.stop]
                self.maximum_action_input_error = max(
                    self.maximum_action_input_error,
                    float((changed_action_input - native_action_input).abs().max().detach().cpu()),
                )
            except KeyError:
                pass

            guided_velocity = velocity_fn(model_state, timestep)
            output = [item.clone() for item in guided_velocity]
            native_action_output = None
            if action is not None:
                native_action_output = guided_velocity[self.sample_index][action.start : action.stop].clone()

            output_vision = output[self.sample_index][vision.start : vision.stop]
            sampler_vision = noisy_state[self.sample_index][vision.start : vision.stop]
            if sigma > torch.finfo(sampler_vision.dtype).eps:
                desired_velocity = (sampler_vision - target) / sigma
                output_vision[selected] = desired_velocity[selected].to(output_vision.dtype)

            if action is not None and native_action_output is not None:
                changed_action_output = output[self.sample_index][action.start : action.stop]
                self.maximum_action_output_error = max(
                    self.maximum_action_output_error,
                    float((changed_action_output - native_action_output).abs().max().detach().cpu()),
                )
            return output

        return interventional


class SamplerVelocityWrapper:
    """Apply a velocity transformation after CFG without vendoring a sampler."""

    def __init__(self, sampler: Any, transform: Callable[[VelocityFunction], VelocityFunction]) -> None:
        self.sampler = sampler
        self.transform = transform

    def __call__(self, velocity_fn: VelocityFunction, *args: Any, **kwargs: Any) -> Any:
        return self.sampler(self.transform(velocity_fn), *args, **kwargs)


class SamplerInitialStateCapture:
    """Capture the native flat initial noise before delegating unchanged."""

    def __init__(self, sampler: Any) -> None:
        self.sampler = sampler
        self.initial_state: list[Tensor] | None = None

    def __call__(self, velocity_fn: VelocityFunction, initial_state: list[Tensor], *args: Any, **kwargs: Any) -> Any:
        self.initial_state = [item.detach().clone() for item in initial_state]
        return self.sampler(velocity_fn, initial_state, *args, **kwargs)


@dataclass
class GuidedX0Recorder:
    """Record post-CFG clean estimates while returning velocity bit-for-bit."""

    layout_capture: PreparedLayoutCapture
    num_train_timesteps: float = 1000.0
    store_on_cpu: bool = True
    records: list[dict[str, Any]] = field(default_factory=list, init=False)

    def wrap_velocity(self, velocity_fn: VelocityFunction) -> VelocityFunction:
        def recording(noisy_state: list[Tensor], timestep: Tensor) -> list[Tensor]:
            if self.layout_capture.layout is None:
                raise RuntimeError("prepared layout has not been captured")
            velocity = velocity_fn(noisy_state, timestep)
            sigma = float(timestep.reshape(-1)[0].detach().double().cpu()) / self.num_train_timesteps
            sample_records: list[dict[str, Tensor]] = []
            for sample_index, (state, predicted_velocity) in enumerate(zip(noisy_state, velocity, strict=True)):
                clean = state - sigma * predicted_velocity
                modalities: dict[str, Tensor] = {}
                for item in self.layout_capture.layout.samples[sample_index]:
                    value = clean[item.start : item.stop].reshape(item.shape).detach()
                    if self.store_on_cpu:
                        value = value.cpu()
                    else:
                        value = value.clone()
                    modalities[item.name] = value
                sample_records.append(modalities)
            self.records.append({"sigma": sigma, "samples": sample_records})
            return velocity

        return recording
