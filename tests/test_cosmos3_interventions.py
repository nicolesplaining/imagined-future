from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from imagined_future.cosmos3_interventions import (
    Cosmos3FlatLayout,
    GuidedFutureClamp,
    GuidedX0Recorder,
    NativeStateInsertion,
    PreparedLayoutCapture,
    SamplerInitialStateCapture,
    VisionStateRecorder,
    gaussian_target_on_mask,
    interpolated_target,
    permuted_future_target,
    temporal_mask,
)


@dataclass
class Plan:
    has_lidar: bool = False
    has_action: bool = True
    has_sound: bool = False


def prepared() -> tuple[list[Plan], SimpleNamespace]:
    data = SimpleNamespace(
        num_vision_items_per_sample=None,
        num_lidar_items_per_sample=None,
        x0_tokens_vision=[torch.zeros(2, 3, 2, 2)],
        x0_tokens_lidar=None,
        x0_tokens_action=[torch.zeros(4, 2)],
        x0_tokens_sound=None,
    )
    return [Plan()], data


def test_flat_layout_matches_vision_then_action_order() -> None:
    plans, data = prepared()
    layout = Cosmos3FlatLayout.from_prepared(plans, data)

    vision = layout.modality(0, "vision_0")
    action = layout.modality(0, "action")
    assert (vision.start, vision.stop, vision.shape) == (0, 24, (2, 3, 2, 2))
    assert (action.start, action.stop, action.shape) == (24, 32, (4, 2))


def test_temporal_mask_selects_only_requested_frames() -> None:
    mask = temporal_mask((2, 3, 2, 2), (1, 2), device=torch.device("cpu"))
    shaped = mask.reshape(2, 3, 2, 2)
    assert not shaped[:, 0].any()
    assert shaped[:, 1:].all()

    batched = temporal_mask((1, 2, 3, 2, 2), (1, 2), device=torch.device("cpu"))
    batched_shaped = batched.reshape(1, 2, 3, 2, 2)
    assert not batched_shaped[:, :, 0].any()
    assert batched_shaped[:, :, 1:].all()


def test_layout_capture_is_a_noop_builder() -> None:
    plans, data = prepared()
    capture = PreparedLayoutCapture()
    result = capture(model=None, net=None, cond_tokens=[], sequence_plans=plans, gen_data_clean=data)

    assert result is None
    assert capture.layout is not None


def test_guided_clamp_changes_model_future_but_not_action_coordinates() -> None:
    plans, data = prepared()
    capture = PreparedLayoutCapture()
    capture(model=None, net=None, cond_tokens=[], sequence_plans=plans, gen_data_clean=data)
    target = torch.arange(24, dtype=torch.float32)
    path_noise = torch.full((24,), 10.0)
    clamp = GuidedFutureClamp(capture, target, path_noise, (1, 2))
    native_state = torch.zeros(32)
    native_state[24:] = 7.0
    observed: list[torch.Tensor] = []

    def velocity_fn(state: list[torch.Tensor], _timestep: torch.Tensor) -> list[torch.Tensor]:
        observed.append(state[0].clone())
        return [state[0] + 1.0]

    result = clamp.wrap_velocity(velocity_fn)([native_state], torch.tensor([[500.0]]))[0]
    observed_vision = observed[0][:24].reshape(2, 3, 2, 2)
    expected_path = (0.5 * target + 0.5 * path_noise).reshape(2, 3, 2, 2)

    assert torch.equal(observed_vision[:, 0], torch.zeros_like(observed_vision[:, 0]))
    assert torch.equal(observed_vision[:, 1:], expected_path[:, 1:])
    assert torch.equal(observed[0][24:], native_state[24:])
    assert torch.equal(result[24:], native_state[24:] + 1.0)
    expected_velocity = ((native_state[:24] - target) / 0.5).reshape(2, 3, 2, 2)
    assert torch.equal(result[:24].reshape(2, 3, 2, 2)[:, 1:], expected_velocity[:, 1:])
    assert clamp.maximum_action_input_error == 0.0
    assert clamp.maximum_action_output_error == 0.0


def test_guided_x0_recorder_is_velocity_identity() -> None:
    plans, data = prepared()
    capture = PreparedLayoutCapture()
    capture(model=None, net=None, cond_tokens=[], sequence_plans=plans, gen_data_clean=data)
    recorder = GuidedX0Recorder(capture)
    state = [torch.arange(32, dtype=torch.float32)]
    velocity = [torch.full((32,), 2.0)]

    result = recorder.wrap_velocity(lambda _state, _time: velocity)(state, torch.tensor([[250.0]]))

    assert result[0] is velocity[0]
    assert recorder.records[0]["sigma"] == 0.25
    assert torch.equal(recorder.records[0]["samples"][0]["vision_0"].reshape(-1), state[0][:24] - 0.5)
    assert torch.equal(recorder.records[0]["samples"][0]["action"].reshape(-1), state[0][24:] - 0.5)


def test_guided_clamp_can_target_one_denoising_call() -> None:
    plans, data = prepared()
    capture = PreparedLayoutCapture()
    capture(model=None, net=None, cond_tokens=[], sequence_plans=plans, gen_data_clean=data)
    target = torch.arange(24, dtype=torch.float32)
    path_noise = torch.full((24,), 10.0)
    clamp = GuidedFutureClamp(capture, target, path_noise, (1, 2), active_call_indices=(1,))
    native_state = [torch.arange(32, dtype=torch.float32)]
    observed: list[torch.Tensor] = []

    def velocity_fn(state: list[torch.Tensor], _timestep: torch.Tensor) -> list[torch.Tensor]:
        observed.append(state[0].clone())
        return [state[0] + 1.0]

    wrapped = clamp.wrap_velocity(velocity_fn)
    first = wrapped(native_state, torch.tensor([[900.0]]))[0]
    second = wrapped(native_state, torch.tensor([[500.0]]))[0]

    assert torch.equal(observed[0], native_state[0])
    assert torch.equal(first, native_state[0] + 1.0)
    assert not torch.equal(observed[1][:24], native_state[0][:24])
    assert torch.equal(observed[1][24:], native_state[0][24:])
    assert torch.equal(second[24:], native_state[0][24:] + 1.0)
    assert clamp.calls == [0.9, 0.5]
    assert clamp.clamped_call_indices == [1]


def test_sampler_initial_state_capture_delegates_without_copying_argument() -> None:
    observed = []

    def sampler(velocity_fn, initial_state, marker):
        del velocity_fn
        observed.append(initial_state)
        return marker

    wrapper = SamplerInitialStateCapture(sampler)
    state = [torch.arange(4)]
    result = wrapper(lambda *_: [], state, "done")

    assert result == "done"
    assert observed[0] is state
    assert wrapper.initial_state is not state
    assert torch.equal(wrapper.initial_state[0], state[0])


def test_gaussian_target_matches_geometry_only_on_mask() -> None:
    recipient = torch.arange(1, 13, dtype=torch.float64)
    donor = torch.flip(recipient, dims=(0,)) * 0.7
    mask = torch.tensor([False, False, True, True, True, True, True, True, True, True, False, False])

    target = gaussian_target_on_mask(recipient, donor, mask, seed=19)

    assert torch.equal(target[~mask], recipient[~mask])
    assert torch.linalg.vector_norm(target[mask]) == pytest.approx(torch.linalg.vector_norm(donor[mask]))
    assert torch.linalg.vector_norm(target[mask] - recipient[mask]) == pytest.approx(
        torch.linalg.vector_norm(donor[mask] - recipient[mask])
    )


def test_interpolated_target_moves_only_masked_coordinates() -> None:
    mask = temporal_mask((2, 3, 2, 2), (1, 2), device=torch.device("cpu"))
    recipient = torch.zeros(24)
    donor = torch.full((24,), 4.0)

    assert torch.equal(interpolated_target(recipient, donor, mask, alpha=0.0), recipient)
    half = interpolated_target(recipient, donor, mask, alpha=0.5)
    assert torch.equal(half[mask], torch.full((16,), 2.0))
    assert torch.equal(half[~mask], recipient[~mask])
    with pytest.raises(ValueError):
        interpolated_target(recipient, donor, mask, alpha=1.5)


def test_permuted_future_target_keeps_current_frame() -> None:
    shape = (2, 3, 2, 2)
    target = torch.arange(24, dtype=torch.float32)
    swapped = permuted_future_target(target, shape, [2, 1]).reshape(shape)
    original = target.reshape(shape)

    assert torch.equal(swapped[:, 0], original[:, 0])
    assert torch.equal(swapped[:, 1], original[:, 2])
    assert torch.equal(swapped[:, 2], original[:, 1])
    with pytest.raises(ValueError):
        permuted_future_target(target, shape, [0, 1])


def test_vision_state_recorder_is_velocity_identity() -> None:
    plans, data = prepared()
    capture = PreparedLayoutCapture()
    capture(model=None, net=None, cond_tokens=[], sequence_plans=plans, gen_data_clean=data)
    recorder = VisionStateRecorder(capture)
    state = torch.arange(32, dtype=torch.float32)
    velocity = torch.full((32,), 3.0)

    output = recorder.wrap_velocity(lambda _state, _timestep: [velocity])([state], torch.tensor([500.0]))
    assert output[0] is velocity
    assert torch.equal(recorder.states[0], state[:24])


def test_native_state_insertion_replaces_one_call_only() -> None:
    plans, data = prepared()
    capture = PreparedLayoutCapture()
    capture(model=None, net=None, cond_tokens=[], sequence_plans=plans, gen_data_clean=data)
    donor_states = [torch.full((24,), 5.0), torch.full((24,), 9.0)]
    insertion = NativeStateInsertion(capture, donor_states, (1, 2), call_index=1)
    mask = temporal_mask((2, 3, 2, 2), (1, 2), device=torch.device("cpu"))
    seen: list[torch.Tensor] = []

    def velocity_fn(state, _timestep):
        seen.append(state[0].clone())
        return [torch.full((32,), 2.0)]

    wrapped = insertion.wrap_velocity(velocity_fn)
    state = torch.zeros(32)
    first = wrapped([state], torch.tensor([800.0]))
    assert torch.equal(first[0], torch.full((32,), 2.0))
    assert torch.equal(seen[0], state)

    second = wrapped([state], torch.tensor([500.0]))
    assert torch.equal(seen[1][:24][mask], torch.full((16,), 9.0))
    assert torch.equal(seen[1][:24][~mask], state[:24][~mask])
    sampler_clean = state[:24] - 0.5 * second[0][:24]
    model_clean = 9.0 - 0.5 * 2.0
    assert torch.allclose(sampler_clean[mask], torch.full((16,), model_clean))
    assert torch.equal(second[0][24:], torch.full((8,), 2.0))
    assert insertion.inserted_call_indices == [1]
    assert insertion.maximum_action_output_error == 0.0
