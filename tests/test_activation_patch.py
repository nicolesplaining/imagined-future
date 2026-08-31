from __future__ import annotations

import pytest
import torch

from imagined_future.activation_patch import capture_module_outputs, transplant_module_output


class AddOne(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


def test_capture_and_transplant_are_call_aligned_and_temporally_local() -> None:
    module = AddOne()
    donor_inputs = [torch.full((1, 4, 1, 1, 2), value) for value in (2.0, 4.0)]
    with capture_module_outputs({"block": module}, (1, 3)) as bank:
        for value in donor_inputs:
            module(value)

    recipient_inputs = [torch.zeros_like(value) for value in donor_inputs]
    with transplant_module_output(module, bank.calls["block"], (1, 3)) as transplant:
        outputs = [module(value) for value in recipient_inputs]
        transplant.validate_complete()

    for output, donor_input in zip(outputs, donor_inputs, strict=True):
        assert torch.all(output[:, (1, 3)] == donor_input[:, (1, 3)] + 1)
        assert torch.all(output[:, (0, 2)] == 1)
    assert transplant.calls == 2
    assert all(distance > 0 for distance in transplant.patch_l2)


def test_transplant_rejects_a_call_count_mismatch() -> None:
    module = AddOne()
    donor = [torch.zeros(1, 1, 1, 1, 1)]
    with transplant_module_output(module, donor, (0,)) as transplant:
        with pytest.raises(RuntimeError, match="expected 1"):
            transplant.validate_complete()


def test_transplant_scale_interpolates_recipient_and_donor() -> None:
    module = AddOne()
    donor = [torch.full((1, 1, 1, 1, 1), 5.0)]
    with transplant_module_output(module, donor, (1,), scale=0.25) as transplant:
        output = module(torch.zeros(1, 2, 1, 1, 1))
        transplant.validate_complete()

    assert output[0, 1, 0, 0, 0].item() == 2.0
    assert transplant.donor_recipient_l2 == [4.0]
    assert transplant.patch_l2 == [1.0]
