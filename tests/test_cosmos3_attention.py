from __future__ import annotations

from types import SimpleNamespace

import torch

from imagined_future.cosmos3_attention import (
    ActionQueryFutureKVExcluder,
    AttentionRuntimeOps,
)


def fake_ops(observed_key_lengths: list[int]) -> AttentionRuntimeOps:
    def attention(query, key, value):
        observed_key_lengths.append(key.shape[1])
        mean = value.mean(dim=1, keepdim=True)
        return mean.expand(query.shape[0], query.shape[1], *mean.shape[2:])

    def from_mode_splits(causal, full, original):
        return {**original, "causal_seq": causal, "full_only_seq": full}

    def get_all(pack):
        return torch.cat((pack["causal_seq"], pack["full_only_seq"]), dim=0)

    return AttentionRuntimeOps(
        attention=attention,
        from_mode_splits=from_mode_splits,
        get_all_seq=get_all,
        get_causal_seq=lambda pack: (pack["causal_seq"], torch.tensor([0, 2])),
        get_full_only_seq=lambda pack: (pack["full_only_seq"], torch.tensor([0, 6])),
    )


def packs():
    causal = torch.zeros(2, 1, 2)
    full = torch.arange(12, dtype=torch.float32).reshape(6, 1, 2)
    metadata = {"_full_indices": torch.arange(2, 8)}
    pack = {"causal_seq": causal, "full_only_seq": full, **metadata}
    return pack


def native_output(pack):
    return {
        **pack,
        "causal_seq": pack["causal_seq"].flatten(-2, -1),
        "full_only_seq": pack["full_only_seq"].flatten(-2, -1),
    }


def test_zero_gate_is_exact_attention_noop() -> None:
    observed = []
    excluder = ActionQueryFutureKVExcluder(
        num_layers=2,
        action_tokens=2,
        video_latent_frames=2,
        device=torch.device("cpu"),
        ops=fake_ops(observed),
    )
    pack = packs()
    native = native_output(pack)

    def original(*_args, **_kwargs):
        return native, None

    output, _ = excluder.wrap(0, original)(pack, pack, pack, SimpleNamespace(is_three_way=False))

    assert torch.equal(output["full_only_seq"], native["full_only_seq"])
    assert observed == []


def test_selected_layer_replaces_only_action_query_rows() -> None:
    observed = []
    excluder = ActionQueryFutureKVExcluder(
        num_layers=2,
        action_tokens=2,
        video_latent_frames=2,
        device=torch.device("cpu"),
        ops=fake_ops(observed),
    )
    excluder.set_layers([1])
    pack = packs()
    native = native_output(pack)

    def original(*_args, **_kwargs):
        return native, "kv"

    output, stored = excluder.wrap(1, original)(pack, pack, pack, SimpleNamespace(is_three_way=False))

    assert stored == "kv"
    assert torch.equal(output["causal_seq"], native["causal_seq"])
    assert torch.equal(output["full_only_seq"][:4], native["full_only_seq"][:4])
    assert not torch.equal(output["full_only_seq"][4:6], native["full_only_seq"][4:6])
    assert excluder.active_layers() == {"action": [1], "nonfuture": []}


def test_nonfuture_scope_replaces_current_and_action_but_not_future_rows() -> None:
    excluder = ActionQueryFutureKVExcluder(
        num_layers=2,
        action_tokens=2,
        video_latent_frames=2,
        device=torch.device("cpu"),
        ops=fake_ops([]),
    )
    excluder.set_layers([0], scope="nonfuture")
    pack = packs()
    native = native_output(pack)

    def original(*_args, **_kwargs):
        return native, None

    output, _ = excluder.wrap(0, original)(pack, pack, pack, SimpleNamespace(is_three_way=False))

    assert not torch.equal(output["full_only_seq"][:2], native["full_only_seq"][:2])
    assert torch.equal(output["full_only_seq"][2:4], native["full_only_seq"][2:4])
    assert not torch.equal(output["full_only_seq"][4:6], native["full_only_seq"][4:6])
    assert excluder.active_layers() == {"action": [], "nonfuture": [0]}


def test_records_and_content_patches_future_kv_without_changing_token_count() -> None:
    observed = []
    excluder = ActionQueryFutureKVExcluder(
        num_layers=2,
        action_tokens=2,
        video_latent_frames=2,
        device=torch.device("cpu"),
        ops=fake_ops(observed),
    )
    recipient = packs()
    recipient_native = native_output(recipient)

    def recipient_original(*_args, **_kwargs):
        return recipient_native, None

    with excluder.activate([0], mode="record", cache_id="self"):
        recorded, _ = excluder.wrap(0, recipient_original)(
            recipient, recipient, recipient, SimpleNamespace(is_three_way=False)
        )
    assert torch.equal(recorded["full_only_seq"], recipient_native["full_only_seq"])
    assert excluder.cache_summary("self") == {"0": 1}
    assert observed == []

    donor = packs()
    donor["full_only_seq"] = donor["full_only_seq"].clone()
    donor["full_only_seq"][2:4] += 100
    donor_native = native_output(donor)

    def donor_original(*_args, **_kwargs):
        return donor_native, None

    with excluder.activate([0], mode="patch", cache_id="self"):
        patched, _ = excluder.wrap(0, donor_original)(
            donor, donor, donor, SimpleNamespace(is_three_way=False)
        )
    assert observed == [8]
    assert torch.equal(patched["full_only_seq"][:4], donor_native["full_only_seq"][:4])
    assert not torch.equal(patched["full_only_seq"][4:6], donor_native["full_only_seq"][4:6])


def test_layer_validation_rejects_duplicates_and_bounds() -> None:
    excluder = ActionQueryFutureKVExcluder(
        num_layers=2,
        action_tokens=2,
        video_latent_frames=2,
        device=torch.device("cpu"),
        ops=fake_ops([]),
    )

    for layers in ([0, 0], [-1], [2]):
        try:
            excluder.set_layers(layers)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid layers were accepted: {layers}")

    with excluder.activate([0]):
        assert excluder.active_layers() == {"action": [0], "nonfuture": []}
    assert excluder.active_layers() == {"action": [], "nonfuture": []}

    try:
        excluder.set_layers([0], scope="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid exclusion scope was accepted")
