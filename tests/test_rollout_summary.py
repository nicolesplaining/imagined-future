from imagined_future.rollout_summary import (
    compress_predicate_trajectory,
    first_true_steps,
    predicate_name,
)


def _snapshot(first: bool, second: bool):
    return {
        "predicates": [
            {"predicate": "on", "arguments": [{"name": "mug"}, {"name": "plate"}], "value": first},
            {"predicate": "close", "arguments": [{"name": "drawer"}], "value": second},
        ]
    }


def _row(index: int, current: tuple[bool, bool], endpoint: tuple[bool, bool]):
    return {
        "query_index": index,
        "query_step": index * 16,
        "executed_count": 16,
        "current": _snapshot(*current),
        "endpoint": _snapshot(*endpoint),
    }


def test_predicate_name_includes_ordered_arguments():
    item = {"predicate": "on", "arguments": [{"name": "mug"}, {"name": "plate"}]}
    assert predicate_name(item) == "on(mug,plate)"


def test_trajectory_compression_and_first_true_steps():
    rows = [
        _row(0, (False, False), (False, False)),
        _row(1, (False, False), (True, False)),
        _row(2, (True, False), (True, False)),
        _row(3, (True, False), (True, True)),
    ]
    transitions = compress_predicate_trajectory(rows)
    assert [row["query_index"] for row in transitions] == [0, 1, 2, 3]
    assert first_true_steps(rows) == {"on(mug,plate)": 32, "close(drawer)": 64}


def test_repeated_signatures_are_removed():
    rows = [_row(0, (False, False), (False, False)), _row(1, (False, False), (False, False))]
    assert len(compress_predicate_trajectory(rows)) == 1
