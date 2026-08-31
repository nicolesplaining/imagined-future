from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from imagined_future.cosmos_config import deterministic_tokenizer_enabled, libero_policy_config


def test_enables_official_deterministic_tokenizer_mode(monkeypatch) -> None:
    class FakePolicyEvalConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module_name = "cosmos_policy.experiments.robot.libero.run_libero_eval"
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(PolicyEvalConfig=FakePolicyEvalConfig))
    monkeypatch.delenv("DETERMINISTIC", raising=False)

    config = libero_policy_config()

    assert config.deterministic is True
    assert config.randomize_seed is False
    assert os.environ["DETERMINISTIC"] == "True"
    assert deterministic_tokenizer_enabled()
