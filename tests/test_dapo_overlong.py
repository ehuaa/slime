"""CPU tests for DAPO raw filtering and soft-overlong reward shaping."""

from argparse import Namespace

import pytest

from slime.rollout.filter_hub.dynamic_sampling_filters import check_raw_reward_nonzero_std
from slime.rollout.rm_hub import dapo_overlong
from slime.utils.types import Sample


NUM_GPUS = 0


def _sample(*, response: str, length: int, reward: float | None = None, raw_reward: float | None = None):
    metadata = {} if raw_reward is None else {"raw_reward": raw_reward}
    return Sample(response=response, response_length=length, reward=reward, metadata=metadata)


@pytest.fixture
def dapo_args():
    return Namespace(rollout_max_response_len=100)


@pytest.fixture(autouse=True)
def fake_accuracy_reward(monkeypatch):
    monkeypatch.setattr(
        dapo_overlong,
        "get_deepscaler_rule_based_reward",
        lambda response, label: 1 if response == "correct" else 0,
    )
    monkeypatch.setattr(dapo_overlong, "OVERLONG_BUFFER", 20)


@pytest.mark.unit
def test_training_reward_uses_dapo_plus_minus_one_scale(dapo_args):
    correct = _sample(response="correct", length=50)
    wrong = _sample(response="wrong", length=50)

    assert dapo_overlong._reward_one(dapo_args, correct, evaluation=False) == 1.0
    assert dapo_overlong._reward_one(dapo_args, wrong, evaluation=False) == -1.0
    assert correct.metadata["raw_reward"] == 1.0
    assert wrong.metadata["raw_reward"] == 0.0


@pytest.mark.unit
def test_overlong_penalty_does_not_change_raw_accuracy(dapo_args):
    correct_at_cap = _sample(response="correct", length=100)
    wrong_at_cap = _sample(response="wrong", length=100)

    assert dapo_overlong._reward_one(dapo_args, correct_at_cap, evaluation=False) == 0.0
    assert dapo_overlong._reward_one(dapo_args, wrong_at_cap, evaluation=False) == -2.0
    assert correct_at_cap.metadata["raw_reward"] == 1.0
    assert wrong_at_cap.metadata["raw_reward"] == 0.0


@pytest.mark.unit
def test_eval_reward_remains_zero_one_accuracy(dapo_args):
    wrong_at_cap = _sample(response="wrong", length=100)
    assert dapo_overlong._reward_one(dapo_args, wrong_at_cap, evaluation=True) == 0.0


@pytest.mark.unit
def test_raw_filter_drops_all_correct_even_when_shaped_rewards_differ():
    args = Namespace(reward_key=None)
    samples = [
        _sample(response="correct", length=50, reward=1.0, raw_reward=1.0),
        _sample(response="correct", length=90, reward=0.5, raw_reward=1.0),
    ]

    output = check_raw_reward_nonzero_std(args, samples)
    assert output.keep is False
    assert output.reason == "zero_std_raw_reward_1.0"


@pytest.mark.unit
def test_raw_filter_keeps_mixed_accuracy_group():
    args = Namespace(reward_key=None)
    samples = [
        _sample(response="correct", length=100, reward=0.0, raw_reward=1.0),
        _sample(response="wrong", length=50, reward=-1.0, raw_reward=0.0),
    ]

    assert check_raw_reward_nonzero_std(args, samples).keep is True


@pytest.mark.unit
def test_raw_filter_fails_fast_when_reward_contract_is_missing():
    args = Namespace(reward_key=None)
    with pytest.raises(ValueError, match="raw_reward"):
        check_raw_reward_nonzero_std(args, [_sample(response="wrong", length=50, reward=-1.0)])
