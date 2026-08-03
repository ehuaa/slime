"""CPU unit tests for the optional zero2one reward standard."""

from __future__ import annotations

import asyncio
from argparse import Namespace

import pytest

from slime.rollout.rm_hub import async_rm
from slime.rollout.rm_hub.zero2one import get_zero2one_rule_based_reward
from slime.utils.types import Sample


NUM_GPUS = 0


@pytest.mark.unit
def test_last_boxed_numeric_answer_is_used():
    response = r"first \\boxed{41}, then \\boxed{42}"
    assert get_zero2one_rule_based_reward(response, "42") == 1


@pytest.mark.unit
def test_boxed_answer_does_not_require_think_marker():
    assert get_zero2one_rule_based_reward(r"answer: \\boxed{-3}", "-3") == 1


@pytest.mark.unit
def test_decimal_and_scientific_notation_compare_numerically():
    assert get_zero2one_rule_based_reward(r"\\boxed{34.0}", "34") == 1
    assert get_zero2one_rule_based_reward(r"\\boxed{3.4e1}", "34") == 1


@pytest.mark.unit
def test_large_integer_comparison_is_exact():
    ground_truth = "22099999999999998951424"
    wrong = str(int(ground_truth) + 1)
    assert get_zero2one_rule_based_reward(rf"\\boxed{{{ground_truth}}}", ground_truth) == 1
    assert get_zero2one_rule_based_reward(rf"\\boxed{{{wrong}}}", ground_truth) == 0


@pytest.mark.unit
def test_last_number_fallback_is_opt_in_and_uses_post_think_text():
    response = "<think>wrong intermediate 99</think> final answer 42"
    assert get_zero2one_rule_based_reward(response, "42") == 0
    assert get_zero2one_rule_based_reward(response, "42", extract_last_number=True) == 1
    assert get_zero2one_rule_based_reward(response, "99", extract_last_number=True) == 0


@pytest.mark.unit
def test_invalid_or_empty_answer_returns_zero():
    assert get_zero2one_rule_based_reward("no numeric answer", "42") == 0
    assert get_zero2one_rule_based_reward(r"\\boxed{42}", "") == 0


@pytest.mark.unit
def test_reward_model_struct_label_is_unwrapped():
    # `--label-key reward_model` hands the whole verl struct through as the label.
    assert get_zero2one_rule_based_reward(r"\\boxed{45}", {"ground_truth": 45, "style": "rule"}) == 1
    assert get_zero2one_rule_based_reward(r"\\boxed{33}", {"ground_truth": "33", "style": "rule"}) == 1
    assert get_zero2one_rule_based_reward(r"\\boxed{34}", {"ground_truth": "33", "style": "rule"}) == 0
    assert get_zero2one_rule_based_reward(r"\\boxed{34}", {"style": "rule"}) == 0


@pytest.mark.unit
def test_rm_type_dispatch_and_metadata_fallback():
    args = Namespace(custom_rm_path=None, rm_type="zero2one")
    sample = Sample(
        response="<think>work 7</think> answer 42",
        label="42",
        metadata={"extract_last_number": True},
    )
    assert asyncio.run(async_rm(args, sample)) == 1
