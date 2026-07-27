"""CPU unit tests for PPO policy-loss clipping."""

import math

import pytest
import torch

from slime.utils.ppo_utils import compute_policy_loss


NUM_GPUS = 0


@pytest.mark.unit
def test_dual_clip_caps_large_negative_advantage_loss():
    # ppo_kl = old_log_prob - new_log_prob, so -log(20) gives ratio 20.
    ppo_kl = torch.tensor([-math.log(20.0)])
    advantages = torch.tensor([-1.0])

    unclipped, _ = compute_policy_loss(ppo_kl, advantages, 0.2, 0.28)
    dual_clipped, _ = compute_policy_loss(ppo_kl, advantages, 0.2, 0.28, 10.0)

    torch.testing.assert_close(unclipped, torch.tensor([20.0]))
    torch.testing.assert_close(dual_clipped, torch.tensor([10.0]))


@pytest.mark.unit
def test_dual_clip_does_not_change_positive_advantage_clip_higher():
    ppo_kl = torch.tensor([-math.log(20.0)])
    advantages = torch.tensor([1.0])

    loss, _ = compute_policy_loss(ppo_kl, advantages, 0.2, 0.28, 10.0)
    torch.testing.assert_close(loss, torch.tensor([-1.28]))


@pytest.mark.unit
def test_dual_clip_requires_c_greater_than_one():
    with pytest.raises(AssertionError, match="greater than 1.0"):
        compute_policy_loss(torch.tensor([0.0]), torch.tensor([-1.0]), 0.2, 0.28, 1.0)
