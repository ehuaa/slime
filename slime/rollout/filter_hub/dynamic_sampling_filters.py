import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_raw_reward_nonzero_std", "check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = bool(torch.tensor(rewards, dtype=torch.float64).std() > 1e-6)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_raw_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    """Keep groups with non-constant raw rewards before reward shaping.

    DAPO dynamic sampling filters on task correctness (for example ``acc``),
    while soft-overlong punishment is applied only to the reward used for
    training. Custom reward functions using this filter must store the
    unshaped score in ``sample.metadata["raw_reward"]``.
    """
    try:
        rewards = [sample.metadata["raw_reward"] for sample in samples]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "check_raw_reward_nonzero_std requires "
            'sample.metadata["raw_reward"] on every sample'
        ) from exc

    keep = bool(torch.tensor(rewards, dtype=torch.float64).std() > 1e-6)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_raw_reward_{round(rewards[0], 1)}",
    )
