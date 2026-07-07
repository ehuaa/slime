"""DAPO-style Soft Overlong Punishment on top of the deepscaler rule-based reward.

Motivation (observed on DeepSeek-V2 021A, 2026-07-04): with plain deepscaler rewards,
truncated samples score 0 and fully-truncated groups have zero reward std, so the
check_reward_nonzero_std dynamic filter silently DROPS exactly the generations that most
need negative feedback. Meanwhile long-but-correct samples are reinforced at full weight.
Net effect over ~27 steps: response length and truncation ratio climb monotonically,
raw_reward stays flat, and eval degrades (eval responses truncate too).

Fix (DAPO, https://arxiv.org/abs/2503.14476, "Soft Overlong Punishment"): subtract a
length penalty that ramps linearly inside a buffer zone right below the response cap:

    length <= L_max - buffer : penalty 0
    L_max - buffer < length  : penalty -(length - (L_max - buffer)) / buffer   (0 -> -1)
    truncated (>= L_max)     : penalty -1

Consequences:
  * finished-but-long samples now carry a graded negative signal — and because their
    rewards differ inside a group, such groups pass the nonzero-std filter and actually
    deliver the length-pressure gradient (they were previously dropped as all-1.0);
  * truncated members of mixed groups fall from 0 to -1, doubling their disadvantage.

EVAL SAFETY: eval scores must stay pure accuracy. Eval samples are tagged via the
eval dataset config (scripts/eval-config-deepseek-v2.yaml sets
``metadata_overrides: {is_eval: true}``, merged into sample.metadata by
slime/utils/eval_config.py) -- no slime core change required. For tagged samples we
return the raw deepscaler reward with no penalty. An ``evaluation`` kwarg is also
honored in case a caller forwards it explicitly.

Wire with (keep --rm-type deepscaler for anything that bypasses the custom path):
    --custom-rm-path slime.rollout.rm_hub.dapo_overlong.custom_rm
Buffer defaults to 16000 tokens; override with env DAPO_OVERLONG_BUFFER.
"""

import os

from slime.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward

OVERLONG_BUFFER = int(os.environ.get("DAPO_OVERLONG_BUFFER", 16000))


def _reward_one(args, sample: Sample, evaluation: bool) -> float:
    base = float(get_deepscaler_rule_based_reward(sample.response, sample.label))

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if evaluation or metadata.get("is_eval"):
        return base

    max_len = args.rollout_max_response_len
    threshold = max_len - OVERLONG_BUFFER
    length = sample.response_length or 0
    if length <= threshold:
        return base
    penalty = -min(length - threshold, OVERLONG_BUFFER) / OVERLONG_BUFFER
    return base + penalty


async def custom_rm(args, sample, evaluation: bool = False, **kwargs):
    # batched_async_rm hands us the whole list when --custom-rm-path is set.
    if isinstance(sample, list):
        return [_reward_one(args, s, evaluation) for s in sample]
    return _reward_one(args, sample, evaluation)
