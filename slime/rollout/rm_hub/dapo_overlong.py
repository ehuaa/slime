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

The unshaped 0/1 accuracy is stored in ``sample.metadata["raw_reward"]`` for
DAPO dynamic sampling and metrics. The training reward maps that accuracy to
-1/+1 before adding the length penalty, matching the standard DAPO reward
scale. Dynamic sampling must filter on ``raw_reward`` rather than this shaped
training reward.

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
from .zero2one import get_zero2one_rule_based_reward

OVERLONG_BUFFER = int(os.environ.get("DAPO_OVERLONG_BUFFER", 16000))


def _get_accuracy(args, sample: Sample) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or getattr(args, "rm_type", None) or "deepscaler").strip()

    if rm_type == "deepscaler":
        return float(get_deepscaler_rule_based_reward(sample.response, sample.label))
    if rm_type == "zero2one":
        return float(
            get_zero2one_rule_based_reward(
                sample.response,
                sample.label,
                extract_last_number=bool(metadata.get("extract_last_number", False)),
            )
        )
    raise ValueError(f"dapo_overlong supports rm_type deepscaler or zero2one, got {rm_type!r}")


def _reward_one(args, sample: Sample, evaluation: bool) -> float:
    accuracy = _get_accuracy(args, sample)
    if accuracy not in (0.0, 1.0):
        raise ValueError(f"DAPO accuracy reward must be binary, got {accuracy}")

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    metadata["raw_reward"] = accuracy
    if evaluation or metadata.get("is_eval"):
        return accuracy

    base_reward = 2.0 * accuracy - 1.0

    max_len = args.rollout_max_response_len
    threshold = max_len - OVERLONG_BUFFER
    length = sample.response_length or 0
    if length <= threshold:
        return base_reward
    penalty = -min(length - threshold, OVERLONG_BUFFER) / OVERLONG_BUFFER
    return base_reward + penalty


async def custom_rm(args, sample, evaluation: bool = False, **kwargs):
    # batched_async_rm hands us the whole list when --custom-rm-path is set.
    if isinstance(sample, list):
        return [_reward_one(args, s, evaluation) for s in sample]
    return _reward_one(args, sample, evaluation)
