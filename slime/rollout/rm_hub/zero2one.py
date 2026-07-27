"""Integer-focused 0/1 reward with optional final-number fallback.

This scorer follows the ``zero2one_reward`` extraction behavior used by the
provided verl recipe: prefer the final ``\\boxed{...}`` occurrence anywhere in
the response, and optionally fall back to the last standalone number after the
final ``</think>`` marker. Numeric comparison uses :class:`decimal.Decimal`
instead of ``float`` so large integer labels remain exact.
"""

import re
from decimal import Decimal, InvalidOperation


_BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
_LAST_NUMBER_RE = re.compile(r"(?<![\\a-zA-Z])(-?\d+(?:\.\d+)?)\b")


def _extract_last_number(text: str) -> str | None:
    matches = _LAST_NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def _numbers_equal(prediction: str, ground_truth: str | int | float) -> bool:
    try:
        return Decimal(prediction.strip()) == Decimal(str(ground_truth).strip())
    except (InvalidOperation, ValueError):
        return False


def get_zero2one_rule_based_reward(
    response: str,
    label: str | int | float,
    *,
    extract_last_number: bool = False,
) -> int:
    """Return 1 when the extracted numeric answer equals ``label``, else 0."""
    if label is None or str(label).strip() == "":
        return 0

    boxed_matches = _BOXED_RE.findall(response.strip())
    if boxed_matches:
        return int(_numbers_equal(boxed_matches[-1], label))

    if not extract_last_number:
        return 0

    answer_part = response
    think_end = response.rfind("</think>")
    if think_end != -1:
        answer_part = response[think_end + len("</think>") :]

    prediction = _extract_last_number(answer_part)
    return int(prediction is not None and _numbers_equal(prediction, label))
