from slime.utils.types import Sample
from slime.utils.metric_utils import has_repetition

__all__ = ["mask_truncated_samples"]


def mask_truncated_samples(args, data):
    """Mask only *repetitive* truncated samples; leave clean truncated samples alone.

    OPD has a built-in length disincentive: when the student overshoots the
    teacher's natural stopping point, teacher_logp << student_logp for those
    tokens, so advantage turns negative. Zeroing the loss on all truncated
    samples silently removes this signal and lets response length drift up.

    We therefore only zero-mask samples that are both truncated AND repetitive
    (the degenerate loop case that caused htmq2aul collapse). Non-repetitive
    truncated samples keep their full KL loss so the natural length penalty
    remains active.
    """

    def _walk(node):
        if isinstance(node, list):
            for item in node:
                _walk(item)
        elif node.status == Sample.Status.TRUNCATED and has_repetition(node.response):
            node.remove_sample = True

    _walk(data)
