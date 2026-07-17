from slime.utils.types import Sample

__all__ = ["mask_truncated_samples"]


def mask_truncated_samples(args, data):
    """Flag truncated (no-EOS) samples so training zeroes their loss.

    A truncated sequence never shows the model a stop decision, so training on
    it (dense per-token KL under OPD) pushes toward "never stop" and feeds the
    length/repetition runaway. Flagging ``remove_sample`` (instead of deleting)
    keeps group structure and batch geometry intact; the train-data builder
    turns the flag into an all-zero loss mask.
    """

    def _walk(node):
        if isinstance(node, list):
            for item in node:
                _walk(item)
        elif node.status == Sample.Status.TRUNCATED:
            node.remove_sample = True

    _walk(data)
