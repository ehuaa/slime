# Pad MLA value head-dim up to the query head-dim so FlashAttention/cuDNN can serve
# multi-latent attention (MLA) on GPUs where the fused/flash kernels reject qk != v.
#
# WHY: DeepSeek-style MLA has head_dim_qk = qk_nope(128) + qk_rope(64) = 192, but
# head_dim_v = 128 (qk != v). TransformerEngine's backend selection on A100 (sm80):
#   - FlashAttention-2 : disabled — "does not support MLA" (requires head_dim_qk == head_dim_v)
#   - FlashAttention-3 : disabled — requires sm90 (Hopper); supports 192/128 there
#   - cuDNN FusedAttn  : disabled — no sub-backend supports 192/128 on sm80
#   -> only UnfusedDotProductAttention remains, which materializes the O(N^2) scores
#      matrix and OOMs at long sequences (e.g. 32k). Training then crashes with
#      "No dot product attention backend is available for the provided inputs".
#
# FIX (mirrors what upstream Megatron-LM later merged into MLA, and what ms-swift did
# via _patch_mla_attention before its #8422 mcore-bridge refactor): zero-pad V from 128
# to 192 so head_dim_qk == head_dim_v == 192, let flash/cuDNN run, then trim each head's
# output back to 128. Padding V with zeros is mathematically exact: the extra output
# dims are weighted sums of zeros (discarded), and softmax_scale uses head_dim_qk=192
# which is unchanged. On Hopper you can drop this (flash/FA3 handle MLA natively).
#
# Wraps megatron.core.extensions.transformer_engine.TEDotProductAttention.forward. The
# wrap is a no-op unless head_dim_qk != head_dim_v, so non-MLA (GQA) models are
# unaffected. apply_mla_v_pad_patch() is idempotent and is called both at import time
# (driver) and explicitly from get_model_provider_func (guaranteed to run inside each
# Ray train actor before the model's first forward, where the import side-effect alone
# is not reliable).

import logging
import warnings

logger = logging.getLogger(__name__)


def apply_mla_v_pad_patch() -> bool:
    """Idempotently wrap TEDotProductAttention.forward to pad MLA V up to the Q head-dim.

    Returns True if the patch is in place (already or newly applied), False on failure.
    """
    try:
        import torch.nn.functional as F
        from megatron.core.extensions.transformer_engine import TEDotProductAttention
    except ImportError as exc:
        warnings.warn(
            f"slime MLA v-pad attention patch not applied — Megatron/TE import failed ({exc!r}). "
            "MLA models (head_dim_qk != head_dim_v) may fall back to the unfused attention "
            "backend and OOM at long sequence lengths on non-Hopper GPUs.",
            stacklevel=2,
        )
        return False

    if getattr(TEDotProductAttention.forward, "_slime_mla_v_pad", False):
        return True  # already applied

    _orig_te_dpa_forward = TEDotProductAttention.forward

    def _mla_v_pad_forward(self, query, key, value, *args, **kwargs):
        # query/value last dim = per-head hidden size (head_dim_qk / head_dim_v).
        qd = query.shape[-1]
        vd = value.shape[-1] if value is not None else qd
        if value is None or qd == vd:
            return _orig_te_dpa_forward(self, query, key, value, *args, **kwargs)

        # MLA: pad V head-dim up to Q head-dim so qk == v and flash/cuDNN engages.
        value = F.pad(value, [0, qd - vd])
        saved_v = getattr(self, "hidden_size_per_attention_head_v", None)
        if saved_v is not None:
            # TE selects/validates the backend against this attribute, not just the
            # tensor shape — it must match the padded V dim.
            self.hidden_size_per_attention_head_v = value.shape[-1]
        try:
            out = _orig_te_dpa_forward(self, query, key, value, *args, **kwargs)
        finally:
            if saved_v is not None:
                self.hidden_size_per_attention_head_v = saved_v

        # out last dim is the flattened per-head output (num_heads * padded_v_dim).
        # Reshape to expose the head axis, trim padded_v_dim -> orig v dim, flatten back.
        lead = out.shape[:-1]
        out = out.reshape(*lead, -1, qd)[..., :vd].reshape(*lead, -1)
        return out

    _mla_v_pad_forward._slime_mla_v_pad = True
    TEDotProductAttention.forward = _mla_v_pad_forward
    logger.warning(
        "slime MLA v-pad attention patch applied to TEDotProductAttention.forward "
        "(pads head_dim_v up to head_dim_qk when they differ, enabling flash/cuDNN for MLA)."
    )
    return True


# Best-effort apply at import time (covers the driver and any import-based entry). The
# authoritative application is the explicit call in get_model_provider_func.
apply_mla_v_pad_patch()
