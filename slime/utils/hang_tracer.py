"""Lightweight per-rank event tracer for debugging distributed (NCCL/gloo) hangs.

Adapted from the sglang `debug-distributed-hang` methodology: every rank writes
an append-only, structured event log; you then `diff` the logs across ranks to
find the *first* point where ranks diverge (one rank enters a collective the
others never reach, or gets stuck in a synchronous gloo collective while the
rest run ahead into the next NCCL collective).

Gated entirely by env vars, so it is ZERO overhead when unset:

    SLIME_HANG_TRACE=1            enable phase-level tracing (cheap: ~10 ev/rollout)
    SLIME_HANG_TRACE_COLL=1       also trace every collective enqueue (heavy)
    SLIME_HANG_TRACE_DIR=<dir>    output dir (default /tmp/slime_hang_trace)

Each rank appends to  <dir>/trace_rank{rank}_pid{pid}.log :

    <mono_ts> seq=<n> <EVENT> k=v k=v ...

Find the first divergence (example: DP-src rank 0 vs a rank that ran ahead):

    d=/tmp/slime_hang_trace
    diff <(grep ' PHASE ' $d/trace_rank000_*.log | awk '{$1="";$2="";print}') \
         <(grep ' PHASE ' $d/trace_rank008_*.log | awk '{$1="";$2="";print}')

CRITICAL: this module must never trigger CUDA synchronization. It only logs
CPU-side metadata (tensor .shape / .numel / .dtype). Do NOT add .cpu(), .item(),
.tolist() on live GPU tensors here -- during a forward pass that would either
deadlock or mask the very hang we are chasing.
"""

import os
import threading
import time

_ENABLED = os.environ.get("SLIME_HANG_TRACE") == "1"
# Collective-level tracing is heavy; require both flags.
TRACE_COLL_ENABLED = _ENABLED and os.environ.get("SLIME_HANG_TRACE_COLL") == "1"
_DIR = os.environ.get("SLIME_HANG_TRACE_DIR", "/tmp/slime_hang_trace")

_lock = threading.Lock()
_seq = 0
_fh = None
_rank = None


def enabled() -> bool:
    return _ENABLED


def _resolve_rank():
    global _rank
    if _rank is not None:
        return _rank
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            _rank = dist.get_rank()
    except Exception:
        pass
    # -1 until torch.distributed is up; do not cache -1 so the file gets the
    # real rank once init completes.
    return _rank if _rank is not None else -1


def _get_fh():
    global _fh
    rank = _resolve_rank()
    if _fh is None and rank >= 0:
        os.makedirs(_DIR, exist_ok=True)
        path = os.path.join(_DIR, f"trace_rank{rank:03d}_pid{os.getpid()}.log")
        _fh = open(path, "a", buffering=1)  # line-buffered text
        _fh.write(f"{time.monotonic():.3f} seq=0 TRACE_OPEN rank={rank} pid={os.getpid()}\n")
    return _fh


def trace(event: str, **fields) -> None:
    """Append one structured event line. No-op unless SLIME_HANG_TRACE=1."""
    if not _ENABLED:
        return
    global _seq
    with _lock:
        fh = _get_fh()
        if fh is None:
            return
        _seq += 1
        parts = [f"{time.monotonic():.3f}", f"seq={_seq}", event]
        for k, v in fields.items():
            parts.append(f"{k}={v}")
        fh.write(" ".join(parts) + "\n")
        fh.flush()


def _group_tag(ranks) -> str:
    """Short, sync-free identifier for a process group."""
    try:
        rs = list(ranks)
        if not rs:
            return "g=?"
        return f"g={len(rs)}:{rs[0]}-{rs[-1]}"
    except Exception:
        return "g=?"


def _shapes(args) -> str:
    """CPU-only tensor shape summary; never touches tensor data (no sync)."""
    import torch

    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(f"{tuple(a.shape)}{a.dtype}".replace("torch.", ""))
        elif isinstance(a, (list, tuple)) and a and isinstance(a[0], torch.Tensor):
            out.append(f"[{len(a)}x{tuple(a[0].shape)}]")
    return "sh=" + (",".join(out) if out else "-")


def trace_coll_call(method: str, ranks, args) -> None:
    """Log BEFORE a collective is enqueued. Pairs with trace_coll_ret.

    A CALL with no matching RET on a rank means it is stuck *inside* a
    synchronous collective (e.g. gloo gather_object). A CALL+RET that then
    hangs later means the op enqueued fine and a peer never arrived.
    """
    if not TRACE_COLL_ENABLED:
        return
    trace(f"COLL_CALL m={method}", **{"_g": _group_tag(ranks), "_s": _shapes(args)})


def trace_coll_ret(method: str, ranks) -> None:
    if not TRACE_COLL_ENABLED:
        return
    trace(f"COLL_RET m={method}", **{"_g": _group_tag(ranks)})
