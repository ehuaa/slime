# Debug helper: record CUDA allocation history and dump a memory snapshot when the
# process is about to hit an OOM. The snapshot (a pickle) contains every live block
# with the allocating stack trace, so it pinpoints what is holding the memory
# (weights vs grads vs activations vs MoE buffers vs recompute, etc.).
#
# Analyze with torch.cuda._memory_viz or a small script that sums live blocks by
# size/frame. NOT for production runs (recording has overhead).

import logging
import os
import socket

logger = logging.getLogger(__name__)

_state = {"attached": False, "dumped": False}


def enable_oom_snapshot(max_entries: int = 300000) -> None:
    """Start recording alloc history and dump a snapshot on the first OOM (idempotent)."""
    if _state["attached"]:
        return
    try:
        import torch
    except ImportError:
        return
    try:
        torch.cuda.memory._record_memory_history(max_entries=max_entries)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[oom-snapshot] _record_memory_history failed: {exc!r}")
        return

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "x"))
    host = socket.gethostname()
    path = f"/root/slime/oom_snap_{host}_rank{rank}.pickle"

    def _observer(device, alloc_size, device_allocated, device_free):
        if _state["dumped"]:
            return
        _state["dumped"] = True
        try:
            torch.cuda.memory._dump_snapshot(path)
            logger.warning(
                f"[oom-snapshot] DUMPED {path} | failing_alloc={alloc_size/2**20:.1f}MiB "
                f"dev_allocated={device_allocated/2**30:.2f}GiB dev_free={device_free/2**30:.2f}GiB"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[oom-snapshot] dump failed: {exc!r}")

    try:
        torch._C._cuda_attach_out_of_memory_observer(_observer)
        _state["attached"] = True
        logger.warning(f"[oom-snapshot] observer attached; will dump to {path} on OOM")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[oom-snapshot] attach observer failed: {exc!r}")
