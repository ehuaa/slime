"""Post-build smoke test for the Qwen3.8-27B cu129 image.

Fails the build instead of a 90-minute rollout.  The TE arch check is the one
that matters most -- it is what makes this image A100-capable at all.
"""

import dataclasses
import glob
import subprocess

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.version.cuda.startswith("12"), "expected a CUDA 12 base image"

import megatron.core  # noqa: F401

print("megatron.core ok")

import sglang

assert "sglang-v0.5.17" in sglang.__file__, sglang.__file__
from sglang.srt.server_args import ServerArgs

fields = {f.name for f in dataclasses.fields(ServerArgs)}
assert "enable_linear_replayssm_spec" in fields, "ReplaySSM spec switch missing"
print("sglang", sglang.__file__, "+ replayssm_spec")

import deep_gemm  # noqa: F401
import sgl_kernel  # noqa: F401

print("sgl_kernel / deep_gemm ok")

so = glob.glob(
    "/usr/local/lib/python3.12/dist-packages/transformer_engine/**/libtransformer_engine.so",
    recursive=True,
)
assert so, "libtransformer_engine.so not found"
cuobjdump = sorted(glob.glob("/usr/local/cuda*/bin/cuobjdump"))[-1]
elf = subprocess.run([cuobjdump, "--list-elf", so[0]], capture_output=True, text=True).stdout
archs = sorted(
    {t for line in elf.splitlines() for t in line.replace(".", " ").split() if t.startswith("sm_")}
)
print("TE archs:", " ".join(archs))
assert "sm_80" in archs, "TE has no sm_80 cubin -- A100 will die in rmsnorm"
