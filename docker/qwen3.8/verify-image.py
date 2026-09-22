"""Post-build smoke test for the Qwen3.8-27B cu129 image.

Fails the build instead of a 90-minute rollout.  The TE arch check is the one
that matters most -- it is what makes this image A100-capable at all.
"""

import dataclasses
import glob
import pathlib
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

# sgl_kernel dlopens an arch-specific common_ops.so, which needs libcuda.so.1 --
# injected by nvidia-container-runtime at run time and absent during docker build.
# Without a driver, fall back to the static checks that catch what actually goes
# wrong here: PyPI ships cu13-only builds of these two whose .so files link
# libnvrtc.so.13 and fail to import on a cu129 image.
import ctypes
import importlib.metadata as md

try:
    ctypes.CDLL("libcuda.so.1")
    has_driver = True
except OSError:
    has_driver = False

if has_driver:
    import deep_gemm  # noqa: F401
    import sgl_kernel  # noqa: F401

    print("sgl_kernel / deep_gemm import ok")
else:
    print("no CUDA driver (build time) -- checking versions and linkage instead")
    for dist, want in (("sglang-kernel", "+cu129"), ("sgl-deep-gemm", "+cu129")):
        got = md.version(dist)
        assert want in got, f"{dist}=={got}, expected a {want} build (PyPI ships cu13 only)"
        print(f"  {dist}=={got}")
    sos = glob.glob("/usr/local/lib/python3.12/dist-packages/sgl_kernel/**/*.so", recursive=True)
    sos += glob.glob("/usr/local/lib/python3.12/dist-packages/deep_gemm/**/*.so", recursive=True)
    assert sos, "no sgl_kernel/deep_gemm .so found"
    # readelf -d, not a read_bytes() scan: these .so files run to hundreds of MB.
    bad = [
        so
        for so in sos
        if "libnvrtc.so.13"
        in subprocess.run(["readelf", "-d", so], capture_output=True, text=True).stdout
    ]
    assert not bad, f"these .so still link CUDA 13: {bad}"
    print(f"  {len(sos)} .so files, none linking libnvrtc.so.13")

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
