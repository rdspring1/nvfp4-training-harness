"""Probe: which CUDA kernels does each TransformerEngine NVFP4 path launch?

TransformerEngine exposes no standalone amax or quantize entry point -- `NVFP4Quantizer`
and `tex.split_quantize` fuse amax and quantize into one call, and a 2D weight call
launches three kernels. To chart TE per kernel against the torchao Triton/CuTeDSL
kernels, we must run TE's fused call under the profiler and attribute CUDA self-time by
kernel name. This script fixes that name -> kernel mapping empirically, so the collection
script does not have to guess it (and can be re-run when TE updates).

Shapes are the ones the torchao benchmark README publishes a TE breakdown for, so the
printed times are directly checkable against it:

    linear   (2048, 7168)   amax 4.47   quantize 12.98
    grouped  E=4 x 2048 x 7168   amax 25.81   quantize 38.32
    2D weight (2048, 7168)  quantize_transpose 13.80  amax 5.40  zero_amax 1.34

Run from the torchao submodule root so its `torchao` package shadows the stale
site-packages copy:

    cd third_party/torchao && \
        PYTHONPATH=. python ../../deepseek_v3/kernel_analysis/nvfp4_671b_te_kernel_probe.py
"""

import os

import torch
from torch.profiler import ProfilerActivity, profile

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

WARMUP = 15
ITERS = 50

M, N = 2048, 7168
EXPERTS = 4


def per_kernel_us(fn, warmup=WARMUP, iters=ITERS):
    """{cuda kernel name: self-time us per call}, memcpy/memset excluded.

    Same accounting as benchmarks/prototype/nvfp4_training/bench_utils.kernel_time_us,
    but broken out by kernel instead of summed.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = {}
    for event in prof.key_averages():
        us = getattr(event, "self_device_time_total", 0) or getattr(
            event, "self_cuda_time_total", 0
        )
        key = event.key.lower()
        if us and "memcpy" not in key and "memset" not in key:
            out[event.key] = us / iters
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def rht_quantizer():
    """The activation quantizer: RHT + post-RHT amax + row/col quantize, one call.

    Matches test_hadamard_quantize_row_col.py's TE conformance test exactly, so the
    kernels timed here are the ones torchao is tested bitwise against.
    """
    quantizer = te.NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_random_sign_mask=True,
        stochastic_rounding=False,
    )
    quantizer.optimize_for_gemm = True
    return quantizer


def weight_quantizer():
    """The 2D weight quantizer: no RHT, 16x16 block scaling."""
    quantizer = te.NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_2d_quantization=True,
    )
    quantizer.optimize_for_gemm = True
    return quantizer


def show(label, kernels):
    print(f"\n=== {label} ===")
    for name, us in kernels.items():
        print(f"  {us:9.3f} us  {name}")
    print(f"  {sum(kernels.values()):9.3f} us  TOTAL ({len(kernels)} kernels)")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    torch.manual_seed(123)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"warmup {WARMUP}, iters {ITERS}\n")

    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    Ag = torch.randn(EXPERTS * M, N, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    linear_q = rht_quantizer()
    grouped_qs = [rht_quantizer() for _ in range(EXPERTS)]
    weight_q = weight_quantizer()

    paths = [
        (f"linear activation ({M}, {N})", lambda: linear_q(A)),
        (
            f"grouped activation (E={EXPERTS}, {M}, {N}) split_quantize",
            lambda: tex.split_quantize(Ag, [M] * EXPERTS, grouped_qs),
        ),
        (f"2D weight ({M}, {N})", lambda: weight_q(W)),
    ]

    for math_mode in ("standard", "fast"):
        if math_mode == "fast":
            os.environ["NVTE_USE_FAST_MATH"] = "1"
        else:
            os.environ.pop("NVTE_USE_FAST_MATH", None)
        print(f"\n{'#' * 70}\n# NVTE_USE_FAST_MATH={'1' if math_mode == 'fast' else 'unset'}\n{'#' * 70}")
        for label, fn in paths:
            show(f"{label} [{math_mode}]", per_kernel_us(fn))

    os.environ.pop("NVTE_USE_FAST_MATH", None)


if __name__ == "__main__":
    main()
