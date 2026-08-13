"""Forward-only grouped GEMM: bf16 vs TorchAO NVFP4 (CuteDSL and Triton backends).

One grouped GEMM at the DSV3-671B gate-projection shape (K=7168, N=2048, E=8),
forward only, uniform tokens/expert. `nvfp4_full` quantizes activations and weights
on every call, so this measures the per-call quantization tax against the bf16
reference and locates the forward crossover. Produces Table 2 in
nvfp4_grouped_gemm_crossover.md.

    PYTHONPATH=third_party/torchao python \\
        deepseek_v3/kernel_analysis/nvfp4_grouped_gemm_fwd.py <bf16|cutedsl|triton>
"""

import sys

import torch

from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
    _to_nvfp4_rht_rs_then_scaled_grouped_mm,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

K = 7168
N = 2048
E = 8
SIGN = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)
WARMUP = 5
ITERS = 20

TOKENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
PREF = {"cutedsl": KernelPreference.AUTO, "triton": KernelPreference.TRITON}


def bench(backend, tpe):
    rows = E * tpe
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    offs = (torch.arange(1, E + 1, device="cuda", dtype=torch.int32) * tpe).contiguous()
    sr_seed = torch.zeros(1, device="cuda", dtype=torch.int64)

    if backend == "bf16":
        def step():
            return torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)
    else:
        pref = PREF[backend]

        def step():
            return _to_nvfp4_rht_rs_then_scaled_grouped_mm(
                x, w, SIGN, sr_seed, offs=offs, kernel_preference=pref
            )

    with torch.no_grad():
        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            step()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def main():
    backend = sys.argv[1]
    torch.manual_seed(0)
    print(f"# backend={backend} fwd-only K={K} N={N} E={E}")
    print(f"{'tok/exp':>7} {'rows':>7} {'ms':>8}")
    for tpe in TOKENS:
        print(f"{tpe:>7} {E * tpe:>7} {bench(backend, tpe):>8.3f}")


if __name__ == "__main__":
    main()
