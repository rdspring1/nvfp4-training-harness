"""Grouped NVFP4 quantize kernels: CuteDSL vs Triton over tokens/expert.

The three forward-path grouped quantize ops now have both a Triton and a CuteDSL
backend, and `_resolve_backends` picks CuteDSL under the default AUTO preference.
This sweeps both at the DSV3-671B activation dim (N=7168) to attribute the
end-to-end training numbers (Table 4) to per-kernel device time.

Reports **device kernel time** via the upstream `kernel_time_us` helper, which
excludes host/custom-op dispatch — NVFP4 training runs these under CUDA graphs, so
device time is the metric that carries. The two backends' public ops take identical
arguments, so the same call site drives both.

Produces the CuteDSL-vs-Triton table in nvfp4_grouped_gemm_crossover.md. Run from
the torchao repo root:

    cd third_party/torchao && \\
        PYTHONPATH=. python ../../deepseek_v3/kernel_analysis/nvfp4_grouped_quant_backends.py
"""

import torch
from tabulate import tabulate

from benchmarks.prototype.nvfp4_training.bench_utils import kernel_time_us
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
    cutedsl_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
    triton_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    VARYING_FIRST_DIM,
)
from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_cutedsl import (
    cutedsl_group_weight_quantize_2d,
)
from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
    triton_group_weight_quantize_2d,
)
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
    cutedsl_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
    triton_group_rht_quantize_row_col,
)
from torchao.utils import is_sm_at_least_100

dev = torch.device("cuda")
SIGN = [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1]

E = 8
N_ACT = 7168  # DSV3-671B gate/up activation dim
TOKENS = [512, 1024, 2048, 4096, 8192]
WEIGHT_SHAPES = [("gate/up", 2048, 7168), ("down", 7168, 2048)]


def act_args(e, m, n):
    total = e * m
    A = torch.randn(total, n, dtype=torch.bfloat16, device=dev)
    offsets = torch.arange(1, e + 1, dtype=torch.int32, device=dev) * m
    return A, offsets, total


def amax_pair(e, m, n):
    A, offsets, total = act_args(e, m, n)
    args = (A, SIGN, offsets, e, total, n, VARYING_FIRST_DIM)
    return (
        kernel_time_us(lambda: triton_group_rht_amax(*args)),
        kernel_time_us(lambda: cutedsl_group_rht_amax(*args)),
    )


def quant_pair(e, m, n):
    A, offsets, total = act_args(e, m, n)
    amax = A.view(e, m, n).float().abs().amax(dim=(1, 2)).contiguous()
    args = (
        A, SIGN, offsets, e, total, n, VARYING_FIRST_DIM, amax, amax, None, False,
    )
    return (
        kernel_time_us(lambda: triton_group_rht_quantize_row_col(*args)),
        kernel_time_us(lambda: cutedsl_group_rht_quantize_row_col(*args)),
    )


def weight_pair(e, m, n):
    W = torch.randn((e, m, n), dtype=torch.bfloat16, device=dev)
    amax = W.float().abs().amax(dim=(1, 2)).contiguous()
    return (
        kernel_time_us(lambda: triton_group_weight_quantize_2d(W, amax, e)),
        kernel_time_us(lambda: cutedsl_group_weight_quantize_2d(W, amax, e)),
    )


def main():
    if not is_sm_at_least_100():
        raise RuntimeError("requires SM100+ (Blackwell)")
    torch.manual_seed(0)

    for name, fn in [
        ("rht_amax (act)", amax_pair),
        ("rht_quantize_row_col (act)", quant_pair),
    ]:
        rows = []
        for tpe in TOKENS:
            t, c = fn(E, tpe, N_ACT)
            rows.append([tpe, E * tpe, round(t, 1), round(c, 1), round(t / c, 2)])
        print(f"\n### {name}  (E={E}, N={N_ACT})")
        print(tabulate(rows, headers=["tok/exp", "rows", "triton_us",
                                      "cutedsl_us", "speedup"]))

    rows = []
    for proj, m, n in WEIGHT_SHAPES:
        t, c = weight_pair(E, m, n)
        rows.append([proj, f"{m}x{n}", round(t, 1), round(c, 1), round(t / c, 2)])
    print(f"\n### weight quantize_2d  (E={E}, per-expert weight)")
    print(tabulate(rows, headers=["proj", "shape", "triton_us",
                                  "cutedsl_us", "speedup"]))


if __name__ == "__main__":
    main()
