"""Grouped-GEMM crossover: bf16 vs NVFP4 (pure kernel, pre-quantized) vs NVFP4
(quantize+GEMM), random data, one grouped GEMM at 671B gate-proj shape
(K=7168, N=2048, E=8). Sweeps tokens/expert. Produces the two tables in
nvfp4_grouped_gemm_crossover.md.

  pure  = 4-bit tensor-core matmul only (inputs pre-quantized once)  -> compute crossover
  full  = pure + on-the-fly RHT quantize of x and weights           -> real training cost

Run from the repo root so `torchao` is importable:
    PYTHONPATH=. python deepseek_v3/nvfp4_grouped_gemm_crossover.py
"""

import torch
import torch.nn.functional as F

from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
    _scaled_grouped_mm,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
    triton_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
    triton_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
    triton_group_weight_quantize_2d,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    VARYING_FIRST_DIM,
)

K = 7168
N = 2048
E = 8
TOKENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
SIGN = list(1 if i % 2 == 0 else -1 for i in range(16))
FP4 = torch.float4_e2m1fn_x2
WARMUP, ITERS = 5, 20


def time_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def quantize(x, w, offs):
    M = x.shape[0]
    logical = offs[-1:]
    xc_amax, xr_amax = triton_group_rht_amax(
        x, SIGN, offs, E, M, K, VARYING_FIRST_DIM, logical_packed_length=logical
    )
    xr_codes, xr_sf, _, _ = triton_group_rht_quantize_row_col(
        x, SIGN, offs, E, M, K, VARYING_FIRST_DIM, xr_amax, xc_amax,
        rng_state=None, enable_stochastic_rounding=False, logical_packed_length=logical,
    )
    w_amax = w.float().abs().amax(dim=(1, 2))
    w_codes, w_sf, _, _ = triton_group_weight_quantize_2d(w, w_amax, E)
    return xr_codes, xr_sf, xr_amax, w_codes, w_sf, w_amax


def gemm(xr_codes, xr_sf, xr_amax, w_codes, w_sf, w_amax, offs):
    return _scaled_grouped_mm(
        xr_codes.view(FP4),
        w_codes.view(FP4).transpose(-2, -1),
        xr_sf, xr_amax, w_sf.flatten(1), w_amax, offs,
    )


def main():
    torch.manual_seed(0)
    print(f"# grouped GEMM  K={K} N={N} E={E}  (gate proj)  random data")
    print(f"{'tok/exp':>7} {'M':>7} {'bf16':>8} {'nvfp4_pure':>11} {'nvfp4_full':>11}"
          f" {'pure/bf16':>9} {'full/bf16':>9}")
    for tpe in TOKENS:
        M = E * tpe
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = (0.02 * torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16))
        offs = torch.arange(tpe, M + 1, tpe, device="cuda", dtype=torch.int32)

        wt = w.transpose(-2, -1).contiguous()  # (E, K, N) for bf16 grouped mm
        t_bf16 = time_ms(lambda: torch._grouped_mm(x, wt, offs=offs))

        q = quantize(x, w, offs)
        t_pure = time_ms(lambda: gemm(*q, offs))
        t_full = time_ms(lambda: gemm(*quantize(x, w, offs), offs))

        print(f"{tpe:>7} {M:>7} {t_bf16:>8.3f} {t_pure:>11.3f} {t_full:>11.3f}"
              f" {t_pure/t_bf16:>9.2f} {t_full/t_bf16:>9.2f}")


if __name__ == "__main__":
    main()
