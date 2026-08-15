"""TorchAO NVFP4 quantize kernels vs TransformerEngine, at DSV3 expert dims.

TorchAO's CuteDSL NVFP4 kernels are a port of TE's; this answers whether the port is
correct and whether it is competitive. Two passes over the same shapes:

  conformance -- per-tensor byte-difference percentages, torchao vs TE, for the FP4
    codes and the FP8 scale factors. Run under both NVTE_USE_FAST_MATH settings: TE's
    default inserts an IEEE divide and a bf16 accumulator round-trip "for bit-wise
    compatibility with unfused kernels", while its fast-math path uses
    reciprocal_approximate_ftz, which is the rcp.approx torchao emits. Expect all zeros
    under NVTE_USE_FAST_MATH=1.

  timing -- device kernel time via torchao's own bench_utils.kernel_time_us, so the
    numbers are directly comparable to the tables in torchao's benchmark README.

TE's amax entry points have no Python binding (nvte_hadamard_transform_amax is reachable
only inside a quantizer that also casts), so the amax kernels are attributed from the
profiler by name rather than compared head to head. TE 2.19 has no grouped 2D weight
quantize at all (extensions/cast.cpp:154), so that row has no TE column.

Lives here rather than in third_party/torchao because torchao must gain no TE dependency.

Run from the repo root (torchao is editable-installed):
    python deepseek_v3/kernel_analysis/nvfp4_quantize_kernels_vs_te.py
    NVTE_USE_FAST_MATH=1 python deepseek_v3/kernel_analysis/nvfp4_quantize_kernels_vs_te.py
Produces the tables in nvfp4_quantize_kernels_vs_te.md.
"""

import os
import sys
from pathlib import Path

import torch

# torchao's `benchmarks` package is not installed (only `torchao` is, editable), so reach
# it through the submodule checkout. Reusing its kernel_time_us keeps these numbers
# directly comparable to the tables in torchao's own benchmark README.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "torchao"))

# transformer_engine.pytorch must be imported before transformer_engine_torch: it loads
# libtransformer_engine.so, without which the compiled ext fails on an undefined symbol.
import transformer_engine.pytorch  # noqa: F401  isort:skip
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

from benchmarks.prototype.nvfp4_training.bench_utils import kernel_time_us
from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)
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
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
    cutedsl_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
    triton_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
    triton_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_cutedsl import (
    cutedsl_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_triton import (
    triton_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    DEFAULT_SIGN_VECTOR,
)
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_cutedsl import (
    cutedsl_weight_quantize_2d,
)
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
    triton_weight_quantize_2d,
)

DEVICE = torch.device("cuda")
SV = list(DEFAULT_SIGN_VECTOR)
LOCAL_EXPERTS = 4
FAST_MATH = os.environ.get("NVTE_USE_FAST_MATH", "") == "1"


def make_quantizer(*, with_rht: bool, with_2d: bool = False):
    """TE quantizer configured to emit what torchao emits.

    optimize_for_gemm=True is the fair setting: torchao always emits swizzled scales, and
    TE emits plain unless asked. with_post_rht_amax is not optional -- TE raises without
    it when with_rht is set.
    """
    q = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=with_rht,
        with_post_rht_amax=with_rht,
        with_2d_quantization=with_2d,
        with_random_sign_mask=True,
    )
    q.optimize_for_gemm = True
    return q


def rht_fusion_eligible(rows: int, cols: int, *, grouped: bool) -> bool:
    """TE's is_eligible_for_rht_cast_fusion (quantizer.cpp:1929).

    The unfused path materializes a full (cols, rows) bf16 intermediate and runs three
    kernels, so an unfused row is off by a large factor, not a small one -- worth
    flagging rather than silently comparing.
    """
    return rows % (128 if grouped else 64) == 0 and cols % 128 == 0


def pct_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Percentage of differing bytes between two tensors of equal byte length."""
    av = a.flatten().contiguous().view(torch.uint8)
    bv = b.flatten().contiguous().view(torch.uint8)
    assert av.numel() == bv.numel(), f"{av.numel()} vs {bv.numel()} bytes"
    return 100.0 * (av != bv).float().mean().item()


# ---------------------------------------------------------------------------
# Linear (single-tensor) comparison
# ---------------------------------------------------------------------------


def compare_linear(M: int, N: int, seed: int = 0):
    """torchao vs TE on the fused RHT quantize and the 2D weight quantize."""
    torch.manual_seed(seed)
    A = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    col_amax, row_amax = triton_rht_amax(A, SV)

    q_rht = make_quantizer(with_rht=True)
    # nvfp4_quantize_with_amax skips TE's amax pass, which is what makes it a
    # like-for-like counterpart to torchao's quantize-only op.
    te = tex.nvfp4_quantize_with_amax(A, q_rht, row_amax.reshape(1), col_amax.reshape(1))

    rows = []
    for name, op in (("triton", triton_rht_quantize_row_col), ("cutedsl", cutedsl_rht_quantize_row_col)):
        cf, csf, rf, rsf = op(A, col_amax, row_amax, SV)
        rows.append(
            {
                "kernel": f"rht_quantize/{name}",
                "row codes": pct_diff(rf, te._rowwise_data),
                "col codes": pct_diff(cf, te._columnwise_data),
                "row sf": pct_diff(rsf, te._rowwise_scale_inv),
                "col sf": pct_diff(csf, te._columnwise_scale_inv),
            }
        )

    W = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    w_amax = W.float().abs().max()
    q_2d = make_quantizer(with_rht=False, with_2d=True)
    te_w = tex.nvfp4_quantize_with_amax(
        W, q_2d, w_amax.reshape(1), w_amax.reshape(1)
    )
    for name, op in (("triton", triton_weight_quantize_2d), ("cutedsl", cutedsl_weight_quantize_2d)):
        codes, sf, t_codes, t_sf = op(W, w_amax)
        rows.append(
            {
                "kernel": f"weight_2d/{name}",
                "row codes": pct_diff(codes, te_w._rowwise_data),
                "col codes": pct_diff(t_codes, te_w._columnwise_data),
                "row sf": pct_diff(sf, te_w._rowwise_scale_inv),
                "col sf": pct_diff(t_sf, te_w._columnwise_scale_inv),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Grouped comparison
# ---------------------------------------------------------------------------


def grouped_metadata(offsets: torch.Tensor):
    """torchao's cumulative int32 row-ends -> TE's int64 device metadata.

    Both are passed: supplying tensor_offsets short-circuits TE's
    build_grouped_tensor_offsets, which otherwise launches an extra kernel inside the
    timed region on every call.
    """
    ends = offsets.to(torch.int64)
    tensor_offsets = torch.cat([ends.new_zeros(1), ends])
    return torch.diff(tensor_offsets), tensor_offsets


def compare_grouped(E: int, M: int, N: int, seed: int = 0):
    """torchao vs TE on the grouped fused RHT quantize."""
    torch.manual_seed(seed)
    psl = E * M
    A = torch.randn((psl, N), dtype=torch.bfloat16, device=DEVICE)
    offsets = torch.arange(1, E + 1, dtype=torch.int32, device=DEVICE) * M
    lpl = offsets[-1:]
    first_dims, tensor_offsets = grouped_metadata(offsets)

    args = (A, SV, offsets, E, psl, N, VARYING_FIRST_DIM)
    col_amax, row_amax = triton_group_rht_amax(*args, logical_packed_length=lpl)

    q = make_quantizer(with_rht=True)
    te = tex.nvfp4_group_quantize_with_amax(
        A,
        q,
        E,
        first_dims,
        rowwise_amax=row_amax,
        columnwise_amax=col_amax,
        tensor_offsets=tensor_offsets,
    )
    te_parts = te.split_into_quantized_tensors()

    rows = []
    for name, op in (
        ("triton", triton_group_rht_quantize_row_col),
        ("cutedsl", cutedsl_group_rht_quantize_row_col),
    ):
        qa, sfa, qd, sfd = op(
            *args, row_amax, col_amax, None, False, logical_packed_length=lpl
        )
        # TE returns one quantized tensor per group. Codes concatenate along the packed
        # axis; scales are already swizzled per group (optimize_for_gemm), and torchao's
        # columnwise buffer is likewise a flat concatenation of per-group blocked
        # buffers, so the byte sequences line up directly.
        te_row_codes = torch.cat([p._rowwise_data for p in te_parts], dim=0)
        te_row_sf = torch.cat(
            [p._rowwise_scale_inv.flatten() for p in te_parts]
        )
        te_col_codes = torch.cat([p._columnwise_data for p in te_parts], dim=1)
        te_col_sf = torch.cat(
            [p._columnwise_scale_inv.flatten() for p in te_parts]
        )
        rows.append(
            {
                "kernel": f"group_rht_quantize/{name}",
                "row codes": pct_diff(qa, te_row_codes),
                "col codes": pct_diff(qd, te_col_codes),
                "row sf": pct_diff(sfa, te_row_sf),
                "col sf": pct_diff(sfd, te_col_sf),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_linear(M: int, N: int):
    torch.manual_seed(0)
    A = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    col_amax, row_amax = triton_rht_amax(A, SV)
    q = make_quantizer(with_rht=True)
    ra, ca = row_amax.reshape(1), col_amax.reshape(1)
    return {
        "triton": kernel_time_us(lambda: triton_rht_quantize_row_col(A, col_amax, row_amax, SV)),
        "cutedsl": kernel_time_us(lambda: cutedsl_rht_quantize_row_col(A, col_amax, row_amax, SV)),
        "te": kernel_time_us(lambda: tex.nvfp4_quantize_with_amax(A, q, ra, ca)),
        "fused": rht_fusion_eligible(M, N, grouped=False),
    }


def time_weight_2d(M: int, N: int):
    torch.manual_seed(0)
    W = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    amax = W.float().abs().max()
    q = make_quantizer(with_rht=False, with_2d=True)
    a = amax.reshape(1)
    return {
        "triton": kernel_time_us(lambda: triton_weight_quantize_2d(W, amax)),
        "cutedsl": kernel_time_us(lambda: cutedsl_weight_quantize_2d(W, amax)),
        "te": kernel_time_us(lambda: tex.nvfp4_quantize_with_amax(W, q, a, a)),
        "fused": True,  # non-RHT 2D swizzle fusion needs only rows%128 and cols%128
    }


def time_grouped_rht(E: int, M: int, N: int):
    torch.manual_seed(0)
    psl = E * M
    A = torch.randn((psl, N), dtype=torch.bfloat16, device=DEVICE)
    offsets = torch.arange(1, E + 1, dtype=torch.int32, device=DEVICE) * M
    lpl = offsets[-1:]
    first_dims, tensor_offsets = grouped_metadata(offsets)
    args = (A, SV, offsets, E, psl, N, VARYING_FIRST_DIM)
    col_amax, row_amax = triton_group_rht_amax(*args, logical_packed_length=lpl)
    # No TE column: the grouped TE invocation here is unvalidated (see main()), and an
    # unvalidated configuration can be fast for the wrong reason. Timing it would be
    # worse than not timing it.
    del first_dims, tensor_offsets
    return {
        "triton": kernel_time_us(
            lambda: triton_group_rht_quantize_row_col(
                *args, row_amax, col_amax, None, False, logical_packed_length=lpl
            )
        ),
        "cutedsl": kernel_time_us(
            lambda: cutedsl_group_rht_quantize_row_col(
                *args, row_amax, col_amax, None, False, logical_packed_length=lpl
            )
        ),
        "te": None,
        "fused": rht_fusion_eligible(M, N, grouped=True),
    }


def time_grouped_amax(E: int, M: int, N: int):
    """torchao's grouped amax. TE has no counterpart reachable from PyTorch."""
    torch.manual_seed(0)
    psl = E * M
    A = torch.randn((psl, N), dtype=torch.bfloat16, device=DEVICE)
    offsets = torch.arange(1, E + 1, dtype=torch.int32, device=DEVICE) * M
    lpl = offsets[-1:]
    args = (A, SV, offsets, E, psl, N, VARYING_FIRST_DIM)
    return {
        "triton": kernel_time_us(
            lambda: triton_group_rht_amax(*args, logical_packed_length=lpl)
        ),
        "cutedsl": kernel_time_us(
            lambda: cutedsl_group_rht_amax(*args, logical_packed_length=lpl)
        ),
        "te": None,
        "fused": rht_fusion_eligible(M, N, grouped=True),
    }


def time_grouped_weight_2d(E: int, M: int, N: int):
    """torchao's grouped 2D weight quantize. TE 2.19 cannot do this at all:
    NVTE_CHECK(!with_2d_quantization, "2D scaling grouped quant kernel is not ready yet")
    at extensions/cast.cpp:154, and the grouped non-RHT path is unimplemented too (:158).
    """
    torch.manual_seed(0)
    W = torch.randn((E, M, N), dtype=torch.bfloat16, device=DEVICE)
    amax = W.float().abs().amax(dim=(1, 2)).contiguous()
    return {
        "cutedsl": kernel_time_us(lambda: cutedsl_group_weight_quantize_2d(W, amax, E)),
        "te": None,
        "fused": True,
    }


# ---------------------------------------------------------------------------


def fmt(x, width=9, prec=3):
    return "n/a".rjust(width) if x is None else f"{x:{width}.{prec}f}"


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    print(f"NVTE_USE_FAST_MATH={'1' if FAST_MATH else '<unset>'}   "
          f"device={torch.cuda.get_device_name(0)}")

    shapes = get_deepseek_v3_weight_shapes(factorized_experts=LOCAL_EXPERTS)

    print("\n=== conformance: % of bytes differing from TE (single-tensor ops only) ===")
    print("grouped ops are omitted: TE's grouped path at E=1 does not reproduce TE's own")
    print("single-tensor output under this invocation, so the grouped TE configuration")
    print("here is unvalidated and any number it produced would be meaningless.")
    hdr = f"{'model':<11} {'projection':<16} {'kernel':<26} " \
          f"{'row codes':>10} {'col codes':>10} {'row sf':>8} {'col sf':>8}"
    print(hdr)
    for s in shapes:
        for r in compare_linear(s.m, s.n):
            print(f"{s.model:<11} {s.projection:<16} {r['kernel']:<26} "
                  f"{r['row codes']:>10.4f} {r['col codes']:>10.4f} "
                  f"{r['row sf']:>8.4f} {r['col sf']:>8.4f}")

    print("\n=== timing: device kernel time (us) ===")
    print(f"{'model':<11} {'projection':<16} {'op':<24} {'E':>3} {'M':>6} {'N':>6} "
          f"{'triton':>9} {'cutedsl':>9} {'te':>9} {'cutedsl/te':>11} {'path':>7}")
    benches = (
        ("rht_quantize", lambda s: time_linear(s.m, s.n), 1),
        ("weight_quantize_2d", lambda s: time_weight_2d(s.m, s.n), 1),
        ("group_rht_amax", lambda s: time_grouped_amax(s.experts, s.m, s.n), None),
        ("group_rht_quantize", lambda s: time_grouped_rht(s.experts, s.m, s.n), None),
        ("group_weight_quantize_2d",
         lambda s: time_grouped_weight_2d(s.experts, s.m, s.n), None),
    )
    for s in shapes:
        for op, fn, e_override in benches:
            r = fn(s)
            e = e_override or s.experts
            ratio = (
                f"{r['cutedsl'] / r['te']:.2f}x" if r.get("te") else "n/a"
            )
            print(f"{s.model:<11} {s.projection:<16} {op:<24} {e:>3} {s.m:>6} {s.n:>6} "
                  f"{fmt(r.get('triton'))} {fmt(r.get('cutedsl'))} {fmt(r.get('te'))} "
                  f"{ratio:>11} {'fused' if r['fused'] else 'UNFUSED':>7}")


if __name__ == "__main__":
    main()
