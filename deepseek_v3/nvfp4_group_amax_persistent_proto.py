"""Prototype #1c: per-group-CTA persistent grouped RHT amax.

Launch grid = num_groups * CTAS_PER_GROUP. Each CTA is bound to ONE group and
strides over that group's tiles keeping an ELEMENTWISE cumulative max (exactly like
the single kernel _hadamard_amax_kernel) -- no in-loop reduction -- so it can use
warp_specialize=True. One atomic per CTA after the loop. This is the single kernel
generalized: num_groups=1, CTAS_PER_GROUP=num_sms recovers it. All groups run
concurrently (no serialization).

    cd third_party/torchao && PYTHONPATH=. python ../../deepseek_v3/nvfp4_group_amax_persistent_proto.py
"""

import torch
import triton
import triton.language as tl
from tabulate import tabulate

from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
    triton_group_rht_amax,
    _group_rht_amax_triton_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    BLOCK_M as GBM, BLOCK_N as GBN, VARYING_FIRST_DIM,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    _compute_pid, get_rht_matrix,
)
from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.utils import is_sm_at_least_100

dev = torch.device("cuda")
SIGN = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)
N_ACT = 7168


@triton.jit
def _pergroup_cta_amax(
    a_ptr, b_ptr, offsets_ptr, row_amax_ptr, col_amax_ptr, N,
    CTAS_PER_GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr, MAX_M: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[MAX_M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[16, 16], strides=[16, 1], block_shape=[16, 16])
    hadamard = b_desc.load([0, 0])

    pid = tl.program_id(0)
    g = pid // CTAS_PER_GROUP
    local = pid % CTAS_PER_GROUP

    g_start = tl.where(g == 0, 0, tl.load(offsets_ptr + g - 1))
    g_end = tl.load(offsets_ptr + g)
    num_pid_m = tl.cdiv(g_end - g_start, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_N * num_pid_m
    num_tiles = num_pid_m * num_pid_n

    cum_col = tl.zeros((BLOCK_N * BLOCK_M // 16, 16), dtype=tl.float32)
    cum_row = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for tile_id in tl.range(
        local, num_tiles, CTAS_PER_GROUP,
        flatten=False, warp_specialize=True, num_stages=NUM_STAGES,
    ):
        pid_n, pid_m = _compute_pid(tile_id, num_pid_in_group, num_pid_n, GROUP_SIZE_N)
        a = a_desc.load([g_start + pid_m * BLOCK_M, pid_n * BLOCK_N])
        a_t = tl.trans(a)
        a_t_r = tl.reshape(a_t, [BLOCK_N * BLOCK_M // 16, 16])
        a_t_rht = tl.dot(a_t_r, hadamard).to(tl.bfloat16)
        cum_col = tl.maximum(cum_col, tl.abs(a_t_rht), propagate_nan=tl.PropagateNan.ALL)
        cum_row = tl.maximum(cum_row, tl.abs(a.to(tl.float32)), propagate_nan=tl.PropagateNan.ALL)

    col = tl.max(tl.max(cum_col, axis=1), axis=0)
    col_nan = tl.max(tl.max((cum_col != cum_col).to(tl.int32), axis=1), axis=0)
    col = tl.where(col_nan != 0, float("nan"), col)
    tl.atomic_max(col_amax_ptr + g, col.to(tl.float32))

    row = tl.max(tl.max(cum_row, axis=1), axis=0)
    row_nan = tl.max(tl.max((cum_row != cum_row).to(tl.int32), axis=1), axis=0)
    row = tl.where(row_nan != 0, float("nan"), row)
    tl.atomic_max(row_amax_ptr + g, row.to(tl.float32))


def _set_alloc():
    triton.set_allocator(
        lambda size, align, stream: torch.empty(max(size, 1), dtype=torch.int8, device=dev))


def pergroup_amax(A, offsets, e, *, bm=128, bn=128, ns=3, nw=8):
    M, N = A.shape
    B = get_rht_matrix(SIGN, dev, torch.bfloat16, 16)
    row = torch.zeros((e,), dtype=torch.float32, device=dev)
    col = torch.zeros((e,), dtype=torch.float32, device=dev)
    num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ctas_per_group = max(1, num_sms // e)
    _set_alloc()
    _pergroup_cta_amax[(e * ctas_per_group,)](
        A, B, offsets, row, col, N,
        CTAS_PER_GROUP=ctas_per_group, BLOCK_M=bm, BLOCK_N=bn,
        GROUP_SIZE_N=8, MAX_M=M, NUM_STAGES=ns, num_warps=nw)
    return col, row


def build(e, m, n):
    A = torch.randn(e * m, n, dtype=torch.bfloat16, device=dev)
    offsets = (torch.arange(1, e + 1, dtype=torch.int32, device=dev) * m).contiguous()
    return A, offsets


def tiled(A, offsets, e):
    M, N = A.shape
    B = get_rht_matrix(SIGN, dev, torch.bfloat16, 16)
    r = torch.zeros((e,), dtype=torch.float32, device=dev)
    c = torch.zeros((e,), dtype=torch.float32, device=dev)
    lpl = offsets[-1:]
    grid = (triton.cdiv(M, GBM) * triton.cdiv(N, GBN),)

    def run():
        _group_rht_amax_triton_kernel[grid](
            A, B, offsets, r, c, M, N, num_tensors=e, SHAPE_REP=VARYING_FIRST_DIM,
            BLOCK_M=GBM, BLOCK_N=GBN, RHT_SIZE=16,
            logical_packed_length_ptr=lpl, num_warps=8, num_stages=3)
    return benchmark_cuda_function_in_microseconds(run)


def main():
    if not is_sm_at_least_100():
        raise RuntimeError("requires SM100+")
    torch.manual_seed(0)

    print("### bitwise validation vs tiled grouped kernel (oracle)")
    ok = True
    for e, m, n in [(2, 128, 256), (8, 512, 7168), (4, 2048, 2048), (2, 8192, 7168), (16, 256, 2048)]:
        A, offsets = build(e, m, n)
        ec, er = triton_group_rht_amax(A, list(SIGN), offsets, e, A.shape[0], n, VARYING_FIRST_DIM)
        ac, ar = pergroup_amax(A, offsets, e)
        c = torch.equal(ac, ec); r = torch.equal(ar, er); ok = ok and c and r
        print(f"  E={e:>3} M={m:>5} N={n:>5}: col={'OK' if c else 'FAIL'} row={'OK' if r else 'FAIL'}"
              + ("" if (c and r) else f"  d={(ac-ec).abs().max():.2e}/{(ar-er).abs().max():.2e}"))
    print(f"\nbitwise: {'ALL OK' if ok else 'MISMATCH'}\n")

    # Tile-shape sweep (mirrors the single kernel's autotune space, plus 128x128).
    CONFIGS = [
        (128, 32, 3, 8), (128, 64, 3, 8), (128, 64, 4, 8), (128, 128, 3, 8),
        (64, 32, 3, 4), (64, 64, 3, 4), (64, 128, 3, 8), (64, 64, 4, 8),
    ]
    E = 8
    rows = []
    for tpe in [512, 1024, 2048, 4096, 8192]:
        A, offsets = build(E, tpe, N_ACT)
        total = E * tpe
        tt = tiled(A, offsets, E)
        best_t, best_cfg = 1e9, None
        for (bm, bn, ns, nw) in CONFIGS:
            _set_alloc()
            try:
                t = benchmark_cuda_function_in_microseconds(
                    lambda: pergroup_amax(A, offsets, E, bm=bm, bn=bn, ns=ns, nw=nw))
            except Exception:
                continue
            if t < best_t:
                best_t, best_cfg = t, (bm, bn, ns, nw)
        rows.append([tpe, total, round(tt, 2), round(tt / total * 1000, 3),
                     round(best_t, 2), round(best_t / total * 1000, 3),
                     round(tt / best_t, 2), f"{best_cfg[0]}x{best_cfg[1]}/s{best_cfg[2]}/w{best_cfg[3]}"])
    print("### grouped tiled vs per-group-CTA persistent (#1c, best tile), E=8, N=7168")
    print(tabulate(rows, headers=["tok/exp", "rows", "tiled_us", "tiled_ns/row",
                                  "pers_us", "pers_ns/row", "speedup", "best_cfg"]))


if __name__ == "__main__":
    main()
