"""Standalone repro: NVFP4 block-scale layout when the contraction dim is grouped.

No torchao, no TransformerEngine, no torchtitan -- torch only. Run on sm100:

    python scaled_grouped_mm_kgrouped_repro.py

Background: `nvfp4_module_te_vs_ao.md` traces a 0.070 training-loss gap in an
NVFP4 MoE down to the weight-gradient GEMM. That is the only GEMM in the recipe
whose *contraction* dimension is the grouped one (`offs` partitions M in
`A(N,M) @ B(M,K) -> (G,N,K)`), and it disagrees with a dequantized reference by
relL2 0.41 at 4 groups while being exact at 1.

This reproduces it with no quantizer in the loop. FP4 codes are drawn straight
from the e2m1 grid and every block scale is a power of two, so dequantization is
*exact*: the fp64 reference is the correct answer, not an approximation of it,
and the only error the kernel is entitled to is bf16 output rounding (~2e-3).

Conclusion, which the two controls below establish: **the kernel is right and
the scale layout torchao hands it is wrong.** For a grouped contraction dim each
group's block scales must be swizzled independently and the flattened blocked
buffers concatenated -- which is what `_check_scales_blocked` documents as
`rounded_up_per_group(K/blocksize, 4)`
(`aten/src/ATen/native/cuda/GroupedBlas.cpp:401-403`). torchao swizzles once
over the whole packed M
(`group_rht_quantize_row_col_triton.py:305-313`), which coincides with the
required layout only at a single group.

Two controls make that attribution rather than a guess:
  * G=1 uses the same packing, swizzle and call, and is exact. So the packing
    and swizzle constructed here are correct.
  * With every block scale set to 1.0, all group counts are exact. So the codes,
    `offs` handling and accumulation are all fine, and only scale *addressing*
    is implicated -- a uniform scale is the one case where mis-indexing cannot
    show up.
"""

import torch
import torch.nn.functional as F

# e2m1: sign in bit 3, then the 8 magnitudes. Note the grid is not uniform.
E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_TABLE = torch.tensor(E2M1_GRID + [-v for v in E2M1_GRID], dtype=torch.float64)

N, K, M = 256, 256, 1024
GROUPS = (1, 2, 4, 8)
BF16_ROUNDING = 1e-2  # bf16 carries ~3 decimal digits; past this is not rounding

SCALE_RECIPE = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
SWIZZLE = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """Pack a (R, C) block-scale array into SWIZZLE_32_4_4 byte order.

    Tiles of 128 rows x 4 scale-columns; within a tile the 128 rows are
    traversed as (4, 32) transposed to (32, 4). Requires R % 128 == 0 and
    C % 4 == 0, which every shape here satisfies.
    """
    R, C = scales.shape
    blocks = scales.view(R // 128, 128, C // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(R, C)


def blocked_layout(scales: torch.Tensor, groups: int, variant: str) -> torch.Tensor:
    """The three candidate readings of "block scales for a grouped contraction dim".

    ``whole``      one swizzle over the full M, letting the kernel slice out each
                   group's columns. This is what torchao builds.
    ``per-group``  each group swizzled independently, flattened blocked buffers
                   concatenated end to end. This is the documented contract.
    ``interleaved``the same per-group swizzle but concatenated along the column
                   axis instead of flat -- included because it is the natural
                   thing to write and it is wrong, at row granularity.
    """
    R, C = scales.shape
    if variant == "whole":
        return to_blocked(scales)
    cols = C // groups
    parts = [to_blocked(scales[:, g * cols:(g + 1) * cols]) for g in range(groups)]
    if variant == "interleaved":
        return torch.cat(parts, dim=1)
    return torch.cat([p.reshape(-1) for p in parts]).view(R, C)


def make_operand(rows: int, device, seed: int, uniform_scales: bool):
    """Random e2m1 codes plus block scales, and the exact values they denote.

    Scales vary per (row, 16-element block) unless *uniform_scales*: a uniform
    scale is precisely the case where a scale-indexing bug cannot show up, which
    is what makes it useful as a control. Powers of two keep both the e4m3 scale
    and the dequantized product exactly representable.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    nibbles = torch.randint(0, 16, (rows, M), generator=g, device=device, dtype=torch.uint8)
    exponents = (torch.zeros(rows, M // 16, device=device) if uniform_scales
                 else torch.randint(-3, 4, (rows, M // 16), generator=g, device=device))
    scales = torch.pow(2.0, exponents.double())

    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    exact = E2M1_TABLE.to(device)[nibbles.long()] * scales.repeat_interleave(16, dim=1)
    return packed, scales.to(torch.float8_e4m3fn), exact


def run(device, groups: int, variant: str, operands) -> float:
    (a_codes, a_scales, a_exact), (b_codes, b_scales, b_exact) = operands
    rows = M // groups
    offs = torch.arange(rows, M + 1, rows, device=device, dtype=torch.int32)
    out = F.scaled_grouped_mm(
        a_codes.view(torch.float4_e2m1fn_x2),
        b_codes.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
        scale_a=[blocked_layout(a_scales, groups, variant), torch.ones(groups, device=device)],
        scale_recipe_a=SCALE_RECIPE,
        scale_b=[blocked_layout(b_scales, groups, variant), torch.ones(groups, device=device)],
        scale_recipe_b=SCALE_RECIPE,
        swizzle_a=SWIZZLE,
        swizzle_b=SWIZZLE,
        offs=offs,
        output_dtype=torch.bfloat16,
    )
    ref = torch.stack([
        a_exact[:, g * rows:(g + 1) * rows] @ b_exact[:, g * rows:(g + 1) * rows].t()
        for g in range(groups)
    ])
    return float((out.double() - ref).norm() / ref.norm())


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
        print("SM100+ required")
        return 2
    device = torch.device("cuda")
    varying = tuple(make_operand(r, device, s, uniform_scales=False)
                    for r, s in ((N, 0), (K, 1)))
    uniform = tuple(make_operand(r, device, s, uniform_scales=True)
                    for r, s in ((N, 0), (K, 1)))

    print(f"A ({N}x{M}) @ B.T ({M}x{K}) -> (G,{N},{K}), contraction dim M grouped by offs")
    print("relL2 against the exact dequantized product; bf16 output rounding is ~2e-3\n")
    variants = ("whole", "interleaved", "per-group")
    print(f"{'groups':>7} {'rows/grp':>9} " + " ".join(f"{v:>12}" for v in variants)
          + f" {'uniform':>9}")

    results = {}
    for G in GROUPS:
        row = [run(device, G, v, varying) for v in variants]
        control = run(device, G, "whole", uniform)
        results[G] = dict(zip(variants, row)) | {"uniform": control}
        print(f"{G:>7} {M // G:>9} " + " ".join(f"{r:>12.5f}" for r in row)
              + f" {control:>9.5f}")

    multi = [g for g in GROUPS if g > 1]
    good = [v for v in variants if all(results[g][v] < BF16_ROUNDING for g in multi)]
    controls_ok = (all(results[g]["uniform"] < BF16_ROUNDING for g in GROUPS)
                   and all(results[1][v] < BF16_ROUNDING for v in variants))

    print()
    if not controls_ok:
        print("INCONCLUSIVE: a control failed, so this script's layout construction "
              "cannot be trusted to attribute the difference.")
        return 2
    print("Controls pass: exact at one group for every layout, and exact at every "
          "group count when all block scales are equal.")
    if good == ["per-group"]:
        print("The kernel requires each group's block scales swizzled independently "
              "and concatenated flat. torchao supplies one whole-M swizzle, so the "
              "wgrad GEMM reads every group's scales from the wrong offset.")
        return 1
    print(f"Unexpected: layouts accepted at G>1 = {good or 'none'}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
