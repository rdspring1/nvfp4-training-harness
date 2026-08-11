#!/usr/bin/env python3
"""Bitwise audit of the TorchAO NVFP4 *forward* path against TransformerEngine.

Motivation
----------
NVFP4 GroupedExperts training tracks bf16 to ~step 50, then holds a small but
consistently positive loss bias that grows into real divergence. A persistent
one-directional bias is the signature of a systematic quantization error, so
every forward-path stage is compared bitwise against TE, which is the oracle
TorchAO's kernels were written to match (see the comment at
torchao/prototype/moe_training/nvfp4_training/hadamard_utils.py:241-246).

Scope is forward only. The forward activation row path uses RTNE with no
stochastic rounding and no RHT, so bitwise equality is actually attainable.

Stages
------
0   Calibrate: TE CUDA kernel vs TE's own pure-PyTorch NVFP4QuantizerRef.
1   Per-tensor (global) scale parity.
2   Activation rowwise quantize -- forward GEMM LHS.
3a  Weight 2D quantize -- forward GEMM RHS.
3b  Isolated arithmetic discriminators for the two suspect expressions in the
    weight quantizer (see below).
4   Forward grouped GEMM on identical operands.
5   Full GroupedExperts forward + forward determinism.
6   Padding-tail probe for the converter's offs[-1] = A.shape[0] rewrite.

The two suspects driving stage 3
--------------------------------
TorchAO has two NVFP4 block-quantize helpers that are *not* arithmetically
identical, and only one of them matches TE:

                     global_encode_scale        pvscale
  activations        tl.div_rn(448*6, amax)     vmax * (ges * (1/6))
  (hadamard_utils)
  weights            (448*6) / amax             (vmax / 6) * ges
  (quantize_2d)
  TE reference       torch.div(448*6, amax)     vmax * (ges * (1/6))

So the activation path matches TE on both counts and the weight path matches
on neither. The weight quantizer runs on every expert weight on every forward
step, which is exactly the shape of a systematic bias.

Usage
-----
  python nvfp4_forward_te_audit.py --stage 3b
  python nvfp4_forward_te_audit.py --all --model debugmodel
"""

from __future__ import annotations

import argparse
import sys

import torch
import triton
import triton.language as tl

from nvfp4_audit_common import (
    DEEPSEEK_V3_MODEL_SHAPES,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    FP32_MAX,
    ao_dequant,
    ao_mods,
    as_u8,
    build_grouped_experts,
    compare_nibbles,
    compare_e4m3,
    compare_exact,
    e4m3,
    get_deepseek_v3_weight_shapes,
    make_groups,
    moe_dims,
    print_table,
    record,
    require_blackwell,
    sign_vector,
    te_extract,
    te_mods,
    te_quantize,
    te_ref_quantize,
    unpack_fp4,
)


# --------------------------------------------------------------------------
# stage 3b -- isolated arithmetic discriminators
# --------------------------------------------------------------------------


def stage_3b_pvscale_order(device, models) -> None:
    """Suspect (B): pvscale association order, pure fp32 PyTorch.

    TE ref (quantization_ref_nvfp4.py:748-751) and TorchAO activations both do
    vmax * (ges * (1/6)). TorchAO weights do (vmax / 6) * ges. Both are exact
    in fp32 PyTorch, so this measures the real divergence with no emulation.

    NOTE: vmax must come from a real weight tensor, not from an independent
    random draw. vmax and amax are strongly correlated (vmax <= amax, and the
    block holding the tensor max gives pvscale exactly 448), and the ties this
    is looking for live precisely in that correlation. An uncorrelated probe
    reports zero divergence and is simply under-powered.
    """
    torch.manual_seed(0)
    for shape in get_deepseek_v3_weight_shapes(factorized_experts=1):
        if shape.model not in models:
            continue
        w = torch.randn(shape.m, shape.n, dtype=torch.bfloat16, device=device)
        amax = w.float().abs().amax()

        # 16x16 block maxima, matching the weight quantizer's 2D tiling.
        blocks = w.float().abs().unfold(0, 16, 16).unfold(1, 16, 16)
        vmax = torch.amax(blocks, dim=(-1, -2))

        ges = (FP8_E4M3_MAX * FP4_E2M1_MAX) / amax
        te_order = e4m3(vmax * (ges * (1.0 / FP4_E2M1_MAX)))
        ao_wgt_order = e4m3((vmax / FP4_E2M1_MAX) * ges)

        ok, ulp, nd, nt, note = compare_e4m3(ao_wgt_order, te_order)
        record(
            "3b-order",
            f"{shape.model} {shape.m}x{shape.n}",
            shape.projection,
            ok,
            ulp,
            nd,
            nt,
            note or "pvscale assoc order",
        )


@triton.jit
def _div_probe(amax_ptr, out_rn_ptr, out_plain_ptr, N, BLOCK: tl.constexpr):
    """Both division forms side by side in one kernel, same inputs."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    # Same constants the real kernels declare as tl.constexpr locals.
    NUMER: tl.constexpr = 448.0 * 6.0
    amax = tl.load(amax_ptr + off, mask=m, other=1.0)
    num = tl.full(amax.shape, NUMER, tl.float32)
    tl.store(out_rn_ptr + off, tl.div_rn(num, amax), mask=m)
    tl.store(out_plain_ptr + off, num / amax, mask=m)


def stage_3b_div_lowering(device, models) -> None:
    """Suspect (A): tl.div_rn vs Triton's plain '/'.

    PyTorch division is correctly-rounded IEEE, so plain '/' as Triton lowers
    it cannot be emulated from PyTorch. This runs both forms in one real
    Triton kernel and measures the divergence in the resulting e4m3 scale
    bytes, which is the quantity that actually reaches the GEMM.
    """
    torch.manual_seed(0)
    for shape in get_deepseek_v3_weight_shapes(factorized_experts=1):
        if shape.model not in models:
            continue
        # Realistic amax values: one per expert-sized weight draw.
        n_probe = 4096
        amaxes = torch.empty(n_probe, dtype=torch.float32, device=device)
        for i in range(0, n_probe, 256):
            w = torch.randn(256, shape.n, dtype=torch.bfloat16, device=device)
            amaxes[i : i + 256] = w.float().abs().amax(dim=1)

        out_rn = torch.empty_like(amaxes)
        out_plain = torch.empty_like(amaxes)
        _div_probe[(triton.cdiv(n_probe, 256),)](
            amaxes, out_rn, out_plain, n_probe, BLOCK=256
        )

        ok, _, nd, nt, _ = compare_exact(out_plain, out_rn)
        record(
            "3b-div",
            f"{shape.model} {shape.m}x{shape.n}",
            "global_encode_scale",
            ok,
            0,
            nd,
            nt,
            "plain '/' vs tl.div_rn, fp32 bits (input to the question)",
            info=True,
        )

        # Propagate both through to the e4m3 scale byte that reaches the GEMM.
        vmax = amaxes * torch.rand(n_probe, device=device).clamp_min(1e-3)
        s_rn = e4m3(vmax * (out_rn * (1.0 / FP4_E2M1_MAX)))
        s_plain = e4m3((vmax / FP4_E2M1_MAX) * out_plain)
        ok, ulp, nd, nt, note = compare_e4m3(s_plain, s_rn)
        record(
            "3b-div",
            f"{shape.model} {shape.m}x{shape.n}",
            "e4m3 scale byte",
            ok,
            ulp,
            nd,
            nt,
            note or "combined (A)+(B) effect",
        )


# --------------------------------------------------------------------------
# stage 0 -- calibrate the harness
# --------------------------------------------------------------------------


def stage_0_calibrate(device, models) -> None:
    """TE CUDA kernel vs TE's own reference. Must be bitwise.

    TE asserts this equality in its own suite. If it fails here the harness is
    wrong -- nibble order, scale slicing, or the byte-view convention -- and no
    TorchAO comparison downstream would mean anything.
    """
    torch.manual_seed(0)
    for M, N in ((256, 256), (512, 2048), (1408, 2048)):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        for two_d in (False, True):
            kind = "2D 16x16" if two_d else "1D 1x16"
            ref = te_ref_quantize(x, two_d=two_d)
            ref_codes = unpack_fp4(ref.data.view(torch.uint8))
            ref_sf, ref_amax = as_u8(ref.scale), ref.global_amax_row
            t = te_quantize(x, two_d=two_d, amax=ref_amax)
            got_codes, got_sf = te_extract(t, M, N)
            rs = ref_sf[:M, : N // 16]

            ok, ulp, nd, nt, note = compare_nibbles(got_codes, ref_codes[:M, :N])
            record("0-calib", f"{M}x{N}", f"codes {kind}", ok, ulp, nd, nt, note)
            ok, ulp, nd, nt, note = compare_e4m3(got_sf, rs)
            record("0-calib", f"{M}x{N}", f"scales {kind}", ok, ulp, nd, nt, note)


# --------------------------------------------------------------------------
# stage 1 -- per-tensor (global) scale parity
# --------------------------------------------------------------------------


def stage_1_global_scale(device, models) -> None:
    """TorchAO's decode scale vs the reciprocal of TE's encode scale.

    TorchAO hands scaled_grouped_mm per_tensor_amax_to_scale(amax) = amax/(448*6),
    while its own Triton kernels encode with div_rn(448*6, amax) and TE does the
    same. amax/(448*6) and 1/((448*6)/amax) are two fp32 roundings of the same
    quantity, so this measures whether the GEMM's decode scale is the exact
    inverse of the encode scale actually used.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

    torch.manual_seed(0)
    for model in models:
        dim, hidden, _ = moe_dims(model)
        for name, (m, n) in (("w1/w3", (hidden, dim)), ("w2", (dim, hidden))):
            w = torch.randn(m, n, dtype=torch.bfloat16, device=device)
            amax = w.float().abs().amax().reshape(1)

            ao_decode = per_tensor_amax_to_scale(amax)
            te_encode = (FP8_E4M3_MAX * FP4_E2M1_MAX) / amax
            te_decode = 1.0 / te_encode

            ok, _, nd, nt, _ = compare_exact(ao_decode, te_decode)
            rel = ((ao_decode - te_decode).abs() / te_decode).max().item()
            record(
                "1-gscale",
                f"{model} {m}x{n}",
                f"{name} decode scale",
                ok,
                0,
                nd,
                nt,
                f"amax/(448*6) vs 1/div(448*6,amax); max rel {rel:.2e}",
            )


# --------------------------------------------------------------------------
# stage 2 -- activation rowwise quantize (forward GEMM LHS)
# --------------------------------------------------------------------------


def stage_2_activations(device, models, num_groups=4, rows_per_group=256) -> None:
    """triton_group_rht_quantize_row_col row outputs vs TE, per expert group.

    Only the row outputs matter for forward: they carry no RHT and no
    stochastic rounding, so they are plain RTNE NVFP4 and directly comparable.
    """
    (
        triton_group_rht_amax,
        triton_group_rht_quantize_row_col,
        _,
        VARYING_FIRST_DIM,
        from_blocked,
    ) = ao_mods()

    torch.manual_seed(0)
    sv = sign_vector()
    for model in models:
        dim, hidden, _ = moe_dims(model)
        for name, K in (("w1/w3 in", dim), ("w2 in", hidden)):
            sizes, offs = make_groups(num_groups, rows_per_group, device)
            M = sum(sizes)
            A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            logical = offs[-1:]

            col_amax, row_amax = triton_group_rht_amax(
                A, sv, offs, num_groups, M, K, VARYING_FIRST_DIM,
                logical_packed_length=logical,
            )
            qa, sfa, _, _ = triton_group_rht_quantize_row_col(
                A, sv, offs, num_groups, M, K, VARYING_FIRST_DIM,
                row_amax, col_amax,
                rng_state=None, enable_stochastic_rounding=False,
                logical_packed_length=logical,
            )

            ao_codes = unpack_fp4(qa)
            ao_sf = as_u8(from_blocked(sfa, M, K // 16))

            # The grouped amax kernel must agree with a plain per-group amax;
            # a wrong amax would desync the per-tensor scale on both sides.
            ref_amax = torch.stack([A[s:e].float().abs().amax()
                                    for s, e in zip([0] + list(offs[:-1].tolist()), offs.tolist())])
            ok, _, nd, nt, _ = compare_exact(row_amax, ref_amax)
            record("2-act", f"{model} {M}x{K}", f"{name} row amax", ok, 0, nd, nt,
                   "grouped kernel vs per-group torch")

            bad_codes = bad_sf = 0
            tot_codes = tot_sf = 0
            worst_ulp = 0
            notes = set()
            start = 0
            for g, sz in enumerate(sizes):
                end = start + sz
                t = te_quantize(A[start:end].contiguous(), two_d=False, amax=row_amax[g])
                te_codes, te_sf = te_extract(t, sz, K)

                ok, ulp, nd, nt, note = compare_nibbles(ao_codes[start:end], te_codes)
                bad_codes += nd
                tot_codes += nt
                ok2, ulp2, nd2, nt2, note2 = compare_e4m3(ao_sf[start:end], te_sf)
                bad_sf += nd2
                tot_sf += nt2
                worst_ulp = max(worst_ulp, ulp2)
                if note2:
                    notes.add(note2.split(" ")[0])
                start = end

            record("2-act", f"{model} {M}x{K}", f"{name} codes", bad_codes == 0,
                   0, bad_codes, tot_codes, "vs TE per group")
            record("2-act", f"{model} {M}x{K}", f"{name} block scales", bad_sf == 0,
                   worst_ulp, bad_sf, tot_sf, " ".join(sorted(notes)) or "vs TE per group")


# --------------------------------------------------------------------------
# stage 3a -- weight 2D quantize (forward GEMM RHS)
# --------------------------------------------------------------------------


def stage_3a_weights(device, models, num_experts=4) -> None:
    """triton_group_weight_quantize_2d vs TE 2D quantize, per expert."""
    _, _, triton_group_weight_quantize_2d, _, from_blocked = ao_mods()

    torch.manual_seed(0)
    for model in models:
        dim, hidden, _ = moe_dims(model)
        for name, (N, K) in (("w1/w3", (hidden, dim)), ("w2", (dim, hidden))):
            W = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=device)
            # Exactly what nvfp4_grouped_mm.py:189 computes.
            w_amax = W.float().abs().amax(dim=(1, 2))
            codes, sf, _, _ = triton_group_weight_quantize_2d(W, w_amax, num_experts)

            bad_codes = bad_sf = tot_codes = tot_sf = 0
            worst_ulp = 0
            notes = set()
            for e in range(num_experts):
                t = te_quantize(W[e].contiguous(), two_d=True, amax=w_amax[e])
                te_codes, te_sf = te_extract(t, N, K)

                ao_codes = unpack_fp4(codes[e])
                ao_sf = as_u8(from_blocked(sf[e].reshape(-1), N, K // 16))

                _, _, nd, nt, _ = compare_nibbles(ao_codes, te_codes)
                bad_codes += nd
                tot_codes += nt
                _, ulp2, nd2, nt2, note2 = compare_e4m3(ao_sf, te_sf)
                bad_sf += nd2
                tot_sf += nt2
                worst_ulp = max(worst_ulp, ulp2)
                if note2:
                    notes.add(note2.split(" ")[0])

            record("3a-wgt", f"{model} {N}x{K}", f"{name} codes", bad_codes == 0,
                   0, bad_codes, tot_codes, "vs TE per expert")
            record("3a-wgt", f"{model} {N}x{K}", f"{name} block scales", bad_sf == 0,
                   worst_ulp, bad_sf, tot_sf, " ".join(sorted(notes)) or "vs TE per expert")


# --------------------------------------------------------------------------
# stage 3c -- attribute the weight divergence to a single expression
# --------------------------------------------------------------------------


def stage_3c_localize(device, models, num_experts=8) -> None:
    """Which fp32 expression does each kernel actually implement?

    Both association orders are recomputed in fp32 PyTorch (where division is
    correctly rounded) and matched against the two kernels. If the AO kernel
    matches the AO order and the TE kernel matches the TE order, the
    association order is the whole story and the div_rn/plain-'/' difference
    contributes nothing on top.

    Also reports the direction of the divergence and its effect on
    reconstruction error, which is what distinguishes a systematic bias from
    symmetric tie-breaking.
    """
    _, _, triton_group_weight_quantize_2d, _, from_blocked = ao_mods()

    torch.manual_seed(0)
    for model in models:
        dim, hidden, _ = moe_dims(model)
        for name, (N, K) in (("w1/w3", (hidden, dim)), ("w2", (dim, hidden))):
            W = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=device)
            amax = W.float().abs().amax(dim=(1, 2))
            codes, sf, _, _ = triton_group_weight_quantize_2d(W, amax, num_experts)

            n_ao_order = n_te_order = n_cross = n_tot = 0
            lo = hi = 0
            err_ao = err_te = 0.0
            bias_ao = bias_te = 0.0
            for e in range(num_experts):
                w = W[e].float()
                vmax = torch.amax(w.abs().unfold(0, 16, 16).unfold(1, 16, 16), dim=(-1, -2))
                ges = (FP8_E4M3_MAX * FP4_E2M1_MAX) / amax[e]
                ao_order = e4m3((vmax / FP4_E2M1_MAX) * ges)
                te_order = e4m3(vmax * (ges * (1.0 / FP4_E2M1_MAX)))

                ao_sf = from_blocked(sf[e].reshape(-1), N, K // 16)
                t = te_quantize(W[e].contiguous(), two_d=True, amax=amax[e])
                te_sf = as_u8(t._rowwise_scale_inv)[:N, : K // 16]

                # 2D scales repeat across each 16-row band; compare one row per band.
                ao_band = as_u8(ao_sf)[::16].contiguous()
                te_band = te_sf[::16].contiguous()
                n_ao_order += compare_e4m3(ao_band, as_u8(ao_order))[2]
                n_te_order += compare_e4m3(te_band, as_u8(te_order))[2]
                n_cross += compare_e4m3(ao_band, te_band)[2]
                n_tot += ao_band.numel()

                d = ao_band.to(torch.int32) - te_band.to(torch.int32)
                lo += int((d < 0).sum())
                hi += int((d > 0).sum())

                a_dq = ao_dequant(codes[e], ao_sf, amax[e])
                t_dq = ao_dequant(as_u8(t._rowwise_data)[:N, : K // 2],
                                te_sf.view(torch.float8_e4m3fn), amax[e])
                err_ao += ((a_dq - w).norm() / w.norm()).item()
                err_te += ((t_dq - w).norm() / w.norm()).item()
                bias_ao += (a_dq - w).mean().item()
                bias_te += (t_dq - w).mean().item()

            sh = f"{model} {N}x{K}"
            record("3c-loc", sh, f"{name} AO kernel == AO order", n_ao_order == 0,
                   0, n_ao_order, n_tot, "(vmax/6)*ges")
            record("3c-loc", sh, f"{name} TE kernel == TE order", n_te_order == 0,
                   0, n_te_order, n_tot, "vmax*(ges/6)")
            record("3c-loc", sh, f"{name} AO kernel == TE kernel", n_cross == 0,
                   1 if n_cross else 0, n_cross, n_tot,
                   f"lo={lo} hi={hi}; relL2 AO {err_ao / num_experts:.8e} "
                   f"TE {err_te / num_experts:.8e}; bias AO {bias_ao / num_experts:.3e} "
                   f"TE {bias_te / num_experts:.3e}")

# --------------------------------------------------------------------------
# stage 4 -- forward grouped GEMM on identical operands
# --------------------------------------------------------------------------


def stage_4_gemm(device, models, num_experts=4, rows_per_group=256) -> None:
    """Does scaled_grouped_mm compute what the quantized operands say?

    The same NVFP4 codes and scales that the forward GEMM consumes are also
    dequantized and multiplied per group in bf16. Agreement bounds the GEMM's
    own contribution, separating it from the quantization stages above.
    """
    import torch.nn.functional as F

    (
        triton_group_rht_amax,
        triton_group_rht_quantize_row_col,
        triton_group_weight_quantize_2d,
        VARYING_FIRST_DIM,
        from_blocked,
    ) = ao_mods()
    from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

    torch.manual_seed(0)
    sv = sign_vector()
    for model in models:
        dim, hidden, _ = moe_dims(model)
        N, K = hidden, dim
        sizes, offs = make_groups(num_experts, rows_per_group, device)
        M = sum(sizes)
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        W = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device=device) * 0.02

        col_amax, row_amax = triton_group_rht_amax(
            A, sv, offs, num_experts, M, K, VARYING_FIRST_DIM, logical_packed_length=offs[-1:]
        )
        qa, sfa, _, _ = triton_group_rht_quantize_row_col(
            A, sv, offs, num_experts, M, K, VARYING_FIRST_DIM, row_amax, col_amax,
            rng_state=None, enable_stochastic_rounding=False, logical_packed_length=offs[-1:],
        )
        w_amax = W.float().abs().amax(dim=(1, 2))
        wq, wsf, _, _ = triton_group_weight_quantize_2d(W, w_amax, num_experts)

        out = F.scaled_grouped_mm(
            qa.view(torch.float4_e2m1fn_x2),
            wq.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
            scale_a=[sfa, per_tensor_amax_to_scale(row_amax)],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[wsf.flatten(1), per_tensor_amax_to_scale(w_amax)],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            offs=offs,
            output_dtype=torch.bfloat16,
        )

        ref = torch.empty_like(out, dtype=torch.float32)
        start = 0
        for g, sz in enumerate(sizes):
            end = start + sz
            a_dq = ao_dequant(qa[start:end], from_blocked(sfa, M, K // 16)[start:end], row_amax[g])
            w_dq = ao_dequant(wq[g], from_blocked(wsf[g].reshape(-1), N, K // 16), w_amax[g])
            ref[start:end] = a_dq @ w_dq.t()
            start = end

        a, b = out.float(), ref
        sqnr = 20 * torch.log10(b.norm() / (a - b).norm())
        record("4-gemm", f"{model} {M}x{K}x{N}", "scaled_grouped_mm", True, 0, 0, 0,
               f"vs dequant+bf16 matmul: SQNR {sqnr:.2f} dB; "
               f"signed mean {(a - b).mean():.3e}", info=True)


def stage_5_grouped_experts(device, models, num_experts=4, rows_per_group=256) -> None:
    """Run the converted NVFP4 GroupedExperts forward against bf16.

    The headline metric is the *signed mean* of the output error, not SQNR.
    The reported training symptom is a persistent one-directional loss offset,
    so a forward path that is merely noisy is exonerated while one that is
    skewed is not.
    """
    torch.manual_seed(0)
    for model in models:
        dim, hidden, n_exp = moe_dims(model)
        n_exp = min(n_exp, num_experts)
        counts = torch.full((n_exp,), rows_per_group, dtype=torch.int32, device=device)
        M = int(counts.sum())
        x = torch.randn(M, dim, dtype=torch.bfloat16, device=device)

        ref = build_grouped_experts(model, device, nvfp4=False)
        # Trim to the expert count actually exercised.
        with torch.no_grad():
            for n, p in list(ref.named_parameters()):
                setattr(ref, n, torch.nn.Parameter(
                    torch.randn(n_exp, *p.shape[1:], dtype=torch.bfloat16, device=device) * 0.02
                ))
        ref.num_experts = n_exp

        q = build_grouped_experts(model, device, nvfp4=True)
        q.num_experts = n_exp
        with torch.no_grad():
            for n, p in ref.named_parameters():
                getattr(q, n).data = p.data.clone()

        out_bf16 = ref(x, counts)
        out_q1 = q(x, counts)
        out_q2 = q(x, counts)

        sh = f"{model} {M}x{dim}"
        ok, _, nd, nt, _ = compare_exact(out_q1, out_q2)
        record("5-experts", sh, "forward determinism", ok, 0, nd, nt,
               "two forwards, no SR in fwd")

        finite = bool(torch.isfinite(out_q1).all())
        record("5-experts", sh, "output finite", finite, 0,
               int((~torch.isfinite(out_q1)).sum()), out_q1.numel(), "")

        a, b = out_q1.float(), out_bf16.float()
        sqnr = 20 * torch.log10(b.norm() / (a - b).norm())
        record("5-experts", sh, "nvfp4 vs bf16", True, 0, 0, 0,
               f"SQNR {sqnr:.2f} dB; signed mean {(a - b).mean():.3e}; "
               f"relL2 {((a - b).norm() / b.norm()):.4e}", info=True)


# --------------------------------------------------------------------------
# stage 6 -- padding-tail and empty-expert probes
# --------------------------------------------------------------------------


def stage_6_pad_tail(device, models, rows_per_group=256) -> None:
    """Probe the converter's offs[-1] = A.shape[0] rewrite (nvfp4.py:410-411).

    The rewrite folds the dispatcher's capacity tail into the last expert's
    group. That is safe only if those rows are exactly zero. It is checked
    here directly against the real permute path rather than assumed.

    Also probes an expert that receives zero routed tokens: the permute
    kernel clamps every per-expert count up to the alignment, so such an
    expert still presents a 128-row all-zero group to the quantizer.
    """
    from torchao.prototype.moe_training.ep.permute import permute_and_pad

    (
        triton_group_rht_amax,
        triton_group_rht_quantize_row_col,
        _,
        VARYING_FIRST_DIM,
        _,
    ) = ao_mods()

    torch.manual_seed(0)
    sv = sign_vector()
    for model in models:
        dim, _, _ = moe_dims(model)
        n_exp = 4
        for label, counts in (
            ("balanced", [rows_per_group] * n_exp),
            ("one empty expert", [rows_per_group, 0, rows_per_group, rows_per_group]),
        ):
            n_tok = sum(counts)
            x = torch.randn(max(n_tok, 1), dim, dtype=torch.bfloat16, device=device)
            cnt = torch.tensor(counts, dtype=torch.int64, device=device)
            _, padded, _, padded_counts, _ = permute_and_pad(x, cnt, 1, n_exp, 128)

            offs = torch.cumsum(padded_counts.to(torch.int32), dim=0, dtype=torch.int32)
            real_end = int(offs[-1])
            tail = padded[real_end:]
            sh = f"{model} {label}"
            record("6-tail", sh, "capacity tail is zero", bool((tail == 0).all()), 0,
                   int((tail != 0).sum()), max(tail.numel(), 1),
                   f"tail rows {tail.shape[0]} of {padded.shape[0]}")

            # The converter's rewrite: extend the last group over the tail.
            ext = offs.clone()
            ext[-1] = padded.shape[0]
            M = padded.shape[0]
            _, row_amax_ext = triton_group_rht_amax(
                padded.contiguous(), sv, ext, n_exp, M, dim, VARYING_FIRST_DIM,
                logical_packed_length=ext[-1:],
            )
            # Same computation without the fold, on the truncated activation.
            trunc = padded[:real_end].contiguous()
            _, row_amax_trunc = triton_group_rht_amax(
                trunc, sv, offs, n_exp, real_end, dim, VARYING_FIRST_DIM,
                logical_packed_length=offs[-1:],
            )
            ok, _, nd, nt, _ = compare_exact(row_amax_ext, row_amax_trunc)
            record("6-tail", sh, "last-expert amax unchanged", ok, 0, nd, nt,
                   "folded tail vs truncated")

            # An all-zero group must not produce NaN/Inf scales.
            col_amax, row_amax = triton_group_rht_amax(
                padded.contiguous(), sv, ext, n_exp, M, dim, VARYING_FIRST_DIM,
                logical_packed_length=ext[-1:],
            )
            qa, sfa, qd, sfd = triton_group_rht_quantize_row_col(
                padded.contiguous(), sv, ext, n_exp, M, dim, VARYING_FIRST_DIM,
                row_amax, col_amax, rng_state=None, enable_stochastic_rounding=False,
                logical_packed_length=ext[-1:],
            )
            bad = int((~torch.isfinite(sfa.float())).sum()) + int(
                (~torch.isfinite(sfd.float())).sum()
            )
            record("6-tail", sh, "block scales finite", bad == 0, 0, bad,
                   sfa.numel() + sfd.numel(),
                   f"zero-token experts: {int((cnt == 0).sum())}")


# --------------------------------------------------------------------------


STAGES = {
    "0": stage_0_calibrate,
    "1": stage_1_global_scale,
    "2": stage_2_activations,
    "3a": stage_3a_weights,
    "3b": lambda dev, models: (
        stage_3b_pvscale_order(dev, models),
        stage_3b_div_lowering(dev, models),
    ),
    "3c": stage_3c_localize,
    "4": stage_4_gemm,
    "5": stage_5_grouped_experts,
    "6": stage_6_pad_tail,
}


def main() -> int:
    known_models = [m.model for m in DEEPSEEK_V3_MODEL_SHAPES]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", action="append", choices=sorted(STAGES), default=None)
    p.add_argument("--all", action="store_true", help="run every stage")
    p.add_argument(
        "--model",
        action="append",
        choices=known_models,
        default=None,
        help="DSV3 flavor(s); default debugmodel",
    )
    args = p.parse_args()

    if (rc := require_blackwell()) is not None:
        return rc

    models = args.model or ["debugmodel"]
    stages = sorted(STAGES) if args.all else (args.stage or ["3b"])
    device = torch.device("cuda", 0)

    print(f"device : {torch.cuda.get_device_name(0)} sm{''.join(map(str, torch.cuda.get_device_capability()))}")
    print(f"models : {', '.join(models)}")
    print(f"stages : {', '.join(stages)}")

    for name in stages:
        STAGES[name](device, models)

    return 0 if print_table() else 1


if __name__ == "__main__":
    raise SystemExit(main())
