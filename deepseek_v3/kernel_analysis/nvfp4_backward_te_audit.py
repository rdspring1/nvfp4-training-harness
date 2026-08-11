#!/usr/bin/env python3
"""Audit of the TorchAO NVFP4 *backward* path against TransformerEngine.

Motivation
----------
The forward path was audited first and exonerated (see
nvfp4_forward_te_audit.md): every discrepancy vs TE is at most 1 ULP with no
consistent sign, and the full GroupedExperts forward is bitwise deterministic
and unbiased against bf16. So the persistent positive training-loss bias lives
in backward, which is untested and holds all the higher-risk machinery.

What backward does (nvfp4_grouped_mm.py:234-332). The row path carries no RHT;
the col path carries RHT:

  dgrad = dy_row  @ weight_t^T     dy_row: plain NVFP4, SR ON
                                   weight_t: 2D, no RHT, no SR (from forward)
  wgrad = dy_col  @ x_col^T        dy_col: RHT, SR ON
                                   x_col:  RHT, no SR (from forward)

The whole RHT/columnwise path was NOT covered by the forward audit, which
compared only the row outputs. These stages close that gap.

Why SR is not compared bitwise
------------------------------
TE exposes no Python-level SR seed (csrc/quantizer.cpp:2579-2603 pulls from the
default CUDA generator passing std::nullopt), its random number is a function of
CUDA launch geometry rather than element index
(quantize_transpose_nvfp4.cuh:344-351), and TE's own SR output differs between
its fused and unfused RHT paths. TorchAO uses unrelated Philox indexing. So the
arithmetic is checked bitwise with SR OFF, and the randomness is checked
statistically -- which keeps "the rounding math is wrong" distinguishable from
"the RNG is wrong".

Stages
------
B0  Calibrate RHT + columnwise: TE kernel vs TE NVFP4QuantizerRef, atol=0.
B1  dgrad operands (dy_row, weight_t), SR off, bitwise vs TE.
B2  wgrad operands (dy_col, x_col), SR off, bitwise vs TE. First test of the
    RHT/columnwise path in this repo.
B3  RHT cancellation in wgrad: does RHT(dy^T) @ RHT(x^T)^T recover dy^T x?
B4  SR round-up probability as a function of position within each FP4 interval.
B5  SR decorrelation (row vs col, tile vs tile).
B6  Full GroupedExperts backward vs bf16: SQNR, signed mean, magnitude ratio.

Usage
-----
  python nvfp4_backward_te_audit.py --stage B0
  python nvfp4_backward_te_audit.py --all --model debugmodel
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nvfp4_audit_common import (  # noqa: E402
    DEEPSEEK_V3_MODEL_SHAPES,
    FP4_E2M1_GRID,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    ao_dequant,
    ao_mods,
    as_u8,
    build_grouped_experts,
    compare_nibbles,
    compare_e4m3,
    compare_exact,
    make_groups,
    moe_dims,
    print_table,
    record,
    require_blackwell,
    sign_vector,
    te_dequant,
    te_extract,
    te_extract_col,
    te_mods,
    te_quantize,
    te_ref_quantize,
    unpack_fp4,
)

# The anchor trick: with a per-16-vector max of 6.0 and a per-tensor amax of
# 6*448, ges = 1, pvscale = 1, and encode_scale collapses to exactly 1.0, so
# input values pass through unscaled and land directly on the FP4 grid. This is
# what makes an exact round-up-probability measurement possible.
_IDENTITY_SCALE_AMAX = FP4_E2M1_MAX * FP8_E4M3_MAX


def _rng_state(seed: int, col_off: int, row_off: int, device) -> torch.Tensor:
    """[col_seed, col_offset, row_seed, row_offset], the layout backward builds
    at nvfp4_grouped_mm.py:279."""
    return torch.tensor(
        [seed, col_off, seed ^ 1, row_off], dtype=torch.int64, device=device
    )


def _quantize_group(A, offs, n_groups, K, *, row_amax, col_amax, rng=None):
    """Drive the grouped RHT quantizer the way backward does."""
    (_, triton_group_rht_quantize_row_col, _, VARYING_FIRST_DIM, _) = ao_mods()
    M = A.shape[0]
    return triton_group_rht_quantize_row_col(
        A, sign_vector(), offs, n_groups, M, K, VARYING_FIRST_DIM,
        row_amax, col_amax,
        rng_state=rng, enable_stochastic_rounding=rng is not None,
        logical_packed_length=offs[-1:],
    )


def _stable_seed(*parts) -> int:
    """Deterministic seed. Python's hash() on strings is randomized per
    process (PYTHONHASHSEED), which would make the audit unreproducible."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFF


def _grad_like(M, N, device, seed=0):
    """Gradient-shaped data: heavy-tailed, unlike the near-Gaussian activations
    the forward audit used."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(M, N, generator=g, device=device, dtype=torch.float32)
    heavy = x * torch.exp(torch.randn(M, 1, generator=g, device=device) * 0.8)
    return heavy.to(torch.bfloat16)


# --------------------------------------------------------------------------
# B0 -- calibrate RHT + columnwise
# --------------------------------------------------------------------------


def stage_b0_calibrate(device, models) -> None:
    """TE CUDA kernel vs TE's own reference, with RHT and columnwise on.

    Mirrors tests/pytorch/nvfp4/test_nvfp4_rht_quantize_exact.py:136-152, which
    TE asserts at atol=0/rtol=0. If this is not bitwise the harness is wrong and
    nothing downstream means anything.
    """
    _, _, NVFP4Quantizer, _, _ = te_mods()
    from transformer_engine.pytorch.tensor.nvfp4_tensor import get_wgrad_sign_vector

    te_sv = get_wgrad_sign_vector(device).flatten().tolist()
    ao_sv = sign_vector()
    same = [int(v) for v in te_sv] == [int(v) for v in ao_sv]
    record("B0-calib", "-", "TE sign vector == torchtitan", same, 0,
           sum(int(a) != int(b) for a, b in zip(te_sv, ao_sv)), len(ao_sv),
           "must match or RHT comparison is meaningless")

    torch.manual_seed(0)
    for M, N in ((256, 256), (512, 2048), (1408, 2048)):
        x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        ref = te_ref_quantize(x, with_rht=True, columnwise=True)
        t = te_quantize(x, rowwise=True, columnwise=True, with_rht=True,
                        amax=ref.global_amax_row, col_amax=ref.global_amax_col)

        got_c, got_sc = te_extract_col(t, M, N)
        ref_c = unpack_fp4(ref.data_t.view(torch.uint8))
        ref_sc = as_u8(ref.scale_t)
        h, w = ref_sc.shape

        ok, ulp, nd, nt, note = compare_nibbles(got_c, ref_c[:N, :M])
        record("B0-calib", f"{M}x{N}", "col codes (RHT)", ok, ulp, nd, nt, note)
        ok, ulp, nd, nt, note = compare_e4m3(got_sc[:h, :w], ref_sc)
        record("B0-calib", f"{M}x{N}", "col scales (RHT)", ok, ulp, nd, nt, note)
        ok, _, nd, nt, _ = compare_exact(t._amax_columnwise, ref.global_amax_col)
        record("B0-calib", f"{M}x{N}", "col amax (post-RHT)", ok, 0, nd, nt, "")


# --------------------------------------------------------------------------
# B1 -- dgrad operands, SR off, bitwise
# --------------------------------------------------------------------------


def stage_b1_dgrad(device, models, n_groups=4, rows_per_group=256) -> None:
    """dy_row (dgrad LHS) and weight_t (dgrad RHS) vs TE."""
    (triton_group_rht_amax, _, triton_group_weight_quantize_2d,
     VARYING_FIRST_DIM, from_blocked) = ao_mods()

    for model in models:
        dim, hidden, _ = moe_dims(model)

        # --- dy_row: grad_output quantized rowwise, no RHT ---
        for label, N in (("dy w1/w3", hidden), ("dy w2", dim)):
            sizes, offs = make_groups(n_groups, rows_per_group, device)
            M = sum(sizes)
            dy = _grad_like(M, N, device, seed=_stable_seed(model, N))
            col_amax, row_amax = triton_group_rht_amax(
                dy, sign_vector(), offs, n_groups, M, N, VARYING_FIRST_DIM,
                logical_packed_length=offs[-1:],
            )
            qa, sfa, _, _ = _quantize_group(dy, offs, n_groups, N,
                                            row_amax=row_amax, col_amax=col_amax)
            ao_codes = unpack_fp4(qa)
            ao_sf = as_u8(from_blocked(sfa, M, N // 16))

            bad_c = bad_s = tot_c = tot_s = worst = 0
            start = 0
            for g, sz in enumerate(sizes):
                end = start + sz
                t = te_quantize(dy[start:end].contiguous(), amax=row_amax[g])
                te_c, te_s = te_extract(t, sz, N)
                bad_c += compare_nibbles(ao_codes[start:end], te_c)[2]
                tot_c += sz * N
                _, u, nd, nt, _ = compare_e4m3(ao_sf[start:end], te_s)
                bad_s += nd
                tot_s += nt
                worst = max(worst, u)
                start = end
            sh = f"{model} {M}x{N}"
            record("B1-dgrad", sh, f"{label} row codes", bad_c == 0, 0, bad_c, tot_c, "SR off")
            record("B1-dgrad", sh, f"{label} row scales", bad_s == 0, worst, bad_s, tot_s, "SR off")

        # --- weight_t: the transposed weight, dgrad RHS ---
        for label, (Nw, Kw) in (("w1/w3", (hidden, dim)), ("w2", (dim, hidden))):
            E = 4
            W = torch.randn(E, Nw, Kw, dtype=torch.bfloat16, device=device) * 0.02
            w_amax = W.float().abs().amax(dim=(1, 2))
            _, _, wt_codes, wt_sf = triton_group_weight_quantize_2d(W, w_amax, E)

            bad_c = bad_s = tot_c = tot_s = worst = 0
            for e in range(E):
                t = te_quantize(W[e].t().contiguous(), two_d=True, amax=w_amax[e])
                te_c, te_s = te_extract(t, Kw, Nw)
                bad_c += compare_nibbles(unpack_fp4(wt_codes[e]), te_c)[2]
                tot_c += Kw * Nw
                ao_s = as_u8(from_blocked(wt_sf[e].reshape(-1), Kw, Nw // 16))
                _, u, nd, nt, _ = compare_e4m3(ao_s, te_s)
                bad_s += nd
                tot_s += nt
                worst = max(worst, u)
            sh = f"{model} {Kw}x{Nw}"
            record("B1-dgrad", sh, f"{label} weight_t codes", bad_c == 0, 0, bad_c, tot_c,
                   "vs TE 2D on W.T")
            record("B1-dgrad", sh, f"{label} weight_t scales", bad_s == 0, worst, bad_s, tot_s,
                   "vs TE 2D on W.T")


# --------------------------------------------------------------------------
# B2 -- wgrad operands, SR off, bitwise
# --------------------------------------------------------------------------


def stage_b2_wgrad(device, models, n_groups=4, rows_per_group=256) -> None:
    """dy_col and x_col -- the RHT/columnwise path, untested until now."""
    triton_group_rht_amax, _, _, VARYING_FIRST_DIM, from_blocked = ao_mods()

    for model in models:
        dim, hidden, _ = moe_dims(model)
        for label, K in (("x_col (w1/w3 in)", dim), ("dy_col (w1/w3 out)", hidden)):
            sizes, offs = make_groups(n_groups, rows_per_group, device)
            M = sum(sizes)
            A = _grad_like(M, K, device, seed=_stable_seed(model, K, "c"))
            col_amax, row_amax = triton_group_rht_amax(
                A, sign_vector(), offs, n_groups, M, K, VARYING_FIRST_DIM,
                logical_packed_length=offs[-1:],
            )
            _, _, qd, sfd = _quantize_group(A, offs, n_groups, K,
                                            row_amax=row_amax, col_amax=col_amax)
            ao_codes = unpack_fp4(qd)
            ao_sf = as_u8(from_blocked(sfd, K, M // 16))

            bad_c = bad_s = tot_c = tot_s = worst = bad_a = 0
            start = 0
            for g, sz in enumerate(sizes):
                end = start + sz
                t = te_quantize(A[start:end].contiguous(), rowwise=True,
                                columnwise=True, with_rht=True,
                                amax=row_amax[g], col_amax=col_amax[g])
                te_c, te_s = te_extract_col(t, sz, K)
                bad_c += compare_nibbles(ao_codes[:, start:end], te_c)[2]
                tot_c += K * sz
                _, u, nd, nt, _ = compare_e4m3(ao_sf[:, start // 16:end // 16], te_s)
                bad_s += nd
                tot_s += nt
                worst = max(worst, u)
                bad_a += compare_exact(col_amax[g].reshape(1), t._amax_columnwise)[2]
                start = end
            sh = f"{model} {M}x{K}"
            record("B2-wgrad", sh, f"{label} amax", bad_a == 0, 0, bad_a, n_groups,
                   "post-RHT col amax")
            record("B2-wgrad", sh, f"{label} codes", bad_c == 0, 0, bad_c, tot_c, "RHT, SR off")
            record("B2-wgrad", sh, f"{label} scales", bad_s == 0, worst, bad_s, tot_s,
                   "RHT, SR off")


# --------------------------------------------------------------------------
# B3 -- RHT cancellation in wgrad
# --------------------------------------------------------------------------


def stage_b3_rht_cancellation(device, models, n_groups=4, rows_per_group=256) -> None:
    """wgrad contracts RHT(dy^T) against RHT(x^T) and relies on H H^T = I.

    With SR off, dequantize both columnwise operands, contract, and compare to
    dy^T @ x. Reports signed mean as well as SQNR -- imperfect cancellation
    would be one-directional, which is the symptom being chased.
    """
    triton_group_rht_amax, _, _, VARYING_FIRST_DIM, from_blocked = ao_mods()

    for model in models:
        dim, hidden, _ = moe_dims(model)
        K, N = dim, hidden
        sizes, offs = make_groups(n_groups, rows_per_group, device)
        M = sum(sizes)
        x = _grad_like(M, K, device, seed=11)
        dy = _grad_like(M, N, device, seed=22)

        xc_amax, xr_amax = triton_group_rht_amax(
            x, sign_vector(), offs, n_groups, M, K, VARYING_FIRST_DIM,
            logical_packed_length=offs[-1:])
        _, _, x_col, x_sf = _quantize_group(x, offs, n_groups, K,
                                            row_amax=xr_amax, col_amax=xc_amax)
        dc_amax, dr_amax = triton_group_rht_amax(
            dy, sign_vector(), offs, n_groups, M, N, VARYING_FIRST_DIM,
            logical_packed_length=offs[-1:])
        _, _, dy_col, dy_sf = _quantize_group(dy, offs, n_groups, N,
                                              row_amax=dr_amax, col_amax=dc_amax)

        x_sf_p = from_blocked(x_sf, K, M // 16)
        dy_sf_p = from_blocked(dy_sf, N, M // 16)

        start = 0
        for g, sz in enumerate(sizes):
            end = start + sz
            xg = ao_dequant(x_col[:, start // 2:end // 2],
                            x_sf_p[:, start // 16:end // 16], xc_amax[g])
            dg = ao_dequant(dy_col[:, start // 2:end // 2],
                            dy_sf_p[:, start // 16:end // 16], dc_amax[g])
            got = dg @ xg.t()                                   # (N, K)
            ref = dy[start:end].float().t() @ x[start:end].float()
            sqnr = 20 * torch.log10(ref.norm() / (got - ref).norm())
            rel = (got - ref).norm() / ref.norm()
            record("B3-rht", f"{model} g{g} {N}x{K}", "RHT cancels in wgrad", True, 0, 0, 0,
                   f"SQNR {sqnr:.2f} dB; relL2 {rel:.4e}; "
                   f"signed mean {(got - ref).mean():.3e}; ref mean {ref.mean():.3e}",
                   info=True)
            start = end


# --------------------------------------------------------------------------
# B4 -- SR round-up probability vs position in the FP4 interval
# --------------------------------------------------------------------------


def stage_b4_sr_probability(device, models, n_draws=64, n_t=17) -> None:
    """P(round up) must equal the value's fractional position in its interval.

    Deliberately stronger than TE's own SR criterion
    (test_nvfp4_sr_quantize.py:281-282 only asserts averaged SR RMSE beats
    round-nearest, which a biased RNG still passes) and stronger than the
    existing single-midpoint 50/50 check
    (test_hadamard_quantize_row_col.py:583).

    The e2m1 grid is NOT uniform -- widths 0.5,0.5,0.5,0.5,1,1,2 -- so a rounder
    that implicitly assumes uniform spacing is biased only in the three wide
    intervals, which neither existing test would catch. Each interval is swept
    separately.

    Identity encode scale is forced by anchoring every 16-vector at 6.0 and
    passing a per-tensor amax of 6*448, so inputs land directly on the grid.
    """
    n_groups, rows = 2, 128
    M, K = n_groups * rows, 256
    _, offs = make_groups(n_groups, rows, device)
    amax = torch.full((n_groups,), _IDENTITY_SCALE_AMAX, dtype=torch.float32, device=device)
    grid = FP4_E2M1_GRID
    keep = (torch.arange(K, device=device) % 16) != 0

    for lo_i in range(len(grid) - 1):
        a, b = grid[lo_i], grid[lo_i + 1]
        max_dev = 0.0
        worst_t = 0.0
        dev_sum = 0.0
        var_sum = 0.0
        n_pts = 0
        for ti in range(n_t):
            t = ti / (n_t - 1)
            val = a + t * (b - a)
            A = torch.empty(M, K, dtype=torch.bfloat16, device=device)
            A[:, :] = torch.tensor(val, dtype=torch.bfloat16)
            A[:, 0::16] = FP4_E2M1_MAX          # anchor -> vmax = 6 -> encode_scale = 1
            # bf16 may not represent `val` exactly; measure the position the
            # kernel actually sees rather than the one we asked for.
            t_actual = (A[0, 1].float().item() - a) / (b - a)

            ups = tot = 0
            for d in range(n_draws):
                # Offsets are varied per interval as well as per draw, so the
                # seven intervals are independent samples. With a shared stream
                # they return bit-identical decisions -- the same elements round
                # up at a given t regardless of interval, which is correct
                # behaviour but makes the rows redundant as evidence.
                off = 1000 + d + 7919 * lo_i + 104729 * ti
                qa, _, _, _ = _quantize_group(
                    A, offs, n_groups, K, row_amax=amax, col_amax=amax,
                    rng=_rng_state(0x5EED, off, off + 31337, device))
                c = unpack_fp4(qa)[:, keep]
                ups += int((c == lo_i + 1).sum())
                tot += c.numel()
            p = ups / tot
            dev = p - t_actual
            dev_sum += dev
            var_sum += max(t_actual * (1 - t_actual), 1e-9) / tot
            n_pts += 1
            if abs(dev) > abs(max_dev):
                max_dev, worst_t = dev, t_actual

        # Binomial standard error, propagated across the sweep points.
        se_mean = (var_sum / (n_pts * n_pts)) ** 0.5
        se_worst = (max(worst_t * (1 - worst_t), 1e-9) / tot) ** 0.5
        mean_dev = dev_sum / n_pts
        ok = abs(max_dev) <= 4 * se_worst and abs(mean_dev) <= 4 * se_mean
        record("B4-sr", f"[{a}, {b}] w={b - a}", "P(up) vs position", ok, 0, 0, 0,
               f"mean dev {mean_dev:+.2e} (SE {se_mean:.1e}, {mean_dev / se_mean:+.1f}sig); "
               f"max dev {max_dev:+.2e} at t={worst_t:.3f} (SE {se_worst:.1e}); "
               f"{n_draws}x{tot // n_draws} samples/pt")


def stage_b4b_sr_col(device, models, n_draws=48, n_bins=16) -> None:
    """Same question for the RHT/columnwise path, which is the wgrad operand.

    The RHT scrambles values, so the position within the FP4 interval cannot be
    dialled in directly. Instead it is *measured*: the post-RHT value and the
    encode scale are reconstructed in PyTorch (the scale path is bitwise-verified
    against TE by stage B2), each element's true fractional position is derived,
    and elements are binned by it.
    """
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import get_rht_matrix

    triton_group_rht_amax, _, _, VARYING_FIRST_DIM, from_blocked = ao_mods()
    n_groups, rows, K = 2, 256, 256
    M = n_groups * rows
    sizes, offs = make_groups(n_groups, rows, device)
    A = _grad_like(M, K, device, seed=99)

    col_amax, row_amax = triton_group_rht_amax(
        A, sign_vector(), offs, n_groups, M, K, VARYING_FIRST_DIM,
        logical_packed_length=offs[-1:])
    _, _, qd_rtne, sfd = _quantize_group(A, offs, n_groups, K,
                                         row_amax=row_amax, col_amax=col_amax)
    sf = from_blocked(sfd, K, M // 16).view(torch.float8_e4m3fn).float()

    # scaled = RHT(A^T) * encode_scale, exactly what the kernel rounds.
    B = get_rht_matrix(tuple(sign_vector()), device, torch.bfloat16, 16)
    rht = torch.empty(K, M, dtype=torch.float32, device=device)
    start = 0
    for sz in sizes:
        end = start + sz
        rht[:, start:end] = (
            A[start:end].t().reshape(-1, 16).to(torch.bfloat16) @ B
        ).reshape(K, sz).float()
        start = end
    gds = torch.empty(M, dtype=torch.float32, device=device)
    start = 0
    for g, sz in enumerate(sizes):
        gds[start:start + sz] = col_amax[g] / (FP4_E2M1_MAX * FP8_E4M3_MAX)
        start += sz
    enc = 1.0 / (sf.repeat_interleave(16, dim=1) * gds.unsqueeze(0))
    scaled = (rht * enc).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

    grid = torch.tensor(FP4_E2M1_GRID, device=device)
    mag = scaled.abs()
    hi = torch.bucketize(mag, grid, right=True).clamp(1, len(FP4_E2M1_GRID) - 1)
    lo = hi - 1
    a_v, b_v = grid[lo], grid[hi]
    t_true = ((mag - a_v) / (b_v - a_v)).clamp(0, 1)

    up_count = torch.zeros_like(t_true)
    for d in range(n_draws):
        _, _, qd, _ = _quantize_group(
            A, offs, n_groups, K, row_amax=row_amax, col_amax=col_amax,
            rng=_rng_state(0xC01, 500 + d, 900 + d, device))
        up_count += (unpack_fp4(qd) & 0x07).float().eq(hi.float()).float()
    p_emp = up_count / n_draws

    edges = torch.linspace(0, 1, n_bins + 1, device=device)
    idx = torch.bucketize(t_true, edges, right=True).clamp(1, n_bins) - 1
    max_dev = 0.0
    worst = (0.0, 0.0, 0)
    tot_n = 0
    wsum = 0.0
    for k in range(n_bins):
        m = idx == k
        n = int(m.sum())
        if n < 1000:
            continue
        pt, pe = float(t_true[m].mean()), float(p_emp[m].mean())
        dev = pe - pt
        tot_n += n
        wsum += dev * n
        if abs(dev) > abs(max_dev):
            max_dev, worst = dev, (pt, pe, n * n_draws)
    se = (0.25 / max(worst[2], 1)) ** 0.5
    record("B4-sr", f"RHT col {M}x{K}", "P(up) vs measured position",
           abs(max_dev) <= max(4 * se, 5e-3), 0, 0, 0,
           f"max dev {max_dev:+.2e} at t={worst[0]:.3f} (SE {se:.1e}); "
           f"weighted mean dev {wsum / max(tot_n, 1):+.2e}; {n_draws} draws")


# --------------------------------------------------------------------------
# B5 -- SR decorrelation
# --------------------------------------------------------------------------


def stage_b5_decorrelation(device, models, n_draws=32) -> None:
    """Row and col enumerate identical linear_idx sets
    (group_rht_quantize_row_col_triton.py:137-175); they are decorrelated only
    by seed vs seed^1 plus two independent offsets (nvfp4_grouped_mm.py:279).
    Check the SR decisions are actually independent.
    """
    n_groups, rows, K = 2, 128, 256
    M = n_groups * rows
    _, offs = make_groups(n_groups, rows, device)
    amax = torch.full((n_groups,), _IDENTITY_SCALE_AMAX, dtype=torch.float32, device=device)

    # Every non-anchor element sits exactly at a midpoint, so its SR decision is
    # a fair coin. The 6.0 anchors are deterministic and MUST be masked out --
    # leaving them in makes every draw share a constant component and fabricates
    # a ~0.055 correlation between any two draws.
    A = torch.full((M, K), 1.25, dtype=torch.bfloat16, device=device)
    A[:, 0::16] = FP4_E2M1_MAX
    keep = (torch.arange(K, device=device) % 16) != 0

    def corr(u, v):
        u = u - u.mean()
        v = v - v.mean()
        d = u.norm() * v.norm()
        return float((u @ v) / d) if float(d) > 0 else 0.0

    def row_bits(seed, off):
        qa, _, _, _ = _quantize_group(
            A, offs, n_groups, K, row_amax=amax, col_amax=amax,
            rng=_rng_state(seed, off, off + 31337, device))
        return (unpack_fp4(qa)[:, keep] == 3).float().flatten()

    draws = [row_bits(0x5EED, 100 + d) for d in range(n_draws)]
    n_el = draws[0].numel()
    se = n_el ** -0.5     # SE of a correlation estimate on n_el samples

    dd = max(abs(corr(draws[i], draws[i + 1])) for i in range(n_draws - 1))
    record("B5-decorr", f"{M}x{K}", "draw-to-draw independence", dd < 5 * se, 0, 0, 0,
           f"max |corr| consecutive draws {dd:.5f} (SE {se:.5f})")

    sp = max(abs(corr(draws[i][:-1], draws[i][1:])) for i in range(n_draws))
    record("B5-decorr", f"{M}x{K}", "spatial whiteness", sp < 5 * se, 0, 0, 0,
           f"max |corr| adjacent elements {sp:.5f} (SE {se:.5f})")

    # The production decorrelation mechanism is seed vs seed^1 with independent
    # offsets (nvfp4_grouped_mm.py:279). Exercise exactly that key/counter
    # difference through the row output, so the RHT does not confound it.
    rc = max(
        abs(corr(row_bits(0x5EED, 200 + d), row_bits(0x5EED ^ 1, 800 + d)))
        for d in range(min(n_draws, 8))
    )
    record("B5-decorr", f"{M}x{K}", "seed vs seed^1 streams", rc < 5 * se, 0, 0, 0,
           f"max |corr| {rc:.5f} (SE {se:.5f}) -- the row/col decorrelation mechanism")

    # Sanity floor: identical seed AND offset must reproduce bit-identically.
    same = torch.equal(row_bits(0x5EED, 4242), row_bits(0x5EED, 4242))
    record("B5-decorr", f"{M}x{K}", "same rng_state reproduces", same, 0, 0, 0,
           "confirms the correlation probe can see a perfect correlation")


# --------------------------------------------------------------------------
# B6 -- GroupedExperts backward vs bf16
# --------------------------------------------------------------------------


def stage_b6_backward(device, models, n_exp=4, rows_per_group=256) -> None:
    """Full converted module backward.

    The magnitude ratio ||g_nvfp4|| / ||g_bf16|| is the metric that maps onto
    "needs ~11% more steps": a ratio consistently below 1 is an effective
    learning-rate reduction.
    """
    for model in models:
        dim, hidden, avail = moe_dims(model)
        E = min(avail, n_exp)
        counts = torch.full((E,), rows_per_group, dtype=torch.int32, device=device)
        M = int(counts.sum())

        torch.manual_seed(7)
        ref = build_grouped_experts(model, device, nvfp4=False, num_experts=E)
        with torch.no_grad():
            for n, p in list(ref.named_parameters()):
                p.copy_(torch.randn_like(p) * 0.02)
        q = build_grouped_experts(model, device, nvfp4=True, num_experts=E)
        with torch.no_grad():
            for n, p in ref.named_parameters():
                getattr(q, n).data = p.data.clone()

        x_ref = _grad_like(M, dim, device, seed=3).requires_grad_(True)
        x_q = x_ref.detach().clone().requires_grad_(True)

        out_r = ref(x_ref, counts)
        out_q = q(x_q, counts)
        # A FIXED grad_output for both. Calling .backward() on each model's own
        # loss would feed them different upstream gradients -- they differ by the
        # forward's ~25% relative error -- and that contaminates the comparison
        # with forward error instead of isolating backward.
        gout = _grad_like(*out_r.shape, device, seed=5)
        out_r.backward(gout)
        out_q.backward(gout)

        sh = f"{model} {M}x{dim}"
        # CAVEAT: the two models' intermediate activations already differ by the
        # forward's ~25% relative error, so these rows measure backward error
        # COMPOUNDED with forward divergence, not the backward kernels alone.
        # Stage B6b isolates the wgrad primitive on identical operands.
        for name, gq, gr in [("grad_input", x_q.grad, x_ref.grad)] + [
            (n, getattr(q, n).grad, getattr(ref, n).grad) for n, _ in ref.named_parameters()
        ]:
            a, b = gq.float(), gr.float()
            sqnr = 20 * torch.log10(b.norm() / (a - b).norm())
            ratio = (a.norm() / b.norm()).item()
            record("B6-bwd", sh, name, True, 0, 0, 0,
                   f"SQNR {sqnr:.2f} dB; |g| ratio {ratio:.5f}; "
                   f"signed mean {(a - b).mean():.3e}; ref mean {b.mean():.3e}",
                   info=True)

        # How much of the wgrad error is stochastic rounding? Average many
        # backward passes with the same inputs: SR noise is zero-mean (B4), so
        # it averages away and what survives is the deterministic part.
        n_avg = 32
        acc = torch.zeros_like(ref.w1_EFD.grad)
        for _ in range(n_avg):
            q.zero_grad(set_to_none=True)
            o = q(x_q.detach().clone().requires_grad_(True), counts)
            o.backward(gout)
            acc += q.w1_EFD.grad
        acc /= n_avg
        b = ref.w1_EFD.grad.float()
        sqnr = 20 * torch.log10(b.norm() / (acc.float() - b).norm())
        record("B6-bwd", sh, f"w1 wgrad averaged over {n_avg}", True, 0, 0, 0,
               f"SQNR {sqnr:.2f} dB; |g| ratio {(acc.float().norm() / b.norm()):.5f} "
               f"(single-pass ratio above); SR noise averages out, deterministic "
               f"forward divergence does not",
               info=True)


def stage_b6b_wgrad_isolated(device, models, n_groups=4, rows_per_group=256) -> None:
    """The wgrad primitive alone, on identical operands.

    B6 compares two modules whose intermediate activations already differ by the
    forward error, so it cannot separate a backward defect from compounded
    forward divergence. Here dy and x are fixed and shared, which is the only
    way to attribute error to the backward path itself. SR on vs off isolates
    the stochastic-rounding contribution.
    """
    triton_group_rht_amax, _, _, VARYING_FIRST_DIM, from_blocked = ao_mods()

    for model in models:
        dim, hidden, _ = moe_dims(model)
        K, N = dim, hidden
        sizes, offs = make_groups(n_groups, rows_per_group, device)
        M = sum(sizes)
        for dist, mk in (
            ("gaussian", lambda c: torch.randn(M, c, dtype=torch.bfloat16, device=device)),
            ("heavy-tailed", lambda c: _grad_like(M, c, device, seed=_stable_seed(model, c, dist))),
        ):
            x, dy = mk(K), mk(N)
            xc, xr = triton_group_rht_amax(x, sign_vector(), offs, n_groups, M, K,
                                           VARYING_FIRST_DIM, logical_packed_length=offs[-1:])
            dc, dr = triton_group_rht_amax(dy, sign_vector(), offs, n_groups, M, N,
                                           VARYING_FIRST_DIM, logical_packed_length=offs[-1:])
            for sr in (False, True):
                rng = _rng_state(1234, 7, 9, device) if sr else None
                _, _, xcol, xsf = _quantize_group(x, offs, n_groups, K,
                                                  row_amax=xr, col_amax=xc)
                _, _, dcol, dsf = _quantize_group(dy, offs, n_groups, N,
                                                  row_amax=dr, col_amax=dc, rng=rng)
                xp = from_blocked(xsf, K, M // 16)
                dp = from_blocked(dsf, N, M // 16)
                sq = ra = 0.0
                start = 0
                for g, sz in enumerate(sizes):
                    end = start + sz
                    xg = ao_dequant(xcol[:, start // 2:end // 2],
                                    xp[:, start // 16:end // 16], xc[g])
                    dg = ao_dequant(dcol[:, start // 2:end // 2],
                                    dp[:, start // 16:end // 16], dc[g])
                    ref = dy[start:end].float().t() @ x[start:end].float()
                    got = dg @ xg.t()
                    sq += float(20 * torch.log10(ref.norm() / (got - ref).norm()))
                    ra += float(got.norm() / ref.norm())
                    start = end
                record("B6b-wgrad", f"{model} {N}x{K}",
                       f"{dist}, SR {'on' if sr else 'off'}", True, 0, 0, 0,
                       f"mean SQNR {sq / n_groups:.2f} dB; mean |g| ratio {ra / n_groups:.4f}",
                       info=True)


# --------------------------------------------------------------------------


STAGES = {
    "B0": stage_b0_calibrate,
    "B1": stage_b1_dgrad,
    "B2": stage_b2_wgrad,
    "B3": stage_b3_rht_cancellation,
    "B4": lambda dev, models: (
        stage_b4_sr_probability(dev, models),
        stage_b4b_sr_col(dev, models),
    ),
    "B5": stage_b5_decorrelation,
    "B6": lambda dev, models: (
        stage_b6_backward(dev, models),
        stage_b6b_wgrad_isolated(dev, models),
    ),
}


def main() -> int:
    known = [m.model for m in DEEPSEEK_V3_MODEL_SHAPES]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", action="append", choices=sorted(STAGES), default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", action="append", choices=known, default=None)
    args = p.parse_args()

    if (rc := require_blackwell()) is not None:
        return rc

    models = args.model or ["debugmodel"]
    stages = sorted(STAGES) if args.all else (args.stage or ["B0"])
    device = torch.device("cuda", 0)

    cap = "".join(map(str, torch.cuda.get_device_capability()))
    print(f"device : {torch.cuda.get_device_name(0)} sm{cap}")
    print(f"models : {', '.join(models)}")
    print(f"stages : {', '.join(stages)}")

    for name in stages:
        STAGES[name](device, models)

    return 0 if print_table() else 1


if __name__ == "__main__":
    raise SystemExit(main())
