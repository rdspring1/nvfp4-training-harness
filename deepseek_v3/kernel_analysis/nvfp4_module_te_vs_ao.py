"""Module-level NVFP4 GroupedExperts comparison: TorchAO vs TransformerEngine.

The 200M-token 16B runs put TE 0.070 below TorchAO in training loss, on a stable
plateau, with the sign one-sided at 39/39 logged steps (see
`../run_logs/eager_runner/deepseek_v3_16b_fsdp4_ep4_te_nvfp4_tail15_compile_200m_gbs128/`).
The forward and backward audits already cleared the TorchAO *kernels* against
TE's quantizers tensor by tensor, so the gap has to live somewhere those audits
did not look: the assembled module.

This is that comparison, on one GPU, at one step. bf16 `GroupedExperts` is the
reference; TorchAO and TE subclasses run with bit-identical weights, input and
upstream gradient, so any difference is the quantize+GEMM primitive.

The stages narrow from the module to the call, in the order they were written:

  M1/M2  Do the two backends carry the same amount of NVFP4 error? If TE's
         forward error is far below TorchAO's, TE is not quantizing everything
         it is supposed to and the "TE wins" result is an artifact, not a
         finding. This is a verdict check, not informational.
  M3/M4  Split each backend's gradient error into the part that averages away
         (SR noise) and the part that does not. Only the second compounds over
         382 steps, so only the second can produce a stable loss offset. M4 then
         tests the seed hypothesis: TorchAO pins one `_sr_seed` per module for
         the whole run and varies only the two Philox counter words per call,
         where TE draws full Philox state from the CUDA generator every call.
  M5/M6  M2 puts TorchAO's wgrad at ~2x TE's error. M5 collapses the three-GEMM
         chain to one `_grouped_mm` to ask whether it is the primitive; M6 then
         feeds one set of TorchAO operands to two consumers -- fp32 dequant and
         the shipped `F.scaled_grouped_mm` -- to ask whether it is the
         quantization or the GEMM. Sweep `--experts` here: the answer depends on
         the group count. See nvfp4_module_te_vs_ao.md.

    python deepseek_v3/kernel_analysis/nvfp4_module_te_vs_ao.py --model debugmodel
    python deepseek_v3/kernel_analysis/nvfp4_module_te_vs_ao.py --model 16B --draws 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# `te_moe_overrides` lives one level up, next to run_titan.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nvfp4_audit_common import (  # noqa: E402
    build_grouped_experts,
    moe_dims,
    print_table,
    record,
    require_blackwell,
)


def _te_grouped_experts(model: str, device, num_experts: int):
    """The TE subclass, built exactly as `build_grouped_experts` builds TorchAO's.

    Not folded into the shared helper: that one takes `nvfp4: bool` and is called
    from both existing audits, and this is the only caller that needs a third
    backend.
    """
    from te_moe_overrides.te_nvfp4 import _get_te_grouped_experts_cls

    from torchtitan.models.common.moe import GroupedExperts

    dim, hidden, _ = moe_dims(model)
    cls = _get_te_grouped_experts_cls(GroupedExperts)
    mod = cls(cls.Config(dim=dim, hidden_dim=hidden, num_experts=num_experts))
    mod = mod.to(device=device, dtype=torch.bfloat16)
    mod._init_self_buffers(buffer_device=device)
    return mod


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b||."""
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm())


def _gain(a: torch.Tensor, b: torch.Tensor) -> float:
    """<a, b> / <b, b>: the component of *a* along *b*.

    This is the metric that maps onto training speed. ||a||/||b|| counts
    orthogonal noise as signal and reads ~1 even when the useful component is
    attenuated; a gain below 1 is an effective learning-rate cut, which is what a
    stable loss offset looks like.
    """
    a, b = a.float(), b.float()
    return float((a * b).sum() / (b * b).sum())


def _make_models(model: str, device, num_experts: int):
    """bf16 reference plus both quantized backends, sharing one set of weights."""
    torch.manual_seed(7)
    ref = build_grouped_experts(model, device, nvfp4=False, num_experts=num_experts)
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p) * 0.02)

    ao = build_grouped_experts(model, device, nvfp4=True, num_experts=num_experts)
    te = _te_grouped_experts(model, device, num_experts)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            getattr(ao, name).data = p.data.clone()
            getattr(te, name).data = p.data.clone()
    return ref, ao, te


def _inputs(model: str, device, num_experts: int, rows_per_group: int):
    dim, hidden, _ = moe_dims(model)
    counts = torch.full((num_experts,), rows_per_group, dtype=torch.int32, device=device)
    M = int(counts.sum())
    torch.manual_seed(3)
    x = torch.randn(M, dim, dtype=torch.bfloat16, device=device)
    dy = torch.randn(M, dim, dtype=torch.bfloat16, device=device)
    return counts, x, dy, M


def stage_m1_forward(device, models, num_experts, rows_per_group) -> None:
    """Forward error of each backend against bf16, on identical weights and input.

    The verdict: TE and TorchAO must be within 2x of each other. The backward
    audit measured TorchAO's module forward at relL2 ~0.25; a TE number an order
    of magnitude below that would mean TE is silently keeping an operand in
    higher precision, which would explain the loss gap as a missing quantization
    rather than a better one.
    """
    for model in models:
        dim, _, avail = moe_dims(model)
        E = min(avail, num_experts)
        ref, ao, te = _make_models(model, device, E)
        counts, x, _, M = _inputs(model, device, E, rows_per_group)
        sh = f"{model} {M}x{dim}"

        with torch.no_grad():
            out_r = ref(x, counts)
            out_a = ao(x, counts)
            out_t = te(x, counts)

        e_a, e_t = _rel_l2(out_a, out_r), _rel_l2(out_t, out_r)
        ratio = e_t / e_a
        record(
            "M1-fwd", sh, "TE vs TorchAO error ratio", 0.5 <= ratio <= 2.0, 0, 0, 0,
            f"relL2 vs bf16: TorchAO {e_a:.5f}, TE {e_t:.5f} (ratio {ratio:.3f}); "
            f"gain vs bf16: TorchAO {_gain(out_a, out_r):.5f}, TE {_gain(out_t, out_r):.5f}",
        )
        record(
            "M1-fwd", sh, "TE vs TorchAO", True, 0, 0, 0,
            f"relL2 {_rel_l2(out_t, out_a):.5f} between the two quantized outputs",
            info=True,
        )


def stage_m2_backward(device, models, num_experts, rows_per_group) -> None:
    """Single-step dgrad and wgrad for both backends, against bf16.

    A fixed upstream gradient for all three models: letting each backward from
    its own loss feeds them gradients that already differ by the forward error
    (the mistake corrected in the backward audit's B6).
    """
    for model in models:
        dim, _, avail = moe_dims(model)
        E = min(avail, num_experts)
        ref, ao, te = _make_models(model, device, E)
        counts, x, dy, M = _inputs(model, device, E, rows_per_group)
        sh = f"{model} {M}x{dim}"

        grads = {}
        for tag, mod in (("bf16", ref), ("TorchAO", ao), ("TE", te)):
            xi = x.detach().clone().requires_grad_(True)
            mod.zero_grad(set_to_none=True)
            mod(xi, counts).backward(dy)
            grads[tag] = {"grad_input": xi.grad.clone()} | {
                n: p.grad.clone() for n, p in mod.named_parameters()
            }

        for name in grads["bf16"]:
            b = grads["bf16"][name]
            note = "; ".join(
                f"{tag} relL2 {_rel_l2(grads[tag][name], b):.5f} gain {_gain(grads[tag][name], b):.5f}"
                for tag in ("TorchAO", "TE")
            )
            record("M2-bwd", sh, name, True, 0, 0, 0, note, info=True)


def _bias_and_noise(mod, x, counts, dy, ref_grad, name, n_draws, reseed=False):
    """Split a backend's wgrad error into its deterministic and stochastic parts.

    SR noise is zero-mean (backward audit B4), so averaging n draws with the same
    inputs leaves the deterministic error. That residue is the only part that
    survives 382 optimizer steps; the per-draw error is what SR trades away to
    get it.
    """
    acc = torch.zeros_like(ref_grad, dtype=torch.float32)
    per_draw = 0.0
    for _ in range(n_draws):
        if reseed:
            # Before the forward, not the backward: the seed is captured into the
            # autograd ctx at quantization time (nvfp4_grouped_mm.py:227).
            mod._sr_seed = torch.randint(
                -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=ref_grad.device
            )
        mod.zero_grad(set_to_none=True)
        mod(x.detach().clone().requires_grad_(True), counts).backward(dy)
        g = getattr(mod, name).grad
        acc += g.float()
        per_draw += _rel_l2(g, ref_grad)
    acc /= n_draws
    return _rel_l2(acc, ref_grad), _gain(acc, ref_grad), per_draw / n_draws


def stage_m3_bias_vs_noise(device, models, num_experts, rows_per_group, n_draws) -> None:
    """The deterministic (compounding) part of each backend's wgrad error."""
    for model in models:
        dim, _, avail = moe_dims(model)
        E = min(avail, num_experts)
        ref, ao, te = _make_models(model, device, E)
        counts, x, dy, M = _inputs(model, device, E, rows_per_group)
        sh = f"{model} {M}x{dim}"

        ref.zero_grad(set_to_none=True)
        ref(x.detach().clone().requires_grad_(True), counts).backward(dy)

        for tag, mod in (("TorchAO", ao), ("TE", te)):
            for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
                b = getattr(ref, name).grad
                bias, gain, noise = _bias_and_noise(
                    mod, x, counts, dy, b, name, n_draws
                )
                record(
                    "M3-bias", sh, f"{tag} {name}", True, 0, 0, 0,
                    f"mean of {n_draws} draws: relL2 {bias:.5f}, gain {gain:.5f}; "
                    f"per-draw relL2 {noise:.5f}",
                    info=True,
                )


def stage_m4_sr_seed(device, models, num_experts, rows_per_group, n_draws) -> None:
    """TorchAO with its `_sr_seed` re-drawn every step, vs the shipped fixed seed.

    Shipped, `_sr_seed` is drawn once in `_init_self_buffers` and only the two
    32-bit counter words are refreshed per backward
    (`nvfp4_grouped_mm.py:273-279`); TE pulls seed *and* offset from the CUDA
    generator on every cast (`csrc/extensions/cast.cpp:177`). If pinning the key
    correlates SR draws across steps, the fixed-seed mean will sit further from
    bf16 than the re-drawn one at the same draw count. If the two agree, the seed
    is not the mechanism and the difference is elsewhere in the recipe.
    """
    for model in models:
        dim, _, avail = moe_dims(model)
        E = min(avail, num_experts)
        ref, ao, _ = _make_models(model, device, E)
        counts, x, dy, M = _inputs(model, device, E, rows_per_group)
        sh = f"{model} {M}x{dim}"

        ref.zero_grad(set_to_none=True)
        ref(x.detach().clone().requires_grad_(True), counts).backward(dy)
        b = ref.w1_EFD.grad

        stats = {}
        for tag, reseed in (("fixed seed (shipped)", False), ("re-drawn seed", True)):
            stats[tag] = _bias_and_noise(
                ao, x, counts, dy, b, "w1_EFD", n_draws, reseed=reseed
            )
        for tag, (bias, gain, noise) in stats.items():
            record(
                "M4-seed", sh, f"TorchAO w1 wgrad, {tag}", True, 0, 0, 0,
                f"mean of {n_draws} draws: relL2 {bias:.5f}, gain {gain:.5f}; "
                f"per-draw relL2 {noise:.5f}",
                info=True,
            )
        fixed, redrawn = stats["fixed seed (shipped)"][0], stats["re-drawn seed"][0]
        record(
            "M4-seed", sh, "seed pinning changes the bias", True, 0, 0, 0,
            f"fixed/re-drawn bias ratio {fixed / redrawn:.4f} "
            f"(1.00 = the pinned seed is not a bias source at this draw count)",
            info=True,
        )


def stage_m5_single_gemm(device, models, num_experts, rows_per_group) -> None:
    """One `_grouped_mm` call, not the three-GEMM chain.

    M2 shows TorchAO's module wgrad at ~2x TE's error with a much lower gain.
    That is either the grouped-GEMM primitive itself or something about how the
    chain feeds it. Driving the seam directly on identical A, B_t and upstream
    gradient separates the two: matching numbers here move the cause into the
    module assembly, diverging numbers put it in the primitive and contradict
    the tensor-level backward audit.
    """
    for model in models:
        dim, hidden, avail = moe_dims(model)
        E = min(avail, num_experts)
        ref, ao, te = _make_models(model, device, E)
        counts, x, _, M = _inputs(model, device, E, rows_per_group)
        offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
        sh = f"{model} {M}x{dim}"

        torch.manual_seed(11)
        dy = torch.randn(M, hidden, dtype=torch.bfloat16, device=device)

        grads = {}
        for tag, mod in (("bf16", ref), ("TorchAO", ao), ("TE", te)):
            a = x.detach().clone().requires_grad_(True)
            w = ref.w1_EFD.detach().clone().requires_grad_(True)
            mod._grouped_mm(A=a, B_t=w.transpose(-2, -1), offs=offs).backward(dy)
            grads[tag] = {"dgrad": a.grad.clone(), "wgrad": w.grad.clone()}

        for name in ("dgrad", "wgrad"):
            b = grads["bf16"][name]
            note = "; ".join(
                f"{tag} relL2 {_rel_l2(grads[tag][name], b):.5f} gain {_gain(grads[tag][name], b):.5f}"
                for tag in ("TorchAO", "TE")
            )
            record("M5-gemm", sh, f"single grouped_mm {name}", True, 0, 0, 0, note, info=True)


def stage_m6_wgrad_consumers(device, models, num_experts, rows_per_group) -> None:
    """One set of TorchAO wgrad operands, two consumers.

    M5 puts the divergence in the primitive, which contradicts the backward
    audit: that audit found TorchAO's operands bitwise-equal to TE's and
    measured the wgrad at ~0.15 relL2 -- but it measured it by dequantizing the
    codes and doing the matmul in fp32, never through
    ``F.scaled_grouped_mm``. So quantize once and feed the same codes and scales
    to both consumers. If the fp32 dequant path lands near TE and the kernel
    path lands near M5's TorchAO number, the defect is in how the wgrad GEMM
    consumes the columnwise scales, not in the quantization.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

    from nvfp4_audit_common import ao_dequant, ao_mods, make_groups, sign_vector
    from nvfp4_backward_te_audit import _quantize_group, _rng_state

    triton_group_rht_amax, _, _, VARYING_FIRST_DIM, from_blocked = ao_mods()
    # The recipe the wgrad GEMM is called with (nvfp4_grouped_mm.py:36-37).
    scale_recipe = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
    swizzle = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]

    for model in models:
        dim, hidden, avail = moe_dims(model)
        E = min(avail, num_experts)
        K, N = dim, hidden
        sizes, offs = make_groups(E, rows_per_group, device)
        M = sum(sizes)

        torch.manual_seed(13)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        dy = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        xc, xr = triton_group_rht_amax(x, sign_vector(), offs, E, M, K,
                                       VARYING_FIRST_DIM, logical_packed_length=offs[-1:])
        dc, dr = triton_group_rht_amax(dy, sign_vector(), offs, E, M, N,
                                       VARYING_FIRST_DIM, logical_packed_length=offs[-1:])
        # x columnwise comes from the forward (no SR); dy columnwise from the
        # backward (SR), exactly as _NVFP4GroupedMM pairs them.
        _, _, xcol, xsf = _quantize_group(x, offs, E, K, row_amax=xr, col_amax=xc)
        _, _, dcol, dsf = _quantize_group(dy, offs, E, N, row_amax=dr, col_amax=dc,
                                          rng=_rng_state(1234, 7, 9, device))

        kernel = F.scaled_grouped_mm(
            dcol.view(torch.float4_e2m1fn_x2),
            xcol.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
            scale_a=[dsf, per_tensor_amax_to_scale(dc)],
            scale_recipe_a=scale_recipe,
            scale_b=[xsf, per_tensor_amax_to_scale(xc)],
            scale_recipe_b=scale_recipe,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
            offs=offs,
            output_dtype=torch.bfloat16,
        )

        xp = from_blocked(xsf, K, M // 16)
        dp = from_blocked(dsf, N, M // 16)
        sh = f"{model} {N}x{K}"
        start = 0
        keys = ["dequant fp32", "scaled_grouped_mm", "kernel vs its own operands"]
        errs = dict.fromkeys(keys, 0.0)
        gains = dict.fromkeys(keys, 0.0)
        by_group = {k: [] for k in keys}
        for g, sz in enumerate(sizes):
            end = start + sz
            xg = ao_dequant(xcol[:, start // 2:end // 2],
                            xp[:, start // 16:end // 16], xc[g])
            dg = ao_dequant(dcol[:, start // 2:end // 2],
                            dp[:, start // 16:end // 16], dc[g])
            ref = dy[start:end].float().t() @ x[start:end].float()
            dequant = dg @ xg.t()
            for tag, got, base in (("dequant fp32", dequant, ref),
                                   ("scaled_grouped_mm", kernel[g], ref),
                                   # The kernel against what its own operands
                                   # dequantize to: this drops the quantization
                                   # error out of both sides, so what is left is
                                   # only how the GEMM consumes them.
                                   ("kernel vs its own operands", kernel[g], dequant)):
                errs[tag] += _rel_l2(got, base)
                gains[tag] += _gain(got, base)
                by_group[tag].append(_rel_l2(got, base))
            start = end

        for tag in errs:
            per_group = ", ".join(f"{v:.5f}" for v in by_group[tag])
            record("M6-wgrad", sh, tag, True, 0, 0, 0,
                   f"mean over {E} groups: relL2 {errs[tag] / E:.5f}, "
                   f"gain {gains[tag] / E:.5f}; per group [{per_group}]", info=True)


STAGES = {
    "m1": stage_m1_forward,
    "m2": stage_m2_backward,
    "m5": stage_m5_single_gemm,
    "m6": stage_m6_wgrad_consumers,
    "m3": stage_m3_bias_vs_noise,
    "m4": stage_m4_sr_seed,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", default=None,
                    help="DSV3 flavor; repeatable (default: debugmodel)")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--rows-per-group", type=int, default=256)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--stage", action="append", default=None,
                    choices=["m1", "m2", "m3", "m4", "m5", "m6"])
    args = ap.parse_args()

    code = require_blackwell()
    if code is not None:
        return code

    device = torch.device("cuda")
    models = args.model or ["debugmodel"]
    stages = args.stage or list(STAGES)
    for name in stages:
        fn = STAGES[name]
        if name in ("m3", "m4"):
            fn(device, models, args.experts, args.rows_per_group, args.draws)
        else:
            fn(device, models, args.experts, args.rows_per_group)

    return 0 if print_table() else 1


if __name__ == "__main__":
    raise SystemExit(main())
