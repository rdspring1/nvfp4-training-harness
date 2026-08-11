"""Shared harness for the NVFP4 forward and backward audits vs TransformerEngine.

Extracted from nvfp4_forward_te_audit.py so the backward audit reuses one copy of
the byte-comparison logic. Nothing here is audit-stage-specific.

Bitwise means equality of raw bytes. On mismatch the helpers report the ULP
distribution and its direction rather than relaxing to SQNR: a systematic
one-directional 1-ULP shift and a symmetric tie-break look identical under SQNR
and are completely different bugs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TORCHAO_ROOT = _REPO_ROOT / "third_party" / "torchao"
sys.path.insert(0, str(_TORCHAO_ROOT / "benchmarks" / "prototype" / "nvfp4_training"))

from deepseek_v3_shapes import (  # noqa: E402
    DEEPSEEK_V3_MODEL_SHAPES,
    get_deepseek_v3_weight_shapes,
)

__all__ = [
    "DEEPSEEK_V3_MODEL_SHAPES",
    "get_deepseek_v3_weight_shapes",
    "FP4_E2M1_MAX",
    "FP8_E4M3_MAX",
    "FP32_MAX",
    "FP4_E2M1_GRID",
    "Row",
    "ROWS",
    "record",
    "print_table",
    "as_u8",
    "compare_e4m3",
    "compare_codes",
    "compare_nibbles",
    "compare_exact",
    "e4m3",
    "te_mods",
    "unpack_fp4",
    "te_quantize",
    "te_extract",
    "te_extract_col",
    "te_dequant",
    "te_ref_quantize",
    "ao_mods",
    "sign_vector",
    "moe_dims",
    "make_groups",
    "ao_dequant",
    "build_grouped_experts",
    "require_blackwell",
]

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
FP32_MAX = torch.finfo(torch.float32).max

FP4_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
"""Positive e2m1 values. Note the spacing is NOT uniform -- widths are
0.5, 0.5, 0.5, 0.5, 1, 1, 2 -- so any stochastic rounder that implicitly assumes
a uniform grid is biased only in the three wide intervals."""


# --------------------------------------------------------------------------
# result table
# --------------------------------------------------------------------------


@dataclass
class Row:
    stage: str
    shape: str
    tensor: str
    bitwise: bool
    max_ulp: int
    n_differ: int
    n_total: int
    note: str = ""
    info: bool = False
    """Informational rows describe an input to a question, not a verdict, and
    do not count as failures (e.g. 'these two fp32 expressions differ' is only
    interesting if the difference survives to the quantized output)."""


ROWS: list[Row] = []


def record(stage, shape, tensor, bitwise, max_ulp=0, n_differ=0, n_total=0, note="", info=False):
    ROWS.append(Row(stage, shape, tensor, bitwise, max_ulp, n_differ, n_total, note, info))


def print_table() -> bool:
    """Render the results table. Returns True if every non-informational row is bitwise."""
    if not ROWS:
        print("no results")
        return True
    w = [
        max(len(r.stage) for r in ROWS) + 1,
        max(len(r.shape) for r in ROWS) + 1,
        max(len(r.tensor) for r in ROWS) + 1,
    ]
    hdr = (
        f"{'stage':<{w[0]}} {'shape':<{w[1]}} {'tensor':<{w[2]}} "
        f"{'bitwise':<8} {'maxULP':>7} {'differ':>16}  note"
    )
    print("\n" + hdr)
    print("-" * (len(hdr) + 8))
    for r in ROWS:
        frac = f"{r.n_differ}/{r.n_total}" if r.n_total else "-"
        pct = f" ({100.0 * r.n_differ / r.n_total:.3f}%)" if r.n_total and r.n_differ else ""
        mark = "(info)" if r.info else ("yes" if r.bitwise else "NO")
        print(
            f"{r.stage:<{w[0]}} {r.shape:<{w[1]}} {r.tensor:<{w[2]}} "
            f"{mark:<8} {r.max_ulp:>7} {frac + pct:>16}  {r.note}"
        )
    checks = [r for r in ROWS if not r.info]
    n_bad = sum(1 for r in checks if not r.bitwise)
    print("-" * (len(hdr) + 8))
    print(f"{len(checks)} checks, {n_bad} not bitwise ({len(ROWS) - len(checks)} informational)\n")
    return n_bad == 0


# --------------------------------------------------------------------------
# byte-level comparison
# --------------------------------------------------------------------------


def as_u8(t: torch.Tensor) -> torch.Tensor:
    """Raw byte view of a tensor, regardless of its logical dtype."""
    if t.dtype == torch.uint8:
        return t.contiguous()
    return t.contiguous().view(torch.uint8)


def compare_e4m3(got: torch.Tensor, ref: torch.Tensor) -> tuple[bool, int, int, int, str]:
    """Compare two e4m3 scale tensors as raw bytes.

    Positive e4m3 bytes are magnitude-monotonic, so the unsigned byte delta is a
    true ULP distance for the non-negative block scales used here. The note
    reports directional skew, which is what separates a systematic bias from
    symmetric tie-breaking.
    """
    g = as_u8(got).flatten().to(torch.int32)
    r = as_u8(ref).flatten().to(torch.int32)
    assert g.shape == r.shape, f"shape mismatch {tuple(g.shape)} vs {tuple(r.shape)}"
    d = g - r
    n_differ = int((d != 0).sum())
    n_total = d.numel()
    if n_differ == 0:
        return True, 0, 0, n_total, ""
    n_low = int((d < 0).sum())
    n_high = int((d > 0).sum())
    skew = "got<ref" if n_low > n_high else ("got>ref" if n_high > n_low else "symmetric")
    return False, int(d.abs().max()), n_differ, n_total, f"{skew} lo={n_low} hi={n_high}"


def compare_codes(got: torch.Tensor, ref: torch.Tensor) -> tuple[bool, int, int, int, str]:
    """Compare PACKED FP4 codes (2 nibbles per byte) at nibble granularity.

    Use compare_nibbles for already-unpacked code arrays -- passing unpacked
    input here makes the high-nibble half of the comparison trivially equal and
    doubles the denominator.
    """
    g = as_u8(got).flatten()
    r = as_u8(ref).flatten()
    assert g.shape == r.shape, f"shape mismatch {tuple(g.shape)} vs {tuple(r.shape)}"
    n_differ = int(((g & 0x0F) != (r & 0x0F)).sum()) + int(((g >> 4) != (r >> 4)).sum())
    n_total = 2 * g.numel()
    if n_differ == 0:
        return True, 0, 0, n_total, ""
    return False, 1, n_differ, n_total, "fp4 nibbles"


def compare_nibbles(got: torch.Tensor, ref: torch.Tensor) -> tuple[bool, int, int, int, str]:
    """Compare UNPACKED FP4 code arrays (one code per byte, values 0-15)."""
    g = as_u8(got).flatten()
    r = as_u8(ref).flatten()
    assert g.shape == r.shape, f"shape mismatch {tuple(g.shape)} vs {tuple(r.shape)}"
    n_differ = int((g != r).sum())
    if n_differ == 0:
        return True, 0, 0, g.numel(), ""
    return False, 1, n_differ, g.numel(), "fp4 codes"


def compare_exact(got: torch.Tensor, ref: torch.Tensor, label: str = "") -> tuple:
    """Bitwise comparison of same-dtype tensors via raw bytes."""
    g = as_u8(got).flatten()
    r = as_u8(ref).flatten()
    assert g.shape == r.shape, f"{label}: shape mismatch"
    n_differ = int((g != r).sum())
    return (n_differ == 0), 0, n_differ, g.numel(), ""


def e4m3(x: torch.Tensor) -> torch.Tensor:
    """The tail shared by TE and TorchAO: cap at 448, cast to e4m3."""
    return torch.clamp(
        torch.minimum(x, torch.tensor(FP32_MAX, device=x.device)), max=FP8_E4M3_MAX
    ).to(torch.float8_e4m3fn)


# --------------------------------------------------------------------------
# TransformerEngine oracle
# --------------------------------------------------------------------------


def te_mods():
    """Import TE lazily so pure-arithmetic stages run without it."""
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch import NVFP4Quantizer
    from transformer_engine.pytorch.custom_recipes import utils as te_utils
    from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import (
        NVFP4QuantizerRef,
    )

    return te, tex, NVFP4Quantizer, NVFP4QuantizerRef, te_utils


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    """Split packed FP4 bytes into one nibble per column.

    Nibble order matches TE's own test helper
    (tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py:98): low nibble is the
    even element.
    """
    r = x.repeat_interleave(2, dim=1)
    r[:, 0::2] &= 0x0F
    r[:, 1::2] >>= 4
    return r


def te_quantize(
    x,
    *,
    two_d: bool = False,
    rowwise: bool = True,
    columnwise: bool = False,
    with_rht: bool = False,
    amax: torch.Tensor | None = None,
    col_amax: torch.Tensor | None = None,
):
    """TE NVFP4 quantize of a 2D bf16 tensor, stochastic rounding always off.

    When amax is given it is injected via tex.nvfp4_quantize_with_amax so both
    sides use an identical per-tensor scale; that keeps a *scale* difference
    from ever being mistaken for a *rounding* difference.

    with_rht=True forces with_post_rht_amax=True (TE rejects the other
    combination, csrc/quantizer.cpp:2624-2628) and applies the RHT to the
    columnwise/transposed output only.
    """
    te, tex, NVFP4Quantizer, _, _ = te_mods()
    q = NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=rowwise,
        columnwise=columnwise,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=with_rht,
        with_post_rht_amax=with_rht,
        with_2d_quantization=two_d,
        stochastic_rounding=False,
    )
    if amax is None:
        return q(x)
    a = amax.reshape(1).to(torch.float32).contiguous()
    c = a.clone() if col_amax is None else col_amax.reshape(1).to(torch.float32).contiguous()
    return tex.nvfp4_quantize_with_amax(x, q, a, c)


def te_extract(t, M: int, N: int):
    """Rowwise (unpacked codes (M,N), scale bytes (M,N//16)) from an NVFP4Tensor.

    TE pads the scale to [roundup(M,128), roundup(ceil(N/16),4)], so slice to the
    logical extent before comparing.
    """
    return (
        unpack_fp4(t._rowwise_data.view(torch.uint8))[:M, :N],
        as_u8(t._rowwise_scale_inv)[:M, : N // 16],
    )


def te_extract_col(t, M: int, N: int):
    """Columnwise outputs, which TE stores transposed as [N, M].

    nvfp4_tensor_storage.py:316-324: columnwise_shape = (K, M), byte shape
    (K, M//2). The columnwise scale pads to [roundup(N,128), roundup(ceil(M/16),4)].
    """
    return (
        unpack_fp4(t._columnwise_data.view(torch.uint8))[:N, :M],
        as_u8(t._columnwise_scale_inv)[:N, : M // 16],
    )


_FP4_LUT = None


def te_dequant(codes_unpacked: torch.Tensor, scale_bytes: torch.Tensor, amax: torch.Tensor):
    """Dequantize unpacked FP4 nibbles + e4m3 block scales.

    TE's NVFP4Tensor.dequantize() refuses columnwise data outright
    (nvfp4_tensor_storage.py:52-53), so this mirrors the local helper TE uses in
    its own SR test (tests/pytorch/nvfp4/test_nvfp4_sr_quantize.py:22-65).
    """
    global _FP4_LUT
    if _FP4_LUT is None or _FP4_LUT.device != codes_unpacked.device:
        vals = list(FP4_E2M1_GRID) + [-v for v in FP4_E2M1_GRID]
        _FP4_LUT = torch.tensor(vals, dtype=torch.float32, device=codes_unpacked.device)
    v = _FP4_LUT[codes_unpacked.long()]
    sf = scale_bytes.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return v * sf * (amax.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX))


def te_ref_quantize(x, *, two_d: bool = False, with_rht: bool = False, columnwise: bool = False):
    """TE's pure-PyTorch reference quantizer -- the calibration oracle.

    Round-to-nearest only; TE has no SR reference (quantization_ref_nvfp4.py has
    no stochastic_rounding parameter at all).
    """
    _, _, _, NVFP4QuantizerRef, te_utils = te_mods()
    kwargs = dict(
        dtype=te_utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=columnwise,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(16, 16) if two_d else (1, 16),
    )
    if with_rht:
        kwargs["with_rht"] = True
    ref = NVFP4QuantizerRef(**kwargs)
    return ref.quantize(x)


# --------------------------------------------------------------------------
# TorchAO forward/backward kernels
# --------------------------------------------------------------------------


def ao_mods():
    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
        triton_group_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
        VARYING_FIRST_DIM,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
        triton_group_weight_quantize_2d,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
        triton_group_rht_quantize_row_col,
    )
    from torchao.prototype.mx_formats.utils import from_blocked

    return (
        triton_group_rht_amax,
        triton_group_rht_quantize_row_col,
        triton_group_weight_quantize_2d,
        VARYING_FIRST_DIM,
        from_blocked,
    )


def sign_vector() -> list[int]:
    """The fixed RHT basis the TorchTitan converter uses on every rank."""
    from torchtitan.components.quantization.nvfp4 import _HARDCODED_SIGN_VECTOR

    return list(_HARDCODED_SIGN_VECTOR)


def moe_dims(model: str) -> tuple[int, int, int]:
    """(dim, moe_hidden_dim, num_local_experts) for a DSV3 flavor."""
    m = next(s for s in DEEPSEEK_V3_MODEL_SHAPES if s.model == model)
    return m.dim, m.moe_hidden_dim, m.local_experts


def make_groups(num_groups: int, rows_per_group: int, device):
    """128-aligned equal token groups, as the pad-128 dispatcher produces."""
    sizes = [rows_per_group] * num_groups
    offs = torch.cumsum(
        torch.tensor(sizes, dtype=torch.int32, device=device), dim=0, dtype=torch.int32
    )
    return sizes, offs


def ao_dequant(codes, sf_e4m3, amax):
    """Dequantize TorchAO packed codes + plain (de-swizzled) e4m3 scales."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    return (
        NVFP4Tensor(
            codes.contiguous(),
            sf_e4m3.contiguous(),
            16,
            torch.bfloat16,
            per_tensor_scale=per_tensor_amax_to_scale(amax.reshape(())),
            is_swizzled_scales=False,
        )
        .dequantize()
        .float()
    )


def build_grouped_experts(model: str, device, nvfp4: bool, num_experts: int | None = None):
    """Build a TorchTitan GroupedExperts, optionally the NVFP4 subclass."""
    from torchtitan.components.quantization.nvfp4 import _get_nvfp4_grouped_experts_cls
    from torchtitan.models.common.moe import GroupedExperts

    dim, hidden, n_exp = moe_dims(model)
    if num_experts is not None:
        n_exp = num_experts
    cls = _get_nvfp4_grouped_experts_cls(GroupedExperts) if nvfp4 else GroupedExperts
    cfg = cls.Config(dim=dim, hidden_dim=hidden, num_experts=n_exp)
    mod = cls(cfg).to(device=device, dtype=torch.bfloat16)
    if nvfp4:
        mod._init_self_buffers(buffer_device=device)
    return mod


def require_blackwell() -> int | None:
    """Return a nonzero exit code if this box cannot run NVFP4, else None."""
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    if torch.cuda.get_device_capability() < (10, 0):
        print("SM100+ required for NVFP4", file=sys.stderr)
        return 2
    return None
