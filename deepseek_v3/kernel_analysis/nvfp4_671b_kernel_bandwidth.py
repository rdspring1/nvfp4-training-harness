"""Effective memory bandwidth of the six NVFP4 training kernels at DeepSeek-V3 671B,
EP=64, across TransformerEngine / CuTeDSL / Triton and standard / fast math.

Six kernels, one chart each:

    1 linear_amax          RHT global amax
    2 linear_quantize_1d   fused RHT row+col quantize
    3 linear_weight_2d     2D weight quantize (no RHT)
    4 grouped_amax         grouped RHT global amax
    5 grouped_quantize_1d  grouped fused RHT row+col quantize
    6 grouped_weight_2d    grouped 2D weight quantize

Layout (671B: 256 routed experts, top-k 8, seq 4096, dim 7168, moe_hidden_dim 2048):

    bs 8 / ep 64  ->  tokens/rank   = 8 * 4096                  = 32,768
                      local experts = 256 / 64                  = 4
                      tokens/expert = (8 * 4096 * 64 * 8) / 256 = 65,536

Triton and CuTeDSL are timed with the submodule's own `kernel_time_us`, at the same
call signatures its bench_* scripts use, so these numbers are comparable with the
torchao benchmark README. TransformerEngine exposes no standalone amax or quantize
entry point -- it fuses them into one call -- so its per-kernel times are attributed
from the profiler by CUDA kernel name; see nvfp4_671b_te_kernel_probe.py, which fixes
that mapping and reproduces the README's published TE breakdown.

Run from the torchao submodule root so its `torchao` and `benchmarks` packages shadow
the stale site-packages copy:

    cd third_party/torchao && \
        PYTHONPATH=. python ../../deepseek_v3/kernel_analysis/nvfp4_671b_kernel_bandwidth.py \
            --csv ../../deepseek_v3/kernel_analysis/nvfp4_671b_64n_bandwidth.csv
"""

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from tabulate import tabulate
from tqdm import tqdm

from nvfp4_671b_te_kernel_probe import per_kernel_us, rht_quantizer, weight_quantizer

from benchmarks.prototype.nvfp4_training.bench_utils import kernel_time_us
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    SAME_BOTH_DIMS,
    VARYING_FIRST_DIM,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
    triton_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
    triton_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.utils import is_sm_at_least_100

import transformer_engine_torch as tex

device = torch.device("cuda")

BACKENDS = ("te", "cutedsl", "triton")
MATH_MODES = ("standard", "fast")

# The 16-entry RHT sign vector the kernels and TE both use.
RHT_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)

LOCAL_EXPERTS = 4  # 256 routed experts / ep 64
TOKENS_PER_RANK = 32_768  # bs 8 * seq 4096
TOKENS_PER_EXPERT = 65_536  # (bs 8 * seq 4096 * ep 64 * top-k 8) / 256

DIM = 7168  # 671B hidden dim
MOE_HIDDEN = 2048  # 671B moe_hidden_dim
DENSE_HIDDEN = 18432  # 671B dense-MLP intermediate dim


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    label: str
    experts: int = 0  # 0 == not grouped

    @property
    def rows(self) -> int:
        return (self.experts or 1) * self.m

    @property
    def elements(self) -> int:
        return self.rows * self.n


# Activation shapes: (tokens, hidden). Weight shapes: (out_features, in_features).
LINEAR_ACT = [
    Shape(TOKENS_PER_RANK, DIM, "hidden-state input"),
    Shape(TOKENS_PER_RANK, DENSE_HIDDEN, "dense-MLP down input"),
]
LINEAR_WEIGHT = [
    Shape(DENSE_HIDDEN, DIM, "dense-MLP gate/up weight"),
    Shape(DIM, DENSE_HIDDEN, "dense-MLP down weight"),
]
GROUPED_ACT = [
    Shape(TOKENS_PER_EXPERT, DIM, "gate/up (w1/w3) input", LOCAL_EXPERTS),
    Shape(TOKENS_PER_EXPERT, MOE_HIDDEN, "down (w2) input", LOCAL_EXPERTS),
]
GROUPED_WEIGHT = [
    Shape(MOE_HIDDEN, DIM, "gate/up (w1/w3) weight", LOCAL_EXPERTS),
    Shape(DIM, MOE_HIDDEN, "down (w2) weight", LOCAL_EXPERTS),
]


# --------------------------------------------------------------------------------------
# Byte accounting, copied from the submodule bench scripts so the GB/s stay comparable.
# --------------------------------------------------------------------------------------


def amax_bytes(shape: Shape) -> int:
    """Reads the whole bf16 input; the scalar (or 2E scalar) outputs are negligible."""
    return shape.elements * 2


def quantize_bytes(shape: Shape) -> int:
    """bf16 read + rowwise and colwise FP4 codes + both e4m3 block-scale sets.

    Matches `_rowcol_bytes` (bench_hadamard_quantize_row_col.py) and `_weight_bytes`
    (bench_quantize_2d.py); their swizzle-tile decomposition is algebraically this.
    """
    elements = shape.elements
    return elements * 2 + elements + 2 * (elements // 16)


# --------------------------------------------------------------------------------------
# Chart definitions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Chart:
    key: str
    title: str
    shapes: list
    bytes_fn: Callable[[Shape], int]
    # Which bucket of TE's fused call this chart's TE bar comes from.
    te_path: str  # "linear" | "grouped" | "weight2d"
    te_bucket: str  # "amax" | "quantize"
    te_estimated: bool = False


CHARTS = [
    Chart("linear_amax", "Linear RHT amax", LINEAR_ACT, amax_bytes, "linear", "amax"),
    Chart(
        "linear_quantize_1d",
        "Linear RHT row+col quantize",
        LINEAR_ACT,
        quantize_bytes,
        "linear",
        "quantize",
    ),
    Chart(
        "linear_weight_2d",
        "Linear 2D weight quantize",
        LINEAR_WEIGHT,
        quantize_bytes,
        "weight2d",
        "quantize",
    ),
    Chart(
        "grouped_amax", "Grouped RHT amax", GROUPED_ACT, amax_bytes, "grouped", "amax"
    ),
    Chart(
        "grouped_quantize_1d",
        "Grouped RHT row+col quantize",
        GROUPED_ACT,
        quantize_bytes,
        "grouped",
        "quantize",
    ),
    Chart(
        "grouped_weight_2d",
        "Grouped 2D weight quantize",
        GROUPED_WEIGHT,
        quantize_bytes,
        "weight2d",
        "quantize",
        # TE has no grouped 2D weight kernel; its bar is E x the single-expert time.
        te_estimated=True,
    ),
]


# --------------------------------------------------------------------------------------
# TransformerEngine: attribute the fused call's CUDA self-time by kernel name.
# --------------------------------------------------------------------------------------

# Substrings that identify each kernel in the (heavily mangled) profiler names. Verified
# by nvfp4_671b_te_kernel_probe.py. Order matters: the grouped names contain the linear
# ones as substrings, so classification is always done within one call's kernel set.
TE_BUCKETS = {
    "linear": {
        "amax": ("HadamardAmaxTmaKernel", "ZeroAmaxKernel"),
        "quantize": ("row_col_rht_gemm_device",),
    },
    "grouped": {
        "amax": ("GroupHadamardAmaxTmaKernel", "MultiZeroAmaxKernel"),
        "quantize": ("group_row_col_rht_gemm_device",),
    },
    "weight2d": {
        # TE computes its own weight amax; the torchao 2D kernels consume a precomputed
        # one, so counting TE's amax pass would credit them with skipping work they
        # never do. Excluded from the chart, still reported for the record.
        "amax": ("amax_kernel<", "zero_amax_kernel("),
        "quantize": ("quantize_transpose_nvfp4_2D_kernel",),
    },
}


def te_split(path: str, kernels: dict) -> dict:
    """Sum a TE call's per-kernel times into {'amax': us, 'quantize': us}.

    Raises if any kernel is unclassified -- a TE update that renames or adds a kernel
    must fail loudly rather than silently drop time from a bar.
    """
    buckets = {name: 0.0 for name in TE_BUCKETS[path]}
    for name, us in kernels.items():
        for bucket, needles in TE_BUCKETS[path].items():
            if any(needle in name for needle in needles):
                buckets[bucket] += us
                break
        else:
            raise RuntimeError(
                f"unclassified TE kernel on the {path} path ({us:.3f} us): {name[:160]}"
            )
    return buckets


def te_measure(path: str, shape: Shape, fast: bool) -> dict:
    """Run TE's fused call for ``path`` at ``shape`` and return its bucket times."""
    set_fast_math(fast)
    if path == "linear":
        A = torch.randn(shape.m, shape.n, dtype=torch.bfloat16, device=device)
        quantizer = rht_quantizer()
        fn = lambda: quantizer(A)  # noqa: E731
    elif path == "grouped":
        A = torch.randn(shape.rows, shape.n, dtype=torch.bfloat16, device=device)
        quantizers = [rht_quantizer() for _ in range(shape.experts)]
        fn = lambda: tex.split_quantize(A, [shape.m] * shape.experts, quantizers)  # noqa: E731
    elif path == "weight2d":
        # TE has no grouped 2D kernel: time one expert, scale by E at the call site.
        W = torch.randn(shape.m, shape.n, dtype=torch.bfloat16, device=device)
        quantizer = weight_quantizer()
        fn = lambda: quantizer(W)  # noqa: E731
    else:
        raise ValueError(path)
    result = te_split(path, per_kernel_us(fn))
    set_fast_math(False)
    return result


def set_fast_math(enabled: bool) -> None:
    """TE reads NVTE_USE_FAST_MATH through plain std::getenv per call, uncached."""
    if enabled:
        os.environ["NVTE_USE_FAST_MATH"] = "1"
    else:
        os.environ.pop("NVTE_USE_FAST_MATH", None)


# --------------------------------------------------------------------------------------
# Triton / CuTeDSL runners, at the submodule bench scripts' call signatures.
# --------------------------------------------------------------------------------------


def make_torchao_runner(
    chart: Chart, backend: str, shape: Shape, fast: bool
) -> Optional[Callable[[], object]]:
    if backend == "cutedsl" and not cutedsl_nvfp4_kernels_available():
        return None
    sv = list(RHT_SIGN_VECTOR)
    grouped = bool(shape.experts)

    if chart.te_path == "weight2d":
        if grouped:
            weights = torch.randn(
                (shape.experts, shape.m, shape.n), dtype=torch.bfloat16, device=device
            )
            global_amax = weights.float().abs().amax(dim=(1, 2))
            if backend == "triton":
                from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
                    triton_group_weight_quantize_2d as op,
                )
            else:
                from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_cutedsl import (
                    cutedsl_group_weight_quantize_2d as op,
                )
            return lambda: op(weights, global_amax, shape.experts)
        w = torch.randn(shape.m, shape.n, dtype=torch.bfloat16, device=device)
        global_amax = w.float().abs().max()
        if backend == "triton":
            from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
                triton_weight_quantize_2d as op,
            )
        else:
            from torchao.prototype.moe_training.nvfp4_training.quantize_2d_cutedsl import (
                cutedsl_weight_quantize_2d as op,
            )
        return lambda: op(w, global_amax)

    if grouped:
        A = torch.randn((shape.rows, shape.n), dtype=torch.bfloat16, device=device)
        offsets = (
            torch.arange(1, shape.experts + 1, dtype=torch.int32, device=device)
            * shape.m
        )
        logical_packed_length = offsets[-1:]
        if chart.te_bucket == "amax":
            if backend == "triton":
                op = triton_group_rht_amax
            else:
                from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
                    cutedsl_group_rht_amax as op,
                )
            return lambda: op(
                A,
                sv,
                offsets,
                shape.experts,
                shape.rows,
                shape.n,
                SAME_BOTH_DIMS,
                logical_packed_length=logical_packed_length,
            )
        # Grouped quantize: feed every backend the same precomputed per-group amaxes.
        col_amax, row_amax = triton_group_rht_amax(
            A,
            sv,
            offsets,
            shape.experts,
            shape.rows,
            shape.n,
            SAME_BOTH_DIMS,
            logical_packed_length=logical_packed_length,
        )
        if backend == "triton":
            from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
                triton_group_rht_quantize_row_col as op,
            )
        else:
            from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
                cutedsl_group_rht_quantize_row_col as op,
            )
        return lambda: op(
            A,
            sv,
            offsets,
            shape.experts,
            shape.rows,
            shape.n,
            VARYING_FIRST_DIM,
            row_amax,
            col_amax,
            None,
            False,
            logical_packed_length=logical_packed_length,
            use_fast_math=fast,
        )

    x = torch.randn(shape.m, shape.n, dtype=torch.bfloat16, device=device)
    if chart.te_bucket == "amax":
        if backend == "triton":
            return lambda: triton_rht_amax(x, sign_vector=sv)
        from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_cutedsl import (
            cutedsl_rht_amax,
        )

        return lambda: cutedsl_rht_amax(x, sv)
    col_amax, row_amax = triton_rht_amax(x, sign_vector=sv)
    if backend == "triton":
        from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_triton import (
            triton_rht_quantize_row_col,
        )

        return lambda: triton_rht_quantize_row_col(
            x,
            col_global_amax=col_amax,
            row_global_amax=row_amax,
            sign_vector=sv,
            stochastic_rounding=False,
            use_fast_math=fast,
        )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_cutedsl import (
        cutedsl_rht_quantize_row_col,
    )

    return lambda: cutedsl_rht_quantize_row_col(
        x, col_amax, row_amax, sv, use_fast_math=fast
    )


# --------------------------------------------------------------------------------------


def get_peak_mem_bw_gbps() -> Optional[float]:
    props = torch.cuda.get_device_properties(device)
    clock_khz = getattr(props, "memory_clock_rate", 0)
    bus_bits = getattr(props, "memory_bus_width", 0)
    if clock_khz <= 0 or bus_bits <= 0:
        return None
    return ((bus_bits / 8.0) * (clock_khz * 1e3) * 2.0) / 1e9


@dataclass
class Row:
    chart: str
    kernel: str
    label: str
    experts: int
    m: int
    n: int
    backend: str
    math: str
    us: float
    moved_bytes: int
    gbps: float
    estimated: bool
    te_amax_us: float = 0.0  # TE's excluded weight-amax pass, recorded for the record
    extras: dict = field(default_factory=dict)


def run(charts: list) -> list:
    rows = []
    work = [(c, s, m) for c in charts for s in c.shapes for m in MATH_MODES]
    for chart, shape, math_mode in tqdm(work, desc="measuring"):
        fast = math_mode == "fast"
        moved = chart.bytes_fn(shape)

        te_buckets = te_measure(chart.te_path, shape, fast)
        te_us = te_buckets[chart.te_bucket]
        if chart.te_estimated:
            te_us *= shape.experts
        rows.append(
            Row(
                chart.key,
                chart.title,
                shape.label,
                shape.experts,
                shape.m,
                shape.n,
                "te",
                math_mode,
                te_us,
                moved,
                (moved / 1e9) / (te_us / 1e6),
                chart.te_estimated,
                te_buckets.get("amax", 0.0)
                if chart.te_path == "weight2d"
                else 0.0,
            )
        )

        for backend in ("cutedsl", "triton"):
            runner = make_torchao_runner(chart, backend, shape, fast)
            if runner is None:
                continue
            us = kernel_time_us(runner)
            rows.append(
                Row(
                    chart.key,
                    chart.title,
                    shape.label,
                    shape.experts,
                    shape.m,
                    shape.n,
                    backend,
                    math_mode,
                    us,
                    moved,
                    (moved / 1e9) / (us / 1e6),
                    False,
                )
            )
        torch.cuda.empty_cache()
    return rows


def print_results(rows: list, charts: list, peak: Optional[float]) -> None:
    for chart in charts:
        subset = [r for r in rows if r.chart == chart.key]
        if not subset:
            continue
        print(f"\n### {chart.title}  [{chart.key}]")
        if chart.te_estimated:
            print("    TE has no grouped 2D weight kernel; its column is E x the "
                  "single-expert time (ESTIMATE).")
        table = []
        for shape in chart.shapes:
            for math_mode in MATH_MODES:
                cells = {
                    r.backend: r
                    for r in subset
                    if r.m == shape.m and r.n == shape.n and r.math == math_mode
                }
                row = [shape.label, shape.experts or "-", shape.m, shape.n, math_mode]
                for backend in BACKENDS:
                    r = cells.get(backend)
                    row.append(round(r.us, 2) if r else "n/a")
                for backend in BACKENDS:
                    r = cells.get(backend)
                    row.append(round(r.gbps, 1) if r else "n/a")
                best = max((r.gbps for r in cells.values()), default=0)
                row.append(
                    round(best / peak * 100.0, 1) if peak and best else "n/a"
                )
                table.append(row)
        headers = (
            ["shape", "E", "M", "N", "math"]
            + [f"{b}_us" for b in BACKENDS]
            + [f"{b}_gbps" for b in BACKENDS]
            + ["pct_peak"]
        )
        print(tabulate(table, headers=headers))


def write_csv(rows: list, path: str, peak: Optional[float]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "chart", "kernel", "shape", "E", "M", "N", "backend", "math",
                "us", "bytes", "gbps", "peak_gbps", "estimated", "te_excluded_amax_us",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.chart, r.kernel, r.label, r.experts, r.m, r.n, r.backend, r.math,
                    f"{r.us:.4f}", r.moved_bytes, f"{r.gbps:.3f}",
                    f"{peak:.1f}" if peak else "", int(r.estimated),
                    f"{r.te_amax_us:.4f}" if r.te_amax_us else "",
                ]
            )
    print(f"\nwrote {len(rows)} rows to {path}")


def main() -> None:
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise SystemExit("NVFP4 grouped quantization requires SM100+")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chart",
        choices=("all", *[c.key for c in CHARTS]),
        default="all",
    )
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    charts = CHARTS if args.chart == "all" else [c for c in CHARTS if c.key == args.chart]

    torch.random.manual_seed(123)
    peak = get_peak_mem_bw_gbps()
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"peak memory bandwidth: {peak:.1f} GB/s" if peak else "peak: n/a")
    print(
        f"671B EP=64: {LOCAL_EXPERTS} local experts, {TOKENS_PER_RANK:,} tokens/rank, "
        f"{TOKENS_PER_EXPERT:,} tokens/expert"
    )

    rows = run(charts)
    print_results(rows, charts, peak)
    if args.csv:
        write_csv(rows, args.csv, peak)


if __name__ == "__main__":
    main()
