"""Sweep tokens/expert (grouped) and tokens/rank (linear) against effective bandwidth.

The bar charts in nvfp4_671b_kernel_bandwidth.py put *shape* on the x-axis, which is a
label rather than a scale: two shapes can differ in both M and N, so no ordering of them
carries physical meaning. This puts a scalar that actually drives the kernel on x instead
-- the token count -- at a fixed model hidden dim N, which is apples-to-apples by
construction: N is a model constant and the token count is exactly what the parallel
layout varies.

The three cells of the 671B EP grid are three points on this axis, and are annotated as
such:

    layout          bs / ep    tokens/rank   tokens/expert
    671B 12-layer    4 / 16         16,384           8,192
    671B 16n         8 / 32         32,768          32,768
    671B 64n         8 / 64         32,768          65,536

Only the four token-dependent kernels are swept. The 2D weight kernels quantize expert
weights and have no token dependence at all; they stay bar charts.

Run from the torchao submodule root:

    cd third_party/torchao && \
        PYTHONPATH=. python ../../deepseek_v3/kernel_analysis/nvfp4_671b_token_sweep.py \
            --csv ../../deepseek_v3/kernel_analysis/nvfp4_671b_token_sweep.csv
"""

import argparse
import csv

import torch
from tqdm import tqdm

from nvfp4_671b_kernel_bandwidth import (
    CHARTS,
    MATH_MODES,
    Shape,
    amax_bytes,
    get_peak_mem_bw_gbps,
    make_torchao_runner,
    quantize_bytes,
    te_measure,
)

from benchmarks.prototype.nvfp4_training.bench_utils import kernel_time_us
from torchao.utils import is_sm_at_least_100

TOKENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

LOCAL_EXPERTS = 4  # 256 routed experts / ep 64

# (family, x-axis name, hidden dims, {layout label: token count})
FAMILIES = [
    (
        "linear",
        "tokens/rank",
        [(7168, "hidden-state input"), (18432, "dense-MLP down input")],
        {"671B 12-layer": 16_384, "671B 16n": 32_768, "671B 64n": 32_768},
    ),
    (
        "grouped",
        "tokens/expert",
        [(7168, "gate/up (w1/w3) input"), (2048, "down (w2) input")],
        {"671B 12-layer": 8_192, "671B 16n": 32_768, "671B 64n": 65_536},
    ),
]

# The amax and quantize charts of a family, keyed by the bucket te_measure returns.
CHART_BY_FAMILY = {
    ("linear", "amax"): "linear_amax",
    ("linear", "quantize"): "linear_quantize_1d",
    ("grouped", "amax"): "grouped_amax",
    ("grouped", "quantize"): "grouped_quantize_1d",
}
CHART_LOOKUP = {c.key: c for c in CHARTS}


def sweep(families, tokens):
    rows = []
    work = [
        (family, x_name, n, label, cells, count, math_mode)
        for family, x_name, dims, cells in families
        for n, label in dims
        for count in tokens
        for math_mode in MATH_MODES
    ]
    for family, x_name, n, label, cells, count, math_mode in tqdm(work, desc="sweeping"):
        fast = math_mode == "fast"
        experts = LOCAL_EXPERTS if family == "grouped" else 0
        shape = Shape(count, n, label, experts)
        bytes_by_bucket = {
            "amax": amax_bytes(shape),
            "quantize": quantize_bytes(shape),
        }

        def record(bucket, backend, us):
            moved = bytes_by_bucket[bucket]
            rows.append(
                {
                    "chart": CHART_BY_FAMILY[(family, bucket)],
                    "family": family,
                    "bucket": bucket,
                    "x_name": x_name,
                    "shape": label,
                    "E": experts,
                    "tokens": count,
                    "N": n,
                    "backend": backend,
                    "math": math_mode,
                    "us": us,
                    "bytes": moved,
                    "gbps": (moved / 1e9) / (us / 1e6),
                }
            )

        # One TE profile per size feeds both this family's charts -- TE fuses amax and
        # quantize into a single call, so splitting them costs nothing extra.
        for bucket, us in te_measure(family, shape, fast).items():
            record(bucket, "te", us)

        for bucket in ("amax", "quantize"):
            chart = CHART_LOOKUP[CHART_BY_FAMILY[(family, bucket)]]
            for backend in ("cutedsl", "triton"):
                runner = make_torchao_runner(chart, backend, shape, fast)
                if runner is not None:
                    record(bucket, backend, kernel_time_us(runner))
        torch.cuda.empty_cache()
    return rows


def write_csv(rows, path, peak):
    fields = [
        "chart", "family", "bucket", "x_name", "shape", "E", "tokens", "N",
        "backend", "math", "us", "bytes", "gbps", "peak_gbps",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "peak_gbps": f"{peak:.1f}" if peak else ""})
    print(f"\nwrote {len(rows)} rows to {path}")


def write_layout_csv(path):
    """The three EP-grid cells, so the plotter can annotate them without hardcoding."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "layout", "tokens"])
        for family, _, _, cells in FAMILIES:
            for layout, count in cells.items():
                writer.writerow([family, layout, count])
    print(f"wrote layout cells to {path}")


def main():
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise SystemExit("NVFP4 grouped quantization requires SM100+")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--family", choices=("all", "linear", "grouped"), default="all"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=max(TOKENS),
        help="Cap the sweep; the largest grouped points allocate several GB per backend.",
    )
    args = parser.parse_args()

    families = (
        FAMILIES if args.family == "all" else [f for f in FAMILIES if f[0] == args.family]
    )
    tokens = [t for t in TOKENS if t <= args.max_tokens]

    torch.random.manual_seed(123)
    peak = get_peak_mem_bw_gbps()
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"peak memory bandwidth: {peak:.1f} GB/s" if peak else "peak: n/a")
    print(f"sweeping {tokens}")

    rows = sweep(families, tokens)
    write_csv(rows, args.csv, peak)
    write_layout_csv(args.csv.replace(".csv", "_layouts.csv"))


if __name__ == "__main__":
    main()
