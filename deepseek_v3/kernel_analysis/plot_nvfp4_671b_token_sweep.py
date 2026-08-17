"""Plot the token sweep: effective bandwidth vs tokens, one chart per (kernel, N).

x is a scale rather than a label -- the token count, log2, at a fixed model hidden dim.
The three cells of the 671B EP grid are marked on it. Six series: backend (hue) x math
mode (solid / dashed + open marker).

    python deepseek_v3/kernel_analysis/plot_nvfp4_671b_token_sweep.py \
        --csv deepseek_v3/kernel_analysis/nvfp4_671b_token_sweep.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BACKENDS = [
    ("te", "TransformerEngine", "#2a78d6"),
    ("cutedsl", "CuTeDSL", "#eb6834"),
    ("triton", "Triton", "#1baf7a"),
]

TITLES = {
    "linear_amax": "Linear RHT amax",
    "linear_quantize_1d": "Linear RHT row+col quantize",
    "grouped_amax": "Grouped RHT amax",
    "grouped_quantize_1d": "Grouped RHT row+col quantize",
}

FLAT_THRESHOLD = 0.03


def load(path):
    rows = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            row["tokens"] = int(row["tokens"])
            row["N"] = int(row["N"])
            row["us"] = float(row["us"])
            row["gbps"] = float(row["gbps"])
            row["peak_gbps"] = float(row["peak_gbps"]) if row["peak_gbps"] else None
            rows.append(row)
    return rows


def load_layouts(path):
    cells = defaultdict(list)
    if not Path(path).exists():
        return cells
    with open(path) as handle:
        for row in csv.DictReader(handle):
            cells[row["family"]].append((row["layout"], int(row["tokens"])))
    return cells


def is_flat(rows):
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["tokens"], row["backend"])][row["math"]] = row["us"]
    for modes in by_key.values():
        if "standard" in modes and "fast" in modes:
            if abs(modes["fast"] - modes["standard"]) / modes["standard"] > FLAT_THRESHOLD:
                return False
    return True


def plot(key, n, rows, layouts, out_dir):
    flat = is_flat(rows)
    modes = ["standard"] if flat else ["standard", "fast"]
    peak = next((r["peak_gbps"] for r in rows if r["peak_gbps"]), None)
    family = rows[0]["family"]
    x_name = rows[0]["x_name"]
    shape_label = rows[0]["shape"]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Layout markers first, so the curves draw over them.
    for layout, tokens in sorted(layouts.get(family, []), key=lambda c: c[1]):
        ax.axvline(tokens, color=BASELINE, linewidth=1.0, linestyle=":", zorder=1)
        ax.text(
            tokens,
            1.004,
            layout.replace("671B ", ""),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_MUTED,
        )

    for backend, name, color in BACKENDS:
        for math_mode in modes:
            points = sorted(
                (
                    (r["tokens"], r["gbps"])
                    for r in rows
                    if r["backend"] == backend and r["math"] == math_mode
                ),
            )
            if not points:
                continue
            xs, ys = zip(*points)
            fast = math_mode == "fast"
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.0,
                linestyle="--" if fast else "-",
                marker="o",
                markersize=5.5,
                markerfacecolor=SURFACE if fast else color,
                markeredgecolor=color,
                markeredgewidth=1.6,
                zorder=3,
            )

    if peak:
        ax.axhline(peak, linestyle="--", linewidth=1.2, color=BASELINE, zorder=2)
        ax.text(
            max(r["tokens"] for r in rows),
            peak,
            f"  HBM peak {peak:,.0f} GB/s",
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=INK_MUTED,
        )
        ax.set_ylim(0, peak * 1.1)

    ticks = sorted({r["tokens"] for r in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{t // 1024}k" if t >= 1024 else str(t) for t in ticks], fontsize=9
    )
    ax.minorticks_off()
    ax.set_xlabel(x_name, fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Effective memory bandwidth (GB/s)", fontsize=10, color=INK_SECONDARY)
    ax.set_title(f"{TITLES[key]} — NVFP4", fontsize=13, color=INK, loc="left", pad=26)
    ax.text(
        0,
        1.045,
        f"DeepSeek-V3 671B, {shape_label}, N={n:,}"
        + (f", E={rows[0]['E']} local experts" if family == "grouped" else "")
        + "  (NVIDIA GB200)",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_MUTED,
    )

    handles = [
        Line2D([], [], color=color, linewidth=2.0, marker="o", markersize=5.5, label=name)
        for _, name, color in BACKENDS
    ]
    if not flat:
        handles += [
            Line2D(
                [], [], color=INK_MUTED, linewidth=2.0, marker="o", markersize=5.5,
                label="standard math",
            ),
            Line2D(
                [], [], color=INK_MUTED, linewidth=2.0, linestyle="--", marker="o",
                markersize=5.5, markerfacecolor=SURFACE, markeredgecolor=INK_MUTED,
                label="fast math",
            ),
        ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, -0.14),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    if flat:
        ax.text(
            0,
            -0.245,
            "* Fast math has no variant for this kernel on any backend — measured, it "
            "moves every backend by <3%. Standard math shown.",
            transform=ax.transAxes,
            fontsize=7.8,
            color=INK_MUTED,
            va="top",
        )

    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)

    fig.tight_layout()
    out = out_dir / f"nvfp4_671b_sweep_{key}_n{n}.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (flat={flat})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parent / "nvfp4_671b_token_sweep.csv"),
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.csv).resolve().parent
    rows = load(args.csv)
    layouts = load_layouts(args.csv.replace(".csv", "_layouts.csv"))

    groups = defaultdict(list)
    for row in rows:
        groups[(row["chart"], row["N"])].append(row)
    for (key, n), subset in groups.items():
        plot(key, n, subset, layouts, out_dir)


if __name__ == "__main__":
    main()
