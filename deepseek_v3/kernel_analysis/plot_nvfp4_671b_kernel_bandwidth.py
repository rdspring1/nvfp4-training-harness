"""Plot the six NVFP4 kernel bandwidth charts from nvfp4_671b_kernel_bandwidth.py's CSV.

One chart per kernel. Series are the three backends (hue) x the two math modes
(solid / hatched tint). Charts where fast math provably does not move the kernel --
the amax and 2D weight families, which have no fast-math variant on any backend --
collapse to the three standard bars and say so in a footnote, rather than drawing six
bars in three identical-height pairs.

No GPU needed; reads the CSV only.

    python deepseek_v3/kernel_analysis/plot_nvfp4_671b_kernel_bandwidth.py \
        --csv deepseek_v3/kernel_analysis/nvfp4_671b_64n_bandwidth.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Validated 3-slot categorical palette (dataviz reference palette, light mode):
# all-pairs CVD dE 9.2, normal-vision dE 24.0. The aqua slot sits at 2.74:1 on the
# light surface, so the relief rule applies -- every bar carries a visible value label.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BACKENDS = [("te", "TransformerEngine", "#2a78d6"), ("cutedsl", "CuTeDSL", "#eb6834"),
            ("triton", "Triton", "#1baf7a")]

# Below this relative spread between standard and fast, the kernel has no fast-math
# variant and the fast series is dropped rather than drawn on top of the standard one.
FLAT_THRESHOLD = 0.03


def tint(hex_color, amount=0.45):
    """Blend toward white -- the fast-math fill, same hue, texture carries the mode."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: (c * (1 - amount) + 255 * amount) / 255  # noqa: E731
    return (mix(r), mix(g), mix(b))


def load(path):
    charts = defaultdict(list)
    with open(path) as handle:
        for row in csv.DictReader(handle):
            row["us"] = float(row["us"])
            row["gbps"] = float(row["gbps"])
            row["peak_gbps"] = float(row["peak_gbps"]) if row["peak_gbps"] else None
            row["estimated"] = row["estimated"] == "1"
            charts[row["chart"]].append(row)
    return charts


def is_flat(rows):
    """True when fast math never moves any backend by more than FLAT_THRESHOLD."""
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["shape"], row["backend"])][row["math"]] = row["us"]
    for modes in by_key.values():
        if "standard" in modes and "fast" in modes:
            std, fast = modes["standard"], modes["fast"]
            if abs(fast - std) / std > FLAT_THRESHOLD:
                return False
    return True


def plot_chart(key, rows, out_dir):
    shapes, seen = [], set()
    for row in rows:
        label = row["shape"]
        if label not in seen:
            seen.add(label)
            shapes.append(row)

    flat = is_flat(rows)
    modes = ["standard"] if flat else ["standard", "fast"]
    series = [(b, name, color, m) for b, name, color in BACKENDS for m in modes]

    lookup = {(r["shape"], r["backend"], r["math"]): r for r in rows}
    peak = next((r["peak_gbps"] for r in rows if r["peak_gbps"]), None)
    estimated = any(r["estimated"] for r in rows)

    n = len(series)
    group_width = 0.74
    bar_width = group_width / n
    xs = range(len(shapes))

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for slot, (backend, name, color, math_mode) in enumerate(series):
        offset = -group_width / 2 + bar_width * (slot + 0.5)
        values, positions = [], []
        for i, shape_row in enumerate(shapes):
            entry = lookup.get((shape_row["shape"], backend, math_mode))
            if entry is None:
                continue
            values.append(entry["gbps"])
            positions.append(i + offset)
        fast = math_mode == "fast"
        # An estimated bar (TE's absent grouped 2D kernel) is drawn open with a dashed
        # rule so it cannot be mistaken for a measurement at a glance.
        est = estimated and backend == "te"
        bars = ax.bar(
            positions,
            values,
            width=bar_width * 0.92,  # ~2px surface gap between adjacent bars
            facecolor=tint(color) if (fast or est) else color,
            edgecolor=color,
            linewidth=1.2 if (fast or est) else 0.0,
            linestyle="--" if est else "-",
            hatch="///" if fast else None,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + peak * 0.012,
                f"{value:,.0f}" + (" est." if est else ""),
                ha="center",
                va="bottom",
                fontsize=7.5 if n > 3 else 8.5,
                rotation=90 if n > 3 else 0,
                color=INK_SECONDARY,
            )

    if peak:
        ax.axhline(peak, linestyle="--", linewidth=1.2, color=BASELINE, zorder=2)
        ax.text(
            len(shapes) - 0.5,
            peak,
            f"  HBM peak {peak:,.0f} GB/s",
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=INK_MUTED,
        )
        ax.set_ylim(0, peak * 1.16)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [
            f"{r['shape']}\n"
            + (f"E={r['E']}  " if r["E"] != "0" else "")
            + f"{int(r['M']):,} x {int(r['N']):,}"
            for r in shapes
        ],
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    ax.set_ylabel("Effective memory bandwidth (GB/s)", fontsize=10, color=INK_SECONDARY)
    ax.set_title(
        f"{rows[0]['kernel']} — NVFP4",
        fontsize=13,
        color=INK,
        loc="left",
        pad=22,
    )
    ax.text(
        0,
        1.028,
        "DeepSeek-V3 671B, EP=64, bs=8 — 32,768 tokens/rank, 65,536 tokens/expert, "
        "4 local experts (NVIDIA GB200)",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_MUTED,
    )

    handles = [
        Patch(facecolor=color, edgecolor=color, label=name)
        for _, name, color in BACKENDS
    ]
    if not flat:
        handles += [
            Patch(facecolor=SURFACE, edgecolor=INK_MUTED, label="standard math"),
            Patch(
                facecolor=SURFACE,
                edgecolor=INK_MUTED,
                hatch="///",
                label="fast math (NVTE_USE_FAST_MATH=1)",
            ),
        ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, -0.13),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        handlelength=1.4,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    notes = []
    if flat:
        notes.append(
            "Fast math has no variant for this kernel on any backend — measured, "
            "it moves every backend by <3%. Standard math shown."
        )
    if estimated:
        notes.append(
            "TransformerEngine has no grouped 2D weight kernel; its bar is E x the "
            "single-expert quantize_transpose time (ESTIMATE)."
        )
    if key.endswith("weight_2d"):
        notes.append(
            "TE's own weight-amax pass is excluded: the torchao 2D kernels consume a "
            "precomputed amax."
        )
    if notes:
        # Anchored in axes coords so the block tracks the legend; bbox_inches="tight"
        # crops to it instead of tight_layout reserving a fixed slab of dead space.
        ax.text(
            0,
            -0.235,
            "\n".join(f"* {note}" for note in notes),
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
    out = out_dir / f"nvfp4_671b_64n_{key}.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(series)} series, flat={flat})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parent / "nvfp4_671b_64n_bandwidth.csv"),
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.csv).resolve().parent
    charts = load(args.csv)
    for key, rows in charts.items():
        plot_chart(key, rows, out_dir)


if __name__ == "__main__":
    main()
