#!/usr/bin/env python3
"""Plot the DeepSeek V3 16B NVFP4 MoE loss curves: TransformerEngine vs TorchAO.

Both primary series come from the same torchtitan converter recipe (15% bf16
layer tail, pad_multiple=128) and differ only in the grouped-GEMM backend, so
the gap between them isolates the quantize+GEMM primitive. The archived runs are
plotted dashed as references: they predate the converter migration and the bf16
tail, so they are not controls.

    python deepseek_v3/plot_te_vs_ao_loss.py
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).resolve().parent / "run_logs" / "eager_runner"
OUT_PNG = LOG_DIR / "training_loss_te_vs_ao_nvfp4_200m_tokens.png"

# Same parse as run_titan.py: strip ANSI, then read `step: N ... loss: X`.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(r"step:\s*(\d+).*?loss:\s*([\d.]+)")

# The two arms under test are the only categorical identities: one hue each, from
# a pair validated for CVD separation (protan dE 21.9) and >=3:1 surface contrast.
# The archived runs are context, not identity -- muted ink plus a distinct dash,
# so they never compete with the two series the figure is about.
TE_COLOR = "#0072b2"
AO_COLOR = "#d55e00"
_REF_INK = "#6b6b6b"

# (run directory under run_logs/eager_runner, legend label, color, linestyle)
SERIES = [
    (
        "deepseek_v3_16b_fsdp4_ep4_te_nvfp4_tail15_compile_200m_gbs128",
        "NVFP4 TransformerEngine (15% bf16 tail)",
        TE_COLOR,
        "-",
    ),
    (
        "deepseek_v3_16b_fsdp4_ep4_nvfp4_tail15_compile_200m_gbs128",
        "NVFP4 TorchAO (15% bf16 tail)",
        AO_COLOR,
        "-",
    ),
    (
        "deepseek_v3_16b_fsdp4_ep4_nvfp4_compile_200m_gbs128",
        "NVFP4 TorchAO, archived (no bf16 tail)",
        _REF_INK,
        (0, (5, 2)),
    ),
    (
        "deepseek_v3_16b_fsdp4_ep4_bf16_compile_200m_gbs128",
        "BF16, archived",
        _REF_INK,
        (0, (1, 2)),
    ),
]
TE_LABEL = "NVFP4 TransformerEngine (15% bf16 tail)"
AO_LABEL = "NVFP4 TorchAO (15% bf16 tail)"


def _series(run_dir: Path) -> tuple[list[int], list[float]]:
    logs = sorted(run_dir.glob("*.txt"))
    if len(logs) != 1:
        raise SystemExit(f"expected exactly one .txt log in {run_dir}, found {len(logs)}")
    steps, losses = [], []
    for line in logs[0].read_text(errors="replace").splitlines():
        match = _STEP_RE.search(_ANSI_RE.sub("", line))
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    if not steps:
        raise SystemExit(f"no step/loss pairs parsed from {logs[0]}")
    return steps, losses


def main() -> None:
    # Two panels, not one axis with two scales: the loss curves span 12->4.6, so a
    # 0.076 gap between the arms is invisible on that axis. The top panel carries
    # the trajectory, the bottom panel carries the finding.
    fig, (ax, ax_d) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]}
    )
    curves = {}
    for dirname, label, color, style in SERIES:
        run_dir = LOG_DIR / dirname
        if not run_dir.is_dir():
            print(f"skipping missing run dir: {run_dir}", file=sys.stderr)
            continue
        steps, losses = _series(run_dir)
        curves[label] = dict(zip(steps, losses))
        ax.plot(steps, losses, label=label, color=color, linestyle=style, linewidth=2)

    ax.set_ylabel("Training loss")
    ax.set_title(
        "DeepSeek V3 16B, FSDP4 + EP4, 200M tokens (GBS 128, seq 4096, lbs 8 x ga 4)\n"
        "NVFP4 MoE grouped GEMMs: TransformerEngine vs TorchAO"
    )
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
        ax_d.spines[spine].set_visible(False)

    te, ao = curves.get(TE_LABEL), curves.get(AO_LABEL)
    if not (te and ao):
        raise SystemExit("both TE and AO runs are required for the delta panel")

    shared = sorted(set(te) & set(ao))
    deltas = [te[s] - ao[s] for s in shared]
    ax_d.axhline(0, color="#9a9a9a", linewidth=1)
    ax_d.plot(shared, deltas, color=TE_COLOR, linewidth=2)
    ax_d.fill_between(shared, deltas, 0, color=TE_COLOR, alpha=0.15)
    # Direct-label the endpoint only -- the one number the panel exists to show.
    ax_d.annotate(
        f"{deltas[-1]:+.3f} at step {shared[-1]}",
        xy=(shared[-1], deltas[-1]),
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        color="#333333",
    )
    ax_d.set_xlabel("Step")
    ax_d.set_ylabel("TE - TorchAO")
    ax_d.grid(True, alpha=0.25, linewidth=0.6)
    ax_d.text(
        0.01,
        0.06,
        "below zero = TransformerEngine lower loss",
        transform=ax_d.transAxes,
        fontsize=9,
        color="#6b6b6b",
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")

    # The decision criterion: mean |TE - AO| over the second half of the run.
    tail = [s for s in shared if s >= max(shared) // 2]
    d_tail = [te[s] - ao[s] for s in tail]
    mean_abs = sum(abs(x) for x in d_tail) / len(d_tail)
    signed = sum(d_tail) / len(d_tail)
    print(
        f"steps {tail[0]}-{tail[-1]}: mean |TE-AO| = {mean_abs:.5f}, "
        f"signed mean = {signed:+.5f}, final TE-AO = {d_tail[-1]:+.5f}, "
        f"negative at {sum(x < 0 for x in deltas)}/{len(deltas)} logged steps"
    )


if __name__ == "__main__":
    main()
