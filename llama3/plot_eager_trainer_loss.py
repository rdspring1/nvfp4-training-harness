#!/usr/bin/env python3
"""Regenerate the eager-Trainer training-loss comparison plot.

Parses step/loss pairs from the TorchTitan run logs in run_logs/eager_trainer/
and plots NVFP4 (bf16-tail mixed recipe) vs MXFP8 vs BF16 for the 200M-token,
GBS-128 fsdp4 runs.

    python plot_eager_trainer_loss.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).resolve().parent / "run_logs" / "eager_trainer"
OUT_PNG = LOG_DIR / "training_loss_nvfp4_vs_mxfp8_vs_bf16_eager_compile_200m_tokens.png"

# Same parse as run_titan.py: strip ANSI, then read `step: N ... loss: X`.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(r"step:\s*(\d+).*?loss:\s*([\d.]+)")

# (log filename, legend label, color). Colors match the prior PNG.
SERIES = [
    (
        "20260727_titan_fsdp4_nvfp4_mixed_eager_compile_200m_gbs128_lbs32_ga1.txt",
        "NVFP4 (bf16 tail) eager + compile",
        "C0",
    ),
    (
        "20260713_titan_fsdp4_mxfp8_eager_compile_200m_gbs128_lbs32_ga1.txt",
        "MXFP8 eager + compile",
        "C2",
    ),
    (
        "20260712_titan_fsdp4_bf16_eager_compile_200m_gbs128_lbs32_ga1.txt",
        "BF16 eager + compile",
        "C1",
    ),
]


def parse_log(path: Path) -> tuple[list[int], list[float]]:
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            m = _STEP_RE.search(_ANSI_RE.sub("", line))
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses


def main() -> None:
    plt.figure(figsize=(12, 7))
    for filename, label, color in SERIES:
        steps, losses = parse_log(LOG_DIR / filename)
        plt.plot(steps, losses, label=label, color=color)

    plt.title("Llama 3 8B — 200M Tokens (GBS 128)")
    plt.xlabel("Optimizer step")
    plt.ylabel("Training loss")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
