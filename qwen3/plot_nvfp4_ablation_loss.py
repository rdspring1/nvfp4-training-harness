#!/usr/bin/env python3
"""Plot Qwen3-8B BF16 and NVFP4 SFT ablation loss curves."""

import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).resolve().parent / "run_logs_sft" / "eager_trainer"
OUT_PNG = LOG_DIR / "training_loss_bf16_vs_nvfp4_ablations_eager_compile_sft.png"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(r"step:\s*(\d+).*?loss:\s*([\d.]+)")
SERIES = [
    ("20260801_titan_fsdp4_bf16_eager_compile_sft_gbs4_lbs1_ga1.txt", "BF16", "C1"),
    (
        "20260801_titan_fsdp4_nvfp4_mixed_eager_compile_sft_gbs4_lbs1_ga1.txt",
        "NVFP4, 15% bf16 tail, LR 2e-5",
        "C0",
    ),
    (
        "20260801_titan_fsdp4_nvfp4_mixed_eager_compile_sft_gbs4_lbs1_ga1_lr1e-5.txt",
        "NVFP4, 15% bf16 tail, LR 1e-5",
        "C2",
    ),
    (
        "20260801_titan_fsdp4_nvfp4_mixed_30_eager_compile_sft_gbs4_lbs1_ga1_tail30.txt",
        "NVFP4, 30% bf16 tail, LR 2e-5",
        "C3",
    ),
]


def _series(path: Path) -> tuple[list[int], list[float]]:
    steps, losses = [], []
    for line in path.read_text(errors="replace").splitlines():
        match = _STEP_RE.search(_ANSI_RE.sub("", line))
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    if not steps:
        raise SystemExit(f"no step/loss metrics found in {path}")
    return steps, losses


def main() -> None:
    plt.figure(figsize=(10, 6))
    for filename, label, color in SERIES:
        steps, losses = _series(LOG_DIR / filename)
        plt.plot(steps, losses, label=label, color=color)
    plt.title("Qwen3-8B GSM8K SFT — BF16 vs. NVFP4 Ablations")
    plt.xlabel("Optimizer step")
    plt.ylabel("Training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
