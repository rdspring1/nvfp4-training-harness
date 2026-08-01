#!/usr/bin/env python3
"""Plot the latest Qwen3-8B FSDP4 SFT eager-trainer precision comparison."""

import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).resolve().parent / "run_logs_sft" / "eager_trainer"
OUT_PNG = LOG_DIR / "training_loss_nvfp4_mixed_vs_mxfp8_vs_bf16_eager_compile_sft.png"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(r"step:\s*(\d+).*?loss:\s*([\d.]+)")
SERIES = [
    ("nvfp4_mixed", "NVFP4 (bf16 tail)", "C0"),
    ("mxfp8", "MXFP8", "C2"),
    ("bf16", "BF16", "C1"),
]


def _latest_log(precision: str) -> Path:
    logs = sorted(LOG_DIR.glob(f"*_fsdp4_{precision}_eager_compile_sft_*.txt"))
    if not logs:
        raise SystemExit(f"no {precision} run log found in {LOG_DIR}")
    return logs[-1]


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
    for precision, label, color in SERIES:
        steps, losses = _series(_latest_log(precision))
        plt.plot(steps, losses, label=label, color=color)
    plt.title("Qwen3-8B GSM8K SFT — FSDP4, GBS 4")
    plt.xlabel("Optimizer step")
    plt.ylabel("Training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
