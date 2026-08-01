#!/usr/bin/env python3
"""Plot Qwen3-8B C4 pretraining loss comparisons."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_ROOT = Path(__file__).resolve().parent / "run_logs_pretrain"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(r"step:\s*(\d+).*?loss:\s*([\d.]+)")
SERIES = [
    ("nvfp4_mixed", "NVFP4 (bf16 tail)", "C0"),
    ("mxfp8", "MXFP8", "C2"),
    ("bf16", "BF16", "C1"),
]


def _latest_log(log_dir: Path, precision: str) -> Path:
    logs = sorted(log_dir.glob(f"*_fsdp4_{precision}_eager_compile_200m_*.txt"))
    if not logs:
        raise SystemExit(f"no {precision} run log found in {log_dir}")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane",
        choices=("random_init", "continued_pretraining"),
        help="Plot one initialization lane",
    )
    args = parser.parse_args()
    lanes = (args.lane,) if args.lane else ("random_init", "continued_pretraining")
    for lane in lanes:
        log_dir = LOG_ROOT / lane / "eager_trainer"
        plt.figure(figsize=(10, 6))
        for precision, label, color in SERIES:
            steps, losses = _series(_latest_log(log_dir, precision))
            plt.plot(steps, losses, label=label, color=color)
        plt.title(f"Qwen3-8B C4 {lane.replace('_', ' ').title()} — FSDP4, GBS 64")
        plt.xlabel("Optimizer step")
        plt.ylabel("Training loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = log_dir / "training_loss_nvfp4_mixed_vs_mxfp8_vs_bf16_eager_compile_200m_tokens.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
