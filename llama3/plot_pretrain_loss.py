#!/usr/bin/env python3
"""Plot the completed Llama 3.1 8B continued-pretraining comparison."""

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_DIR = (
    Path(__file__).resolve().parent
    / "run_logs_pretrain"
    / "continued_pretraining"
    / "eager_trainer"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(r"step:\s*(\d+).*?loss:\s*(\S+)")
SERIES = [
    ("nvfp4_mixed", "NVFP4 (BF16 layers 27-31 + lm_head)", "C0"),
    ("mxfp8", "MXFP8", "C2"),
    ("bf16", "BF16", "C1"),
]


def _latest_completed(precision: str) -> Path:
    pattern = f"*_fsdp4_{precision}_eager_compile_200m_gbs128_lbs32_ga1*.txt"
    for path in reversed(sorted(LOG_DIR.glob(pattern))):
        if "Training completed" in path.read_text(errors="replace"):
            return path
    raise SystemExit(f"no completed {precision} run log found in {LOG_DIR}")


def _series(path: Path) -> tuple[list[int], list[float]]:
    steps, losses = [], []
    for line in path.read_text(errors="replace").splitlines():
        if match := _STEP_RE.search(_ANSI_RE.sub("", line)):
            loss = float(match.group(2))
            if not math.isfinite(loss):
                raise SystemExit(f"non-finite loss in {path}")
            steps.append(int(match.group(1)))
            losses.append(loss)
    if not steps:
        raise SystemExit(f"no step/loss metrics found in {path}")
    return steps, losses


def main() -> None:
    plt.figure(figsize=(10, 6))
    for precision, label, color in SERIES:
        steps, losses = _series(_latest_completed(precision))
        plt.plot(steps, losses, label=label, color=color)
    plt.title("Llama 3.1 8B C4 Continued Pretraining - FSDP4, GBS 128")
    plt.xlabel("Optimizer step")
    plt.ylabel("Training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = (
        LOG_DIR
        / "training_loss_nvfp4_mixed_vs_mxfp8_vs_bf16_eager_compile_200m_tokens.png"
    )
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
