#!/usr/bin/env python3
"""Regenerate the DeepSeek V3 16B eager-runner training-loss comparison plot.

Parses step/loss pairs from the TorchTitan run logs under run_logs/eager_runner/
and plots the two NVFP4 arms against the BF16 baseline for the 200M-token,
GBS-128 fsdp4 + ep4 runs.

    python deepseek_v3/plot_eager_runner_loss.py
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).resolve().parent / "run_logs" / "eager_runner"
OUT_PNG = LOG_DIR / "training_loss_nvfp4_vs_bf16_200m_tokens.png"

# Same parse as run_titan.py: strip ANSI, then read `step: N ... loss: X`.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(r"step:\s*(\d+).*?loss:\s*([\d.]+)")

# (run directory under run_logs/eager_runner, legend label, color).
SERIES = [
    (
        "deepseek_v3_16b_fsdp4_ep4_nvfp4_ffn_tail15_compile_200m_gbs128",
        "NVFP4 TorchAO (FFN + experts, 15% bf16 tail)",
        "C0",
    ),
    (
        "deepseek_v3_16b_fsdp4_ep4_te_nvfp4_tail15_compile_200m_gbs128",
        "NVFP4 TransformerEngine (experts, 15% bf16 tail)",
        "C2",
    ),
    (
        "deepseek_v3_16b_fsdp4_ep4_bf16_compile_200m_gbs128",
        "BF16",
        "C1",
    ),
]


def _series(log_path: Path) -> tuple[list[int], list[float]]:
    # Each step is logged once per rank with the same global loss, so keep one
    # point per step rather than drawing the repeats.
    by_step: dict[int, float] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = _STEP_RE.search(_ANSI_RE.sub("", line))
        if match:
            by_step[int(match.group(1))] = float(match.group(2))
    if not by_step:
        raise SystemExit(f"no step/loss metrics found in {log_path}")
    steps = sorted(by_step)
    return steps, [by_step[step] for step in steps]


def _run_log(run_dir: Path) -> Path:
    logs = sorted(run_dir.glob("*.txt"))
    if len(logs) != 1:
        raise SystemExit(f"expected exactly one .txt log in {run_dir}, found {len(logs)}")
    return logs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.replay_log is None:
        raise SystemExit("--output requires --replay-log")

    plt.figure(figsize=(12, 7))
    for dirname, label, color in SERIES:
        steps, losses = _series(_run_log(LOG_DIR / dirname))
        plt.plot(steps, losses, label=label, color=color)

    output = OUT_PNG
    if args.replay_log is not None:
        steps, losses = _series(args.replay_log)
        plt.plot(steps, losses, label="NVFP4 TorchAO replay", color="C3")
        output = args.output or args.replay_log.with_name("training_loss_replay_overlay.png")

    plt.title(
        "DeepSeek V3 16B — C4 — NVFP4 vs BF16 — 200M tokens (GBS 128, seq 4096)"
    )
    plt.xlabel("Optimizer step")
    plt.ylabel("Training loss")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
