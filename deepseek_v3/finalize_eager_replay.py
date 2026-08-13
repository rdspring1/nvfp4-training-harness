#!/usr/bin/env python3
"""Preserve and summarize a completed DeepSeek V3 eager NVFP4 replay."""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = ROOT / "deepseek_v3" / "run_logs" / "eager_runner"
RUN_DIR = LOG_ROOT / "deepseek_v3_16b_fsdp4_ep4_nvfp4_ffn_tail15_compile_200m_gbs128"
REFERENCE = RUN_DIR / "20260811_162959_titan_deepseek_v3_16b_eager_nvfp4_compile.txt"
METRIC = re.compile(r"step:\s*(\d+).*?loss:\s*([\d.]+).*?memory:\s*([\d.]+)GiB.*?tps:\s*([\d,]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def metrics(path: Path):
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        match = METRIC.search(ANSI.sub("", line))
        if match:
            step, loss, memory, tps = match.groups()
            values[int(step)] = (float(loss), float(memory), int(tps.replace(",", "")))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_log", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--torchao-sha", required=True)
    parser.add_argument("--torchtitan-sha", required=True)
    parser.add_argument("--mslk-sha", required=True)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    parser.add_argument("--label", default="TorchAO NVFP4 Triton replay")
    args = parser.parse_args()
    parsed = metrics(args.raw_log)
    if 380 not in parsed or "Training completed" not in args.raw_log.read_text(errors="replace"):
        raise SystemExit("replay did not complete through logged step 380")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log_copy = args.output_dir / "replay.txt"
    shutil.copy2(args.raw_log, log_copy)
    ref = metrics(REFERENCE)
    final_loss, _, final_tps = parsed[380]
    peak_memory = max(memory for _, memory, _ in parsed.values())
    steady = [tps for step, (_, _, tps) in parsed.items() if 50 <= step <= 380]
    replay_mean = sum(steady) / len(steady)
    ref_steady = [tps for step, (_, _, tps) in ref.items() if 50 <= step <= 380]
    ref_mean = sum(ref_steady) / len(ref_steady)
    common = sorted(set(parsed) & set(ref))
    plateau = [parsed[s][0] - ref[s][0] for s in common if 200 <= s <= 380]
    overlay = args.output_dir / "training_loss_four_series_overlay.png"
    subprocess.run([sys.executable, str(ROOT / "deepseek_v3" / "plot_eager_runner_loss.py"), "--replay-log", str(log_copy), "--output", str(overlay)], check=True)
    minutes, seconds = divmod(args.elapsed_seconds, 60)
    report = f"""# DeepSeek V3 16B {args.label}

## Command

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python deepseek_v3/run_titan.py --precision nvfp4 --flavor 16b --compile --expert-parallel-degree 4 --batch-size 8 --global-batch-size 128 --seq-len 4096 --steps 382 --dataset c4 --log-freq 10`

## Inputs

- Raw replay log: `{log_copy.name}`
- TorchAO: `{args.torchao_sha}`; TorchTitan: `{args.torchtitan_sha}`; MSLK: `{args.mslk_sha}`
- Wallclock: {minutes}m {seconds:02d}s

## Metrics

| Metric | Replay | August TorchAO baseline |
| --- | ---: | ---: |
| loss at step 380 | {final_loss:.5f} | {ref[380][0]:.5f} |
| mean TPS/GPU, steps 50-380 | {replay_mean:,.0f} | {ref_mean:,.0f} |
| peak reserved GiB | {peak_memory:.2f} | 147.59 |

Step-380 loss delta (replay - baseline): {final_loss - ref[380][0]:+.5f}. Plateau mean delta over common logged steps 200-380: {sum(plateau) / len(plateau):+.5f}.

## Interpretation

The overlay (`{overlay.name}`) includes historical TorchAO, TE, BF16, and this replay. The baseline used TorchAO `6b062ac5`; this replay used `{args.torchao_sha}`, so performance and loss comparisons are observational rather than an exact source-revision reproduction.
"""
    (args.output_dir / "report.md").write_text(report)


if __name__ == "__main__":
    main()
