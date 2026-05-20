#!/usr/bin/env python3
"""
TransformerEngine low-precision training launcher.

Runs TE-native NVFP4 and MXFP8 experiments on separate GPUs with an 8-hour
wall clock. GPU 0 is reserved for user use.

Usage:
    # Smoke test (5 minutes, 10 steps):
    python run_te.py --smoke

    # Full 8-hour run in background:
    nohup python run_te.py > run_te.log 2>&1 &
    echo $!
    tail -f run_te.log
    tail -f llama3_results/*te_native*.txt
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WALL_HOURS = 8
STEPS_CEILING = 500_000
BATCH_SIZE = 4
SEQ_LEN = 2048
LR = 3e-4
RESULTS_DIR = Path("llama3_results")
SMOKE_STEPS = 3_000

EXPERIMENTS = [
    {
        "name": "nvfp4",
        "gpu": 2,
        "extra": ["--precision", "nvfp4"],
    },
    {
        "name": "mxfp8",
        "gpu": 3,
        "extra": ["--precision", "mxfp8"],
    },
]

# Step-line regex: "  1234   6.8234     9842    1.0M    24.31"
_STEP_RE = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+\S+\s+([\d.]+)\s*$")


def stream_to_file(proc, log_path: Path, prefix: str):
    """Read proc.stdout line by line, write to log_path, and print with prefix."""
    with open(log_path, "w") as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()
            stripped = line.rstrip()
            if stripped:
                print(f"[{prefix}] {stripped}", flush=True)


def launch_experiments(
    wall_hours: float,
    steps: int,
    only: str = None,
    gpu_override: int = None,
    cuda_graphs: bool = False,
    compile_mode: str = None,
):
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wall_seconds = wall_hours * 3600

    base_args = [
        "--data",
        "wikitext",
        "--steps",
        str(steps),
        "--batch-size",
        str(BATCH_SIZE),
        "--seq-len",
        str(SEQ_LEN),
        "--lr",
        str(LR),
    ]

    exps = [e for e in EXPERIMENTS if only is None or e["name"] == only]
    if not exps:
        raise SystemExit(
            f"No experiment named {only!r}. Choices: {[e['name'] for e in EXPERIMENTS]}"
        )

    graph_enabled = cuda_graphs or compile_mode in {"reduce-overhead", "max-autotune"}
    suffix = ""
    if compile_mode:
        suffix = f"_{compile_mode.replace('-', '_')}"
    elif graph_enabled:
        suffix = "_cuda_graphs"

    running = []

    print()
    print("=" * 60)
    print(f"TE native NVFP4 + MXFP8 - {wall_hours:.4g}h wall clock")
    print(f"Batch {BATCH_SIZE} x seq {SEQ_LEN} = {BATCH_SIZE * SEQ_LEN} tok/step")
    print(f"Steps ceiling: {steps:,}")
    print(f"CUDA Graphs: {'enabled' if graph_enabled else 'disabled'}")
    print("=" * 60)
    print()

    for exp in exps:
        log_path = RESULTS_DIR / f"{ts}_te_native_{exp['name']}{suffix}.txt"
        extra = list(exp["extra"])
        if graph_enabled:
            extra.append("--cuda-graphs")
        cmd = [sys.executable, "-u", "te_llama3_train.py"] + base_args + extra
        gpu = gpu_override if gpu_override is not None else exp["gpu"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(
            target=stream_to_file,
            args=(proc, log_path, exp["name"]),
            daemon=True,
        )
        thread.start()

        start = time.monotonic()
        running.append((exp["name"], gpu, proc, log_path, start, thread))
        print(f"  [{exp['name']}] GPU {gpu} PID {proc.pid} -> {log_path}")

    print()

    deadline = time.monotonic() + wall_seconds
    try:
        while time.monotonic() < deadline:
            if all(proc.poll() is not None for _, _, proc, _, _, _ in running):
                print("\nAll experiments finished before wall clock.")
                break
            time.sleep(15)
        else:
            print(f"\n{wall_hours:.4g}h wall clock reached - stopping experiments.")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt - stopping experiments.")

    for name, gpu, proc, log_path, start, thread in running:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print(f"  [{name}] terminated (GPU {gpu})")

    for _, _, _, _, _, thread in running:
        thread.join(timeout=5)

    print()
    print_summary(running)


def parse_log(log_path: Path):
    """Return (last_step, last_loss, last_tok_per_sec, last_mem_gb) or None."""
    last = None
    try:
        with open(log_path) as f:
            for line in f:
                match = _STEP_RE.match(line)
                if match:
                    last = (
                        int(match.group(1)),
                        float(match.group(2)),
                        float(match.group(3)),
                        float(match.group(4)),
                    )
    except FileNotFoundError:
        pass
    return last


def print_summary(running):
    tok_per_step = BATCH_SIZE * SEQ_LEN

    header = (
        f"{'Experiment':<16} {'Steps':>8} {'Final Loss':>12} "
        f"{'Tok/s':>8} {'Total Tokens':>14} {'Log'}"
    )
    sep = "-" * (len(header) + 10)

    print("=" * (len(header) + 10))
    print(" SUMMARY")
    print(sep)
    print(f" {header}")
    print(sep)

    for name, gpu, proc, log_path, start, _ in running:
        result = parse_log(log_path)

        if result is None:
            print(f" {name:<16} {'NO DATA':>8}")
            continue

        step, loss, tok_s, mem = result
        total_tokens = (step + 1) * tok_per_step
        tok_str = f"{total_tokens / 1e6:.1f}M"

        warn = " <smoke floor" if step < SMOKE_STEPS else ""
        print(
            f" {name:<16} {step:>8,} {loss:>12.4f} {tok_s:>8.0f}"
            f" {tok_str:>14}  {log_path.name}{warn}"
        )

    print("=" * (len(header) + 10))
    print()


def main():
    parser = argparse.ArgumentParser(description="TE native NVFP4/MXFP8 launcher")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 5-minute wall clock, 10 steps",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="NAME",
        help=f"Run only this experiment. Choices: {[e['name'] for e in EXPERIMENTS]}",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="N",
        help="Override GPU index for the selected experiment",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Enable CUDA graph capture in te_llama3_train.py",
    )
    parser.add_argument(
        "--compile",
        type=str,
        nargs="?",
        const="reduce-overhead",
        default=None,
        metavar="MODE",
        choices=[
            "reduce-overhead",
            "default",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help=(
            "Compatibility label from run_triton.py. For TE, reduce-overhead "
            "and max-autotune enable CUDA graphs; default and "
            "max-autotune-no-cudagraphs leave graphs disabled."
        ),
    )
    args = parser.parse_args()

    kwargs = dict(
        only=args.only,
        gpu_override=args.gpu,
        cuda_graphs=args.cuda_graphs,
        compile_mode=args.compile,
    )
    if args.smoke:
        print("SMOKE TEST MODE: 5min wall clock, 10 steps")
        launch_experiments(wall_hours=5 / 60, steps=10, **kwargs)
    else:
        launch_experiments(wall_hours=WALL_HOURS, steps=STEPS_CEILING, **kwargs)


if __name__ == "__main__":
    main()
