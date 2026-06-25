#!/usr/bin/env python3
"""
NVFP4 Triton kernel training run.

Runs the Triton-backend experiment on GPU 1 with an 8-hour wall clock.
GPU 0 is reserved for user use.

Usage:
    # Smoke test (5 minutes, 10 steps):
    python run_triton.py --smoke

    # Full 8-hour run in background:
    nohup python run_triton.py > run_triton.log 2>&1 &
    echo $!                                  # note the PID
    tail -f run_triton.log                   # follow launcher log
    tail -f llama3_results/*nvfp4_triton*.txt
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit here before launching
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
WALL_HOURS = 8
STEPS_CEILING = 500_000  # wall clock kills first; this is just a ceiling
BATCH_SIZE = 4
SEQ_LEN = 2048  # 8192 tok/step with batch=4
LR = 3e-4
RESULTS_DIR = ROOT_DIR / "llama3_results"
SMOKE_STEPS = 3_000  # warn if any run ends before this

EXPERIMENTS = [
    {
        "name": "nvfp4_triton",
        "tag": "ao",
        "gpu": 1,
        "extra": ["--quantize", "nvfp4", "--kernel", "triton"],
    },
]

# ---------------------------------------------------------------------------
# Step-line regex: "  1234   6.8234     9842    1.0M    24.31"
# ---------------------------------------------------------------------------
_STEP_RE = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+\S+\s+([\d.]+)\s*$")


def stream_to_file(proc, log_path: Path, prefix: str):
    """Read proc.stdout line by line, write to log_path and print with prefix."""
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

    running = []  # list of (name, gpu, proc, log_path, start_time)

    print(f"\n{'='*60}")
    print(f"NVFP4 Triton — {wall_hours:.4g}h wall clock")
    print(f"Batch {BATCH_SIZE} × seq {SEQ_LEN} = {BATCH_SIZE * SEQ_LEN} tok/step")
    print(f"Steps ceiling: {steps:,}")
    print(f"{'='*60}\n")

    for exp in exps:
        compile_suffix = f"_{compile_mode.replace('-', '_')}" if compile_mode else ""
        log_path = RESULTS_DIR / f"{ts}_{exp['tag']}_{exp['name']}{compile_suffix}.txt"
        script = SCRIPT_DIR / "ao_llama3_train.py"
        extra = exp["extra"]
        if compile_mode:
            extra = extra + ["--compile", compile_mode]
        cmd = [sys.executable, "-u", str(script)] + base_args + extra
        gpu = gpu_override if gpu_override is not None else exp["gpu"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        t = threading.Thread(
            target=stream_to_file,
            args=(proc, log_path, exp["name"]),
            daemon=True,
        )
        t.start()

        start = time.monotonic()
        running.append((exp["name"], gpu, proc, log_path, start, t))
        print(f"  [{exp['name']}]  GPU {gpu}  PID {proc.pid}" f"  → {log_path}")

    print()

    # Wait up to wall_seconds, polling every 15s
    deadline = time.monotonic() + wall_seconds
    try:
        while time.monotonic() < deadline:
            if all(p.poll() is not None for _, _, p, _, _, _ in running):
                print("\nAll experiments finished before wall clock.")
                break
            time.sleep(15)
        else:
            print(f"\n{wall_hours:.4g}h wall clock reached — stopping experiments.")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — stopping experiments.")

    # Graceful termination
    for name, gpu, proc, log_path, start, t in running:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print(f"  [{name}] terminated (GPU {gpu})")

    # Wait for streaming threads to flush
    for _, _, _, _, _, t in running:
        t.join(timeout=5)

    print()
    print_summary(running)


def parse_log(log_path: Path):
    """Return (last_step, last_loss, last_tok_per_sec, last_mem_gb) or None."""
    last = None
    try:
        with open(log_path) as f:
            for line in f:
                m = _STEP_RE.match(line)
                if m:
                    last = (
                        int(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                        float(m.group(4)),
                    )
    except FileNotFoundError:
        pass
    return last


def print_summary(running):
    tok_per_step = BATCH_SIZE * SEQ_LEN

    header = f"{'Experiment':<16} {'Steps':>8} {'Final Loss':>12} {'Tok/s':>8} {'Total Tokens':>14} {'Log'}"
    sep = "─" * (len(header) + 10)

    print("═" * (len(header) + 10))
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
        total_tokens = step * tok_per_step
        tok_str = f"{total_tokens / 1e6:.1f}M"

        warn = " ⚠ <smoke floor" if step < SMOKE_STEPS else ""
        print(
            f" {name:<16} {step:>8,} {loss:>12.4f} {tok_s:>8.0f}"
            f" {tok_str:>14}  {log_path.name}{warn}"
        )

    print("═" * (len(header) + 10))
    print()


def main():
    parser = argparse.ArgumentParser(description="NVFP4 Triton training launcher")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 7-second wall clock, 10 steps (verify launch + logging)",
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
        help="torch.compile mode for ao_llama3_train.py experiments (default when flag given: reduce-overhead)",
    )
    args = parser.parse_args()

    kwargs = dict(only=args.only, gpu_override=args.gpu, compile_mode=args.compile)
    if args.smoke:
        print("SMOKE TEST MODE: 5min wall clock, 10 steps")
        launch_experiments(wall_hours=5 / 60, steps=10, **kwargs)
    else:
        launch_experiments(wall_hours=WALL_HOURS, steps=STEPS_CEILING, **kwargs)


if __name__ == "__main__":
    main()
