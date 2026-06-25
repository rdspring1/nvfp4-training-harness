#!/usr/bin/env python3
"""
NVFP4 FSDP2 + TP training launcher.

Runs distributed TorchAO NVFP4 Triton experiments sequentially because each
experiment consumes its full TP x FSDP2 world size.

Usage:
    # 10-step eager WikiText smoke over 4xTP, 4xFSDP2, then 2xTP/2xFSDP2.
    python run_multi.py --smoke

    # 10-step torch.compile reduce-overhead smoke.
    python run_multi.py --smoke --compile

    # Run one shape explicitly.
    python run_multi.py --smoke --only tp2_fsdp2
"""

import argparse
import datetime
import os
import re
import subprocess
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
WALL_HOURS = 8
SMOKE_WALL_HOURS = 10 / 60
STEPS_CEILING = 500_000
SEQ_LEN = 2048
LR = 3e-4
RESULTS_DIR = ROOT_DIR / "llama3_results"
SMOKE_STEPS = 10

# Conservative first-jump batch sizes from reduce-overhead 10-step memory smoke.
EXPERIMENTS = [
    {
        "name": "tp4",
        "tag": "ao_multi",
        "tp": 4,
        "fsdp": 1,
        "batch_size": 32,
    },
    {
        "name": "fsdp4",
        "tag": "ao_multi",
        "tp": 1,
        "fsdp": 4,
        "batch_size": 4,
    },
    {
        "name": "tp2_fsdp2",
        "tag": "ao_multi",
        "tp": 2,
        "fsdp": 2,
        "batch_size": 8,
    },
]

_STEP_RE = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+\S+\s+([\d.]+)\s*$")


def visible_gpu_count() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        devices = [d for d in cuda_visible_devices.split(",") if d.strip()]
        return len(devices)
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def stream_to_file(proc, log_path: Path, prefix: str):
    with open(log_path, "w") as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()
            stripped = line.rstrip()
            if stripped:
                print(f"[{prefix}] {stripped}", flush=True)


def parse_log(log_path: Path):
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


def experiment_label(exp: dict, compile_mode: str | None) -> str:
    suffix = f"_{compile_mode.replace('-', '_')}" if compile_mode else ""
    return f"{exp['name']}{suffix}"


def build_cmd(exp: dict, steps: int, compile_mode: str | None, data: str):
    world_size = exp["tp"] * exp["fsdp"]
    batch_size = exp["batch_size"]
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(world_size),
        str(SCRIPT_DIR / "ao_llama3_fsdp2_tp_train.py"),
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(SEQ_LEN),
        "--lr",
        str(LR),
        "--tp-size",
        str(exp["tp"]),
        "--fsdp-size",
        str(exp["fsdp"]),
        "--quantize",
        "nvfp4",
        "--kernel",
        "triton",
    ]
    if data != "synthetic":
        cmd += ["--data", data]
    if compile_mode:
        cmd += ["--compile", compile_mode]
    return cmd


def terminate_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def launch_experiments(
    wall_hours: float,
    steps: int,
    only: str | None = None,
    compile_mode: str | None = None,
    data: str = "wikitext",
    allow_insufficient_gpus: bool = False,
):
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wall_seconds = wall_hours * 3600
    available_gpus = visible_gpu_count()

    exps = [exp for exp in EXPERIMENTS if only is None or exp["name"] == only]
    if not exps:
        raise SystemExit(
            f"No experiment named {only!r}. Choices: {[e['name'] for e in EXPERIMENTS]}"
        )

    print()
    print("=" * 72)
    print(f"NVFP4 FSDP2 + TP - {wall_hours:.4g}h wall clock per experiment")
    batch_sizes = ", ".join(f"{exp['name']}={exp['batch_size']}" for exp in exps)
    print(f"Batch sizes per DP replica: {batch_sizes}")
    print(f"Seq length: {SEQ_LEN}")
    print(f"Steps ceiling: {steps:,}")
    print(f"Compile: {compile_mode or 'eager'}")
    print(f"Data: {data}")
    print(f"Visible GPUs: {available_gpus}")
    print("=" * 72)
    print()

    results = []
    for exp in exps:
        world_size = exp["tp"] * exp["fsdp"]
        label = experiment_label(exp, compile_mode)
        log_path = RESULTS_DIR / f"{ts}_{exp['tag']}_{label}.txt"
        if available_gpus < world_size and not allow_insufficient_gpus:
            print(
                f"  [{label}] skipped: needs {world_size} visible GPUs, "
                f"found {available_gpus}"
            )
            results.append((exp, label, "SKIPPED", None, log_path, None))
            continue

        cmd = build_cmd(exp, steps, compile_mode, data)
        env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
        print(
            f"  [{label}] world_size={world_size} batch={exp['batch_size']} "
            f"-> {log_path}"
        )
        print(f"  [{label}] {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(
            target=stream_to_file, args=(proc, log_path, label), daemon=True
        )
        thread.start()

        start = time.monotonic()
        deadline = start + wall_seconds
        status = "OK"
        try:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(15)
            else:
                status = "TIMEOUT"
                print(f"  [{label}] wall clock reached - terminating")
                terminate_process(proc)
        except KeyboardInterrupt:
            status = "INTERRUPTED"
            print(f"  [{label}] KeyboardInterrupt - terminating")
            terminate_process(proc)
            raise
        finally:
            thread.join(timeout=10)

        if proc.returncode not in (0, None) and status == "OK":
            status = f"FAILED({proc.returncode})"
        results.append((exp, label, status, proc.returncode, log_path, start))
        print()

    print_summary(results)

    failures = [result for result in results if result[2].startswith("FAILED")]
    if failures:
        raise SystemExit(1)


def print_summary(results):
    header = (
        f"{'Experiment':<30} {'Status':>12} {'Steps':>8} {'Final Loss':>12} "
        f"{'Tok/s':>8} {'Log'}"
    )
    sep = "-" * (len(header) + 10)

    print("=" * (len(header) + 10))
    print(" SUMMARY")
    print(sep)
    print(f" {header}")
    print(sep)

    for exp, label, status, _returncode, log_path, _start in results:
        parsed = parse_log(log_path)
        if parsed is None:
            print(f" {label:<30} {status:>12} {'NO DATA':>8}  {log_path.name}")
            continue

        last_step, loss, tok_s, _mem = parsed
        completed_steps = last_step + 1
        warn = " <smoke floor" if completed_steps < SMOKE_STEPS else ""
        print(
            f" {label:<30} {status:>12} {completed_steps:>8,} "
            f"{loss:>12.4f} {tok_s:>8.0f}  {log_path.name}{warn}"
        )

    print("=" * (len(header) + 10))
    print()


def main():
    parser = argparse.ArgumentParser(description="NVFP4 FSDP2 + TP launcher")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 10-minute wall clock, 10 steps per experiment",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="NAME",
        help=f"Run only this experiment. Choices: {[e['name'] for e in EXPERIMENTS]}",
    )
    parser.add_argument(
        "--compile",
        type=str,
        nargs="?",
        const="reduce-overhead",
        default=None,
        choices=[
            "reduce-overhead",
            "default",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="torch.compile mode for ao_llama3_fsdp2_tp_train.py",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="wikitext",
        choices=["synthetic", "wikitext"],
        help="Dataset for smoke/full runs. Use synthetic for launch-only checks.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the number of training steps for each selected experiment.",
    )
    parser.add_argument(
        "--allow-insufficient-gpus",
        action="store_true",
        help="Launch torchrun even when visible GPU count is below world size.",
    )
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps must be positive")

    if args.smoke:
        print("SMOKE TEST MODE: 10min wall clock per experiment, 10 steps")
        launch_experiments(
            wall_hours=SMOKE_WALL_HOURS,
            steps=args.steps or 10,
            only=args.only,
            compile_mode=args.compile,
            data=args.data,
            allow_insufficient_gpus=args.allow_insufficient_gpus,
        )
    else:
        launch_experiments(
            wall_hours=WALL_HOURS,
            steps=args.steps or STEPS_CEILING,
            only=args.only,
            compile_mode=args.compile,
            data=args.data,
            allow_insufficient_gpus=args.allow_insufficient_gpus,
        )


if __name__ == "__main__":
    main()
