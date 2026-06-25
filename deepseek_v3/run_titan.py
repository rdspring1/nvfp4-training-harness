#!/usr/bin/env python3
"""Single-GPU TorchTitan launcher for DeepSeek V3 debugmodel."""

import argparse
import datetime
import os
import re
import subprocess
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TORCHTITAN_DIR = ROOT_DIR / "third_party" / "torchtitan"
DEEPSEEK_MODEL_DIR = TORCHTITAN_DIR / "torchtitan" / "models" / "deepseek_v3"
RESULTS_DIR = ROOT_DIR / "deepseek_v3_results"

MODULE = "deepseek_v3"
CONFIG = "deepseek_v3_debugmodel"
LOCAL_BATCH_SIZE = 8
SEQ_LEN = 2048

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(
    r"step:\s*(\d+).*?loss:\s*([\d.]+).*?memory:\s*([\d.]+)GiB"
    r".*?tps:\s*([\d,]+).*?tflops:\s*([\d.,]+)"
)


def _check_torchtitan() -> None:
    if not TORCHTITAN_DIR.exists():
        raise SystemExit(
            "third_party/torchtitan is missing. Run: "
            "git submodule update --init third_party/torchtitan"
        )
    if not DEEPSEEK_MODEL_DIR.exists():
        raise SystemExit(
            "third_party/torchtitan does not include torchtitan/models/deepseek_v3. "
            "Update the submodule to a DeepSeek V3-capable commit."
        )


def _cmd(args: argparse.Namespace) -> list[str]:
    return [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        "-m",
        "torchtitan.train",
        "--module",
        MODULE,
        "--config",
        CONFIG,
        "--training.local_batch_size",
        str(args.batch_size),
        "--training.seq_len",
        str(args.seq_len),
        "--training.steps",
        str(args.steps),
        "--dataloader.dataset",
        "c4_test",
        "--metrics.log_freq",
        str(args.log_freq),
    ]


def _stream_to_file(proc: subprocess.Popen, log_path: Path) -> None:
    with open(log_path, "w") as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()
            stripped = line.rstrip()
            if stripped:
                print(f"[deepseek_v3] {stripped}", flush=True)


def _parse_log(log_path: Path):
    last = None
    try:
        with open(log_path) as f:
            for line in f:
                match = _STEP_RE.search(_ANSI_RE.sub("", line))
                if match:
                    last = (
                        int(match.group(1)),
                        float(match.group(2)),
                        float(match.group(3)),
                        int(match.group(4).replace(",", "")),
                        float(match.group(5).replace(",", "")),
                    )
    except FileNotFoundError:
        pass
    return last


def _print_summary(log_path: Path, batch_size: int, seq_len: int) -> None:
    print()
    print("=" * 96)
    print(
        f"{'Model':<16} {'Steps':>8} {'Final Loss':>12} {'Tps':>10} "
        f"{'TFLOPs':>8} {'Mem(GiB)':>10} {'Tokens':>12}  Log"
    )
    print("-" * 96)

    result = _parse_log(log_path)
    if result is None:
        print(f"{'deepseek_v3':<16} {'NO DATA':>8}  {log_path.name}")
    else:
        step, loss, mem, tps, tflops = result
        tokens = step * batch_size * seq_len
        print(
            f"{'deepseek_v3':<16} {step:>8,} {loss:>12.4f} {tps:>10,} "
            f"{tflops:>8.2f} {mem:>10.2f} {tokens:>12,}  {log_path.name}"
        )
    print("=" * 96)
    print()


def run(args: argparse.Namespace) -> None:
    _check_torchtitan()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be positive")
    if args.log_freq <= 0:
        raise SystemExit("--log-freq must be positive")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"{ts}_titan_deepseek_v3_debugmodel.txt"
    cmd = _cmd(args)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}

    print()
    print("=" * 72)
    print("TorchTitan DeepSeek V3 debugmodel")
    print(f"GPU: {args.gpu}")
    print(f"Batch {args.batch_size} x seq {args.seq_len}")
    print(f"Steps: {args.steps}")
    print(f"Log: {log_path}")
    print("=" * 72)
    print(f"cmd: {' '.join(cmd)}")
    print()

    proc = subprocess.Popen(
        cmd,
        cwd=TORCHTITAN_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    thread = threading.Thread(target=_stream_to_file, args=(proc, log_path), daemon=True)
    thread.start()
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    finally:
        thread.join(timeout=10)

    _print_summary(log_path, args.batch_size, args.seq_len)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TorchTitan DeepSeek V3 debugmodel on one GPU"
    )
    parser.add_argument("--steps", type=int, default=10, help="Training steps")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=LOCAL_BATCH_SIZE,
        help=f"Local batch size (default {LOCAL_BATCH_SIZE})",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=f"Sequence length (default {SEQ_LEN})",
    )
    parser.add_argument("--log-freq", type=int, default=1, help="Metrics log frequency")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
