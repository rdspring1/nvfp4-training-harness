#!/usr/bin/env python3
"""TorchTitan launcher for DeepSeek V3 debugmodel and 16B smoke runs."""

import argparse
import datetime
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
ROOT_DIR = PLUGIN_DIR.parent
TORCHTITAN_DIR = ROOT_DIR / "third_party" / "torchtitan"
DEEPSEEK_MODEL_DIR = TORCHTITAN_DIR / "torchtitan" / "models" / "deepseek_v3"
RESULTS_DIR = ROOT_DIR / "deepseek_v3_results"
NVFP4_OVERRIDE_MODULE = "torchtitan_ao_dsv3.overrides"

MODULE = "deepseek_v3"
FLAVOR_CONFIGS = {
    "debugmodel": "deepseek_v3_debugmodel",
    "16b": "deepseek_v3_16b",
}
FLAVOR_DEFAULTS = {
    "debugmodel": {"batch_size": 8, "seq_len": 2048, "steps": 10},
    "16b": {"batch_size": 1, "seq_len": 1024, "steps": 1},
}
HF_ASSET_PATHS = {
    "16b": TORCHTITAN_DIR / "assets" / "hf" / "deepseek-moe-16b-base",
}
HF_ASSET_REPOS = {
    "16b": "deepseek-ai/deepseek-moe-16b-base",
}

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


def _download_tokenizer(flavor: str) -> None:
    repo_id = HF_ASSET_REPOS[flavor]
    cmd = [
        sys.executable,
        "scripts/download_hf_assets.py",
        "--repo_id",
        repo_id,
        "--assets",
        "tokenizer",
    ]
    print(
        f"DeepSeek V3 {flavor} tokenizer assets are missing; "
        f"downloading tokenizer from {repo_id}."
    )
    try:
        subprocess.run(cmd, cwd=TORCHTITAN_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Failed to download DeepSeek V3 {flavor} tokenizer assets. Run:\n"
            "  cd third_party/torchtitan\n"
            f"  python scripts/download_hf_assets.py --repo_id {repo_id} "
            "--assets tokenizer"
        ) from exc


def _ensure_assets(flavor: str) -> None:
    hf_assets_path = HF_ASSET_PATHS.get(flavor)
    if hf_assets_path is None:
        return
    tokenizer_path = hf_assets_path / "tokenizer.json"
    if not tokenizer_path.exists():
        _download_tokenizer(flavor)

    if not tokenizer_path.exists():
        raise SystemExit(
            f"DeepSeek V3 {flavor} tokenizer assets are still missing after download. "
            "Run:\n"
            "  cd third_party/torchtitan\n"
            "  python scripts/download_hf_assets.py "
            f"--repo_id {HF_ASSET_REPOS[flavor]} --assets tokenizer"
        )


def _parse_gpus(args: argparse.Namespace) -> list[str]:
    if args.gpus is None:
        return [str(args.gpu)] if args.flavor == "debugmodel" else ["0", "1", "2", "3"]

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU index")
    return gpus


def _cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "-m",
        "torchtitan.train",
        "--module",
        MODULE,
        "--config",
        FLAVOR_CONFIGS[args.flavor],
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
    if args.flavor == "16b":
        cmd += [
            "--parallelism.data_parallel_shard_degree",
            "4",
            "--parallelism.expert_parallel_degree",
            "2",
        ]
    if args.nvfp4:
        cmd += ["--override.imports", NVFP4_OVERRIDE_MODULE]
    return cmd


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


def _print_summary(
    log_path: Path, batch_size: int, seq_len: int, data_parallel_degree: int
) -> None:
    global_batch_size = batch_size * data_parallel_degree
    print()
    print("=" * 96)
    print(
        f"{'Model':<16} {'Steps':>8} {'Final Loss':>12} {'Tps':>10} "
        f"{'TFLOPs':>8} {'Mem(GiB)':>10}  Log"
    )
    print("-" * 96)

    result = _parse_log(log_path)
    if result is None:
        print(f"{'deepseek_v3':<16} {'NO DATA':>8}  {log_path.name}")
    else:
        step, loss, mem, tps, tflops = result
        tokens = step * global_batch_size * seq_len
        print(
            f"{'deepseek_v3':<16} {step:>8,} {loss:>12.4f} {tps:>10,} "
            f"{tflops:>8.2f} {mem:>10.2f}  {log_path.name}"
        )
        print(
            f"Tokens: {step:,} (step) * {global_batch_size:,} "
            f"(global batch) * {seq_len:,} (seq_len) = {tokens:,}"
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

    gpus = _parse_gpus(args)
    args.nproc_per_node = len(gpus)
    if args.flavor == "16b" and args.nproc_per_node != 4:
        raise SystemExit("--flavor 16b requires exactly 4 GPUs via --gpus")

    _ensure_assets(args.flavor)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    precision = "nvfp4" if args.nvfp4 else "bf16"
    log_path = RESULTS_DIR / (
        f"{ts}_titan_deepseek_v3_{args.flavor}_{precision}.txt"
    )
    cmd = _cmd(args)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    if args.nvfp4:
        env["PYTHONPATH"] = f"{PLUGIN_DIR}:{os.environ.get('PYTHONPATH', '')}"

    print()
    print("=" * 72)
    print(f"TorchTitan DeepSeek V3 {args.flavor}")
    print(f"GPUs: {','.join(gpus)}")
    print(f"Processes: {args.nproc_per_node}")
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

    _print_summary(log_path, args.batch_size, args.seq_len, args.nproc_per_node)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TorchTitan DeepSeek V3 debugmodel or 16B smoke"
    )
    parser.add_argument(
        "--flavor",
        choices=sorted(FLAVOR_CONFIGS),
        default="debugmodel",
        help="DeepSeek V3 model flavor",
    )
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated visible GPU indices; defaults to --gpu for debugmodel and 0,1,2,3 for 16b",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Local batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length",
    )
    parser.add_argument("--log-freq", type=int, default=1, help="Metrics log frequency")
    parser.add_argument(
        "--nvfp4",
        action="store_true",
        help=f"Enable torchao NVFP4 grouped experts via {NVFP4_OVERRIDE_MODULE}",
    )
    args = parser.parse_args()
    defaults = FLAVOR_DEFAULTS[args.flavor]
    if args.steps is None:
        args.steps = defaults["steps"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.seq_len is None:
        args.seq_len = defaults["seq_len"]
    run(args)


if __name__ == "__main__":
    main()
