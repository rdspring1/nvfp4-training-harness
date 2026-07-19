#!/usr/bin/env python3
"""TorchTitan launcher for DeepSeek V3 debugmodel and 16B smoke runs."""

import argparse
import datetime
import math
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
NVFP4_LINEAR_MODULE = "torchtitan.overrides.nvfp4_linear"
NVFP4_GROUPED_EXPERTS_MODULE = "torchtitan.overrides.nvfp4_grouped_experts"
TE_GROUPED_EXPERTS_MODULE = "te_moe_overrides.te_grouped_experts"
NVFP4_TARGET_MODULES = {
    "linear": [NVFP4_LINEAR_MODULE],
    "grouped-experts": [NVFP4_GROUPED_EXPERTS_MODULE],
    "te-grouped-experts": [TE_GROUPED_EXPERTS_MODULE],
    "both": [NVFP4_LINEAR_MODULE, NVFP4_GROUPED_EXPERTS_MODULE],
}

TRAINER_MODULES = {
    "eager": "deepseek_v3",
    "graph": "graph_trainer.deepseek_v3",
}
FLAVOR_CONFIGS = {
    "eager": {
        "debugmodel": "deepseek_v3_debugmodel",
        "16b": "deepseek_v3_16b",
    },
    "graph": {
        "debugmodel": "graph_trainer_deepseek_v3_debugmodel",
        "16b": "graph_trainer_deepseek_v3_16b",
    },
}
FLAVOR_DEFAULTS = {
    "debugmodel": {"batch_size": 8, "seq_len": 2048, "steps": 10},
    "16b": {"batch_size": 1, "seq_len": 1024, "steps": 1},
}
# 16b MoE routing (torchtitan/models/deepseek_v3/__init__.py:341,359)
MOE_16B_NUM_EXPERTS = 64
MOE_16B_TOP_K = 6
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
        if args.flavor == "debugmodel":
            return ["0", "1"] if args.nvfp4 else [str(args.gpu)]
        return ["0", "1", "2", "3"]

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
        TRAINER_MODULES[args.trainer],
        "--config",
        FLAVOR_CONFIGS[args.trainer][args.flavor],
        "--training.local_batch_size",
        str(args.batch_size),
        "--training.seq_len",
        str(args.seq_len),
        "--training.steps",
        str(args.steps),
        "--dataloader.dataset",
        args.dataset,
        "--metrics.log_freq",
        str(args.log_freq),
    ]
    if args.flavor == "16b":
        cmd += [
            "--parallelism.data_parallel_shard_degree",
            "4",
            "--parallelism.expert_parallel_degree",
            str(args.expert_parallel_degree or 2),
        ]
    elif args.nvfp4:
        cmd += [
            "--parallelism.data_parallel_shard_degree",
            str(args.nproc_per_node),
            "--parallelism.expert_parallel_degree",
            str(args.expert_parallel_degree or 2),
        ]
    if args.nvfp4:
        cmd += ["--override.imports", ",".join(NVFP4_TARGET_MODULES[args.nvfp4_target])]
    if args.global_batch_size is not None:
        cmd += ["--training.global_batch_size", str(args.global_batch_size)]
    if args.trainer == "graph":
        cmd += ["--compile.mode", "aot_fx_trace"]
    elif args.compile:
        cmd += ["--compile.enable"]
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
    completed = False
    try:
        with open(log_path) as f:
            for line in f:
                clean = _ANSI_RE.sub("", line)
                if "Training completed" in clean:
                    completed = True
                match = _STEP_RE.search(clean)
                if match:
                    metric = (
                        int(match.group(1)),
                        float(match.group(2)),
                        float(match.group(3)),
                        int(match.group(4).replace(",", "")),
                        float(match.group(5).replace(",", "")),
                    )
                    if last is None or metric[0] > last[0]:
                        last = metric
                    elif metric[0] == last[0] and metric[2] > last[2]:
                        last = metric
    except FileNotFoundError:
        pass
    return last, completed


def _print_summary(
    log_path: Path,
    batch_size: int,
    seq_len: int,
    data_parallel_degree: int,
    global_batch_size: int | None,
    trainer: str,
    requested_steps: int,
) -> None:
    global_batch_size = global_batch_size or batch_size * data_parallel_degree
    run_name = f"deepseek_v3/{trainer}"
    print()
    print("=" * 96)
    print(
        f"{'Model':<16} {'Steps':>8} {'Final Loss':>12} {'Tps':>10} "
        f"{'TFLOPs':>8} {'Mem(GiB)':>10}  Log"
    )
    print("-" * 96)

    result, completed = _parse_log(log_path)
    if result is None:
        print(f"{run_name:<16} {'NO DATA':>8}  {log_path.name}")
    else:
        metric_step, loss, mem, tps, tflops = result
        steps = requested_steps if completed else metric_step
        tokens = steps * global_batch_size * seq_len
        print(
            f"{run_name:<16} {steps:>8,} {loss:>12.4f} {tps:>10,} "
            f"{tflops:>8.2f} {mem:>10.2f}  {log_path.name}"
        )
        if metric_step != steps:
            print(f"Last metric: step {metric_step:,}")
        print(
            f"Tokens: {steps:,} (step) * {global_batch_size:,} "
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
    if args.global_batch_size is not None and args.global_batch_size <= 0:
        raise SystemExit("--global-batch-size must be positive")
    if args.expert_parallel_degree is not None and args.expert_parallel_degree <= 0:
        raise SystemExit("--expert-parallel-degree must be positive")

    gpus = _parse_gpus(args)
    args.nproc_per_node = len(gpus)
    if args.flavor == "16b" and args.nproc_per_node != 4:
        raise SystemExit("--flavor 16b requires exactly 4 GPUs via --gpus")
    if args.flavor == "debugmodel" and args.nvfp4 and args.nproc_per_node != 2:
        raise SystemExit("--nvfp4 debugmodel requires exactly 2 GPUs via --gpus")

    _ensure_assets(args.flavor)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    precision = "nvfp4" if args.nvfp4 else "bf16"
    compile_suffix = "_compile" if args.trainer == "graph" or args.compile else ""
    log_path = RESULTS_DIR / (
        f"{ts}_titan_deepseek_v3_{args.flavor}_{args.trainer}_{precision}"
        f"{compile_suffix}.txt"
    )
    cmd = _cmd(args)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    if args.nvfp4:
        env["PYTHONPATH"] = f"{PLUGIN_DIR}:{os.environ.get('PYTHONPATH', '')}"

    print()
    print("=" * 72)
    print(f"TorchTitan DeepSeek V3 {args.flavor}")
    print(f"Trainer: {args.trainer}")
    print(f"Compile: {'yes' if args.trainer == 'graph' or args.compile else 'no'}")
    print(f"GPUs: {','.join(gpus)}")
    print(f"Processes: {args.nproc_per_node}")
    print(f"Batch {args.batch_size} x seq {args.seq_len}")
    if args.target_tokens_per_expert is not None:
        ep = args.expert_parallel_degree or 2
        tokens = args.batch_size * args.seq_len
        per_expert = tokens * MOE_16B_TOP_K * ep / MOE_16B_NUM_EXPERTS
        aggregate = tokens * MOE_16B_TOP_K
        print(
            f"MoE M: T={tokens:,} (batch*seq) -> per-expert M={per_expert:,.0f} "
            f"(EP={ep}, {MOE_16B_NUM_EXPERTS // ep} local experts), aggregate/rank={aggregate:,}"
        )
    if args.global_batch_size is not None:
        print(f"Global batch: {args.global_batch_size}")
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

    _print_summary(
        log_path,
        args.batch_size,
        args.seq_len,
        args.nproc_per_node,
        args.global_batch_size,
        args.trainer,
        args.steps,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TorchTitan DeepSeek V3 debugmodel or 16B smoke"
    )
    parser.add_argument(
        "--flavor",
        choices=sorted(FLAVOR_DEFAULTS),
        default="debugmodel",
        help="DeepSeek V3 model flavor",
    )
    parser.add_argument(
        "--trainer",
        choices=sorted(TRAINER_MODULES),
        default="eager",
        help="Trainer implementation to launch",
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
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Global batch size; enables gradient accumulation when larger than local batch times data-parallel degree",
    )
    parser.add_argument(
        "--expert-parallel-degree",
        type=int,
        default=None,
        help="Expert parallelism degree override",
    )
    parser.add_argument(
        "--target-tokens-per-expert",
        type=int,
        default=None,
        help="For --flavor 16b: size local batch so each local expert's grouped-GEMM sees "
        "~this many tokens (per-expert M). Overrides --batch-size",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="c4_test",
        help="TorchTitan dataloader dataset",
    )
    parser.add_argument("--log-freq", type=int, default=1, help="Metrics log frequency")
    parser.add_argument(
        "--nvfp4",
        action="store_true",
        help="Enable torchao NVFP4 overrides (select which with --nvfp4-target)",
    )
    parser.add_argument(
        "--nvfp4-target",
        choices=sorted(NVFP4_TARGET_MODULES),
        default="both",
        help="Which NVFP4 override(s) to import when --nvfp4 is set (default: both)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable TorchTitan compile for the eager trainer; graph trainer compiles by default",
    )
    args = parser.parse_args()
    defaults = FLAVOR_DEFAULTS[args.flavor]
    if args.steps is None:
        args.steps = defaults["steps"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.seq_len is None:
        args.seq_len = defaults["seq_len"]
    if args.target_tokens_per_expert is not None:
        if args.flavor != "16b":
            raise SystemExit("--target-tokens-per-expert is only supported for --flavor 16b")
        if args.target_tokens_per_expert <= 0:
            raise SystemExit("--target-tokens-per-expert must be positive")
        ep = args.expert_parallel_degree or 2
        per_unit_batch = MOE_16B_TOP_K * ep * args.seq_len / MOE_16B_NUM_EXPERTS
        args.batch_size = math.ceil(args.target_tokens_per_expert / per_unit_batch)
    run(args)


if __name__ == "__main__":
    main()
