#!/usr/bin/env python3
"""Run Qwen3-8B SFT smoke tests with eager, FSDP, and TP layouts."""

import argparse
import os
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TORCHTITAN_DIR = ROOT_DIR / "third_party" / "torchtitan"
HF_ASSETS_DIR = TORCHTITAN_DIR / "assets" / "hf" / "Qwen3-8B"

EXPERIMENTS = [
    {"name": "eager", "fsdp": 1, "tp": 1},
    {"name": "fsdp2", "fsdp": 2, "tp": 1},
    {"name": "tp2", "fsdp": 1, "tp": 2},
    {"name": "fsdp2_tp2", "fsdp": 2, "tp": 2},
]


def _check_assets() -> None:
    tokenizer_exists = (HF_ASSETS_DIR / "tokenizer.json").exists()
    weights_exist = any(HF_ASSETS_DIR.glob("*.safetensors"))
    if tokenizer_exists and weights_exist:
        return

    raise SystemExit(
        "Qwen3-8B tokenizer or safetensors are missing. Download them with:\n"
        "  cd third_party/torchtitan\n"
        "  python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-8B "
        "--assets tokenizer safetensors"
    )


def _command(experiment: dict[str, int | str], steps: int) -> list[str]:
    world_size = int(experiment["fsdp"]) * int(experiment["tp"])
    return [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(world_size),
        "--local-ranks-filter",
        "0",
        "-m",
        "torchtitan.train",
        "--module",
        "qwen3",
        "--config",
        "sft_qwen3_8b_math",
        "--parallelism.data_parallel_shard_degree",
        str(experiment["fsdp"]),
        "--parallelism.tensor_parallel_degree",
        str(experiment["tp"]),
        "--training.steps",
        str(steps),
        "--metrics.log_freq",
        "1",
        "--metrics.no-enable-wandb",
        "--checkpoint.load_only",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-8B GSM8K SFT smoke tests"
    )
    parser.add_argument(
        "--only",
        choices=[experiment["name"] for experiment in EXPERIMENTS],
        help="Run one layout instead of the full smoke sweep",
    )
    parser.add_argument(
        "--gpus",
        help="Comma-separated GPU indices; each layout uses the required prefix",
    )
    parser.add_argument("--steps", type=int, default=10, help="Steps per layout")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without launching"
    )
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    experiments = [
        experiment
        for experiment in EXPERIMENTS
        if args.only is None or experiment["name"] == args.only
    ]
    gpus = None
    if args.gpus is not None:
        gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        required = max(
            int(experiment["fsdp"]) * int(experiment["tp"])
            for experiment in experiments
        )
        if len(gpus) < required:
            raise SystemExit(
                f"Selected layouts require {required} GPUs, but --gpus has {len(gpus)}"
            )

    if not args.dry_run:
        _check_assets()

    for experiment in experiments:
        world_size = int(experiment["fsdp"]) * int(experiment["tp"])
        command = _command(experiment, args.steps)
        env = os.environ.copy()
        selected_gpus = gpus[:world_size] if gpus is not None else None
        if selected_gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)

        print(
            f"[{experiment['name']}] world_size={world_size} "
            f"FSDP={experiment['fsdp']} TP={experiment['tp']}"
        )
        if selected_gpus is not None:
            print(f"[{experiment['name']}] GPUs={','.join(selected_gpus)}")
        print(f"[{experiment['name']}] {' '.join(command)}", flush=True)

        if not args.dry_run:
            subprocess.run(command, cwd=TORCHTITAN_DIR, env=env, check=True)


if __name__ == "__main__":
    main()
