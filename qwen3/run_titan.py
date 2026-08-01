#!/usr/bin/env python3
"""Run Qwen3-8B GSM8K SFT eager Trainer comparisons on 4xFSDP."""

import argparse
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TORCHTITAN_DIR = ROOT_DIR / "third_party" / "torchtitan"
HF_ASSETS_DIR = TORCHTITAN_DIR / "assets" / "hf" / "Qwen3-8B"
LOG_DIR = Path(__file__).resolve().parent / "run_logs_sft" / "eager_trainer"
PRECISIONS = {
    "bf16": "sft_qwen3_8b_math",
    "mxfp8": "qwen3_8b_mxfp8",
    "nvfp4-mixed": "qwen3_8b_nvfp4_mixed",
}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_METRIC_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[\d.]+).*?"
    r"memory:\s*(?P<memory>[^\s]+).*?tps:\s*(?P<tps>[\d,]+).*?"
    r"tflops:\s*(?P<tflops>[\d.]+)"
)


def _assets_ready() -> bool:
    return (HF_ASSETS_DIR / "tokenizer.json").exists() and any(
        HF_ASSETS_DIR.glob("*.safetensors")
    )


def _download_assets() -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/download_hf_assets.py",
            "--repo_id",
            "Qwen/Qwen3-8B",
            "--local_dir",
            str(HF_ASSETS_DIR.parent),
            "--assets",
            "tokenizer",
            "safetensors",
        ],
        cwd=TORCHTITAN_DIR,
        check=True,
    )


def _command(precision: str, steps: int) -> list[str]:
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "4",
        "--local-ranks-filter",
        "0",
        "-m",
        "torchtitan.train",
        "--module",
        "qwen3",
        "--config",
        PRECISIONS[precision],
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "1",
        "--training.local_batch_size",
        "1",
        "--training.global_batch_size",
        "4",
        "--training.steps",
        str(steps),
        "--metrics.log_freq",
        "10",
        "--metrics.no-enable-wandb",
        "--checkpoint.load_only",
    ]
    if precision == "bf16":
        command += ["--compile.enable", "--compile.components", "model"]
    return command


def _git_revision(directory: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True
    ).strip()


def _summary(precision: str, command: list[str], log_path: Path, started: datetime) -> str:
    metrics = None
    for line in log_path.read_text(errors="replace").splitlines():
        match = _METRIC_RE.search(_ANSI_RE.sub("", line))
        if match:
            metrics = match.groupdict()
    completed = "Training completed" in log_path.read_text(errors="replace")
    elapsed = datetime.now(UTC) - started
    lines = [
        f"# Qwen3-8B {precision.upper()} GSM8K SFT FSDP4 Run",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONUNBUFFERED=1 " + " ".join(command),
        "```",
        "",
        "## Run Shape",
        "",
        "- Trainer: eager Trainer with model `torch.compile`",
        "- Model/dataset: Qwen3-8B SFT on `openai/gsm8k` (`main/train`)",
        "- Parallelism: FSDP 4, TP 1",
        "- Local/global batch size: 1 / 4 (gradient accumulation 1)",
        "- Sequence length/steps: 2048 / 180",
        f"- TorchTitan revision: `{_git_revision(TORCHTITAN_DIR)}`",
        f"- Harness revision: `{_git_revision(ROOT_DIR)}`",
        "",
        "## Result",
        "",
        f"- Completion marker: {'present' if completed else 'missing'}",
        f"- Wall time: {elapsed}",
    ]
    if metrics:
        lines += [
            "",
            "| Final logged step | Loss | TPS / GPU | TFLOPs / GPU | Peak Reserved Mem |",
            "| ---: | ---: | ---: | ---: | ---: |",
            "| {step} | {loss} | {tps} | {tflops} | {memory} |".format(**metrics),
        ]
    return "\n".join(lines) + "\n"


def _run(precision: str, command: list[str], env: dict[str, str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    tag = f"{stamp}_titan_fsdp4_{precision.replace('-', '_')}_eager_compile_sft_gbs4_lbs1_ga1"
    log_path = LOG_DIR / f"{tag}.txt"
    started = datetime.now(UTC)
    print(f"[{precision}] {' '.join(command)}", flush=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=TORCHTITAN_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        if process.wait() != 0:
            raise SystemExit(f"{precision} run failed; raw log retained at {log_path}")
    summary_path = log_path.with_suffix(".md")
    summary_path.write_text(_summary(precision, command, log_path, started))
    print(f"[{precision}] wrote {log_path} and {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=PRECISIONS, help="Run one precision only")
    parser.add_argument("--steps", type=int, default=180, help="SFT steps per precision")
    parser.add_argument("--gpus", help="Comma-separated four GPU indices")
    parser.add_argument("--download-assets", action="store_true", help="Download missing Qwen3-8B assets")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    args = parser.parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    precisions = [args.only] if args.only else list(PRECISIONS)
    if args.dry_run:
        for precision in precisions:
            print(f"[{precision}] {' '.join(_command(precision, args.steps))}")
        return
    if not _assets_ready():
        if not args.download_assets:
            raise SystemExit("Qwen3-8B assets are missing; rerun with --download-assets")
        _download_assets()
    if not _assets_ready():
        raise SystemExit("Qwen3-8B asset download completed without tokenizer and safetensors")

    env = os.environ.copy()
    if args.gpus:
        gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        if len(gpus) != 4:
            raise SystemExit("--gpus must name exactly four GPU indices")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env["PYTHONUNBUFFERED"] = "1"
    for precision in precisions:
        _run(precision, _command(precision, args.steps), env)


if __name__ == "__main__":
    main()
