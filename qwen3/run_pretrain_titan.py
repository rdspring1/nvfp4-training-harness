#!/usr/bin/env python3
"""Run Qwen3-8B C4 pretraining precision comparisons on 4xFSDP."""

import argparse
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TORCHTITAN_DIR = ROOT_DIR / "third_party" / "torchtitan"
HF_ASSETS_DIR = TORCHTITAN_DIR / "assets" / "hf" / "Qwen3-8B"
LOG_ROOT = Path(__file__).resolve().parent / "run_logs_pretrain"
TOTAL_TOKENS = 200_000_000
LOCAL_BATCH_SIZE = 16
WORLD_SIZE = 4
SEQ_LEN = 2048
PRECISIONS = ("bf16", "mxfp8", "nvfp4-mixed")
LANES = {
    "random_init": {
        "bf16": "qwen3_8b_pretrain",
        "mxfp8": "qwen3_8b_pretrain_mxfp8",
        "nvfp4-mixed": "qwen3_8b_pretrain_nvfp4_mixed",
    },
    "continued_pretraining": {
        "bf16": "qwen3_8b_continue_pretrain",
        "mxfp8": "qwen3_8b_continue_pretrain_mxfp8",
        "nvfp4-mixed": "qwen3_8b_continue_pretrain_nvfp4_mixed",
    },
}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_METRIC_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[\d.]+).*?"
    r"memory:\s*(?P<memory>[^\s]+).*?tps:\s*(?P<tps>[\d,]+).*?"
    r"tflops:\s*(?P<tflops>[\d.]+)"
)


def _steps(total_tokens: int) -> int:
    tokens_per_step = LOCAL_BATCH_SIZE * WORLD_SIZE * SEQ_LEN
    return -(-total_tokens // tokens_per_step)


def _assets_ready() -> bool:
    return (HF_ASSETS_DIR / "tokenizer.json").exists() and any(
        HF_ASSETS_DIR.glob("*.safetensors")
    )


def _command(lane: str, precision: str, steps: int) -> list[str]:
    return [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(WORLD_SIZE),
        "--local-ranks-filter",
        "0",
        "-m",
        "torchtitan.train",
        "--module",
        "qwen3",
        "--config",
        LANES[lane][precision],
        "--parallelism.data_parallel_shard_degree",
        str(WORLD_SIZE),
        "--parallelism.tensor_parallel_degree",
        "1",
        "--training.local_batch_size",
        str(LOCAL_BATCH_SIZE),
        "--training.global_batch_size",
        str(LOCAL_BATCH_SIZE * WORLD_SIZE),
        "--training.seq_len",
        str(SEQ_LEN),
        "--training.steps",
        str(steps),
        "--metrics.log_freq",
        "10",
        "--metrics.no-enable-wandb",
        "--checkpoint.load_only",
    ]


def _git_revision(directory: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True
    ).strip()


def _summary(
    lane: str, precision: str, command: list[str], log_path: Path, started: datetime, steps: int
) -> str:
    content = log_path.read_text(errors="replace")
    metrics = None
    for line in content.splitlines():
        match = _METRIC_RE.search(_ANSI_RE.sub("", line))
        if match:
            metrics = match.groupdict()
    initialization = "random initialization" if lane == "random_init" else "local Qwen3-8B HF checkpoint"
    lines = [
        f"# Qwen3-8B {precision.upper()} C4 {lane.replace('_', ' ').title()}",
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
        f"- Initialization: {initialization}",
        "- Model/dataset: Qwen3-8B on C4",
        "- Parallelism: FSDP 4, TP 1",
        "- Local/global batch size: 16 / 64 (gradient accumulation 1)",
        f"- Sequence length/steps: {SEQ_LEN} / {steps}",
        f"- Tokens requested: {TOTAL_TOKENS:,}",
        f"- Tokens processed: {steps * LOCAL_BATCH_SIZE * WORLD_SIZE * SEQ_LEN:,}",
        f"- TorchTitan revision: `{_git_revision(TORCHTITAN_DIR)}`",
        f"- Harness revision: `{_git_revision(ROOT_DIR)}`",
        "",
        "## Result",
        "",
        f"- Completion marker: {'present' if 'Training completed' in content else 'missing'}",
        f"- Wall time: {datetime.now(UTC) - started}",
    ]
    if metrics:
        lines += [
            "",
            "| Final logged step | Loss | TPS / GPU | TFLOPs / GPU | Peak Reserved Mem |",
            "| ---: | ---: | ---: | ---: | ---: |",
            "| {step} | {loss} | {tps} | {tflops} | {memory} |".format(**metrics),
        ]
    return "\n".join(lines) + "\n"


def _run(
    lane: str,
    precision: str,
    command: list[str],
    env: dict[str, str],
    steps: int,
    tag: str | None,
) -> None:
    log_dir = LOG_ROOT / lane / "eager_trainer"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    run_tag = (
        f"{stamp}_titan_fsdp4_{precision.replace('-', '_')}_eager_compile_"
        f"200m_gbs64_lbs16_ga1"
    )
    if tag:
        run_tag += f"_{re.sub(r'[^a-zA-Z0-9_-]', '_', tag)}"
    log_path = log_dir / f"{run_tag}.txt"
    started = datetime.now(UTC)
    print(f"[{lane}/{precision}] {' '.join(command)}", flush=True)
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
            raise SystemExit(f"{lane}/{precision} failed; raw log retained at {log_path}")
    summary_path = log_path.with_suffix(".md")
    summary_path.write_text(_summary(lane, precision, command, log_path, started, steps))
    print(f"[{lane}/{precision}] wrote {log_path} and {summary_path}", flush=True)


def _completed_log(lane: str, precision: str) -> Path | None:
    log_dir = LOG_ROOT / lane / "eager_trainer"
    logs = sorted(
        log_dir.glob(
            f"*_fsdp4_{precision.replace('-', '_')}_eager_compile_200m_gbs64_lbs16_ga1.txt"
        )
    )
    for log_path in reversed(logs):
        if "Training completed" in log_path.read_text(errors="replace"):
            return log_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES, help="Run one initialization lane")
    parser.add_argument("--only", choices=PRECISIONS, help="Run one precision")
    parser.add_argument("--total-tokens", type=int, default=TOTAL_TOKENS)
    parser.add_argument("--gpus", help="Comma-separated four GPU indices")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs and write missing summaries")
    parser.add_argument("--tag", help="Suffix log filenames for a distinct rerun")
    args = parser.parse_args()
    if args.total_tokens <= 0:
        raise SystemExit("--total-tokens must be positive")
    steps = _steps(args.total_tokens)
    lanes = [args.lane] if args.lane else list(LANES)
    precisions = [args.only] if args.only else list(PRECISIONS)
    if args.dry_run:
        for lane in lanes:
            for precision in precisions:
                print(f"[{lane}/{precision}] {' '.join(_command(lane, precision, steps))}")
        return
    if not _assets_ready():
        raise SystemExit(f"Qwen3-8B tokenizer and safetensors are required in {HF_ASSETS_DIR}")
    env = os.environ.copy()
    if args.gpus:
        gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        if len(gpus) != WORLD_SIZE:
            raise SystemExit("--gpus must name exactly four GPU indices")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env["PYTHONUNBUFFERED"] = "1"
    for lane in lanes:
        for precision in precisions:
            if args.resume and (log_path := _completed_log(lane, precision)):
                summary_path = log_path.with_suffix(".md")
                if not summary_path.exists():
                    started = datetime.fromtimestamp(log_path.stat().st_ctime, UTC)
                    summary_path.write_text(
                        _summary(
                            lane,
                            precision,
                            _command(lane, precision, steps),
                            log_path,
                            started,
                            steps,
                        )
                    )
                print(f"[{lane}/{precision}] already completed: {log_path}", flush=True)
                continue
            _run(
                lane,
                precision,
                _command(lane, precision, steps),
                env,
                steps,
                args.tag,
            )


if __name__ == "__main__":
    main()
