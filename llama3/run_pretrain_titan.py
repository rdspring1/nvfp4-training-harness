#!/usr/bin/env python3
"""Run Llama 3.1 8B C4 continued-pretraining precision comparisons."""

import argparse
import math
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TORCHTITAN_DIR = ROOT_DIR / "third_party" / "torchtitan"
HF_ASSETS_DIR = TORCHTITAN_DIR / "assets" / "hf" / "Llama-3.1-8B"
LOG_ROOT = (
    Path(__file__).resolve().parent / "run_logs_pretrain" / "continued_pretraining"
)
FULL_LOG_DIR = LOG_ROOT / "eager_trainer"
SMOKE_LOG_DIR = LOG_ROOT / "smoke"
TOTAL_TOKENS = 200_000_000
LOCAL_BATCH_SIZE = 32
WORLD_SIZE = 4
SEQ_LEN = 2048
SMOKE_STEPS = 2
PRECISIONS = {
    "bf16": "llama3_8b_continue_pretrain",
    "mxfp8": "llama3_8b_continue_pretrain_mxfp8",
    "nvfp4-mixed": "llama3_8b_continue_pretrain_nvfp4_mixed",
}
STEP1_LIMITS = {"bf16": 6.14, "mxfp8": 6.09, "nvfp4-mixed": 6.10}
CONVERTER_MARKERS = {
    "bf16": None,
    "mxfp8": "Converted Linear layers to MXFP8Linear",
    "nvfp4-mixed": "Converted Linear layers to NVFP4Linear",
}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_METRIC_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>\S+).*?"
    r"grad_norm:\s*(?P<grad_norm>\S+).*?"
    r"memory:\s*(?P<memory>\S+).*?tps:\s*(?P<tps>[\d,]+).*?"
    r"tflops:\s*(?P<tflops>[\d,.]+)"
)


def _steps(total_tokens: int) -> int:
    return -(-total_tokens // (LOCAL_BATCH_SIZE * WORLD_SIZE * SEQ_LEN))


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
            "meta-llama/Llama-3.1-8B",
            "--local_dir",
            str(HF_ASSETS_DIR.parent),
            "--assets",
            "tokenizer",
            "safetensors",
            "config",
        ],
        cwd=TORCHTITAN_DIR,
        check=True,
    )


def _command(precision: str, steps: int, smoke: bool) -> list[str]:
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
        "llama3",
        "--config",
        PRECISIONS[precision],
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
        "1" if smoke else "10",
        "--metrics.no-enable-wandb",
        "--checkpoint.load_only",
    ]


def _metrics(content: str) -> list[dict[str, str]]:
    found = []
    for line in content.splitlines():
        if match := _METRIC_RE.search(_ANSI_RE.sub("", line)):
            found.append(match.groupdict())
    return found


def _validate_log(path: Path, precision: str, smoke: bool) -> dict[str, str]:
    content = path.read_text(errors="replace")
    if "Loading HF safetensors from" not in content:
        raise ValueError("missing HF safetensor loading marker")
    marker = CONVERTER_MARKERS[precision]
    if marker and marker not in content:
        raise ValueError(f"missing precision converter marker: {marker}")
    if not marker and "Converted Linear layers to" in content:
        raise ValueError("BF16 run unexpectedly applied a linear converter")
    if "Training completed" not in content:
        raise ValueError("missing Training completed marker")
    metrics = _metrics(content)
    if not metrics:
        raise ValueError("missing training metrics")
    for metric in metrics:
        for name in ("loss", "grad_norm"):
            if not math.isfinite(float(metric[name])):
                raise ValueError(f"non-finite {name} at step {metric['step']}")
    step1 = next((metric for metric in metrics if int(metric["step"]) == 1), None)
    if step1 is None:
        raise ValueError("missing step-1 loss")
    if smoke and float(step1["loss"]) >= STEP1_LIMITS[precision]:
        raise ValueError(
            f"step-1 loss {step1['loss']} is not below {STEP1_LIMITS[precision]:.2f}"
        )
    return metrics[-1]


def _git_revision(directory: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True
    ).strip()


def _summary(
    precision: str,
    command: list[str],
    log_path: Path,
    started: datetime,
    steps: int,
    smoke: bool,
) -> str:
    final = _validate_log(log_path, precision, smoke)
    tokens = steps * LOCAL_BATCH_SIZE * WORLD_SIZE * SEQ_LEN
    lines = [
        f"# Llama 3.1 8B {precision.upper()} C4 Continued Pretraining",
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
        "- Initialization: local Llama 3.1 8B HF checkpoint",
        "- Dataset: C4",
        "- Parallelism: FSDP 4, TP 1",
        "- Local/global batch size: 32 / 128 (gradient accumulation 1)",
        f"- Sequence length/steps: {SEQ_LEN} / {steps}",
        f"- Tokens processed: {tokens:,}",
        f"- TorchTitan revision: `{_git_revision(TORCHTITAN_DIR)}`",
        f"- Harness revision: `{_git_revision(ROOT_DIR)}`",
        "",
        "## Result",
        "",
        "- Completion marker: present",
        (
            f"- Step-1 loss gate: passed (< {STEP1_LIMITS[precision]:.2f})"
            if smoke
            else "- Finite final metrics: passed"
        ),
        f"- Wall time: {datetime.now(UTC) - started}",
        "",
        "| Final logged step | Loss | Grad norm | TPS / GPU | TFLOPs / GPU | Peak Reserved Mem |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {step} | {loss} | {grad_norm} | {tps} | {tflops} | {memory} |".format(
            **final
        ),
    ]
    return "\n".join(lines) + "\n"


def _run(
    precision: str,
    command: list[str],
    env: dict[str, str],
    steps: int,
    smoke: bool,
    tag: str | None,
) -> Path:
    log_dir = SMOKE_LOG_DIR if smoke else FULL_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    scope = "smoke" if smoke else "200m"
    run_tag = (
        f"{stamp}_titan_fsdp4_{precision.replace('-', '_')}_eager_compile_"
        f"{scope}_gbs128_lbs32_ga1"
    )
    if tag:
        run_tag += f"_{re.sub(r'[^a-zA-Z0-9_-]', '_', tag)}"
    log_path = log_dir / f"{run_tag}.txt"
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
            log_file.flush()
        if process.wait() != 0:
            raise SystemExit(f"{precision} failed; raw log retained at {log_path}")
    try:
        summary = _summary(precision, command, log_path, started, steps, smoke)
    except ValueError as error:
        raise SystemExit(
            f"{precision} validation failed: {error}; see {log_path}"
        ) from error
    log_path.with_suffix(".md").write_text(summary)
    print(f"[{precision}] validated {log_path}", flush=True)
    return log_path


def _latest_valid_log(log_dir: Path, precision: str, smoke: bool) -> Path | None:
    pattern = (
        f"*_fsdp4_{precision.replace('-', '_')}_eager_compile_*_gbs128_lbs32_ga1*.txt"
    )
    for path in reversed(sorted(log_dir.glob(pattern))):
        try:
            _validate_log(path, precision, smoke)
        except ValueError:
            continue
        return path
    return None


def _validate_gpus(gpus_arg: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if gpus_arg:
        gpus = [gpu.strip() for gpu in gpus_arg.split(",") if gpu.strip()]
        if len(gpus) != WORLD_SIZE or len(set(gpus)) != WORLD_SIZE:
            raise SystemExit("--gpus must name exactly four distinct GPU indices")
        try:
            physical_count = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=count",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).splitlines()[0]
            )
            indices = [int(gpu) for gpu in gpus]
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
            raise SystemExit(f"could not validate --gpus: {error}") from error
        if any(index < 0 or index >= physical_count for index in indices):
            raise SystemExit(f"--gpus indices must be in [0, {physical_count - 1}]")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    probe = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        env=env,
        text=True,
    ).strip()
    if int(probe) != WORLD_SIZE:
        raise SystemExit(f"FSDP4 requires exactly four visible GPUs; found {probe}")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=PRECISIONS, help="Run one precision")
    parser.add_argument("--total-tokens", type=int, default=TOTAL_TOKENS)
    parser.add_argument("--gpus", help="Comma-separated four GPU indices")
    parser.add_argument("--smoke", action="store_true", help="Run two-step validation")
    parser.add_argument("--download-assets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tag", help="Suffix log filenames for a distinct rerun")
    args = parser.parse_args()
    if args.total_tokens <= 0:
        raise SystemExit("--total-tokens must be positive")
    steps = SMOKE_STEPS if args.smoke else _steps(args.total_tokens)
    precisions = [args.only] if args.only else list(PRECISIONS)
    if args.dry_run:
        for precision in precisions:
            print(f"[{precision}] {' '.join(_command(precision, steps, args.smoke))}")
        return
    env = _validate_gpus(args.gpus)
    if not _assets_ready() and args.download_assets:
        _download_assets()
    if not _assets_ready():
        raise SystemExit(
            "Llama 3.1 8B assets are missing; rerun with --download-assets"
        )
    if not args.smoke:
        missing = [
            precision
            for precision in precisions
            if _latest_valid_log(SMOKE_LOG_DIR, precision, smoke=True) is None
        ]
        if missing:
            raise SystemExit(
                "valid two-step smoke logs are required before full runs: "
                + ", ".join(missing)
            )
    for precision in precisions:
        log_dir = SMOKE_LOG_DIR if args.smoke else FULL_LOG_DIR
        if args.resume and (
            completed := _latest_valid_log(log_dir, precision, args.smoke)
        ):
            print(f"[{precision}] already completed: {completed}", flush=True)
            continue
        _run(
            precision,
            _command(precision, steps, args.smoke),
            env,
            steps,
            args.smoke,
            args.tag,
        )
    if not args.smoke and args.only is None:
        subprocess.run(
            [sys.executable, "plot_pretrain_loss.py"],
            cwd=Path(__file__).parent,
            check=True,
        )


if __name__ == "__main__":
    main()
