#!/usr/bin/env python3
"""
TorchTitan bf16 launcher — single-GPU and multi-GPU FSDP2 + TP runs.

Subcommands:
    single   one TorchTitan run on a chosen GPU (8 h wall clock default)
    multi    sequential FSDP2 + TP shape sweep (tp4 / fsdp4 / tp2_fsdp2)
             targeting TOTAL_TOKENS tokens (default 200 M)

Usage:
    # Single-GPU smoke (5 min, llama3_debugmodel, 10 steps):
    python run_titan.py single --smoke

    # Single-GPU GraphTrainer smoke with CUDA graphs:
    python run_titan.py single --smoke --graph

    # Multi-GPU smoke over all three shapes (10 min/shape, debugmodel):
    python run_titan.py multi --smoke

    # Multi-GPU GraphTrainer smoke over all three shapes:
    python run_titan.py multi --smoke --graph

    # Full single-GPU 8-hour run in background:
    nohup python run_titan.py single > run_titan_single.log 2>&1 &

    # Full 200M-token multi-shape sweep:
    nohup python run_titan.py multi > run_titan_multi.log 2>&1 &
"""

import argparse
import datetime
import os
import re
import subprocess
import threading
import time
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent  # hosts torchtitan_ao
TORCHTITAN_DIR = PLUGIN_DIR / "third_party" / "torchtitan"
RESULTS_DIR = Path("llama3_results")
NVFP4_OVERRIDE_MODULE = "torchtitan_ao.overrides"
SEQ_LEN = 2048
LR = 3e-4

# single-GPU mode
SINGLE_BATCH = 4
SINGLE_WALL_HOURS = 8
SINGLE_STEPS_CEILING = 500_000
SINGLE_SMOKE_FLOOR = 3_000  # warn if non-smoke run ends below this

# multi-GPU mode
MULTI_WALL_HOURS = 8
MULTI_SMOKE_WALL_HOURS = 10 / 60
MULTI_TOTAL_TOKENS = 200_000_000
MULTI_SMOKE_STEPS = 10
MULTI_EXPERIMENTS = [
    {"name": "tp4", "tp": 4, "fsdp": 1, "batch_size": 32},
    {"name": "fsdp4", "tp": 1, "fsdp": 4, "batch_size": 4},
    {"name": "tp2_fsdp2", "tp": 2, "fsdp": 2, "batch_size": 8},
]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(
    r"step:\s*(\d+).*?loss:\s*([\d.]+).*?memory:\s*([\d.]+)GiB"
    r".*?tps:\s*([\d,]+).*?tflops:\s*([\d.,]+)"
)


def visible_gpu_count() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        return len([d for d in cuda_visible_devices.split(",") if d.strip()])
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
    """Return (last_step, last_loss, last_mem_gib, last_tps, last_tflops) or None."""
    last = None
    try:
        with open(log_path) as f:
            for line in f:
                m = _STEP_RE.search(_ANSI_RE.sub("", line))
                if m:
                    last = (
                        int(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                        int(m.group(4).replace(",", "")),
                        float(m.group(5).replace(",", "")),
                    )
    except FileNotFoundError:
        pass
    return last


def terminate_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def steps_for_total_tokens(
    total_tokens: int, local_batch: int, seq_len: int, fsdp: int
) -> int:
    tokens_per_step = local_batch * seq_len * fsdp
    return -(-total_tokens // tokens_per_step)


def _compile_flags(enabled: bool) -> list[str]:
    if not enabled:
        return []
    # torchtitan's CompileConfig has no `mode` field; apply_compile uses
    # default mode (block.compile(backend=..., fullgraph=True)).
    return ["--compile.enable", "--compile.components", "model"]


def _trainer_module(graph: bool) -> str:
    return "graph_trainer.llama3" if graph else "llama3"


def _trainer_config(config: str, graph: bool) -> str:
    return f"graph_trainer_{config}" if graph else config


def _trainer_label(graph: bool) -> str:
    return "GraphTrainer" if graph else "Trainer"


def _hf_assets_path_for_config(base_config: str) -> str | None:
    # 8B's default tokenizer path is the local HF mirror, which may be
    # unavailable in this environment. Fall back to the bundled tokenizer.
    if (
        base_config == "llama3_8b"
        and not (TORCHTITAN_DIR / "assets/hf/Llama-3.1-8B").exists()
    ):
        return "./tests/assets/tokenizer"
    return None


def _nvfp4_flags(nvfp4: bool) -> list[str]:
    if not nvfp4:
        return []
    # torchtitan registers a custom tyro rule that parses list[str] as
    # comma-separated values, not JSON. Pass the bare module path.
    return ["--override.modules", NVFP4_OVERRIDE_MODULE]


def _precision_tag(nvfp4: bool) -> str:
    return "nvfp4" if nvfp4 else "bf16"


def _plugin_env(extra: dict | None = None, *, tp_degree: int = 1) -> dict:
    env = {**os.environ}
    if extra:
        env.update(extra)
    env["PYTHONPATH"] = f"{PLUGIN_DIR}:{os.environ.get('PYTHONPATH', '')}"
    # Override factory reads this to require local-after-TP dim % 128 == 0.
    env["TORCHTITAN_AO_TP_DEGREE"] = str(tp_degree)
    return env


# ----------------------------------------------------------------------------
# single-GPU mode
# ----------------------------------------------------------------------------


def _single_cmd(
    module: str,
    config: str,
    steps: int,
    dataset: str,
    compile_enabled: bool,
    nvfp4: bool,
    hf_assets_path: str | None = None,
) -> list[str]:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        "-m",
        "torchtitan.train",
        "--module",
        module,
        "--config",
        config,
        "--training.local_batch_size",
        str(SINGLE_BATCH),
        "--training.seq_len",
        str(SEQ_LEN),
        "--training.steps",
        str(steps),
        "--optimizer.lr",
        str(LR),
        "--dataloader.dataset",
        dataset,
        "--metrics.log_freq",
        "10",
    ]
    if hf_assets_path is not None:
        cmd += ["--hf-assets-path", hf_assets_path]
    return cmd + _compile_flags(compile_enabled) + _nvfp4_flags(nvfp4)


def run_single(args):
    smoke = args.smoke
    base_config = "llama3_debugmodel" if smoke else "llama3_8b"
    module = _trainer_module(args.graph)
    config = _trainer_config(base_config, args.graph)
    dataset = "c4_test" if smoke else "c4"
    steps = 10 if smoke else SINGLE_STEPS_CEILING
    wall_hours = 5 / 60 if smoke else SINGLE_WALL_HOURS
    gpu = args.gpu

    precision = _precision_tag(args.nvfp4)
    label_parts = [precision]
    if args.graph:
        label_parts.append("graph")
    elif args.compile:
        label_parts.append("compile")
    label = "_".join(label_parts)
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"{ts}_titan_single_{label}.txt"

    print(f"\n{'='*60}")
    print(
        f"TorchTitan {precision} {_trainer_label(args.graph)} single — "
        f"{wall_hours:.4g}h wall clock"
    )
    print(f"Batch {SINGLE_BATCH} × seq {SEQ_LEN} = {SINGLE_BATCH * SEQ_LEN} tok/step")
    print(f"Steps ceiling: {steps:,}")
    print(f"Mode: {'SMOKE (debugmodel)' if smoke else 'FULL (8B)'}")
    print(f"Module: {module}")
    print(f"Config: {config}")
    print(
        "Compile: "
        + (
            "graph_trainer default (aot_fx_trace + passes)"
            if args.graph
            else ("on" if args.compile else "eager")
        )
    )
    print(f"GPU: {gpu}")
    print(f"{'='*60}\n")

    cmd = _single_cmd(
        module,
        config,
        steps,
        dataset,
        args.compile and not args.graph,
        args.nvfp4,
        _hf_assets_path_for_config(base_config),
    )
    env = _plugin_env({"CUDA_VISIBLE_DEVICES": str(gpu)}, tp_degree=1)
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_path}\n")

    proc = subprocess.Popen(
        cmd,
        cwd=TORCHTITAN_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    t = threading.Thread(
        target=stream_to_file, args=(proc, log_path, label), daemon=True
    )
    t.start()

    start = time.monotonic()
    deadline = start + wall_hours * 3600
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(15)
        else:
            print(f"\n{wall_hours:.4g}h wall clock reached — terminating.")
            terminate_process(proc)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — terminating.")
        terminate_process(proc)
        raise
    finally:
        t.join(timeout=5)

    print()
    _single_summary(log_path, smoke, label)
    if proc.returncode not in (0, None):
        raise SystemExit(proc.returncode)


def _single_summary(log_path: Path, smoke: bool, precision: str):
    tok_per_step = SINGLE_BATCH * SEQ_LEN
    header = (
        f"{'Experiment':<16} {'Steps':>8} {'Final Loss':>12} {'Tps':>10} "
        f"{'TFLOPs':>8} {'Mem(GiB)':>10} {'Total Tokens':>14}  Log"
    )
    sep = "-" * (len(header) + 10)

    print("=" * (len(header) + 10))
    print(" SUMMARY")
    print(sep)
    print(f" {header}")
    print(sep)

    result = parse_log(log_path)
    if result is None:
        print(f" {precision:<16} {'NO DATA':>8}  {log_path.name}")
    else:
        step, loss, mem, tps, tflops = result
        total_tokens = step * tok_per_step
        tok_str = f"{total_tokens / 1e6:.1f}M"
        warn = " <smoke floor" if (not smoke and step < SINGLE_SMOKE_FLOOR) else ""
        print(
            f" {precision:<16} {step:>8,} {loss:>12.4f} {tps:>10,} "
            f"{tflops:>8.2f} {mem:>10.2f} {tok_str:>14}  {log_path.name}{warn}"
        )

    print("=" * (len(header) + 10))
    print()


# ----------------------------------------------------------------------------
# multi-GPU mode
# ----------------------------------------------------------------------------


def _multi_cmd(
    exp: dict,
    steps: int,
    module: str,
    config: str,
    compile_enabled: bool,
    data: str,
    nvfp4: bool,
    hf_assets_path: str | None = None,
) -> list[str]:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(exp["tp"] * exp["fsdp"]),
        "--local-ranks-filter",
        "0",
        "-m",
        "torchtitan.train",
        "--module",
        module,
        "--config",
        config,
        "--parallelism.tensor_parallel_degree",
        str(exp["tp"]),
        "--parallelism.data_parallel_shard_degree",
        str(exp["fsdp"]),
        "--training.local_batch_size",
        str(exp["batch_size"]),
        "--training.seq_len",
        str(SEQ_LEN),
        "--training.steps",
        str(steps),
        "--optimizer.lr",
        str(LR),
        "--dataloader.dataset",
        data,
        "--metrics.log_freq",
        "10",
    ]
    if hf_assets_path is not None:
        cmd += ["--hf-assets-path", hf_assets_path]
    return cmd + _compile_flags(compile_enabled) + _nvfp4_flags(nvfp4)


def _multi_label(exp: dict, compile_enabled: bool, nvfp4: bool, graph: bool) -> str:
    parts = [exp["name"]]
    if nvfp4:
        parts.append("nvfp4")
    if graph:
        parts.append("graph")
    elif compile_enabled:
        parts.append("compile")
    return "_".join(parts)


def run_multi(args):
    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps must be positive")

    smoke = args.smoke
    # torchao NVFP4 quantization requires per-rank weight dims divisible by
    # 128. llama3_debugmodel (dim=256) collapses below that under TP, so we
    # use 8B for any --nvfp4 multi smoke run with a TP shape; the override
    # also skips linears whose local-after-TP shape would fail the
    # divisibility check (see TORCHTITAN_AO_TP_DEGREE below).
    needs_8b = (
        smoke
        and args.nvfp4
        and any(
            e["tp"] > 1
            for e in MULTI_EXPERIMENTS
            if args.only is None or e["name"] == args.only
        )
    )
    base_config = "llama3_8b" if needs_8b or not smoke else "llama3_debugmodel"
    module = _trainer_module(args.graph)
    config = _trainer_config(base_config, args.graph)
    if args.data is None:
        args.data = "c4_test" if smoke else "c4"
    wall_hours = MULTI_SMOKE_WALL_HOURS if smoke else MULTI_WALL_HOURS
    wall_seconds = wall_hours * 3600
    available_gpus = visible_gpu_count()

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    exps = [e for e in MULTI_EXPERIMENTS if args.only is None or e["name"] == args.only]
    if not exps:
        raise SystemExit(
            f"No experiment named {args.only!r}. "
            f"Choices: {[e['name'] for e in MULTI_EXPERIMENTS]}"
        )

    precision = _precision_tag(args.nvfp4)
    print()
    print("=" * 72)
    print(
        f"TorchTitan {precision} {_trainer_label(args.graph)} FSDP2 + TP — "
        f"{wall_hours:.4g}h wall clock per experiment"
    )
    print(f"Module: {module}")
    print(f"Config: {config}")
    batch_sizes = ", ".join(f"{e['name']}={e['batch_size']}" for e in exps)
    print(f"Local batch per DP replica: {batch_sizes}")
    print(f"Seq length: {SEQ_LEN}")
    if args.steps is not None:
        print(f"Steps (override): {args.steps:,}")
    elif smoke:
        print(f"Steps (smoke): {MULTI_SMOKE_STEPS}")
    else:
        print(f"Target total tokens: {args.total_tokens:,} (per-shape steps computed)")
    print(
        "Compile: "
        + (
            "graph_trainer default (aot_fx_trace + passes)"
            if args.graph
            else ("on" if args.compile else "eager")
        )
    )
    print(f"Data: {args.data}")
    print(f"Visible GPUs: {available_gpus}")
    print("=" * 72)
    print()

    results = []
    for exp in exps:
        world_size = exp["tp"] * exp["fsdp"]
        label = _multi_label(exp, args.compile, args.nvfp4, args.graph)
        log_path = RESULTS_DIR / f"{ts}_titan_multi_{label}.txt"

        if args.steps is not None:
            steps = args.steps
        elif smoke:
            steps = MULTI_SMOKE_STEPS
        else:
            steps = steps_for_total_tokens(
                args.total_tokens, exp["batch_size"], SEQ_LEN, exp["fsdp"]
            )

        if available_gpus < world_size and not args.allow_insufficient_gpus:
            print(
                f"  [{label}] skipped: needs {world_size} visible GPUs, found {available_gpus}"
            )
            results.append((exp, label, "SKIPPED", None, log_path, None, steps))
            continue

        hf_assets_path = _hf_assets_path_for_config(base_config)
        cmd = _multi_cmd(
            exp,
            steps,
            module,
            config,
            args.compile and not args.graph,
            args.data,
            args.nvfp4,
            hf_assets_path,
        )
        env = _plugin_env(
            {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
            tp_degree=exp["tp"],
        )
        print(
            f"  [{label}] world_size={world_size} batch={exp['batch_size']} steps={steps:,} -> {log_path}"
        )
        print(f"  [{label}] {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            cwd=TORCHTITAN_DIR,
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
        results.append((exp, label, status, proc.returncode, log_path, start, steps))
        print()

    _multi_summary(results, smoke)
    failures = [r for r in results if r[2].startswith("FAILED")]
    if failures:
        raise SystemExit(1)


def _multi_summary(results, smoke: bool):
    header = (
        f"{'Experiment':<24} {'Status':>12} {'Steps':>8} {'Final Loss':>12} "
        f"{'Tps':>10} {'TFLOPs':>8} {'Mem(GiB)':>10}  Log"
    )
    sep = "-" * (len(header) + 10)

    print("=" * (len(header) + 10))
    print(" SUMMARY")
    print(sep)
    print(f" {header}")
    print(sep)

    for exp, label, status, _rc, log_path, _start, _target_steps in results:
        parsed = parse_log(log_path)
        if parsed is None:
            print(f" {label:<24} {status:>12} {'NO DATA':>8}  {log_path.name}")
            continue

        step, loss, mem, tps, tflops = parsed
        warn = " <smoke floor" if (smoke and step < MULTI_SMOKE_STEPS) else ""
        print(
            f" {label:<24} {status:>12} {step:>8,} {loss:>12.4f} "
            f"{tps:>10,} {tflops:>8.2f} {mem:>10.2f}  {log_path.name}{warn}"
        )

    print("=" * (len(header) + 10))
    print()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="TorchTitan bf16 launcher (single + multi-GPU)"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single", help="One TorchTitan run on a chosen GPU")
    p_single.add_argument(
        "--smoke",
        action="store_true",
        help="5-min wall clock, llama3_debugmodel, 10 steps",
    )
    p_single.add_argument(
        "--gpu",
        type=int,
        default=1,
        metavar="N",
        help="GPU index (default 1; GPU 0 reserved for user)",
    )
    p_single.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Enable torch.compile (default mode) on each TransformerBlock "
            "for eager Trainer runs"
        ),
    )
    p_single.add_argument(
        "--graph",
        action="store_true",
        help="Use graph_trainer.llama3 with GraphTrainer defaults including CUDA graphs",
    )
    p_single.add_argument(
        "--nvfp4",
        action="store_true",
        help=f"Enable torchao NVFP4 via override module {NVFP4_OVERRIDE_MODULE}",
    )
    p_single.set_defaults(func=run_single)

    p_multi = sub.add_parser("multi", help="FSDP2 + TP shape sweep")
    p_multi.add_argument(
        "--smoke",
        action="store_true",
        help=f"10-min wall clock/shape, debugmodel, {MULTI_SMOKE_STEPS} steps",
    )
    p_multi.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="NAME",
        help=f"Run only this shape. Choices: {[e['name'] for e in MULTI_EXPERIMENTS]}",
    )
    p_multi.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Enable torch.compile (default mode) on each TransformerBlock "
            "for eager Trainer runs"
        ),
    )
    p_multi.add_argument(
        "--graph",
        action="store_true",
        help="Use graph_trainer.llama3 with GraphTrainer defaults including CUDA graphs",
    )
    p_multi.add_argument(
        "--data",
        type=str,
        default=None,
        help="--dataloader.dataset value (default: c4_test on --smoke, c4 otherwise)",
    )
    p_multi.add_argument(
        "--steps", type=int, default=None, help="Override per-shape step count"
    )
    p_multi.add_argument(
        "--total-tokens",
        type=int,
        default=MULTI_TOTAL_TOKENS,
        help=f"Token budget for non-smoke runs (default {MULTI_TOTAL_TOKENS:,})",
    )
    p_multi.add_argument(
        "--allow-insufficient-gpus",
        action="store_true",
        help="Launch torchrun even when visible GPU count < world size",
    )
    p_multi.add_argument(
        "--nvfp4",
        action="store_true",
        help=f"Enable torchao NVFP4 via override module {NVFP4_OVERRIDE_MODULE}",
    )
    p_multi.set_defaults(func=run_multi)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
