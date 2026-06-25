#!/usr/bin/env python3
"""
LLaMA 3 training with BF16/TorchAO NVFP4, tensor parallelism, and FSDP2.

The model uses sequence-parallel tensor parallelism for the NVFP4 Triton
linear layers:

    torchrun --standalone --nproc_per_node=4 ao_llama3_fsdp2_tp_train.py \
        --tp-size 2 --fsdp-size 2 --steps 10 --quantize nvfp4 --kernel triton

For BF16 baseline coverage:

    torchrun --standalone --nproc_per_node=4 ao_llama3_fsdp2_tp_train.py \
        --tp-size 2 --fsdp-size 2 --steps 10 --quantize none

For compile coverage:

    torchrun --standalone --nproc_per_node=4 ao_llama3_fsdp2_tp_train.py \
        --tp-size 2 --fsdp-size 2 --steps 10 --quantize nvfp4 --kernel triton \
        --compile reduce-overhead
"""

import argparse
import contextlib
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from ao_llama3_train import (
    LLAMA3_8B,
    LLAMA_SMALL,
    apply_rotary_pos_emb,
    build_rope_freqs,
    data_iterator,
    load_tokenizer,
    prepare_nvfp4_for_cuda_graph,
)


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, x, cos, sin, batch_size: int, seq_len: int):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        full_tokens = batch_size * seq_len
        if q.size(0) != full_tokens:
            raise RuntimeError(
                f"TP attention expected {full_tokens} gathered tokens, got {q.size(0)}"
            )

        local_num_heads = q.size(-1) // self.head_dim
        local_num_kv_heads = k.size(-1) // self.head_dim
        if local_num_heads % local_num_kv_heads != 0:
            raise RuntimeError(
                "Local query heads must be divisible by local KV heads: "
                f"{local_num_heads=} {local_num_kv_heads=}"
            )
        local_num_kv_groups = local_num_heads // local_num_kv_heads

        q = q.reshape(batch_size, seq_len, local_num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, local_num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, local_num_kv_heads, self.head_dim)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if local_num_kv_groups > 1:
            k = (
                k.unsqueeze(2)
                .expand(-1, -1, local_num_kv_groups, -1, -1)
                .reshape(batch_size, local_num_heads, seq_len, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(-1, -1, local_num_kv_groups, -1, -1)
                .reshape(batch_size, local_num_heads, seq_len, self.head_dim)
            )

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .reshape(full_tokens, local_num_heads * self.head_dim)
        )
        return self.o_proj(attn_out)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config["hidden_size"], config["ffn_hidden_size"], bias=False
        )
        self.up_proj = nn.Linear(
            config["hidden_size"], config["ffn_hidden_size"], bias=False
        )
        self.down_proj = nn.Linear(
            config["ffn_hidden_size"], config["hidden_size"], bias=False
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )
        self.mlp = LlamaMLP(config)

    def forward(self, x, cos, sin, batch_size: int, seq_len: int):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, batch_size, seq_len)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            [LlamaBlock(config) for _ in range(config["num_layers"])]
        )
        self.norm = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.lm_head = nn.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )

    def forward(self, input_ids, cos, sin, labels=None, batch_size=None, seq_len=None):
        if batch_size is None or seq_len is None:
            raise RuntimeError("batch_size and seq_len are required for TP attention")

        h = self.embed_tokens(input_ids.reshape(-1))
        for layer in self.layers:
            h = layer(h, cos, sin, batch_size, seq_len)

        h = self.norm(h)
        logits = self.lm_head(h)
        if labels is not None:
            return F.cross_entropy(logits.float(), labels.reshape(-1))
        return logits


def rank0_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def setup_distributed(tp_size: int, fsdp_size: int) -> DeviceMesh:
    expected_world_size = tp_size * fsdp_size
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != expected_world_size:
        raise SystemExit(
            f"Expected WORLD_SIZE={expected_world_size} for "
            f"tp={tp_size}, fsdp={fsdp_size}; got WORLD_SIZE={world_size}. "
            f"Launch with torchrun --nproc_per_node={expected_world_size}."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device_mesh = init_device_mesh(
        "cuda", (fsdp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    torch.manual_seed(1)
    return device_mesh


def apply_nvfp4_quantization(model: nn.Module, kernel: str) -> None:
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
        NVFP4TrainingConfig,
    )
    from torchao.quantization import quantize_
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    kernel_preferences = {
        "torch": KernelPreference.TORCH,
        "auto": KernelPreference.AUTO,
        "triton": KernelPreference.TRITON,
    }
    if hasattr(KernelPreference, "TE"):
        kernel_preferences["te"] = KernelPreference.TE
    elif kernel == "te":
        raise SystemExit("TorchAO KernelPreference.TE is not available")
    kernel_preference = kernel_preferences[kernel]
    quantize_(model, NVFP4TrainingConfig(kernel_preference=kernel_preference))
    rank0_print(
        "Applied TorchAO NVFP4 training quantization "
        f"to all nn.Linear layers (kernel={kernel.upper()})"
    )


def validate_tp_shapes(
    config: dict, tp_size: int, batch_size: int, seq_len: int, quantize: str
) -> None:
    checks = {
        "hidden_size": config["hidden_size"],
        "ffn_hidden_size": config["ffn_hidden_size"],
        "num_attention_heads": config["num_attention_heads"],
        "num_kv_heads": config["num_kv_heads"],
    }
    for name, value in checks.items():
        if value % tp_size != 0:
            raise SystemExit(f"{name}={value} must be divisible by tp_size={tp_size}")

    if quantize != "nvfp4":
        return

    total_tokens = batch_size * seq_len
    local_tokens = total_tokens // tp_size
    if total_tokens % tp_size != 0:
        raise SystemExit(
            f"batch_size * seq_len must be divisible by tp_size: "
            f"{batch_size} * {seq_len} vs {tp_size}"
        )
    if local_tokens % 128 != 0:
        raise SystemExit(
            "NVFP4 TP kernels require local flattened tokens to be a multiple "
            f"of 128, got {local_tokens}"
        )


def apply_tensor_parallel(model: Llama, tp_mesh: DeviceMesh, quantize: str) -> Llama:
    if tp_mesh.size() == 1:
        return model

    if quantize == "nvfp4":
        from torchao.prototype.moe_training.nvfp4_training.nvfp4_tensor_parallel import (
            NVFP4ColwiseParallel,
            NVFP4RowwiseParallel,
        )

        colwise_parallel = NVFP4ColwiseParallel
        rowwise_parallel = NVFP4RowwiseParallel
    else:
        colwise_parallel = ColwiseParallel
        rowwise_parallel = RowwiseParallel

    tp_plan = {}
    for layer_idx in range(len(model.layers)):
        prefix = f"layers.{layer_idx}"
        tp_plan[f"{prefix}.self_attn.q_proj"] = colwise_parallel()
        tp_plan[f"{prefix}.self_attn.k_proj"] = colwise_parallel()
        tp_plan[f"{prefix}.self_attn.v_proj"] = colwise_parallel()
        tp_plan[f"{prefix}.self_attn.o_proj"] = rowwise_parallel()
        tp_plan[f"{prefix}.mlp.gate_proj"] = colwise_parallel()
        tp_plan[f"{prefix}.mlp.up_proj"] = colwise_parallel()
        tp_plan[f"{prefix}.mlp.down_proj"] = rowwise_parallel()

    return parallelize_module(model, tp_mesh, tp_plan)


def apply_fsdp2(model: nn.Module, dp_mesh: DeviceMesh) -> nn.Module:
    if dp_mesh.size() == 1:
        return model
    return fully_shard(model, mesh=dp_mesh)


def shard_flat_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_input = input_ids.reshape(-1)
    flat_labels = labels.reshape(-1)
    local_tokens = flat_input.numel() // tp_size
    start = tp_rank * local_tokens
    end = start + local_tokens
    return flat_input[start:end].contiguous(), flat_labels[start:end].contiguous()


def synthetic_batch(
    config: dict,
    batch_size: int,
    seq_len: int,
    dp_rank: int,
    step: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + dp_rank * 10_000 + step)
    tokens = torch.randint(
        0,
        config["vocab_size"],
        (batch_size, seq_len + 1),
        device=device,
        generator=generator,
    )
    return tokens[:, :seq_len], tokens[:, 1:]


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def reduce_max_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def optimizer_param_groups(model: nn.Module) -> list[dict]:
    from torch.distributed.tensor import DTensor

    local_params = []
    dtensor_params = []
    for param in model.parameters():
        if isinstance(param, DTensor):
            dtensor_params.append(param)
        else:
            local_params.append(param)

    groups = []
    if local_params:
        groups.append({"params": local_params, "foreach": True})
    if dtensor_params:
        groups.append({"params": dtensor_params, "foreach": True})
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="LLaMA 3 training - BF16/TorchAO NVFP4 + FSDP2 + TP"
    )
    parser.add_argument("--small", action="store_true", help="Use small debug model")
    parser.add_argument("--overfit", action="store_true", help="Reuse one fixed batch")
    parser.add_argument("--data", type=str, default=None, choices=["wikitext"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--fsdp-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--quantize",
        type=str,
        default="nvfp4",
        choices=["none", "nvfp4"],
        help="Quantization mode. Use 'none' for the BF16 baseline.",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="triton",
        choices=["torch", "te", "auto", "triton"],
    )
    parser.add_argument(
        "--compile",
        type=str,
        default=None,
        choices=[
            "reduce-overhead",
            "default",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
    )
    parser.add_argument(
        "--profile-start",
        type=int,
        default=None,
        help="Start PyTorch profiler after this many training steps.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Number of training steps to capture with PyTorch profiler.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="traces",
        help="Directory for per-rank profiler traces and key-average summaries.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.profile_steps < 0:
        raise SystemExit("--profile-steps must be non-negative")
    if args.profile_steps and args.profile_start is None:
        raise SystemExit("--profile-start is required when --profile-steps is set")
    if args.profile_start is not None and args.profile_start < 0:
        raise SystemExit("--profile-start must be non-negative")

    torch.set_float32_matmul_precision("high")
    mesh = setup_distributed(args.tp_size, args.fsdp_size)
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]
    dp_rank, tp_rank = mesh.get_coordinate()
    device = torch.device("cuda")

    try:
        config = dict(LLAMA_SMALL if args.small else LLAMA3_8B)
        tokenizer = None
        if args.data:
            tokenizer = load_tokenizer(args.data)
            vocab = tokenizer.vocab_size
            config["vocab_size"] = (vocab + 127) // 128 * 128

        seq_len = min(args.seq_len, config["max_seq_len"])
        validate_tp_shapes(
            config, args.tp_size, args.batch_size, seq_len, args.quantize
        )
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        if args.quantize == "nvfp4":
            precision_str = "TorchAO nvfp4 (master weights fp32, autocast bf16)"
        else:
            precision_str = "BF16 baseline (master weights fp32, autocast bf16)"
        mode = "overfit (fixed synthetic batch)"
        if not args.overfit:
            mode = args.data or "synthetic random"

        rank0_print("=" * 72)
        rank0_print("LLaMA 3 Training - BF16/TorchAO NVFP4 + FSDP2 + TP")
        rank0_print("=" * 72)
        rank0_print(f"Model:      {'small (debug)' if args.small else '8B'}")
        rank0_print(f"Precision:  {precision_str}")
        rank0_print(f"Kernel:     {args.kernel}")
        rank0_print(f"Compile:    {args.compile or 'eager'}")
        rank0_print(f"TP/FSDP2:   {args.tp_size} x {args.fsdp_size}")
        rank0_print(f"Batch size: {args.batch_size} per DP replica")
        rank0_print(f"Seq length: {seq_len}")
        rank0_print(f"Steps:      {args.steps}")
        rank0_print(f"Mode:       {mode}")
        rank0_print(
            f"Device:     {torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
        rank0_print()

        rank0_print("Building model...")
        model = Llama(config).to(device=device)

        if args.quantize == "nvfp4":
            apply_nvfp4_quantization(model, args.kernel)
        else:
            rank0_print(
                "Using BF16 autocast baseline; nn.Linear layers remain unquantized"
            )

        model = apply_tensor_parallel(model, tp_mesh, args.quantize)
        if args.tp_size > 1:
            rank0_print("Applied tensor parallel plan to attention and MLP projections")
        else:
            rank0_print("Tensor parallel disabled (tp_size=1)")

        if args.compile:
            if (
                args.quantize == "nvfp4"
                and args.kernel == "triton"
                and args.compile in {"reduce-overhead", "max-autotune"}
            ):
                sign_vectors = prepare_nvfp4_for_cuda_graph(model, device)
                rank0_print(
                    "Prepared NVFP4 CUDA graph state for "
                    f"{len(sign_vectors)} RHT sign vector(s)"
                )
            rank0_print(f"Compiling model with mode={args.compile!r}...")
            if args.compile == "default":
                model = torch.compile(model)
            else:
                model = torch.compile(model, mode=args.compile)
            rank0_print("Compile done.")

        model = apply_fsdp2(model, dp_mesh)
        if args.fsdp_size > 1:
            rank0_print("Applied FSDP2 over the DP mesh")
        else:
            rank0_print("FSDP2 disabled (fsdp_size=1)")

        num_params = sum(p.numel() for p in model.parameters())
        rank0_print(f"Local parameter handles: {num_params:,}")
        init_mem = reduce_max_float(torch.cuda.max_memory_allocated() / 1e9, device)
        rank0_print(f"Max memory after init: {init_mem:.2f} GB")

        cos, sin = build_rope_freqs(
            head_dim, config["max_seq_len"], config["rope_base"], device
        )
        optimizer = torch.optim.AdamW(
            optimizer_param_groups(model),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        fixed_batch = None
        real_data_iter = None
        if args.overfit:
            fixed_batch = synthetic_batch(
                config, args.batch_size, seq_len, dp_rank, 0, device, args.seed
            )
        elif args.data:
            real_data_iter = data_iterator(
                args.data,
                tokenizer,
                args.batch_size,
                seq_len,
                device,
                dp_rank=dp_rank,
                dp_size=args.fsdp_size,
            )

        rank0_print()
        rank0_print("Training...")
        rank0_print(
            f"{'Step':>5} {'Loss':>10} {'Tok/s':>10} {'Tokens':>10} {'Mem (GB)':>10}"
        )
        rank0_print("-" * 52)

        model.train()
        dtype = torch.bfloat16
        loss = None
        profiler = None
        if args.profile_steps:
            from torch.profiler import (
                ProfilerActivity,
                profile,
                schedule,
                tensorboard_trace_handler,
            )

            rank = dist.get_rank()
            rank_trace_dir = os.path.join(args.profile_dir, f"rank{rank}")
            os.makedirs(rank_trace_dir, exist_ok=True)
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=args.profile_start,
                    warmup=0,
                    active=args.profile_steps,
                    repeat=1,
                ),
                on_trace_ready=tensorboard_trace_handler(rank_trace_dir),
                record_shapes=True,
                profile_memory=True,
            )
            rank0_print(
                "Profiler: "
                f"start={args.profile_start}, steps={args.profile_steps}, "
                f"dir={args.profile_dir}"
            )

        dist.barrier()
        with profiler if profiler is not None else contextlib.nullcontext():
            for step in range(args.steps):
                t0 = time.time()

                if fixed_batch is not None:
                    input_full, labels_full = fixed_batch
                elif real_data_iter is not None:
                    input_full, labels_full = next(real_data_iter)
                else:
                    input_full, labels_full = synthetic_batch(
                        config,
                        args.batch_size,
                        seq_len,
                        dp_rank,
                        step,
                        device,
                        args.seed,
                    )

                if args.quantize == "nvfp4":
                    input_batch, labels_batch = shard_flat_batch(
                        input_full, labels_full, tp_rank, args.tp_size
                    )
                else:
                    input_batch, labels_batch = input_full, labels_full

                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    loss = model(
                        input_batch,
                        cos,
                        sin,
                        labels_batch,
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                    )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                torch.cuda.synchronize()
                mean_loss = reduce_mean(loss.detach().float())
                mem_gb = reduce_max_float(
                    torch.cuda.max_memory_allocated() / 1e9, device
                )
                dt = time.time() - t0
                tok_per_sec = args.batch_size * seq_len * args.fsdp_size / dt
                tokens_seen = (step + 1) * args.batch_size * seq_len * args.fsdp_size
                tok_str = (
                    f"{tokens_seen / 1e6:.1f}M"
                    if tokens_seen >= 1_000_000
                    else f"{tokens_seen / 1e3:.0f}K"
                )
                rank0_print(
                    f"{step:5d} {mean_loss.item():10.4f} {tok_per_sec:10.0f} "
                    f"{tok_str:>10} {mem_gb:10.2f}"
                )
                if profiler is not None:
                    profiler.step()

        if profiler is not None:
            rank = dist.get_rank()
            summary_path = os.path.join(
                args.profile_dir, f"rank{rank}_key_averages.txt"
            )
            with open(summary_path, "w") as f:
                f.write(
                    profiler.key_averages().table(
                        sort_by="cuda_time_total", row_limit=80
                    )
                )
                f.write("\n\n")
                f.write(
                    profiler.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=80
                    )
                )
            rank0_print(
                f"Profiler summaries written to {args.profile_dir}/"
                "rank*_key_averages.txt"
            )

        final_loss = loss.detach().float()
        final_loss = reduce_mean(final_loss)
        peak_mem = reduce_max_float(torch.cuda.max_memory_allocated() / 1e9, device)
        rank0_print()
        rank0_print(f"Training complete. Final loss: {final_loss.item():.4f}")
        rank0_print(f"Peak memory: {peak_mem:.2f} GB")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
