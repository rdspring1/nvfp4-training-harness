#!/usr/bin/env python3
"""
LLaMA 3 training with TransformerEngine, tensor parallelism, and FSDP2.

The TP path uses TE-native sequence parallel linear layers:

    torchrun --standalone --nproc_per_node=4 te_llama3_fsdp2_tp_train.py \
        --tp-size 2 --fsdp-size 2 --steps 10 --precision nvfp4

The model keeps high-precision trainable parameters and applies TE low-precision
recipes through te.autocast(), matching te_llama3_train.py.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor
from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling
from transformer_engine.pytorch.attention.multi_head_attention import (
    apply_rotary_pos_emb,
)

from te_llama3_train import LLAMA3_8B, LLAMA_SMALL, data_iterator, load_tokenizer


class LlamaAttention(nn.Module):
    def __init__(self, config: dict, layer_number: int, tp_group, tp_size: int):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.tp_size = tp_size
        self.local_num_heads = self.num_heads // tp_size
        self.local_num_kv_heads = self.num_kv_heads // tp_size

        self.q_proj = te_parallel_linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            tp_group,
            tp_size,
            "column",
        )
        self.k_proj = te_parallel_linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            tp_group,
            tp_size,
            "column",
        )
        self.v_proj = te_parallel_linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            tp_group,
            tp_size,
            "column",
        )
        self.o_proj = te_parallel_linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            tp_group,
            tp_size,
            "row",
        )

        self.core_attn = te.DotProductAttention(
            num_attention_heads=self.local_num_heads,
            kv_channels=self.head_dim,
            num_gqa_groups=self.local_num_kv_heads,
            attention_dropout=0.0,
            attn_mask_type="causal",
            qkv_format="bshd",
            layer_number=layer_number,
        )

    def forward(self, x, rotary_pos_emb, batch_size: int, seq_len: int):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        full_tokens = batch_size * seq_len
        if q.size(0) != full_tokens:
            raise RuntimeError(
                f"TP attention expected {full_tokens} gathered tokens, got {q.size(0)}"
            )

        q = q.view(batch_size, seq_len, self.local_num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.local_num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.local_num_kv_heads, self.head_dim)

        q = apply_rotary_pos_emb(q, rotary_pos_emb, tensor_format="bshd", fused=True)
        k = apply_rotary_pos_emb(k, rotary_pos_emb, tensor_format="bshd", fused=True)

        attn_out = self.core_attn(q, k, v)
        attn_out = attn_out.view(full_tokens, self.local_num_heads * self.head_dim)
        return self.o_proj(attn_out)


class LlamaMLP(nn.Module):
    def __init__(self, config: dict, tp_group, tp_size: int):
        super().__init__()
        self.gate_proj = te_parallel_linear(
            config["hidden_size"],
            config["ffn_hidden_size"],
            tp_group,
            tp_size,
            "column",
        )
        self.up_proj = te_parallel_linear(
            config["hidden_size"],
            config["ffn_hidden_size"],
            tp_group,
            tp_size,
            "column",
        )
        self.down_proj = te_parallel_linear(
            config["ffn_hidden_size"],
            config["hidden_size"],
            tp_group,
            tp_size,
            "row",
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, config: dict, layer_number: int, tp_group, tp_size: int):
        super().__init__()
        self.input_layernorm = te.RMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )
        self.self_attn = LlamaAttention(config, layer_number, tp_group, tp_size)
        self.post_attention_layernorm = te.RMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )
        self.mlp = LlamaMLP(config, tp_group, tp_size)

    def forward(self, x, rotary_pos_emb, batch_size: int, seq_len: int):
        x = x + self.self_attn(
            self.input_layernorm(x), rotary_pos_emb, batch_size, seq_len
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TELlama(nn.Module):
    def __init__(self, config: dict, tp_group, tp_size: int):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.rope = te.RotaryPositionEmbedding(
            dim=config["hidden_size"] // config["num_attention_heads"],
            rotary_base=config["rope_base"],
        )
        self.layers = nn.ModuleList(
            [
                LlamaBlock(
                    config, layer_number=i + 1, tp_group=tp_group, tp_size=tp_size
                )
                for i in range(config["num_layers"])
            ]
        )
        self.norm = te.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.lm_head = te.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )

    def forward(self, input_ids, labels=None, batch_size=None, seq_len=None):
        if batch_size is None or seq_len is None:
            raise RuntimeError("batch_size and seq_len are required for TP attention")

        h = self.embed_tokens(input_ids.reshape(-1))
        rotary_pos_emb = self.rope(seq_len)

        for layer in self.layers:
            h = layer(h, rotary_pos_emb, batch_size, seq_len)

        h = self.norm(h)
        logits = self.lm_head(h)
        if labels is not None:
            return F.cross_entropy(logits.float(), labels.reshape(-1))
        return logits


def te_parallel_linear(
    in_features: int,
    out_features: int,
    tp_group,
    tp_size: int,
    parallel_mode: str,
) -> te.Linear:
    kwargs = {"bias": False}
    if tp_size > 1:
        kwargs.update(
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode=parallel_mode,
            sequence_parallel=True,
        )
    return te.Linear(in_features, out_features, **kwargs)


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


def validate_tp_shapes(
    config: dict, tp_size: int, batch_size: int, seq_len: int
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

    total_tokens = batch_size * seq_len
    if total_tokens % tp_size != 0:
        raise SystemExit(
            f"batch_size * seq_len must be divisible by tp_size: "
            f"{batch_size} * {seq_len} vs {tp_size}"
        )
    local_tokens = total_tokens // tp_size
    if local_tokens % 128 != 0:
        raise SystemExit(
            "TE TP kernels require local flattened tokens to be a multiple "
            f"of 128, got {local_tokens}"
        )


def save_custom_attrs(module: nn.Module) -> dict:
    custom_attrs = {}
    for name, param in module.named_parameters():
        custom_attrs[name] = dict(vars(param))
    return custom_attrs


def restore_custom_attrs(module: nn.Module, custom_attrs: dict) -> None:
    for name, param in module.named_parameters():
        if name not in custom_attrs:
            continue
        for attr_name, attr_value in custom_attrs[name].items():
            setattr(param, attr_name, attr_value)


def apply_fsdp2(model: TELlama, dp_mesh: DeviceMesh) -> TELlama:
    if dp_mesh.size() == 1:
        return model

    custom_attrs = save_custom_attrs(model)
    fully_shard(model.embed_tokens, mesh=dp_mesh)
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model.norm, mesh=dp_mesh)
    fully_shard(model.lm_head, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)
    restore_custom_attrs(model, custom_attrs)
    return model


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


def recipe_for_precision(precision: str):
    return NVFP4BlockScaling() if precision == "nvfp4" else MXFP8BlockScaling()


def main():
    parser = argparse.ArgumentParser(
        description="LLaMA 3 training - TransformerEngine + FSDP2 + TP"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="nvfp4",
        choices=["nvfp4", "nvpf4", "mxfp8", "mxpf8"],
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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    precision = {
        "nvfp4": "nvfp4",
        "nvpf4": "nvfp4",
        "mxfp8": "mxfp8",
        "mxpf8": "mxfp8",
    }[args.precision]
    recipe = recipe_for_precision(precision)

    torch.set_float32_matmul_precision("high")
    mesh = setup_distributed(args.tp_size, args.fsdp_size)
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]
    dp_rank, tp_rank = mesh.get_coordinate()
    tp_group = tp_mesh.get_group() if args.tp_size > 1 else None
    device = torch.device("cuda")

    try:
        config = dict(LLAMA_SMALL if args.small else LLAMA3_8B)
        tokenizer = None
        if args.data:
            tokenizer = load_tokenizer(args.data)
            vocab_multiple = 16 if precision == "nvfp4" else 32
            vocab = tokenizer.vocab_size
            config["vocab_size"] = (
                (vocab + vocab_multiple - 1) // vocab_multiple * vocab_multiple
            )

        seq_len = min(args.seq_len, config["max_seq_len"])
        validate_tp_shapes(config, args.tp_size, args.batch_size, seq_len)

        mode = "overfit (fixed synthetic batch)"
        if not args.overfit:
            mode = args.data or "synthetic random"

        rank0_print("=" * 72)
        rank0_print("LLaMA 3 Training - TransformerEngine + FSDP2 + TP")
        rank0_print("=" * 72)
        rank0_print(f"Model:      {'small (debug)' if args.small else '8B'}")
        rank0_print(
            f"Precision:  {precision.upper()} " "(master weights fp32, autocast bf16)"
        )
        rank0_print(f"Recipe:     {recipe}")
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
        model = TELlama(config, tp_group=tp_group, tp_size=args.tp_size).to(
            device=device
        )

        if args.tp_size > 1:
            rank0_print("Enabled TE sequence-parallel tensor parallel linears")
        else:
            rank0_print("Tensor parallel disabled (tp_size=1)")

        model = apply_fsdp2(model, dp_mesh)
        if args.fsdp_size > 1:
            rank0_print("Applied FSDP2 over the DP mesh")
        else:
            rank0_print("FSDP2 disabled (fsdp_size=1)")

        if args.compile:
            rank0_print(f"Compiling model with mode={args.compile!r}...")
            if args.compile == "default":
                model = torch.compile(model)
            else:
                model = torch.compile(model, mode=args.compile)
            rank0_print("Compile done.")

        num_params = sum(p.numel() for p in model.parameters())
        rank0_print(f"Local parameter handles: {num_params:,}")
        init_mem = reduce_max_float(torch.cuda.max_memory_allocated() / 1e9, device)
        rank0_print(f"Max memory after init: {init_mem:.2f} GB")

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
        autocast_kwargs = {}
        if tp_group is not None:
            autocast_kwargs["amax_reduction_group"] = tp_group

        dist.barrier()
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

            input_local, labels_local = shard_flat_batch(
                input_full, labels_full, tp_rank, args.tp_size
            )

            amp_context = torch.amp.autocast(device_type="cuda", dtype=dtype)
            with amp_context:
                with te.autocast(enabled=True, recipe=recipe, **autocast_kwargs):
                    loss = model(
                        input_local,
                        labels_local,
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                    )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            mean_loss = reduce_mean(loss.detach().float())
            mem_gb = reduce_max_float(torch.cuda.max_memory_allocated() / 1e9, device)
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

        final_loss = loss.detach().float()
        final_loss = reduce_mean(final_loss)
        peak_mem = reduce_max_float(torch.cuda.max_memory_allocated() / 1e9, device)
        rank0_print()
        rank0_print(f"Training complete. Final loss: {final_loss.item():.4f}")
        rank0_print(f"Peak memory: {peak_mem:.2f} GB")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
