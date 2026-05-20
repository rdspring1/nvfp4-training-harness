"""
LLaMA 3 training with pure PyTorch — TorchAO quantization target.

This is the TorchAO-side counterpart to te_llama3_train.py. It uses no
TransformerEngine — only nn.Linear, nn.RMSNorm, F.scaled_dot_product_attention,
and manual RoPE. In bf16 mode it serves as a correctness baseline; with
--quantize it applies TorchAO's quantize_() to swap nn.Linear → NVFP4Linear.

Compare loss curves against te_llama3_train.py (the TE NVFP4 reference oracle)
to validate TorchAO components as they land upstream.

Usage:
    python ao_llama3_train.py --small --overfit          # bf16 baseline
    python ao_llama3_train.py --small --data wikitext    # bf16 + real data
    python ao_llama3_train.py --small --overfit --quantize nvfp4  # TorchAO NVFP4
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data (shared logic with te_llama3_train.py)
# ---------------------------------------------------------------------------
def load_tokenizer(data_source):
    """Load a tokenizer appropriate for the data source."""
    from transformers import AutoTokenizer

    if data_source == "wikitext":
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        return tok
    raise ValueError(f"Unknown data source: {data_source}")


def data_iterator(
    data_source, tokenizer, batch_size, seq_len, device, dp_rank=0, dp_size=1
):
    """Yield (input_ids, labels) batches from a real dataset, packed to seq_len.

    Cycles through the dataset indefinitely, re-shuffling each epoch with a
    different seed so repeated passes see different token orderings.
    """
    from datasets import load_dataset

    if not 0 <= dp_rank < dp_size:
        raise ValueError(
            f"Invalid data-parallel shard: dp_rank={dp_rank}, dp_size={dp_size}"
        )

    buffer = []
    epoch = 0
    while True:
        if data_source == "wikitext":
            ds = load_dataset(
                "wikitext", "wikitext-103-raw-v1", split="train", streaming=True
            )
            ds = ds.shuffle(seed=epoch, buffer_size=10_000)
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        epoch += 1

        doc_idx = 0
        for example in ds:
            text = example["text"].strip()
            if not text:
                continue
            if doc_idx % dp_size != dp_rank:
                doc_idx += 1
                continue
            doc_idx += 1
            tokens = tokenizer.encode(text)
            buffer.extend(tokens)

            while len(buffer) >= batch_size * (seq_len + 1):
                flat = buffer[: batch_size * (seq_len + 1)]
                buffer = buffer[batch_size * (seq_len + 1) :]
                t = torch.tensor(flat, dtype=torch.long, device=device).view(
                    batch_size, seq_len + 1
                )
                yield t[:, :seq_len], t[:, 1 : seq_len + 1]


# ---------------------------------------------------------------------------
# Model configurations (identical to te_llama3_train.py)
# ---------------------------------------------------------------------------
LLAMA3_8B = dict(
    hidden_size=4096,
    ffn_hidden_size=14336,
    num_attention_heads=32,
    num_kv_heads=8,
    num_layers=32,
    vocab_size=128256,
    max_seq_len=2048,
    rope_base=500000.0,
    rms_norm_eps=1e-5,
)

LLAMA_SMALL = dict(
    hidden_size=512,
    ffn_hidden_size=1280,
    num_attention_heads=8,
    num_kv_heads=4,
    num_layers=4,
    vocab_size=32000,
    max_seq_len=512,
    rope_base=500000.0,
    rms_norm_eps=1e-5,
)


# ---------------------------------------------------------------------------
# RoPE — manual implementation replacing te.RotaryPositionEmbedding
# ---------------------------------------------------------------------------
def build_rope_freqs(head_dim, max_seq_len, rope_base, device):
    """Precompute RoPE cos/sin tables: (max_seq_len, head_dim)."""
    freqs = 1.0 / (
        rope_base
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim/2)
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


def apply_rotary_pos_emb(x, cos, sin):
    """Apply RoPE to a (B, S, H, D) tensor."""
    S = x.size(1)
    cos = cos[:S].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
    sin = sin[:S].unsqueeze(0).unsqueeze(2)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# Model — pure PyTorch, no TransformerEngine dependency.
# nn.Linear layers are the quantize_() swap targets.
# ---------------------------------------------------------------------------
class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

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

    def forward(self, x, cos, sin):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Transpose to (B, H, S, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand KV heads for GQA: (B, num_kv_heads, S, D) → (B, num_heads, S, D)
        if self.num_kv_groups > 1:
            k = (
                k.unsqueeze(2)
                .expand(-1, -1, self.num_kv_groups, -1, -1)
                .reshape(B, self.num_heads, S, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(-1, -1, self.num_kv_groups, -1, -1)
                .reshape(B, self.num_heads, S, self.head_dim)
            )

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
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

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Llama(nn.Module):
    """LLaMA 3 model — pure PyTorch, ready for torchao.quantization.quantize_()."""

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

    def forward(self, input_ids, cos, sin, labels=None):
        h = self.embed_tokens(input_ids)

        for layer in self.layers:
            h = layer(h, cos, sin)

        h = self.norm(h)
        logits = self.lm_head(h)

        if labels is not None:
            return F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        return logits


def apply_quantization(model, quantize_mode, kernel):
    """Apply TorchAO quantization to nn.Linear layers."""
    if quantize_mode is None:
        return

    if quantize_mode == "nvfp4":
        from torchao.quantization import quantize_
        from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
            NVFP4TrainingConfig,
        )
        from torchao.quantization.quantize_.common.kernel_preference import (
            KernelPreference,
        )

        kernel_preferences = {
            "torch": KernelPreference.TORCH,
            "auto": KernelPreference.AUTO,
            "triton": KernelPreference.TRITON,
        }
        if hasattr(KernelPreference, "TE"):
            kernel_preferences["te"] = KernelPreference.TE
        elif kernel == "te":
            raise SystemExit("TorchAO KernelPreference.TE is not available")
        kp = kernel_preferences[kernel]
        quantize_(model, NVFP4TrainingConfig(kernel_preference=kp))
        print(
            f"Applied TorchAO NVFP4 training quantization to all nn.Linear layers (kernel={kernel.upper()})"
        )
    else:
        print(f"ERROR: Unknown quantize mode: {quantize_mode}")
        raise SystemExit(1)


def nvfp4_rht_sign_vectors(model):
    """Return every RHT sign vector that can be used by NVFP4 Triton kernels."""
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
        NVFP4Linear,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_tensor_parallel import (
        _TP_RHT_SIGN_VECTOR,
    )

    sign_vectors = [tuple(_TP_RHT_SIGN_VECTOR)]
    for module in model.modules():
        if isinstance(module, NVFP4Linear):
            sign_vectors.append(tuple(module.rht_sign_vector))
    return tuple(dict.fromkeys(sign_vectors))


def prepare_nvfp4_for_cuda_graph(model, device):
    """Prewarm NVFP4 persistent CUDA graph state for the model's sign vectors."""
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        prepare_for_cuda_graph,
    )

    sign_vectors = nvfp4_rht_sign_vectors(model)
    prepare_for_cuda_graph(torch.device(device), sign_vectors=sign_vectors)
    return sign_vectors


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLaMA 3 training — pure PyTorch + TorchAO"
    )
    parser.add_argument(
        "--small", action="store_true", help="Use small debug model instead of 8B"
    )
    parser.add_argument(
        "--overfit", action="store_true", help="Overfit on a single fixed batch"
    )
    parser.add_argument(
        "--data", type=str, default=None, choices=["wikitext"], help="Real dataset"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of training steps"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["nvfp4"],
        help="TorchAO quantization mode (default: none, bf16 baseline)",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="torch",
        choices=["torch", "te", "auto", "triton"],
        help="Quantization kernel backend (default: torch)",
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
        help="torch.compile mode (default: None = eager)",
    )
    args = parser.parse_args()

    config = LLAMA_SMALL if args.small else LLAMA3_8B

    tokenizer = None
    if args.data:
        tokenizer = load_tokenizer(args.data)
        # Pad vocab to multiple of 128 for Triton kernel alignment (M, K, N % 128 == 0)
        vocab = tokenizer.vocab_size
        vocab = (vocab + 127) // 128 * 128
        config = {**config, "vocab_size": vocab}

    seq_len = min(args.seq_len, config["max_seq_len"])
    dtype = torch.bfloat16
    device = "cuda"
    head_dim = config["hidden_size"] // config["num_attention_heads"]

    precision_str = (
        f"TorchAO {args.quantize} (master weights fp32, autocast bf16)"
        if args.quantize
        else "bf16 baseline (master weights fp32, autocast bf16)"
    )
    print(f"{'=' * 60}")
    print(f"LLaMA 3 Training — Pure PyTorch + TorchAO")
    print(f"{'=' * 60}")
    print(f"Model:      {'small (debug)' if args.small else '8B'}")
    print(f"Precision:  {precision_str}")
    print(f"Compile:    {args.compile or 'eager'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seq length: {seq_len}")
    print(f"Steps:      {args.steps}")
    data_mode = (
        "overfit (fixed batch)" if args.overfit else (args.data or "synthetic random")
    )
    print(f"Mode:       {data_mode}")
    print(f"Device:     {torch.cuda.get_device_name(0)}")
    print()

    # Build model — fp32 master weights, bf16 compute via autocast
    print("Building model...")
    model = Llama(config).to(device=device)

    # Apply quantization before optimizer so quantized params are optimized
    apply_quantization(model, args.quantize, args.kernel)

    _CUDA_GRAPH_MODES = {"reduce-overhead", "max-autotune"}
    if args.compile:
        if (
            args.quantize == "nvfp4"
            and args.kernel == "triton"
            and args.compile in _CUDA_GRAPH_MODES
        ):
            # Pre-allocate TMA workspace and warm RHT matrix lru_cache before graph
            # capture. Without this, the first kernel call inside the CUDA graph
            # allocates these persistent tensors from the graph pool and they are
            # flagged as untracked allocations at replay time.
            sign_vectors = prepare_nvfp4_for_cuda_graph(model, device)
            print(
                "Prepared NVFP4 CUDA graph state for "
                f"{len(sign_vectors)} RHT sign vector(s)"
            )
        fullgraph = args.compile in _CUDA_GRAPH_MODES
        print(f"Compiling model with mode={args.compile!r} fullgraph={fullgraph}...")
        if args.compile == "default":
            model = torch.compile(model)
        else:
            model = torch.compile(model, mode=args.compile, fullgraph=fullgraph)
        print("Compile done.")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params / 1e9:.2f}B)")
    print(f"Memory after init: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Precompute RoPE tables
    cos, sin = build_rope_freqs(
        head_dim, config["max_seq_len"], config["rope_base"], device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Set up data source
    if args.overfit:
        fixed_tokens = torch.randint(
            0, config["vocab_size"], (args.batch_size, seq_len + 1), device=device
        )
        real_data_iter = None
    elif args.data:
        real_data_iter = data_iterator(
            args.data, tokenizer, args.batch_size, seq_len, device
        )
    else:
        real_data_iter = None

    # Training loop
    print(f"\nTraining...")
    print(f"{'Step':>5} {'Loss':>10} {'Tok/s':>10} {'Tokens':>10} {'Mem (GB)':>10}")
    print("-" * 52)

    model.train()
    for step in range(args.steps):
        t0 = time.time()

        if args.overfit:
            input_ids, labels = fixed_tokens[:, :seq_len], fixed_tokens[:, 1:]
        elif real_data_iter is not None:
            input_ids, labels = next(real_data_iter)
        else:
            tokens = torch.randint(
                0, config["vocab_size"], (args.batch_size, seq_len + 1), device=device
            )
            input_ids, labels = tokens[:, :seq_len], tokens[:, 1:]

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            loss = model(input_ids, cos, sin, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.time() - t0
        tok_per_sec = args.batch_size * seq_len / dt
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        tokens_seen = (step + 1) * args.batch_size * seq_len
        tok_str = (
            f"{tokens_seen/1e6:.1f}M"
            if tokens_seen >= 1_000_000
            else f"{tokens_seen/1e3:.0f}K"
        )
        print(
            f"{step:5d} {loss.item():10.4f} {tok_per_sec:10.0f} {tok_str:>10} {mem_gb:10.2f}"
        )

    print(f"\nTraining complete. Final loss: {loss.item():.4f}")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    os._exit(0)


if __name__ == "__main__":
    main()
