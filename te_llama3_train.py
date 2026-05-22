"""
LLaMA 3 low-precision training with TransformerEngine.

Trains using TE's NVFP4BlockScaling recipe: 4-bit quantized GEMMs with
stochastic rounding (gradients), random Hadamard transform (inputs/gradients),
and 2D block quantization (weights). Master weights stay in fp32, non-TE ops
run in bf16 via torch.amp.autocast.

Also supports TE's MXFP8BlockScaling recipe through --precision mxfp8.

Usage:
    python te_llama3_train.py --small --overfit          # Overfit test
    python te_llama3_train.py --small --data wikitext   # Real data, small model
    python te_llama3_train.py --data wikitext             # Full 8B + real data
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling
from transformer_engine.pytorch.attention.multi_head_attention import (
    apply_rotary_pos_emb,
)


# ---------------------------------------------------------------------------
# Data
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
                "Salesforce/wikitext",
                "wikitext-103-raw-v1",
                split="train",
                streaming=True,
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
# Model configurations
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
    ffn_hidden_size=1376,
    num_attention_heads=8,
    num_kv_heads=4,
    num_layers=4,
    vocab_size=32000,
    max_seq_len=512,
    rope_base=500000.0,
    rms_norm_eps=1e-5,
)


# ---------------------------------------------------------------------------
# Model — built from TE primitives for component-level swappability.
# Each te.Linear can be individually replaced with a TorchAO equivalent.
# ---------------------------------------------------------------------------
class LlamaAttention(nn.Module):
    def __init__(self, config, layer_number):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = self.hidden_size // self.num_heads

        # Individual projections — each is a separately swappable te.Linear
        self.q_proj = te.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = te.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = te.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = te.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # TE's fused attention kernel (Flash Attention / cuDNN)
        self.core_attn = te.DotProductAttention(
            num_attention_heads=self.num_heads,
            kv_channels=self.head_dim,
            num_gqa_groups=self.num_kv_heads,
            attention_dropout=0.0,
            attn_mask_type="causal",
            qkv_format="bshd",
            layer_number=layer_number,
        )

    def forward(self, x, rotary_pos_emb):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q = apply_rotary_pos_emb(q, rotary_pos_emb, tensor_format="bshd", fused=True)
        k = apply_rotary_pos_emb(k, rotary_pos_emb, tensor_format="bshd", fused=True)

        attn_out = self.core_attn(q, k, v)
        attn_out = attn_out.view(B, S, -1)
        return self.o_proj(attn_out)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU: gate_proj and up_proj are separate for swappability
        self.gate_proj = te.Linear(
            config["hidden_size"], config["ffn_hidden_size"], bias=False
        )
        self.up_proj = te.Linear(
            config["hidden_size"], config["ffn_hidden_size"], bias=False
        )
        self.down_proj = te.Linear(
            config["ffn_hidden_size"], config["hidden_size"], bias=False
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, config, layer_number):
        super().__init__()
        self.input_layernorm = te.RMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )
        self.self_attn = LlamaAttention(config, layer_number)
        self.post_attention_layernorm = te.RMSNorm(
            config["hidden_size"], eps=config["rms_norm_eps"]
        )
        self.mlp = LlamaMLP(config)

    def forward(self, x, rotary_pos_emb):
        x = x + self.self_attn(self.input_layernorm(x), rotary_pos_emb)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TELlama(nn.Module):
    """LLaMA 3 model built from TE primitives (te.Linear, te.RMSNorm, te.DotProductAttention)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])

        self.rope = te.RotaryPositionEmbedding(
            dim=config["hidden_size"] // config["num_attention_heads"],
            rotary_base=config["rope_base"],
        )

        self.layers = nn.ModuleList(
            [
                LlamaBlock(config, layer_number=i + 1)
                for i in range(config["num_layers"])
            ]
        )

        self.norm = te.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.lm_head = te.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )

    def forward(self, input_ids, labels=None):
        h = self.embed_tokens(input_ids)
        rotary_pos_emb = self.rope(h.size(1))

        for layer in self.layers:
            h = layer(h, rotary_pos_emb)

        h = self.norm(h)
        logits = self.lm_head(h)

        if labels is not None:
            return F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def install_fixed_rht_sign_vector():
    """Bind TE's NVFP4 RHT helpers to TorchAO's TP kernel sign vector."""
    import transformer_engine.pytorch.tensor.nvfp4_tensor as nvfp4_tensor

    _TP_RHT_SIGN_VECTOR = (
        1,
        1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        -1,
        -1,
    )

    def fixed_wgrad_sign_vector(device):
        return torch.tensor(_TP_RHT_SIGN_VECTOR, dtype=torch.float32, device=device)

    nvfp4_tensor.get_wgrad_sign_vector = fixed_wgrad_sign_vector
    nvfp4_tensor.get_random_sign_mask_for_rht.cache_clear()
    nvfp4_tensor.get_rht_matrix.cache_clear()
    return _TP_RHT_SIGN_VECTOR


def main():
    parser = argparse.ArgumentParser(
        description="LLaMA 3 8B training with TransformerEngine"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="nvfp4",
        choices=["nvfp4", "nvpf4", "mxfp8", "mxpf8"],
        help="TransformerEngine low-precision recipe to use",
    )
    parser.add_argument(
        "--small", action="store_true", help="Use small debug model instead of 8B"
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Overfit on a single fixed batch (proves training works)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        choices=["wikitext"],
        help="Real dataset to train on (default: synthetic random tokens)",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of training steps"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Capture training step as a CUDA graph and replay each step",
    )
    args = parser.parse_args()
    precision = {
        "nvfp4": "nvfp4",
        "nvpf4": "nvfp4",
        "mxfp8": "mxfp8",
        "mxpf8": "mxfp8",
    }[args.precision]

    config = LLAMA_SMALL if args.small else LLAMA3_8B

    # Override vocab_size to match tokenizer when using real data
    tokenizer = None
    if args.data:
        tokenizer = load_tokenizer(args.data)
        # TE block-scaling recipes require aligned GEMM dimensions.
        vocab_multiple = 16 if precision == "nvfp4" else 32
        vocab = tokenizer.vocab_size
        vocab = (vocab + vocab_multiple - 1) // vocab_multiple * vocab_multiple
        config = {**config, "vocab_size": vocab}

    seq_len = min(args.seq_len, config["max_seq_len"])
    dtype = torch.bfloat16
    device = "cuda"

    if precision == "nvfp4":
        install_fixed_rht_sign_vector()

    recipe = NVFP4BlockScaling() if precision == "nvfp4" else MXFP8BlockScaling()
    precision_label = precision.upper()

    print(f"{'=' * 60}")
    print(f"LLaMA 3 {precision_label} Training with TransformerEngine")
    print(f"{'=' * 60}")
    print(f"Model:      {'small (debug)' if args.small else '8B'}")
    print(f"Precision:  {precision_label} (master weights fp32, autocast bf16)")
    print(f"CUDA Graphs: {'enabled' if args.cuda_graphs else 'disabled'}")
    print(f"Recipe:     {recipe}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seq length: {seq_len}")
    print(f"Steps:      {args.steps}")
    data_mode = (
        "overfit (fixed batch)" if args.overfit else (args.data or "synthetic random")
    )
    print(f"Mode:       {data_mode}")
    print(f"Device:     {torch.cuda.get_device_name(0)}")
    print()

    # Build model — fp32 master weights, fp16 compute via autocast
    print("Building model...")
    model = TELlama(config).to(device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params / 1e9:.2f}B)")
    print(f"Memory after init: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Optimizer (LLaMA 3 recipe: AdamW with β=(0.9, 0.95), wd=0.1)
    # capturable=True required when optimizer.step() is captured inside a CUDA graph
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=args.cuda_graphs,
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

    if args.cuda_graphs:
        # Static input buffers — fixed GPU addresses required for graph replay
        static_ids = torch.zeros(
            args.batch_size, seq_len, dtype=torch.long, device=device
        )
        static_labels = torch.zeros(
            args.batch_size, seq_len, dtype=torch.long, device=device
        )

        # Warmup on a side stream: initializes AdamW state and TE lazy allocations
        # without polluting the graph memory pool
        _ws = torch.cuda.Stream()
        _ws.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_ws):
            for _ in range(3):
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    with te.autocast(enabled=True, recipe=recipe):
                        _loss = model(static_ids, static_labels)
                _loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.current_stream().wait_stream(_ws)

        # Capture — autocast contexts active so TE dispatches the right kernels
        cuda_graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=False)  # grads must be tensors, not None
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            with te.autocast(enabled=True, recipe=recipe):
                with torch.cuda.graph(cuda_graph):
                    static_loss = model(static_ids, static_labels)
                    static_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(
                        set_to_none=False
                    )  # zero inside graph for next step
        torch.cuda.synchronize()
        print("CUDA graph captured.")

    for step in range(args.steps):
        t0 = time.time()

        # Data loading — always outside the graph
        if args.overfit:
            input_ids, labels = fixed_tokens[:, :seq_len], fixed_tokens[:, 1:]
        elif real_data_iter is not None:
            input_ids, labels = next(real_data_iter)
        else:
            tokens = torch.randint(
                0, config["vocab_size"], (args.batch_size, seq_len + 1), device=device
            )
            input_ids, labels = tokens[:, :seq_len], tokens[:, 1:]

        if args.cuda_graphs:
            static_ids.copy_(input_ids)
            static_labels.copy_(labels)
            cuda_graph.replay()
            loss_val = static_loss.item()
        else:
            # Forward pass: TE recipe for GEMMs, bf16 for non-TE ops.
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                with te.autocast(enabled=True, recipe=recipe):
                    loss = model(input_ids, labels)
            # Backward + optimizer step (no GradScaler needed for bf16)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_val = loss.item()

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
            f"{step:5d} {loss_val:10.4f} {tok_per_sec:10.0f} {tok_str:>10} {mem_gb:10.2f}"
        )

    print(f"\nTraining complete. Final loss: {loss_val:.4f}")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Skip Python finalization — avoids PyGILState_Release race with
    # streaming dataset background threads (harmless SIGABRT on exit).
    os._exit(0)


if __name__ == "__main__":
    main()
