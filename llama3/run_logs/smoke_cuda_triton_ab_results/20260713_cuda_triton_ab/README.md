# MXFP8 Compiled-Kernel Smoke: CUDA vs TRITON dim1 Cast A/B

Date: 2026-07-13

## Goal

The committed MXFP8 benchmarks (`smoke_fixed_batch_size_results/`, and the 200M-token runs) were
produced with the torchao MXFP8 **dim1 cast forced to the TRITON kernel**, because the torchao in
use had no compiled `_C` extension (the CUDA `torchao::mxfp8_quantize` op was unavailable). Those
notes flagged that the TRITON numbers were therefore a likely **floor** for MXFP8 throughput.

This suite answers the narrow question directly: **how much throughput does the compiled CUDA dim1
cast kernel recover versus TRITON?** It is a clean A/B — identical torchao, identical batch shapes,
the *only* difference between arms is the dim1 kernel choice
(`torchao/prototype/moe_training/mxfp8_linear.py:66`, `MXFP8Dim1CastKernelChoice.CUDA` vs `.TRITON`).

## Contract and Source State

- Model: Llama 3 8B. Hardware: 4× NVIDIA GB200 (SM100, cap 10.0), FSDP 4, TP 1.
- Sequence length 2048, dataset `c4_test`, 50 steps, `log_freq 10`, LR 3e-4, GA as noted per trainer.
- Eager: local batch 32, global batch 128 ⇒ **GA=1** (physical global 128), `torch.compile` on model.
- Graph (`GraphTrainer`): local batch 16, global batch 128 ⇒ **GA=2** (physical global 64), CUDA graphs.
- Metrics are the **step-50** values (short run; TPS is noisier than the 200M benchmarks). Aggregate
  TPS = per-device TPS × 4.
- Harness (`llama3`) commit: `7aad72c` (branch `nvfp4_linear_titan`).
- TorchTitan: `origin/nvfp4_linear_ao @ e8e39984` (the committed-converter MXFP8 benchmark tree),
  run from a throwaway git worktree.
- Torch: `2.14.0a0+gitd9abf9e`, CUDA 13.3.
- TorchAO: **`0.18.0+gitcb76f29`** (installed from local source `/opt/pytorch/ao`). Its prebuilt
  `_C_mxfp8.cpython-312-aarch64-linux-gnu.so` provides the CUDA `torchao::mxfp8_quantize` op; the op
  registers when `torchao.prototype.mx_formats.kernels` imports (which the MXFP8 path does). No
  torchao rebuild or reinstall was needed — the compiled kernel works in place.

### Difference from the published MXFP8 benchmarks

The published committed-TRITON smoke used a **different torchao** (`f229086c`, branch
`mxfp8_triton_compat`, with the dim1 cast hand-edited to TRITON). This suite uses torchao
`0.18.0+gitcb76f29`, whose dim1 default is already CUDA; the TRITON arm is produced by flipping line
66 back to TRITON. The TRITON arm here **reproduces the published TRITON numbers within ≤0.5%**
(see Cross-check), which is what makes the CUDA numbers trustworthy despite the torchao version diff.

## Commands

Each eager run:

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 \
  -m torchtitan.train --module llama3 --config llama3_8b_mxfp8 \
  --parallelism.tensor_parallel_degree 1 --parallelism.data_parallel_shard_degree 4 \
  --training.local_batch_size 32 --training.global_batch_size 128 \
  --training.seq_len 2048 --training.steps 50 --dataloader.dataset c4_test \
  --metrics.log_freq 10 --optimizer.param-groups.0.optimizer-kwargs.lr 0.0003 \
  --hf-assets-path ./tests/assets/tokenizer --compile.enable --compile.components model
```

Each graph run used the same training args with `--module graph_trainer.llama3 --config
graph_trainer_llama3_8b_mxfp8`, no eager compile flags, default memory policy, and CUDA graphs.

Between the CUDA and TRITON arms, only `mxfp8_linear.py:66` was flipped (auto-reverted to CUDA on
exit). `PYTORCH_CUDA_ALLOC_CONF` was left at default (no `expandable_segments`).

## Results (step 50)

| Trainer | dim1 kernel | Local batch | Phys. global | TPS/GPU | Aggregate TPS | TFLOPs/GPU | Peak reserved | Retries | Loss | Done |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| eager | **CUDA**   | 32 | 128 | **28,905** | 115,620 | 1,394.67 | 179.94 GiB (97.63%) | 0 | 2.788 | ✓ |
| eager | TRITON     | 32 | 128 | 28,259 | 113,036 | 1,363.50 | 177.39 GiB (96.25%) | 0 | 2.844 | ✓ |
| graph | **CUDA**   | 16 | 64  | **27,230** | 108,920 | 1,313.89 | 139.55 GiB (75.72%) | 0 | 2.787 | ✓ |
| graph | TRITON     | 16 | 64  | 26,434 | 105,736 | 1,275.48 | 139.55 GiB (75.72%) | 0 | 2.864 | ✓ |

All four runs reached step 50 with finite loss and `Training completed`.

## Analysis

### 1. CUDA dim1 cast is faster than TRITON (the core result)
- **Eager: 28,905 vs 28,259 TPS/GPU → +2.3%.**
- **Graph: 27,230 vs 26,434 TPS/GPU → +3.0%.**

Modest but real, and consistent with the prior estimate that TRITON was a ~2–5% floor. So the
compiled kernel is the true production-path number and the published TRITON MXFP8 benchmarks were a
~2–3% *underestimate*.

### 2. Cross-check: the TRITON arm reproduces the published committed-TRITON smoke
Despite torchao differing (`0.18.0` here vs `f229086c` published):

| | This TRITON | Published TRITON (`smoke_fixed_batch_size_results`) | Δ |
| --- | ---: | ---: | ---: |
| eager TPS/GPU | 28,259 | 28,401 | −0.5% |
| eager peak mem | 177.39 GiB | 177.49 GiB | ≈0 |
| graph TPS/GPU | 26,434 | 26,410 | +0.1% |
| graph peak mem | 139.55 GiB | 139.55 GiB | 0 |

Near-exact reproduction ⇒ the environment reconstruction is faithful and the CUDA arm is trustworthy.

### 3. Memory
- **Eager:** CUDA uses **+2.55 GiB** more peak reserved (179.94 vs 177.39 GiB) — extra CUDA-kernel
  scratch near the ceiling; still **0 allocator retries**.
- **Graph:** identical (139.55 GiB both) — plenty of headroom at 75.72%.

### 4. Effect on the NVFP4 / MXFP8 / BF16 story
Against the iso-batch BF16 from `smoke_fixed_batch_size_results` (eager 21,776, graph 21,377 TPS/GPU):

| MXFP8 kernel | eager × BF16 | graph × BF16 |
| --- | ---: | ---: |
| TRITON (published) | 1.30× | 1.24× |
| **CUDA (compiled)** | **1.33×** | **1.27×** |

The compiled kernel closes ~half of the eager NVFP4-vs-MXFP8 *throughput* gap (NVFP4 eager is 1.45×
BF16) and **nothing** on memory — MXFP8 still stores BF16 activations for backward (≈BF16 memory),
while NVFP4 checkpoints packed 4-bit activations (~0.60× BF16 memory). **Ordering is unchanged:
NVFP4 leads on both throughput and memory; the dim1 kernel choice is a ~2–3% effect that does not
move the three-way conclusion.**

## Fairness caveats

- **Loss is not a quality signal here.** 50 steps on `c4_test`; the CUDA-vs-TRITON loss deltas
  (2.79 vs 2.84 eager; 2.79 vs 2.86 graph) are warmup/seed noise, not a numerical-quality result.
- **Retries are stochastic near the ceiling.** The published TRITON eager run logged 5 allocator
  retries; both eager arms here logged 0. Retry occurrence at ~96–97% memory varies run-to-run and
  is not a stable kernel property.
- **A/B validity.** Both arms use the identical torchao `0.18.0` and identical batch shapes, so the
  within-suite CUDA-vs-TRITON delta is clean. Absolute cross-suite comparisons inherit the usual
  eager-GA1-vs-graph-GA2 caveat and should use the iso-batch BF16 above, not cross-trainer loss.

## Environment reconstruction (required to run; not committed upstream)

The current environment had drifted from the benchmark environment and could not run torchtitan
until these were installed. **None touched torch or torchao** (verified `2.14.0a0+gitd9abf9e` /
`0.18.0+gitcb76f29` unchanged before and after):

1. `pip install tyro` — hard torchtitan config dependency, was missing.
2. `pip install -r requirements.txt` — installed torchtitan's declared deps that were absent
   (`spmd_types==0.2.1`, `datasets`, `tokenizers`, `torchdata`, `wandb`, `safetensors`, …). `torch`
   was already satisfied by the nightly; torchao is not a torchtitan requirement.

The compiled CUDA kernel itself required **no** install/rebuild — the prebuilt `_C_mxfp8.so` was
already present in the installed torchao and works on GB200 in place.

## Raw logs

`eager_cuda.txt`, `eager_triton.txt`, `graph_cuda.txt`, `graph_triton.txt`.
