# Llama 3 8B Fixed-Batch Smoke Suite + `expandable_segments` A/B

Date: 2026-07-12

## Goal

The sibling `smoke_max_batch_size_results/` suite pushed each precision to its *maximum* batch,
which confounded throughput with batch size and let eager "passes" survive only via allocator
retries at the memory ceiling. This suite instead fixes the batch (eager `lbs32/GA1`,
graph `lbs16/GA2` — the same shapes as the 200M-token benchmarks) and asks a narrower question:

**does `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` remove the MXFP8 allocator-retry stalls,
and at what throughput/memory cost?** MXFP8 is the only subject — at these fixed batches BF16 and
NVFP4 run retry-free; only MXFP8 eager (96%+ of memory) hits retries.

## Contract and Source State

- Model: Llama 3 8B. Hardware: 4× NVIDIA GB200 (SM100, 184.30 GiB usable/GPU), FSDP 4, TP 1.
- Sequence length 2048, dataset `c4_test`, 50 steps, `log_freq 10`, LR 3e-4.
- Eager: local batch 32, global batch 128 ⇒ **GA=1** (physical global 128), `torch.compile` on model.
- Graph (`GraphTrainer`): local batch 16, global batch 128 ⇒ **GA=2** (physical global 64), CUDA graphs.
- Harness (`llama3`) commit: `fbe2de4`.
- TorchTitan `041ec017` (branch `nvfp4_linear_ao`) with local changes described under **Environment
  fixes** below.
- TorchAO `f229086c` with one local change (see **Environment fixes**).
- Metrics are the **step-50** values (short run; TPS is noisier than the 200M benchmarks). Aggregate
  TPS = per-device TPS × 4.

## Results (step 50)

| Precision | Trainer | expand? | Local batch | Phys. global | TPS/GPU | Aggregate TPS | TFLOPs/GPU | Peak reserved | **Retries** | Loss | Done |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| BF16  | eager | – | 32 | 128 | 21,776 | 87,104 | 1,050 | 174.49 GiB (94.68%) | 0 | 2.878 | ✓ |
| NVFP4 | eager | – | 32 | 128 | 31,639 | 126,556 | 1,526 | 103.00 GiB (55.89%) | 0 | 2.789 | ✓ |
| MXFP8 | eager | no  | 32 | 128 | 28,401 | 113,604 | 1,370 | 177.49 GiB (96.31%) | **5** | 2.902 | ✓ |
| MXFP8 | eager | **yes** | 32 | 128 | 28,747 | 114,988 | 1,387 | 167.38 GiB (90.82%) | **0** | 2.827 | ✓ |
| BF16  | graph | – | 16 | 64 | 21,377 | 85,508 | 1,031 | 137.03 GiB (74.35%) | 0 | 2.899 | ✓ |
| NVFP4 | graph | – | 16 | 64 | 26,044 | 104,176 | 1,256 | 61.68 GiB (33.47%) | 0 | 2.856 | ✓ |
| MXFP8 | graph | no  | 16 | 64 | 26,410 | 105,640 | 1,274 | 139.55 GiB (75.72%) | 0 | 2.806 | ✓ |
| MXFP8 | graph | **yes** | 16 | 64 | 26,430 | 105,720 | 1,275 | 137.77 GiB (74.76%) | 0 | 2.822 | ✓ |

All eight runs reached step 50 with finite loss and `Training completed`.

## Analysis

### 1. `expandable_segments` on MXFP8 eager — the core result (clear win)
At `lbs32`, MXFP8 eager sits at 96.31% of memory and the recurring large activation allocation
cannot always find a contiguous block, triggering the allocator's fail→free-cache→retry cycle
(5 retry-warning batches over 50 steps). Enabling `expandable_segments`:

- **Retries 5 → 0** — the stalls are eliminated.
- **Peak reserved 177.49 → 167.38 GiB (−10.1 GiB, −5.5 pts)** — less fragmentation reservation, so
  the same workload fits in materially less memory.
- **TPS/GPU 28,401 → 28,747 (+1.2%)** — a small throughput gain from removing the retry stalls.

So for MXFP8 near the memory ceiling, `expandable_segments` is unambiguously beneficial: it removes
the allocator thrash, frees ~10 GiB, and nudges throughput up.

### 2. `expandable_segments` on MXFP8 graph — neutral, and safe with CUDA graphs
The `GraphTrainer` runs at `lbs16` sit at only 75.72% of memory with **0 retries already**, so there
is nothing for `expandable_segments` to fix: retries 0→0, memory 139.55 → 137.77 GiB (−1.8), TPS
+0.08% (noise). The important negative result: **`expandable_segments` did not break CUDA graphs** —
the run completed normally. It is a no-op here, not a hazard.

### 3. Throughput / memory (context, with caveats)
- **Eager, iso-batch (all `lbs32/GA1`)** is the fair throughput comparison: NVFP4 **1.45×** and MXFP8
  **1.32×** (expandable) the BF16 TPS — consistent with the 200M-token eager benchmarks.
- **Graph (`lbs16/GA2`)**: NVFP4 1.22× and MXFP8 1.24× BF16. Lower ratios than eager because the
  small physical batch (16) underfeeds the GPU.
- **Memory:** NVFP4 is dramatically lighter (eager 103 vs 174 GiB; graph 62 vs 137 GiB).
- **Sanity:** BF16 eager peak (174.49 GiB) matches the prior 200M run exactly and NVFP4 (103.00 vs
  104.95 GiB) matches closely — confirming BF16/NVFP4 are unaffected by the MXFP8 environment changes.
- **Do not read MXFP8's memory as "8-bit, so between BF16 and NVFP4."** All three share identical
  model static (10.10 GiB) and SelectiveAC, so the memory spread is entirely *activations saved for
  backward*. MXFP8 does `ctx.save_for_backward(input_hp, weight_hp)` — it stores the **BF16**
  activations and re-quantizes in backward — so it saves no activation memory (≈BF16, +3 GiB for the
  extra FP8 copies/scales). This holds for the prior override MXFP8 too (~180 GiB), not just the
  committed converter used here. NVFP4's override instead stores **packed 4-bit** activations, giving
  the ~4× activation reduction (and thus ~56% of BF16 memory). The NVFP4≪MXFP8 gap is this design
  difference — NVFP4 checkpoints activations in low precision, MXFP8 does not — not a measurement
  artifact (NVFP4 is also the *fastest*, so the low memory is not bought with recompute).

### 4. Fairness caveats
- **Loss is not a quality signal here.** 50 steps on `c4_test`; step-50 losses (2.79–2.90) are
  warmup noise. Also eager is GA=1 and graph is GA=2 — as documented in the 200M analyses the GA=2
  path is not loss-equivalent to GA=1, so **do not compare loss across trainers**.
- **MXFP8 here is NOT the prior MXFP8.** The original override (`torchtitan.overrides.mxfp8_linear`)
  was lost, so these runs use the committed `MXFP8LinearConverter` with the **TRITON** dim1 cast
  kernel (see below). MXFP8 numbers are internally consistent (the with/without-`expandable` A/B is
  valid — both use the identical path) but are **not comparable** to the prior override-based MXFP8
  benchmarks.

## Environment fixes (required to run; not yet committed upstream)

The pinned submodules were missing local compatibility changes that had been lost. Three small,
additive/local edits were needed — each is documented so the runs are reproducible:

1. **`torchtitan/models/llama3/config_registry.py`** — added `llama3_8b_mxfp8()` and imported
   `MXFP8LinearConverter` (mirrors the existing `Float8` variant). Enables committed-converter MXFP8.
2. **`torchtitan/experiments/graph_trainer/llama3/config_registry.py`** — added
   `graph_trainer_llama3_8b_mxfp8()`.
3. **`torchtitan/experiments/graph_trainer/trainer.py`** — the graph trainer called
   `_maybe_apply_numa_binding_to_current_process(device_index=…)`; the installed PyTorch renamed that
   kwarg to `gpu_index`. One-line rename. (Without this the entire graph trainer fails at init,
   independent of precision.)
4. **`torchao/prototype/moe_training/mxfp8_linear.py`** — the committed MXFP8 path hardcodes the CUDA
   dim1 cast kernel (`torchao::mxfp8_quantize`), but this torchao is a Python-only editable install
   with no compiled `_C` extension, so that op is unavailable. Switched the dim1 cast kernel choice
   from `CUDA` to `TRITON` (numerically equivalent, needs no compiled kernel). This is the reason
   MXFP8 here differs from the prior override runs.

## Raw logs

`bf16_eager.txt`, `nvfp4_eager.txt`, `mxfp8_eager.txt`, `mxfp8_eager_expandable.txt`,
`bf16_graph.txt`, `nvfp4_graph.txt`, `mxfp8_graph.txt`, `mxfp8_graph_expandable.txt`.
