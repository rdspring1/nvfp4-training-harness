# NVFP4 vs bf16 grouped-GEMM crossover (DeepSeek V3 671B)

One grouped GEMM at the 671B gate-projection shape, random data, forward only, on
a single **NVIDIA GB200**. Uniform tokens/expert. Shape: `K=7168`, `N=2048`,
`E=8`. Times are ms/call (warmup 5, 20 iters, CUDA-event timed).

- **`nvfp4_pure`** — the 4-bit tensor-core grouped matmul on **pre-quantized**
  inputs (quantize once, outside the timed loop). Isolates raw compute.
- **`nvfp4_full`** — `nvfp4_pure` + on-the-fly OpenAI-Triton quantization of the
  activations and weights every call (RHT amax + row/col quantize + weight
  quantize). This is what the TorchAO training override actually runs.
- **`te_full`** — TransformerEngine's fused NVFP4 grouped GEMM (`_GroupedLinear`),
  which always quantizes activations + weights inside the call (graph-safe RHT
  cast-fusion kernel). TE's forward is inherently "full"; there is no pre-quantized
  split to isolate.

## Table 1 — Pure compute crossover (bf16 vs NVFP4, no quantization)

| tok/exp | M (rows) | bf16 (ms) | nvfp4_pure (ms) | pure/bf16 | winner |
|--------:|---------:|----------:|----------------:|----------:|:-------|
|     128 |    1,024 |     0.056 |           0.107 |      1.91 | bf16   |
|     256 |    2,048 |     0.055 |           0.108 |      1.96 | bf16   |
|     512 |    4,096 |     0.094 |           0.107 |      1.14 | bf16   |
|   1,024 |    8,192 |     0.177 |           0.108 |  **0.61** | nvfp4  |
|   2,048 |   16,384 |     0.302 |           0.111 |      0.37 | nvfp4  |
|   4,096 |   32,768 |     0.615 |           0.190 |      0.31 | nvfp4  |
|   8,192 |   65,536 |     1.210 |           0.338 |      0.28 | nvfp4  |
|  16,384 |  131,072 |     2.797 |           0.967 |      0.35 | nvfp4  |
|  32,768 |  262,144 |     6.373 |           1.824 |      0.29 | nvfp4  |

**Pure crossover ≈ 512–1024 tokens/expert** (M ≈ 4K–8K rows). Below it the NVFP4
kernel is launch-overhead-bound (flat ~0.107 ms floor) and bf16 wins; above it the
4-bit tensor cores saturate at **~3–3.6× faster** than bf16 (ratio ~0.28–0.35).

## Table 2 — OpenAI-Triton quantization overhead

Overhead = `nvfp4_full − nvfp4_pure` (the RHT/amax/quantize Triton kernels).

| tok/exp | M (rows) | nvfp4_pure (ms) | nvfp4_full (ms) | quant overhead (ms) | overhead ÷ pure GEMM | full/bf16 |
|--------:|---------:|----------------:|----------------:|--------------------:|---------------------:|----------:|
|     128 |    1,024 |           0.107 |           0.982 |               0.875 |                 8.2× |      17.6 |
|     256 |    2,048 |           0.108 |           1.005 |               0.897 |                 8.3× |      18.2 |
|     512 |    4,096 |           0.107 |           1.055 |               0.948 |                 8.9× |      11.3 |
|   1,024 |    8,192 |           0.108 |           1.169 |               1.061 |                 9.8× |       6.6 |
|   2,048 |   16,384 |           0.111 |           1.391 |               1.280 |                11.5× |       4.6 |
|   4,096 |   32,768 |           0.190 |           2.047 |               1.857 |                 9.8× |       3.3 |
|   8,192 |   65,536 |           0.338 |           2.880 |               2.542 |                 7.5× |       2.4 |
|  16,384 |  131,072 |           0.967 |           4.420 |               3.453 |                 3.6× |       1.6 |
|  32,768 |  262,144 |           1.824 |           7.657 |               5.833 |                 3.2× |       1.2 |

Quantization costs **~8–11× the matmul itself** across the mid-range and only
falls to ~3× at the largest sizes. It grows with M (activation quant scales with
row count; weight quant is fixed), so it never becomes negligible.

## Table 3 — TransformerEngine forward (fused quantize+GEMM)

Same shape (K=7168, N=2048, E=8), forward only. `te_full` is TE's fused NVFP4
grouped GEMM; compared against the same bf16 reference and TorchAO's `full/bf16`.

| tok/exp | M (rows) | bf16 (ms) | te_full (ms) | te/bf16 | torchao full/bf16 |
|--------:|---------:|----------:|-------------:|--------:|------------------:|
|     128 |    1,024 |     0.056 |        0.911 |    16.2 |              17.6 |
|     256 |    2,048 |     0.056 |        0.903 |    16.3 |              18.2 |
|     512 |    4,096 |     0.096 |        0.849 |     8.9 |              11.3 |
|   1,024 |    8,192 |     0.178 |        0.867 |     4.9 |               6.6 |
|   2,048 |   16,384 |     0.301 |        0.866 |     2.9 |               4.6 |
|   4,096 |   32,768 |     0.605 |        0.872 |     1.4 |               3.3 |
|   8,192 |   65,536 |     1.210 |        1.195 | **0.99**|               2.4 |
|  16,384 |  131,072 |     2.897 |        1.988 |    0.69 |               1.6 |
|  32,768 |  262,144 |     6.297 |        3.639 |    0.58 |               1.2 |

**TE's forward crosses bf16 at ~8K tokens/expert** — much earlier than TorchAO's
`nvfp4_full` (~50K). TE's fused quantize is nearly flat (~0.85 ms floor to M≈65K),
so it amortizes faster and beats TorchAO's forward at every point. On the forward
path **TE is the stronger backend.**

## Table 3b — TE backward (single grouped GEMM)

TE's backward is well-behaved — cheaper than the forward at large E (the forward
carries the activation RHT quantize):

| config | TE fwd (ms) | TE bwd (ms) | bwd/fwd |
|:-------|------------:|------------:|--------:|
| E=8,  TPE=2048 | 0.73 | 0.94 | 1.3× |
| E=64, TPE=512  | 3.61 | 2.44 | 0.7× |
| E=64, TPE=2048 | 3.38 | 2.55 | 0.8× |

End-to-end 3-GEMM expert MLP fwd+bwd (E=64, 512 tok/exp): **26 ms**, faster than
TorchAO's 36 ms.

Note: TE's graph-safe RHT quant kernel caps at 64 experts/launch
(`kMaxTensorsPerKernel=64`, hard assert from the <4 KB launch-args struct, no
chunking), but that is moot at realistic training EP (~4 experts/GPU). Expert count
did not affect the forward crossover — TE forward at E=64 is as good as E=8.

## How the crossover point moves

| Regime | Crossover (tokens/expert) | Why |
|:-------|:--------------------------|:----|
| **Pure 4-bit GEMM** | **~1K** | raw tensor-core compute; only kernel launch overhead to amortize |
| **TE forward (quant + GEMM)** | **~8K** (measured) | TE's fused quantize is nearly flat (~0.85 ms floor), so it amortizes ~6× earlier than TorchAO's forward |
| **TorchAO full fwd GEMM (quant + GEMM)** | **~50K** (extrapolated; full/bf16 still 1.2 at 32K) | Triton quantization tax is ~8–11× the matmul, pushing the break-even out ~40–60× |
| **Real training fwd+bwd (3-GEMM expert MLP)** | **~16K** (TorchAO, measured) | the quantized activations/weights + RHT are computed once and **reused across the forward + 2 backward GEMMs**, amortizing the quant tax over ~3× more matmul work |

The 4-bit kernels pay off early (~1K tok/expert), but this override's per-call
Triton **re-quantization** is the bottleneck that moves the practical training
crossover out to ~16K tokens/expert. `torch.compile` on the full path gives only a
constant-factor win (best ~25% mid-range, ~5% at large M) and does **not** shift
the crossover — the cost lives inside opaque hand-written Triton kernels that
dynamo cannot fuse into. The high-leverage optimizations are algorithmic:
caching quantized weights across microbatches and a cheaper activation transform.
