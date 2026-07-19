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

## Table 4 — Training fwd+bwd sweep (bf16 vs TorchAO vs TE)

3-GEMM expert MLP, forward+backward, E=8. ms/iter; ratio to bf16 (<1 = beats bf16).
Re-measured with the persistent `rht_amax` dispatch live (see Table 5).

| tok/exp | rows | bf16 | TorchAO | TE | TorchAO/bf16 | TE/bf16 |
|--------:|--------:|------:|--------:|------:|-------------:|--------:|
|     512 |   4,096 |  2.13 |    7.96 |  5.81 |         3.74 |    2.73 |
|   1,024 |   8,192 |  2.80 |    8.00 |  5.81 |         2.86 |    2.08 |
|   2,048 |  16,384 |  4.11 |    8.15 |  5.87 |         1.98 |    1.43 |
|   4,096 |  32,768 |  7.20 |   10.03 |  6.47 |         1.39 | **0.90** ← TE crosses |
|   8,192 |  65,536 | 13.65 |   14.38 |  8.84 |         1.05 |    0.65 |
|  16,384 | 131,072 | 28.84 |   23.36 | 14.93 | **0.81** ← AO crosses | 0.52 |
|  32,768 | 262,144 | 57.74 |   41.38 | 27.42 |         0.72 |    0.48 |

**TE crosses bf16 at ~4K tok/expert; TorchAO at ~16K.** TE beats TorchAO at every
point (post `unbind` fix) and the lead widens with scale — at 32K tok/expert TE is
**1.5× faster than TorchAO and 2.1× faster than bf16** (9.6 vs 6.3 vs 4.5 Mtok/s).
Memory is a tie (~21–23 GiB). So for training, TE is the winner above ~4K
tok/expert; below ~4K bf16 wins; TorchAO only overtakes bf16 at ~16K.

The persistent `rht_amax` dispatch trims TorchAO **~7–10% at ≥8K tok/expert** (32K:
46.2→41.4 ms) — a scale-side efficiency win that narrows the TE↔AO gap but does **not**
move AO's bf16 crossover (still ~16K; interpolated ~10K vs ~11K before). rht_amax is
one of several quant kernels; the larger `rht_quantize_row_col` (unchanged) still
governs the break-even. TE's numbers are unchanged (no code change), matching the
prior sweep within noise.

## How the crossover point moves

| Regime | Crossover (tokens/expert) | Why |
|:-------|:--------------------------|:----|
| **Pure 4-bit GEMM** | **~1K** | raw tensor-core compute; only kernel launch overhead to amortize |
| **TE forward (quant + GEMM)** | **~8K** (measured) | TE's fused quantize is nearly flat (~0.85 ms floor), so it amortizes ~6× earlier than TorchAO's forward |
| **TorchAO full fwd GEMM (quant + GEMM)** | **~50K** (extrapolated; full/bf16 still 1.2 at 32K) | Triton quantization tax is ~8–11× the matmul, pushing the break-even out ~40–60× |
| **Training fwd+bwd — TE (3-GEMM MLP)** | **~4K** (measured) | TE's low, flat fused-quant floor + a cheap grouped backward amortize over the fwd + 2 bwd GEMMs |
| **Training fwd+bwd — TorchAO (3-GEMM MLP)** | **~16K** (measured) | growing Triton quant tax (~8–11× the matmul), only partly amortized over fwd + 2 bwd GEMMs |

The 4-bit kernels pay off early (~1K tok/expert), but the backends diverge on the
per-call quantization tax. **TE** is the stronger training backend: fused, flat
quant → crossover ~4K tok/expert, beating both bf16 (above ~4K) and TorchAO
(everywhere). **TorchAO**'s per-call Triton re-quantization grows with M and pushes
its training crossover out to ~16K; `torch.compile` gives only a constant-factor win
(best ~25% mid-range, ~5% at large M) and does **not** shift it — the cost lives
inside opaque hand-written Triton kernels dynamo cannot fuse into. The high-leverage
TorchAO optimizations are algorithmic: caching quantized weights across microbatches
and a cheaper activation transform.

## Table 5 — Are TorchAO's grouped quant kernels worth grouping?

Each of TorchAO's three forward NVFP4 quant kernels has a single-expert twin. This
compares one **grouped** launch over `E=8` experts against `E ×` the **single**
kernel at the same per-expert size — i.e. the cost of just launching the single
kernel once per expert. `ratio = grouped ÷ (E × single)`; **>1 means grouping is
slower than per-expert launches.** DSV3-671B activation dim `N=7168`. Raw kernels,
pre-allocated buffers (times the kernel body only). For `rht_amax`, grouped µs is
the **dispatched** kernel body — tiled ≤1K avg rows/group, per-group-CTA persistent
above (the fix, see below). Repro: `nvfp4_grouped_kernel_grouping.py`,
`nvfp4_group_amax_persistent_proto.py`, `nvfp4_amax_dispatch_table5.py`.

| kernel | tok/exp | grouped µs | E×single µs | ratio |
|:-------|--------:|-----------:|------------:|------:|
| **rht_amax** (act) |    512 |    43 |   139 | 0.31 |
| rht_amax           |  2,048 |    78 |   186 | 0.42 |
| rht_amax           |  4,096 |   130 |   238 | 0.55 |
| rht_amax           |  8,192 |   232 |   358 | 0.65 |
| **rht_quantize_row_col** (act) |    512 |    65 |   210 | 0.31 |
| rht_quantize_row_col           |  4,096 |   445 |   532 | 0.84 |
| rht_quantize_row_col           |  8,192 |   878 |   953 | 0.92 |
| **weight_quantize_2d** (gate/up 2048×7168) | — | 287 | 350 | 0.82 |
| weight_quantize_2d (down 7168×2048)        | — | 286 | 350 | 0.82 |

**`rht_amax` used to be the one grouped kernel that lost to per-expert launches**
(**1.44× slower** at 8K with the tiled kernel). It no longer does: the public op now
dispatches to a **per-group-CTA persistent kernel** above 1K avg rows/group, and the
grouped µs above are that dispatched path. It stays at/below break-even everywhere
(0.31→0.65) — a **2.25× kernel-body speedup** over the old tiled path at 8K. The other
two group healthily too: `rht_quantize_row_col` stays at/below break-even everywhere
(0.31→0.92, margin shrinking with M but never negative), and `weight_quantize_2d` is a
steady ~0.82 (grouping ~18% faster; no token dependence).

The original `rht_amax` deficit was **not** atomic-max contention or grid overhead — an
E-sweep (`bench_amax_esweep.py`) rules both out. At fixed *total* rows (65536),
grouped time is flat across E=1→64 (501→563 µs): spreading the same work over 64
small groups instead of 1 big one barely helps, so per-group amax-scalar contention
is not the driver. At fixed tok/expert, grouped time is exactly linear in E (~64 µs
per group, zero cross-group interference). The grouped kernel runs at a **constant
~0.0078 µs/row** regardless of M. What actually moves the ratio is the *single*
kernel: it is a persistent kernel that amortizes launch/setup over rows, so its
µs/row falls from 0.019 (M=1024) to 0.0033 (M=65536) — a 5.7× efficiency gain the
grouped tiled kernel never captures. The gap only *looks* like it grows with row
count because the single-kernel baseline keeps improving, not because grouping
degrades.

**The fix — a per-group-CTA persistent kernel.** The single kernel's edge was its
persistent design; the tiled grouped kernel couldn't capture it. Binding
`num_sms // E` CTAs to each group and striding its tiles with an elementwise
cumulative max — the single kernel generalized (num_groups=1 recovers it) — restores
that bandwidth: **~3.5 ns/row at 8K vs the tiled kernel's flat ~8 ns/row**, matching
the single-tensor floor. It's warp-specializable (elementwise max, no in-loop
reduction) and validated bitwise against the tiled oracle. Dispatch is gated at 1024
avg rows/group (`_PERSISTENT_MIN_AVG_ROWS`, the measured crossover); below it the
tiled kernel still wins — too few tiles per group to amortize the persistent prologue.
Note: this closes the kernel-body gap; the public op still carries a fixed eager
dispatch overhead (~320 µs) that is captured away under CUDA graphs in training.

### Tiled vs persistent — the two launch strategies

| | **Tiled (prior)** | **Persistent (fix)** |
|:--|:--|:--|
| Grid | `cdiv(M,128) × cdiv(N,128)` — one CTA per (128,128) tile | `E × (#SMs // E) ≈ #SMs` — CTAs binned by group |
| CTA↔work | occupancy-based; scheduler fills SMs with tile-CTAs | each CTA bound to one group, strides its tiles (`num_stages`-pipelined) |
| Accumulation | per-tile partial amax | elementwise cumulative max held in registers across the stride loop |
| Atomics | one `atomic_max` **per tile** (~#tiles total) | one `atomic_max` **per CTA** (~#SMs total) |
| Tile shape | fixed 128×128 | autotuned (64×128 usually wins) |

The persistent kernel is the single-tensor kernel generalized: `E=1`,
`CTAS_PER_GROUP=#SMs` recovers it exactly. It relies on **one CTA per group**, so
dispatch is guarded by `E ≤ #SMs` *and* avg rows/group `≥ 1024`.

**If `#SMs < E`** (`num_tensors > num_sms`) the guard fails and dispatch **falls back
to the tiled kernel** — otherwise `ctas_per_group = #SMs // E` would be 0, giving an
empty `grid = E × 0` launch. On GB200 (**152 SMs**) with realistic `E ≤ 64` this never
triggers; the fallback is a safety net for small-GPU / pathological-E cases, not a
training path. (The standalone prototype instead clamps `max(1, #SMs // E)`, keeping
one CTA per group and oversubscribing when `E > #SMs` — correct, but a different
choice than the upstreamed fallback.)

`rht_quantize_row_col` is now the largest remaining **absolute** per-launch cost
(878 µs at 8K, 2× the IO — row+col codes and scales) and groups fine.
