# NVFP4 vs bf16 grouped-GEMM crossover (DeepSeek V3 671B)

One grouped GEMM at the 671B gate-projection shape, random data, forward only, on
a single **NVIDIA GB200**. Uniform tokens/expert. Shape: `K=7168`, `N=2048`,
`E=8`. Times are ms/call (warmup 5, 20 iters, CUDA-event timed).

- **`nvfp4_pure`** — the 4-bit tensor-core grouped matmul on **pre-quantized**
  inputs (quantize once, outside the timed loop). Isolates raw compute.
- **`nvfp4_full`** — `nvfp4_pure` + on-the-fly quantization of the activations and
  weights every call (RHT amax + row/col quantize + weight quantize). This is what
  the TorchAO training override actually runs.
- **`te_full`** — TransformerEngine's fused NVFP4 grouped GEMM (`_GroupedLinear`),
  which always quantizes activations + weights inside the call (graph-safe RHT
  cast-fusion kernel). TE's forward is inherently "full"; there is no pre-quantized
  split to isolate.

> **TorchAO now has two quantize backends.** The three grouped quantize ops each
> have a Triton and a CuteDSL implementation, and `_resolve_backends` picks CuteDSL
> per op under the default `kernel_preference=AUTO`; `TRITON` forces the old path.
> Every TorchAO row below is therefore split into **`cutedsl`** (today's default)
> and **`triton`**. The CuteDSL backend is 1.6–1.9× on the two activation RHT ops
> and 1.15× on the weight quantize (Table 2b), which moves both crossovers
> substantially. Tables 2 and 4 were re-measured from scratch for this; Tables 1,
> 3 and 5 carry forward from the prior run (same GPU and shapes; the shared bf16
> reference reproduces within ~5%, so the columns remain comparable).

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

## Table 2 — Quantization overhead, per backend

`nvfp4_full` re-measured on both TorchAO quantize backends against a fresh bf16
reference (which reproduces Table 1's bf16 within ~5%, confirming the two runs are
comparable). Overhead = `nvfp4_full − nvfp4_pure`, with `nvfp4_pure` from Table 1.
Repro: `nvfp4_grouped_gemm_fwd.py <bf16|cutedsl|triton>`.

| tok/exp | M (rows) | bf16 (ms) | full **cutedsl** (ms) | full **triton** (ms) | cutedsl/bf16 | triton/bf16 |
|--------:|---------:|----------:|----------------------:|---------------------:|-------------:|------------:|
|     128 |    1,024 |     0.061 |                 0.727 |                0.660 |         11.9 |        10.8 |
|     256 |    2,048 |     0.057 |                 0.719 |                0.647 |         12.6 |        11.4 |
|     512 |    4,096 |     0.095 |                 0.712 |                0.650 |          7.5 |         6.8 |
|   1,024 |    8,192 |     0.179 |                 0.695 |                0.664 |          3.9 |         3.7 |
|   2,048 |   16,384 |     0.307 |                 0.686 |                0.660 |          2.2 |         2.1 |
|   4,096 |   32,768 |     0.610 |                 0.700 |                0.854 |         1.15 |        1.40 |
|   8,192 |   65,536 |     1.257 |             **1.037** |                1.418 |     **0.82** |        1.13 |
|  16,384 |  131,072 |     3.029 |                 1.789 |            **2.552** |         0.59 |    **0.84** |
|  32,768 |  262,144 |     6.174 |                 3.375 |                4.827 |         0.55 |        0.78 |

Quant overhead per call (`full − pure`, ms) and as a multiple of the pure GEMM:

| tok/exp | pure (ms) | overhead cutedsl | ÷ pure | overhead triton | ÷ pure |
|--------:|----------:|-----------------:|-------:|----------------:|-------:|
|     512 |     0.107 |            0.605 |   5.7× |           0.543 |   5.1× |
|   2,048 |     0.111 |            0.575 |   5.2× |           0.549 |   4.9× |
|   8,192 |     0.338 |            0.699 |   2.1× |           1.080 |   3.2× |
|  32,768 |     1.824 |            1.551 |   0.9× |           3.003 |   1.6× |

**The forward crossover moved from ~50K to ~5K tokens/expert.** Both backends are
far cheaper than the number this doc previously carried (0.98→0.73 ms at the small
end, 7.66→3.38 at 32K), and the quant tax is no longer 8–11× the matmul — it is
~5× at small M and falls *below* the GEMM at 32K on CuteDSL. `nvfp4_full` crosses
bf16 at **~5K tok/expert on CuteDSL** and **~10K on Triton**.

**Below ~4K tok/expert Triton is the faster eager path** (~0.65 vs ~0.71 ms flat),
even though CuteDSL wins on device time at every size (Table 2b). Both are
launch-bound down there — the flat floor is host dispatch, not kernels — and
CuteDSL's custom-op dispatch is the more expensive of the two. The ordering flips
at 4K, where device time starts to dominate, and CuteDSL then widens to 1.43× at
32K. Under CUDA graphs the host term is captured away and the device-time ordering
should hold throughout; the eager floor here is the untraced worst case.

## Table 2b — CuteDSL vs Triton, per grouped quantize kernel

Device kernel time (profiler CUDA self-time, host dispatch excluded) for the three
grouped quantize ops at the DSV3-671B activation dim `N=7168`, `E=8`. Both backends'
public ops take identical arguments, so the same call site drives both.
Repro: `nvfp4_grouped_quant_backends.py`.

| kernel | tok/exp | triton µs | cutedsl µs | speedup |
|:-------|--------:|----------:|-----------:|--------:|
| **rht_amax** (act) |    512 |  39.7 |  15.4 | **2.59** |
| rht_amax           |  1,024 |  40.5 |  23.0 | 1.76 |
| rht_amax           |  2,048 |  65.7 |  40.7 | 1.61 |
| rht_amax           |  4,096 | 115.5 |  69.6 | 1.66 |
| rht_amax           |  8,192 | 216.3 | 129.6 | 1.67 |
| **rht_quantize_row_col** (act) |    512 |  42.3 |  33.0 | 1.28 |
| rht_quantize_row_col           |  1,024 |  84.3 |  48.9 | 1.73 |
| rht_quantize_row_col           |  2,048 | 163.9 |  90.8 | 1.81 |
| rht_quantize_row_col           |  4,096 | 322.8 | 174.0 | 1.86 |
| rht_quantize_row_col           |  8,192 | 640.8 | 336.2 | **1.91** |
| **weight_quantize_2d** (gate/up 2048×7168) | — | 180.4 | 156.2 | 1.15 |
| weight_quantize_2d (down 7168×2048)        | — | 178.6 | 155.7 | 1.15 |

CuteDSL wins on device time at **every** size and op: **1.6–1.9×** on the two
activation RHT kernels (the M-scaling term, and the one that sets the crossover)
and a steady **1.15×** on the token-independent weight quantize. `rht_amax`'s 2.59×
at 512 is the small-M outlier — the Triton persistent kernel has not amortized its
prologue there yet. The per-expert weight *amax* stays Triton on every path (no
CuteDSL twin) and is unchanged.

## Table 3 — TransformerEngine forward (fused quantize+GEMM)

Same shape (K=7168, N=2048, E=8), forward only. `te_full` is TE's fused NVFP4
grouped GEMM. TE is untouched by the TorchAO work, so this column carries forward
unchanged; the `torchao` column is the new Table 2 CuteDSL measurement.

| tok/exp | M (rows) | bf16 (ms) | te_full (ms) | te/bf16 | torchao cutedsl (ms) | AO vs TE |
|--------:|---------:|----------:|-------------:|--------:|---------------------:|---------:|
|     128 |    1,024 |     0.056 |        0.911 |    16.2 |                0.727 | **1.25×** |
|     256 |    2,048 |     0.056 |        0.903 |    16.3 |                0.719 | 1.26× |
|     512 |    4,096 |     0.096 |        0.849 |     8.9 |                0.712 | 1.19× |
|   1,024 |    8,192 |     0.178 |        0.867 |     4.9 |                0.695 | 1.25× |
|   2,048 |   16,384 |     0.301 |        0.866 |     2.9 |                0.686 | 1.26× |
|   4,096 |   32,768 |     0.605 |        0.872 |     1.4 |                0.700 | 1.25× |
|   8,192 |   65,536 |     1.210 |        1.195 | **0.99**|                1.037 | 1.15× |
|  16,384 |  131,072 |     2.897 |        1.988 |    0.69 |                1.789 | 1.11× |
|  32,768 |  262,144 |     6.297 |        3.639 |    0.58 |                3.375 | 1.08× |

**TE's forward crosses bf16 at ~8K tokens/expert; TorchAO-CuteDSL now crosses at
~5K and is faster than TE at every point** (1.1–1.26×). This reverses the previous
reading — TE's flat ~0.85 ms fused-quantize floor used to be well under TorchAO's
~1.0–1.4 ms, and CuteDSL has taken TorchAO's floor to ~0.70 ms. On the forward path
**TorchAO-CuteDSL is now the stronger backend**, and TE's remaining advantage is
confined to the backward (Table 3b, Table 4).

## Table 3b — TE backward (single grouped GEMM)

TE's backward is well-behaved — cheaper than the forward at large E (the forward
carries the activation RHT quantize):

| config | TE fwd (ms) | TE bwd (ms) | bwd/fwd |
|:-------|------------:|------------:|--------:|
| E=8,  TPE=2048 | 0.73 | 0.94 | 1.3× |
| E=64, TPE=512  | 3.61 | 2.44 | 0.7× |
| E=64, TPE=2048 | 3.38 | 2.55 | 0.8× |

**E=64 end-to-end, re-measured** (3-GEMM expert MLP fwd+bwd, ms/iter, same session
and harness as Table 4). The previous reading here — TE 26 ms vs TorchAO 36 ms at
E=64/512 — has inverted:

| E=64 | bf16 | AO-cutedsl | AO-triton | TE |
|:--|---:|---:|---:|---:|
| 512 tok/exp (32,768 rows) | 16.00 | **19.13** | 21.36 | 35.94 |
| 2,048 tok/exp (131,072 rows) | 34.37 | **27.26** | 32.94 | 39.43 |

At E=64 TorchAO is **1.9× faster than TE** at 512 tok/exp and 1.45× at 2,048, and
TE no longer beats bf16 at either point. Compare the same *total* rows at E=8
(Table 4, 4K tok/exp: bf16 7.51, AO 5.76, TE 7.34): high expert counts cost every
backend 2–5×, and cost TE the most. Two caveats — the TE build has moved since the
26 ms figure was taken, so part of this is a TE-side change rather than a TorchAO
gain; and E=64 is exactly TE's `kMaxTensorsPerKernel=64` cap (hard assert from the
<4 KB launch-args struct, no chunking) and exactly CuteDSL's `MAX_GROUPS=64`, so
both are at their group limit here. All of this is moot at realistic training EP
(~4 experts/GPU), which is the E=8 regime of Table 4.

## Table 4 — Training fwd+bwd sweep (bf16 vs TorchAO vs TE vs MXFP8)

3-GEMM expert MLP, forward+backward, E=8. ms/iter; ratio to bf16 (<1 = beats bf16).
**All five columns re-measured in one session** on the same GB200 and the same
harness, so every ratio is like-for-like. TorchAO appears twice: `AO-cutedsl` is
today's `AUTO` default, `AO-triton` forces the old backend.

MXFP8 uses torchtitan's first-class `MXFP8GroupedExpertsConverter` (recipe `mxfp8_rceil`,
e4m3 data + e8m0 block-32 scales), measured on the **CUDA-built** torchao (the `_C_mxfp8`
extension; the editable `USE_CPP=0` build lacks it).
Repro: `nvfp4_grouped_gemm_crossover.py <bf16|torchao|torchao_triton|te|mxfp8>`.
Numerics cross-check at E=8/512 against the bf16 module with identical weights:
TorchAO and TE land on the same SQNR to 0.1 dB (fwd 12.0, dgrad 11.5, wgrad 11.4),
so these are speed differences between two backends computing the same thing.

| tok/exp | rows | bf16 | AO-cutedsl | AO-triton | TE | MXFP8 |
|--------:|--------:|------:|------:|------:|------:|------:|
|     512 |   4,096 |  2.15 |  5.75 |  4.89 |  7.16 |  4.11 |
|   1,024 |   8,192 |  2.87 |  5.79 |  5.10 |  7.17 |  4.05 |
|   2,048 |  16,384 |  4.31 |  5.78 |  5.03 |  7.15 |  4.21 |
|   4,096 |  32,768 |  7.51 |  5.76 |  6.57 |  7.34 |  6.38 |
|   8,192 |  65,536 | 14.31 |  8.34 | 10.69 |  9.30 | 10.85 |
|  16,384 | 131,072 | 30.92 | **14.31** | 19.12 | 14.57 | 20.37 |
|  32,768 | 262,144 | 60.97 | 26.56 | 35.91 | **25.79** | 39.59 |

Ratio to bf16:

| tok/exp | AO-cutedsl | AO-triton | TE | MXFP8 |
|--------:|-----------:|----------:|-------:|------:|
|     512 |       2.67 |      2.27 |   3.33 |  1.91 |
|   1,024 |       2.02 |      1.78 |   2.50 |  1.41 |
|   2,048 |       1.34 |      1.17 |   1.66 | **0.98** ← MXFP8 |
|   4,096 |   **0.77** |  **0.87** | **0.98** |  0.85 |
|   8,192 |       0.58 |      0.75 |   0.65 |  0.76 |
|  16,384 |       0.46 |      0.62 |   0.47 |  0.66 |
|  32,768 |       0.44 |      0.59 |   0.42 |  0.65 |

**TorchAO-CuteDSL has closed the gap to TE.** The two are now within 2–3% at
production M (16K: AO 14.31 vs TE 14.57 — AO ahead; 32K: 26.56 vs 25.79 — TE
ahead), against a 1.4× TE lead in the previous sweep. Peak memory is a tie
(bf16 21.1, AO 22.6 both backends, TE 23.6, MXFP8 23.5 GiB at 32K).

**Every NVFP4 backend now crosses bf16 at ~3–4K tok/expert** — AO-CuteDSL 0.77 and
AO-triton 0.87 at 4K, TE 0.98 (log-interpolated crossings: AO-CuteDSL 2.9K,
AO-triton 3.0K, TE 4.0K). TorchAO's training crossover has moved ~16K → ~8K →
**~3K** across the two rounds of kernel work; it is no longer the late-crossing
backend. Below ~2K bf16 still wins for all three.

**MXFP8 is the earliest crossover (~2K) but no longer the mid-range winner.** Its
single cheap block-scale cast still gives the lowest floor (~4.1 ms flat to 2K, vs
AO-CuteDSL's 5.75), so it is the fastest backend below ~3.5K. Above that the 4-bit
backends pass it and keep pulling away: at 32K, TE 25.8 ≈ AO-CuteDSL 26.6 < AO-triton
35.9 < MXFP8 39.6 < bf16 61.0 ms. **Ordering at scale: TE ≈ AO-cutedsl < AO-triton <
MXFP8 < bf16.** The AO-cutedsl-vs-MXFP8 crossover is ~3.5K, and by 16K AO-CuteDSL is
**1.42× faster than MXFP8**; the previous conclusion that MXFP8 beats TorchAO-NVFP4
at every M ≥ 4K no longer holds.

**The CuteDSL default is worth 1.35× at scale** (35.91 → 26.56 ms at 32K; 1.34× at
16K, 1.28× at 8K), tracking the 1.6–1.9× per-kernel device-time win in Table 2b
diluted by the GEMMs and the backward. Memory is unchanged. Below ~4K the sign
flips — AO-triton is 1.14–1.18× faster there (4.89 vs 5.75 at 512) for the eager
host-dispatch reason in Table 2: those sizes are launch-bound, not kernel-bound.
**If you run eager at ≤2K tok/expert, `kernel_preference=TRITON` is the faster
choice; everywhere else, and under CUDA graphs, take the default.**

### Table 4b — All five backends at scale, repeated

Table 4 is single-shot and stops at 32K. This repeats every point **3× and reports the
min**, and extends to 65,536 tok/expert — a size the CuteDSL backend could not run
before the Int32 fix below. Same harness, same GB200. Run-to-run spread is ≤0.6%
everywhere except bf16 at 16,384, which came in at 7.4%.

| tok/exp | rows | bf16 | **AO-cutedsl** | AO-triton | TE | MXFP8 |
|--------:|--------:|-------:|-----------:|----------:|------:|------:|
|   8,192 |  65,536 |  14.02 |  **8.32** | 10.70 |  9.35 | 10.90 |
|  16,384 | 131,072 |  28.75 | **14.30** | 19.13 | 14.57 | 19.98 |
|  32,768 | 262,144 |  59.68 |     26.30 | 35.92 | **25.70** | 38.51 |
|  65,536 | 524,288 | 121.27 |     50.58 | 69.39 | **48.51** | 76.28 |

Ratio to bf16:

| tok/exp | AO-cutedsl | AO-triton | TE | MXFP8 |
|--------:|-----------:|----------:|------:|------:|
|   8,192 | **0.59** | 0.76 | 0.67 | 0.78 |
|  16,384 | **0.50** | 0.67 | 0.51 | 0.70 |
|  32,768 |     0.44 | 0.60 | **0.43** | 0.65 |
|  65,536 |     0.42 | 0.57 | **0.40** | 0.63 |

**The AO-vs-TE lead changes hands at ~22K tok/expert.** AO-CuteDSL is ahead at 8K
(1.12×) and 16K (1.02×); TE takes it at 32K (1.02×) and 65K (1.04×), widening slowly.
The two are within 4% at every point — effectively co-equal, against the 1.4× TE lead
this doc carried before the CuteDSL backend landed. At 65,536 tok/expert that is 10.4
Mtok/s (AO-CuteDSL) and 10.8 (TE) against bf16's 4.3.

**CuteDSL's margin over the other TorchAO path and over MXFP8 widens with M**, which is
what amortizing a quant tax looks like:

| CuteDSL vs | 8,192 | 16,384 | 32,768 | 65,536 |
|:--|--:|--:|--:|--:|
| AO-triton | 1.29× | 1.34× | 1.37× | 1.37× |
| MXFP8 | 1.31× | 1.40× | 1.46× | **1.51×** |

**The Triton fallback beats MXFP8 too, but only by 2–10%** (1.02×, 1.04×, 1.07×, 1.10×
across the sweep, rising to 1.12× at the `E=4` split below). That is a poor trade for
dropping from 8-bit to 4-bit precision, and it is the one comparison where the ordering
is close enough that the 3× repetition matters. The 4-bit path is only decisively worth
it on CuteDSL.

**Practical rule: where CuteDSL cannot run, prefer MXFP8 over TorchAO-NVFP4.** The
fallbacks to Triton are the TP path (`nvfp4_training.py` keeps `use_cutedsl =
(preference == CUTEDSL)`, so AUTO runs TP on Triton), `E > MAX_GROUPS=64`, and
`N % 256 != 0` for the weight quantize. In those configurations the NVFP4 quant tax eats
nearly all of the 4-bit compute advantage over MX 8-bit.

**Cross-check at the production group split.** The same 524,288 rows as `E=4 × 131,072`
(the DSV3-671B EP=64 layout) rather than `E=8 × 65,536`:

| | bf16 | AO-cutedsl | AO-triton | MXFP8 |
|:--|--:|--:|--:|--:|
| E=4 × 131,072 | 122.61 | **50.29** | 68.48 | 76.62 |
| E=8 × 65,536 | 121.27 | 50.58 | 69.39 | 76.28 |

Every backend lands within ~1.3%, so cost tracks **total rows per rank**, not how they
are split across experts — the same invariance the M/expert derivation predicts. TE was
not measured at `E=4`.

## Fixed — CuteDSL columnwise scale prefix overflowed Int32 (torchao `ca485bcc`)

`cutedsl_group_rht_quantize_row_col` wrote out of bounds, silently corrupting the
columnwise scales, once rows/rank grew past ~342K. **Root-caused and fixed**; this
section is kept because the failure mode is instructive and because anyone on a torchao
before `ca485bcc` still has it.

### Root cause

`_store_grouped_col_sf_u32` addresses a group's swizzled columnwise scale tile by the
span of the preceding groups:

```python
prefix_words = hidden * group_start // cutlass.Int32(64)   # both operands Int32
```

Both operands are `Int32`, so the **product** wraps past 2^31. At 671B `hidden=7168`
that happens the moment `group_start` reaches `2^31 / 7168 =` **299,593 rows**: the
multiply goes negative, the quotient with it, and the scale word is stored far below
the buffer. The quotient itself is harmless (≤ `hidden × tokens / 64`, ~59M words), so
only the multiply needed widening — the index arithmetic after it stays 32-bit.

`compute-sanitizer` named it directly: an invalid **4-byte** `__global__` write from
`Tcgen05GroupRowColFused` (the 4-byte store is the *scale* write; the FP4 store is a
16-byte `STG.E.128`), at an address **112,918,524 bytes _before_** the nearest
allocation. The negative displacement was the tell.

### The damage is silent, and starts well below any visible failure

Past the overflow the kernel stores scales at a negative offset and carries on: the
wrapped address usually still lands inside another live allocation, so training
continues on wrong scales with no signal at all. Columnwise scale output vs the Triton
backend, per group, E=8, hidden 7168 — the ~1.5% baseline is the pre-existing difference
between the two backends, present at every size:

| rows | max(`hidden × group_start`) | | sfd differing, **before** | **after** |
|--:|--:|:--|--:|--:|
| 32,768 | 205,520,896 | | 1.6% | 1.6% |
| 262,144 | 1,644,167,168 | ← live production size | 1.5% | 1.5% |
| 360,448 | 2,260,729,856 | past 2^31 | **12.7%** | 1.8% |
| 458,752 | 2,877,292,544 | past 2^31 | **unusable** | 1.7% |

**Corruption begins at ~342K rows/rank.** Above ~459K the bad address additionally falls
outside a mapped segment and the process dies, but that is the tail of the bug rather
than its boundary — the correctness threshold is the overflow, ~117K rows earlier. An
earlier revision of this note took the visible failure for the threshold and treated
everything under it as clean; that was wrong for the whole 342K–459K band.

**The live production run at 262,144 rows/rank was unaffected** — it sits below the
overflow itself, not merely below the point where the overflow becomes visible. Confirmed
by both the numerics above and zero sanitizer findings.

### After the fix

- `compute-sanitizer`: zero invalid accesses at the shape that previously wrote adrift.
- The 3-GEMM MoE forward at 524,288 rows runs in **50.3 ms** vs the Triton backend's
  68.5 ms; it was previously unusable. This fills in the CuteDSL entries in Table 4b at
  65,536 tok/expert and at E=4 × 131,072.
- No measurable cost — the multiply is once per work item. Grouped
  `rht_quantize_row_col` over 512–8192 tok/expert is unchanged at 1.29×, 1.74×, 1.81×,
  1.87×, 1.92× over Triton (was 1.28×, 1.73×, 1.81×, 1.86×, 1.91×).
- `test_group_rht_quantize_row_col.py` + `test_nvfp4_grouped_mm.py`: 85 passed.

**Not covered by a regression test.** Every existing test shape is far below 299,593
rows; reaching the overflow needs ~6 GB of activations, so nothing in CI would have
caught this and nothing guards it now.

## How the crossover point moves

| Regime | Crossover (tokens/expert) | Why |
|:-------|:--------------------------|:----|
| **Pure 4-bit GEMM** | **~1K** | raw tensor-core compute; only kernel launch overhead to amortize |
| **TE forward (quant + GEMM)** | **~8K** (measured) | TE's fused quantize is nearly flat (~0.85 ms floor) |
| **TorchAO fwd GEMM — CuteDSL** | **~5K** (measured; was ~50K) | CuteDSL quant floor ~0.70 ms, below TE's; quant tax down to ~5× the matmul at small M and <1× at 32K |
| **TorchAO fwd GEMM — Triton** | **~10K** (measured) | same structure at 1.6–1.9× the device time on the two activation RHT ops |
| **Training fwd+bwd — TE (3-GEMM MLP)** | **~4K** (measured) | TE's low, flat fused-quant floor + a cheap grouped backward amortize over the fwd + 2 bwd GEMMs |
| **Training fwd+bwd — TorchAO CuteDSL** | **~3K** (measured; was ~8K, ~16K before that) | CuteDSL grouped quantize kernels; now at TE parity from 8K up |
| **Training fwd+bwd — TorchAO Triton** | **~3K** (measured) | crosses at the same point but on a flatter slope — 0.59 vs 0.44 of bf16 at 32K |
| **Training fwd+bwd — MXFP8 (3-GEMM MLP)** | **~2K** (measured) | MX 8-bit single block-scale cast → flat ~4.1 ms quant floor, the earliest crossover; but 8-bit compute (~1.5× bf16 at scale) yields to both NVFP4 backends above ~5K |

The 4-bit kernels pay off early (~1K tok/expert), and the backends have converged on
the per-call quantization tax that used to separate them. **TorchAO-CuteDSL** and
**TE** are now equivalent training backends at E=8 (within 3% at M ≥ 16K — AO ahead
at 16K, TE ahead at 32K), both crossing bf16 at ~3–4K. TorchAO is the faster of the
two on the forward at every M (Table 3), and **decisively faster at E=64**
(1.45–1.9×, Table 3b); TE's edge is confined to the E=8 backward at the very top end.
TorchAO's training crossover has moved ~16K → ~8K → **~3K** over three rounds of
kernel work: first the persistent `rht_amax` dispatch and autotuned grouped quantize
launches (Table 5, ~15%), then porting all three grouped quantize ops to CuteDSL
(Table 2b, a further 1.35× at scale). `torch.compile` gives only a constant-factor
win (best ~25% mid-range, ~5% at large M) and does **not** shift the crossover — the
cost lives inside opaque hand-written kernels dynamo cannot fuse into.

What is left is no longer kernel-level. With the quantize ops at ~1.15–1.9× of their
Triton twins and TE parity reached, the remaining TorchAO headroom is algorithmic
(caching quantized weights across microbatches, a cheaper activation transform) plus
the eager host-dispatch floor that dominates below ~4K tok/expert — which CUDA graphs,
not a faster kernel, is the fix for.

## Which operating point is realistic? (2025 default vs 2026 frontier)

The crossover only matters relative to the **M/expert** an actual run produces:

```
M/expert            = local_tokens × top_k × EP / num_experts
total routed rows/GPU = local_tokens × top_k   (EP-invariant)
```

**The torchtitan DSV3-671B default is *not* a representative operating point.** It ships
`local_batch=4, seq_len=4096, EP=2` → M = 2·4·4096·8/256 = **1,024 tok/expert** — the
bottom of the table, where bf16 *beats* NVFP4 (AO 2.0×, TE 2.5× at 1K). That's an
`EP=2` small-scale/template artifact (128 experts/GPU, tiny per-expert GEMMs), not the
production layout. The run actually being trained here uses **EP=32 at
`local_batch_size=8`** → M = 32,768 tok/expert over 8 local experts (verified below).

**Summer-2026 frontier MoE configs land deep in the NVFP4-winning regime.** Assuming a
1M-context packed sequence sharded across 64 token-parallel ranks (16,384 local tokens/GPU):

| Model (routed experts, top-k) | EP | E_local | M/expert | rows/GPU | ~AO/bf16 | ~TE/bf16 |
|:------|---:|---:|---:|---:|---:|---:|
| DeepSeek V4-Pro (384, top-6) | 64 | 6 | 16,384 | 98,304 | 0.46 | 0.47 |
| GLM-5.2 (256, top-8) | 32 | 8 | 16,384 | 131,072 | 0.46 | 0.47 |
| GLM-5.2 | 64 | 4 | 32,768 | 131,072 | 0.44 | 0.42 |
| Kimi K3 (896, top-16) | 64 | 14 | 18,725 | 262,144 | 0.46 | 0.46 |
| Kimi K3 | 128 | 7 | 37,449 | 262,144 | ~0.44 | ~0.42 |

Every config sits at **M ≥ 16K tok/expert** — 5–12× past the ~3K crossover — so NVFP4 is
**~2.2× faster than bf16** across the board, on either backend (AO-CuteDSL and TE are
within 3% of each other at every one of these M). Long context + high top-k (6/8/16)
keep per-expert M large *even with heavy expert sharding*: token parallelism cuts
local_tokens but top_k multiplies routed rows back up. **EP is the knob** — it trades
E_local for M/expert at constant total work, so raising EP pushes *deeper* into the
NVFP4 win (GLM: EP 32→64 moves M 16K→32K, AO 0.46→0.44), paying only more all-to-all.
All five have avg rows/group ≫ the 1K persistent-`rht_amax` threshold and
E_local ≪ 152 SMs (and ≪ the CuteDSL `MAX_GROUPS=64` cap), so every kernel fix and
the CuteDSL default are active here.

(Ratios interpolated from Table 4's E=8, DSV3 dims — regime holds regardless of exact
expert dims since the crossover is governed by M.)

### Verified: how M/expert is actually produced (instrumented torchtitan run)

The formula above is not just arithmetic — the token flow was traced end-to-end in
torchtitan (`trainer.py`, `models/common/moe.py`, `token_dispatcher.py`) and confirmed
with an instrumented run, because the M an operating point produces is easy to
mis-derive by a factor of `local_batch` or `EP`.

**One `model.forward` processes the entire local batch — all sequences at once, never
one.** `trainer.py` splits a step into `gradient_accumulation_steps = global_batch /
(local_batch × dp)` microbatches, but **each microbatch is the full `(local_batch,
seq_len)` tensor** (`trainer.py:772,791`); the MoE flattens it to `T = local_batch ×
seq_len` rows before routing (`moe.py:129`). Grad-accum repeats the full-local-batch
forward — it never slices down to a sequence. So "one microbatch = one sequence" is
wrong; `M` scales with `local_batch`, not with 1.

Empirical trace — debugmodel, EP=1, `local_batch=16, seq=4096`, `MOE_INSTR=1`
(env-gated hook at the MoE call site):

```
Trainer: local batch 16, global batch 16, gradient accumulation steps 1, seq 4096
moe_forward   T(pre_routing_tokens)=65536  K=3  routed_slots(T*K)=196608   # every forward
experts_forward e=8  R(sumM)=196608  M_i=[…4902 … 55023 …]                 # ∑M = T×top_k
```

- `T = 65,536 = 16 × 4096` on **every** forward → full local batch, confirmed.
- At EP=1, `∑Mᵢ = T × top_k` exactly (the grouped GEMM sees all locally-routed slots).
- `Mᵢ` is **heavily skewed** — 4.9K–55K vs 24.6K mean (≈2.2× hot/mean), not uniform.

**The EP all-to-all conserves rows per GPU; it does not divide them.** With EP>1
(`token_dispatcher.py:460,475`) a rank ships its `T×top_k` routed slots out to `EP`
ranks (~`T×top_k / EP` to each) and receives a comparable count back for its
`E_local = E/EP` experts. By send≈receive symmetry the grouped GEMM's aggregate stays
pinned near `T×top_k`:

```
∑Mᵢ per expert GPU ≈ local_tokens × top_k          (conserved — NOT ÷ EP)
per-expert M       = ∑Mᵢ / E_local = local_tokens × top_k × EP / E
```

Dividing the aggregate by `EP` gives the per-(source-rank → destination-GPU) shard, not
a per-GPU or per-expert quantity. Raising `EP` gives each GPU *fewer* experts, so it
*concentrates* rows per expert (pushes deeper into NVFP4), at constant aggregate/GPU.

**The configuration actually being trained** is `local_batch_size=8, seq=4096, top-8,
256 experts, EP=32` on 64 GB300s — 32,768 tokens/GPU/step, 2.1M tokens/step globally:

```
EP group tokens   = 32 ranks × 32,768        = 1,048,576
token-expert pairs= 1,048,576 × 8            = 8,388,608
per-expert M      = 8,388,608 / 256          =    32,768 rows   (E_local = 256/32 = 8)
∑Mᵢ per GPU       = 32,768 × 8               =   262,144 rows
```

That is ~11× past the ~3K AO crossover, deep in the NVFP4 win. It is skew-robust: even
a cold expert at ~0.4× mean ≈ 13K is ~4× past crossover, and the 128-row group padding
wastes ≤ 128/32768 ≈ **0.4%** at this scale (vs ~1–3% near the crossover).

`∑Mᵢ/GPU = local_tokens × top_k` is **EP-invariant** (derived above), so `EP` only
re-splits those rows across `E_local` experts; `local_batch_size` and `seq_len` are the
knobs that move the total:

| local_batch (seq=4096) | ∑Mᵢ/GPU | per-expert M @EP=32 | vs AO crossover ~3K |
|--:|--:|--:|:--|
| 16 | 524,288 | 65,536 | 22× — but see the CuteDSL fault above |
| **8 (the live run)** | **262,144** | **32,768** | 11× — deep NVFP4 |
| 4 (torchtitan `deepseek_v3_671b` default) | 131,072 | 16,384 | 5× — deep NVFP4 |
| 1 | 32,768 | 4,096 | ~1× — at the crossover |

**Grouped-GEMM launches per optimizer step.** Each MoE layer issues **3
`torch._grouped_mm`** (w1, w3, w2; `moe.py:97-108`); profiler count on one layer's
fwd+bwd is **9 `aten::_grouped_mm`** (3 fwd + 6 bwd). DSV3-671B has **58 MoE layers**
(`n_layers=61, n_dense=3`), so per optimizer step:

```
≈ 9 × 58 × grad_accum_steps  ≈ 522·g grouped-GEMM launches   (no activation ckpt)
≈ 12 × 58 × grad_accum_steps ≈ 696·g                          (with SelectiveAC recompute)
```

(SelectiveAC re-runs the 3 forward GEMMs in backward — the instrumented run showed the
forward path executing twice per step, matching 12/layer.)

## Table 5 — Are TorchAO's grouped quant kernels worth grouping? (Triton path)

> Scope: everything below concerns the **Triton** backend, which is now the fallback
> rather than the default (Table 2b). It is kept because it is what settled the
> grouping question and motivated the persistent `rht_amax` kernel — and because the
> Triton path is still what runs under `kernel_preference=TRITON`, on the TP path,
> and wherever CuteDSL's `MAX_GROUPS=64` / `N % 256` limits bite. The CuteDSL
> kernels group by construction (one launch over `offsets`), so the grouped-vs-
> per-expert question does not re-arise for them. The `rht_quantize_row_col` and
> `weight_quantize_2d` rows are **pre-autotune** launches (fixed `num_warps=8`);
> the tuned figures are the `triton` column of Table 2b (641 vs 878 µs at 8K).
> Ratios are unaffected in sign — both grouped kernels only got faster.

Each of TorchAO's three forward NVFP4 quant kernels has a single-expert twin. This
compares one **grouped** launch over `E=8` experts against `E ×` the **single**
kernel at the same per-expert size — i.e. the cost of just launching the single
kernel once per expert. `ratio = grouped ÷ (E × single)`; **>1 means grouping is
slower than per-expert launches.** DSV3-671B activation dim `N=7168`. Raw kernels,
pre-allocated buffers (times the kernel body only). For `rht_amax`, grouped µs is
the **dispatched** kernel body — tiled ≤1K avg rows/group, per-group-CTA persistent
above (the fix, see below). Repro: `nvfp4_grouped_kernel_grouping.py`.

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
E-sweep rules both out. At fixed *total* rows (65536),
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

`rht_quantize_row_col` is the largest **absolute** per-launch cost on this path
(878 µs at 8K, 2× the IO — row+col codes and scales) and groups fine. The persistent
strategy does **not** transfer to it: a bitwise-validated per-group-CTA persistent
prototype is **~15% slower** than the tiled kernel (0.85× at 8K). Unlike the launch-bound amax reduction, the
quantize kernels are element-wise maps (output == input size, no accumulator, no
atomics) and compute/IO-bound — their single-tensor twins are already
persistent+autotuned yet the plain tiled grouped kernel still matches them per row, so
there is no launch overhead for persistence to recover. The same holds for
`weight_quantize_2d` (already 0.82, token-independent).

The real lever for these kernels was **autotuning**, not persistence. They shipped
with a fixed `num_warps=8/num_stages=3` launch while their single-tensor twins were
`@triton.autotune`d. Sweeping that config space (all configs bitwise-validated)
showed a **~1.4× speedup** on `rht_quantize_row_col` (882→618 µs at 8K), entirely
from `num_warps=4` (the 128×128 tile was already optimal; 8 warps over-subscribed
registers on this quantize-heavy body). **Done** —
`_group_rht_quantize_row_col_kernel` and `_group_weight_quantize_2d_kernel` both
carry `@triton.autotune` now, and the Table 2b Triton column is the tuned path.
CuteDSL then took the same kernel a further 1.9× (336 vs 641 µs at 8K), which is
where the remaining device-time win came from.
