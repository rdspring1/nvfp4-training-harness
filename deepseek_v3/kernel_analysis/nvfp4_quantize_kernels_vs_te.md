# TorchAO NVFP4 quantize kernels vs TransformerEngine

TorchAO's CuteDSL NVFP4 kernels are a port of TE's. This asks two things: does the port
reproduce TE's arithmetic, and is it competitive. Single **NVIDIA GB200**, TE
2.19.0.dev0, PyTorch 2.15.0a0+git04a7716, Triton 3.8.0, nvidia-cutlass-dsl 4.5.2.
Shapes are the DeepSeek-V3 expert dims at `E = 4` local experts.

Times are device kernel time (`torch.profiler` CUDA self-time via torchao's
`bench_utils.kernel_time_us`), so they are directly comparable to the tables in
torchao's own benchmark README.

TE's amax pass is skipped on both sides: `tex.nvfp4_quantize_with_amax` takes
precomputed amaxes, which is what makes it a like-for-like counterpart to torchao's
quantize-only ops. `optimize_for_gemm=True` so TE emits the swizzled scales torchao
always emits.

## 1. Conformance — torchao sits between TE's two numeric modes

`NVTE_USE_FAST_MATH` selects between two genuinely different TE recipes
(`quantizer.cpp:2676-2686`):

- **default** — IEEE divide for the per-block encode reciprocal, plus a bf16 accumulator
  round-trip "for bit-wise compatibility with unfused kernels".
- **fast math** — `reciprocal_approximate_ftz`, and the fusion kernel drops the bf16
  round-trip so the cast runs on fp32 data.

TorchAO takes **TE's default bf16 rounding and TE's fast-math reciprocal**, so it matches
each mode on the half that mode shares with it. Percentage of differing bytes,
`rht_quantize` at 671B (identical for both torchao backends, which are bitwise equal to
each other):

| TE mode | row codes | col codes | row sf | col sf |
|---|---:|---:|---:|---:|
| default (`NVTE_USE_FAST_MATH` unset) | 0.0000 | **0.0000** | 0.0000 | 0.0000 |
| fast math (`NVTE_USE_FAST_MATH=1`) | **0.0000** | 1.4021 | 0.0000 | 1.8271 |

Reading it:

- **All four scale-factor tensors are bitwise identical to TE in default mode**, at every
  shape and for both backends. So the whole two-level scale chain — `2688/amax` with
  `div_rn`, the `amax * (S_enc/6)` association, the E4M3 cast, no lower clamp — is exact.
- **Columnwise codes are bitwise identical to TE in default mode.** They differ under
  fast math because torchao rounds the Hadamard accumulator to bf16 (deliberately, to
  match TE's shipped default and TransformerEngine's bf16 RHT output tensor) and TE's
  fast-math path does not.
- **Rowwise codes are bitwise identical to TE under fast math**, and differ by ~0.05% in
  default mode — torchao uses `rcp.approx.f32` for the per-block encode reciprocal where
  TE's default uses an IEEE divide. Those differences are all on E2M1 rounding midpoints.
- `weight_quantize_2d` is bitwise identical to TE in **both** modes at 16B and 671B
  (0.0946% at the 256x256 debug model, again midpoint ties).

TorchAO's in-tree tests pin this independently: both backends reproduce a plain-PyTorch
transcription of TE's arithmetic (`torchao/prototype/moe_training/nvfp4_training/
nvfp4_reference.py`) with scales bitwise and codes inside a few-ulp encode-scale bracket.

### Not measured: the grouped ops

TE's grouped path is **omitted from the conformance table**, not because torchao fails it
but because this harness's TE grouped invocation is unvalidated: at `E = 1`,
`tex.nvfp4_group_quantize_with_amax` does not reproduce TE's own single-tensor output on
identical data and identical amaxes (99.9% of codes differ). Until that is understood,
any grouped conformance number here would be meaningless, and the grouped timing column
is withheld for the same reason — an unvalidated configuration can be fast for the wrong
reason.

## 2. Timing — single-tensor ops, TE is ahead

Device kernel time (us), `NVTE_USE_FAST_MATH=1` (the faster TE configuration; its default
is 10-25% slower on `rht_quantize`).

| model | projection | op | M | N | triton | cutedsl | te | cutedsl/te |
|---|---|---|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up | rht_quantize | 256 | 256 | 12.10 | 8.76 | **4.45** | 1.97x |
| debugmodel | gate/up | weight_quantize_2d | 256 | 256 | 15.31 | 9.30 | **6.39** | 1.46x |
| 16B | gate/up | rht_quantize | 1408 | 2048 | 15.97 | **9.23** | 11.67† | 0.79x† |
| 16B | gate/up | weight_quantize_2d | 1408 | 2048 | 20.71 | 9.82 | **7.52** | 1.31x |
| 671B | gate/up | rht_quantize | 2048 | 7168 | 33.31 | 20.54 | **9.61** | 2.14x |
| 671B | gate/up | weight_quantize_2d | 2048 | 7168 | 43.18 | 22.41 | **12.97** | 1.73x |
| 671B | down | rht_quantize | 7168 | 2048 | 33.01 | 20.41 | **9.52** | 2.15x |
| 671B | down | weight_quantize_2d | 7168 | 2048 | 43.24 | 22.37 | **13.37** | 1.67x |

† the 16B `rht_quantize` row is from the default-mode run; under fast math TE lands at
9.01 us (1.04x). It is the one shape where the two are near parity.

**TE is 1.3-2.2x faster than CuteDSL on the single-tensor quantize ops**, and CuteDSL is
in turn 1.5-2.0x faster than Triton. The gap is widest on the fused RHT quantize at 671B
(2.15x) and narrowest on the 2D weight quantize (1.3-1.7x). This is the headline result:
the port is numerically faithful but has real performance headroom against the kernels it
was ported from.

## 3. Timing — grouped ops, no TE counterpart

| model | projection | op | E | M | N | triton | cutedsl |
|---|---|---|---:|---:|---:|---:|---:|
| 16B | gate/up | group_rht_amax | 4 | 1408 | 2048 | 18.31 | **10.26** |
| 16B | gate/up | group_rht_quantize | 4 | 1408 | 2048 | 20.38 | **19.24** |
| 16B | gate/up | group_weight_quantize_2d | 4 | 1408 | 2048 | n/a | 21.18 |
| 671B | gate/up | group_rht_amax | 4 | 2048 | 7168 | 40.55 | **22.94** |
| 671B | gate/up | group_rht_quantize | 4 | 2048 | 7168 | 87.37 | **52.25** |
| 671B | gate/up | group_weight_quantize_2d | 4 | 2048 | 7168 | n/a | 81.17 |
| 671B | down | group_rht_amax | 4 | 7168 | 2048 | 40.69 | **22.19** |
| 671B | down | group_rht_quantize | 4 | 7168 | 2048 | 85.08 | **52.76** |
| 671B | down | group_weight_quantize_2d | 4 | 7168 | 2048 | n/a | 80.79 |

Two structural gaps on TE's side, independent of the invocation problem above:

- **No amax-only entry point is reachable from PyTorch.** There is no binding for any
  `nvte_hadamard_transform*` symbol; `nvte_hadamard_transform_amax` runs only inside a
  quantizer that also casts. A head-to-head on `group_rht_amax` is not possible without
  attributing the amax kernel out of a profiler trace.
- **TE 2.19 cannot do grouped 2D weight quantize at all.**
  `NVTE_CHECK(!with_2d_quantization, "2D scaling grouped quant kernel is not ready yet")`
  (`extensions/cast.cpp:154`); the grouped non-RHT path is likewise unimplemented
  (`:158`). torchao's `group_weight_quantize_2d` has no TE counterpart to lose to.

## Reproducing

```bash
# from the repo root; torchao is editable-installed, TE comes from the base image
python deepseek_v3/kernel_analysis/nvfp4_quantize_kernels_vs_te.py
NVTE_USE_FAST_MATH=1 python deepseek_v3/kernel_analysis/nvfp4_quantize_kernels_vs_te.py
```

This lives outside `third_party/torchao` on purpose: torchao carries no TransformerEngine
dependency, and its in-tree oracle is a plain-PyTorch transcription of TE's arithmetic
rather than a TE import.

## Open questions

1. **Why is TE 2x faster on the fused RHT quantize at 671B?** Both run the SM100 RHT
   cast-fusion kernel on the same shapes and produce the same bytes, so the gap is
   scheduling or epilogue efficiency, not work. Worth an nsys trace on the 2048x7168 case.
2. **The grouped TE invocation.** `nvfp4_group_quantize_with_amax` at `E = 1` disagreeing
   with TE's own single-tensor path points at a missing or misread argument
   (`first_dims` vs `tensor_offsets` semantics, or the amax layout). Resolving it would
   unlock the grouped comparison, which is the case torchao is actually built for.
