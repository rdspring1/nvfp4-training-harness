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

## 1. Conformance — TransformerEngine default

TE default uses correctly rounded division for the per-block encode reciprocal and a
bf16 accumulator round-trip after the RHT. TorchAO now follows that recipe throughout:
`2688/amax` with `div_rn`, `amax * (S_enc/6)`, the E4M3 cast with no lower clamp, and
`div.rn` for the final encode reciprocal.

At E=1 and E=4, both TorchAO backends match TE default byte-for-byte for row/column FP4
codes and scale factors. The E=4 grouped validation initially failed because the harness
passed row-prefix `tensor_offsets`; TE requires element-prefix offsets, computed by
`tex.splits_to_offsets(first_dims, hidden_size)`. Correcting that metadata makes grouped
TE reproduce TE single at E=1 and both TorchAO backends at E=4.

TorchAO's in-tree tests pin this independently: both backends reproduce the plain-PyTorch
TE transcription bitwise, including adversarial E2M1 midpoint inputs.

## 2. Timing — single-tensor ops, TE is ahead

Device kernel time (us), TE default (`NVTE_USE_FAST_MATH` unset).

| model | projection | op | M | N | triton | cutedsl | te | cutedsl/te |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 671B | gate/up | rht_quantize | 2048 | 7168 | 33.997 | 23.588 | **12.222** | 1.93x |
| 671B | gate/up | weight_quantize_2d | 2048 | 7168 | 45.627 | 24.599 | **13.597** | 1.81x |
| 671B | down | rht_quantize | 7168 | 2048 | 33.934 | 23.729 | **12.535** | 1.89x |
| 671B | down | weight_quantize_2d | 7168 | 2048 | 45.607 | 24.275 | **13.397** | 1.81x |

TE default is 1.8-1.9x faster than CuteDSL on the validated single-tensor operations.

## 3. Timing — grouped E=4

| model | projection | op | E | M | N | triton | cutedsl | TE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 671B | gate/up | group_rht_amax | 4 | 2048 | 7168 | 40.018 | **22.862** | n/a |
| 671B | gate/up | group_rht_quantize | 4 | 2048 | 7168 | 92.820 | 62.367 | **47.930** |
| 671B | gate/up | group_weight_quantize_2d | 4 | 2048 | 7168 | 109.452 | 92.629 | **61.390*** |
| 671B | down | group_rht_amax | 4 | 7168 | 2048 | 40.003 | **23.182** | n/a |
| 671B | down | group_rht_quantize | 4 | 7168 | 2048 | 92.297 | 64.122 | **48.003** |
| 671B | down | group_weight_quantize_2d | 4 | 7168 | 2048 | 108.508 | 91.717 | **60.705*** |

`*` TE 2D weight is the summed device time of four single-expert calls; TE 2.19 has no
grouped 2D kernel. Grouped RHT uses TE's actual grouped kernel.

Two structural gaps on TE's side:

- **No amax-only entry point is reachable from PyTorch.** There is no binding for any
  `nvte_hadamard_transform*` symbol; `nvte_hadamard_transform_amax` runs only inside a
  quantizer that also casts. A head-to-head on `group_rht_amax` is not possible without
  attributing the amax kernel out of a profiler trace.
- **TE 2.19 cannot do grouped 2D weight quantize at all.**
  `NVTE_CHECK(!with_2d_quantization, "2D scaling grouped quant kernel is not ready yet")`
  (`extensions/cast.cpp:154`); the grouped non-RHT path is likewise unimplemented
  (`:158`). The table therefore uses four TE single-expert calls as the same-work baseline.

## Reproducing

```bash
# from the repo root; torchao is editable-installed, TE comes from the base image
python deepseek_v3/kernel_analysis/nvfp4_quantize_kernels_vs_te.py
```

This lives outside `third_party/torchao` on purpose: torchao carries no TransformerEngine
dependency, and its in-tree oracle is a plain-PyTorch transcription of TE's arithmetic
rather than a TE import.

## Open questions

1. **Why is TE 2x faster on the fused RHT quantize at 671B?** Both run the SM100 RHT
   cast-fusion kernel on the same shapes and produce the same bytes, so the gap is
   scheduling or epilogue efficiency, not work. Worth an nsys trace on the 2048x7168 case.
