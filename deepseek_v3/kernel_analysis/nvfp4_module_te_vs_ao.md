# NVFP4 GroupedExperts: TorchAO vs TransformerEngine at the module level

Companion to `nvfp4_module_te_vs_ao.py`. Run on GB200 (sm100), torch
`2.14.0a0+gitc451937`, TE `2.19.0.dev0+f1d5f8d`, torchao `nvfp4_moe`, torchtitan
`496e365f`.

```
PYTHONPATH=deepseek_v3/kernel_analysis \
  python deepseek_v3/kernel_analysis/nvfp4_module_te_vs_ao.py --model debugmodel
```

Single GPU, single step. Follows the 200M-token 16B comparison in
`../run_logs/eager_runner/deepseek_v3_16b_fsdp4_ep4_te_nvfp4_tail15_compile_200m_gbs128/`,
which put TE 0.070 below TorchAO in training loss on a stable plateau with the
sign one-sided at 39/39 logged steps.

## Conclusion

**TorchAO's NVFP4 weight gradient is wrong, and the defect is in the grouped
GEMM, not in the quantization.** Feeding one set of TorchAO-quantized wgrad
operands to two consumers:

| consumer of the same codes + scales | relL2 vs bf16 wgrad |
| --- | ---: |
| dequantize, matmul in fp32 | **0.1672** |
| `F.scaled_grouped_mm`, as shipped | **0.4497** |

The kernel disagrees with its own operands by relL2 **0.414** — more than twice
the entire quantization error it is supposed to be carrying.

This is the loss gap. TorchAO's wgrad reaches the optimizer with gain 0.88
against bf16 where TE reaches 0.97 (gain = `<g, g_bf16>/<g_bf16, g_bf16>`, the
component along the true gradient). A persistent 12% attenuation of every MoE
expert weight gradient is an effective learning-rate cut, which is exactly the
shape of a stable, non-compounding loss offset.

The forward path, the dgrad path, and TorchAO's quantizers are all clean. The
two prior audits were right that the operands match TE bitwise; they measured
the wgrad by dequantizing and multiplying in fp32, and so never exercised the
call that is actually broken.

## The gap is single-group-exact and grows with group count

Same operands, varying only how the packed rows are split into expert groups:

| groups x rows | kernel vs its own operands | kernel vs bf16 |
| --- | ---: | ---: |
| 1 x 1024 | **0.00257** | 0.16650 |
| 2 x 512 | 0.33563 | 0.37658 |
| 4 x 256 | 0.41400 | 0.44969 |

At one group the GEMM is exact (0.0026 is fp32-vs-bf16-accumulate noise). At
four groups every group is wrong, not just the tail ones: per-group relL2
`[0.349, 0.481, 0.447, 0.379]`.

That pattern rules out a missing per-group base offset (which would leave group
0 clean) and points at the blocking itself. TorchAO swizzles the columnwise
scale buffer once over the full packed M -- `sfd_storage` is
`(hidden//128, M//64, 32, 16)`
(`group_rht_quantize_row_col_triton.py:305-313`) -- and hands the whole thing to
`F.scaled_grouped_mm` with `offs` partitioning M
(`nvfp4_grouped_mm.py:310-321`). Those two agree only when there is a single
group, so the grouped GEMM is reading each group's block scales from a layout
that was not written for it.

The wgrad is the only GEMM in the recipe whose *contraction* dimension is the
grouped one. The forward and dgrad group the output rows instead, use the
rowwise scale buffer, and both match TE to 3 decimal places.

## Results

### M1 -- forward: the two backends are equivalent

| | relL2 vs bf16 | gain |
| --- | ---: | ---: |
| TorchAO | 0.25163 | 0.95998 |
| TE | 0.25132 | 0.96036 |

Ratio 0.999. This is a verdict check, not a measurement: had TE come in an order
of magnitude lower, "TE wins the loss curve" would have meant TE was skipping a
quantization, and the 200M-token result would have been an artifact. It is not.
The two quantized outputs differ from each other by 0.0736, i.e. they carry the
same amount of error along different realizations.

### M2 -- module backward, fixed upstream gradient

| tensor | TorchAO relL2 / gain | TE relL2 / gain |
| --- | ---: | ---: |
| grad_input (dgrad) | 0.28698 / 0.96023 | 0.28868 / 0.95890 |
| w1_EFD | 0.59234 / 0.88191 | 0.28666 / 0.96651 |
| w2_EDF | 0.59911 / 0.87949 | 0.26725 / 0.96456 |
| w3_EFD | 0.60565 / 0.88610 | 0.28402 / 0.96834 |

dgrad matches. Every wgrad is ~2.1x worse in TorchAO, and the gain gap is the
number that matters: 0.88 vs 0.97.

### M5 -- one `_grouped_mm`, not the three-GEMM chain

| | TorchAO | TE |
| --- | ---: | ---: |
| dgrad | 0.17655 / gain 0.98743 | 0.17670 / gain 0.98737 |
| wgrad | **0.46228** / gain 0.98381 | **0.16752** / gain 0.99201 |

The divergence survives collapsing the chain, so it is the primitive. Note TE's
single-GEMM wgrad (0.1675) equals the fp32 dequant reference (0.1672): TE and
the dequant path are two independent references and they agree against the
TorchAO kernel.

Comparing M2 and M5 also shows the 0.88 module gain is this same defect
compounded through three chained GEMMs, not a separate effect.

### M3 -- the error does not average away

Mean of 16 backward passes with identical inputs, so zero-mean SR noise cancels
and only the deterministic part survives:

| | bias (mean of 16) | per-draw | gain |
| --- | ---: | ---: | ---: |
| TorchAO w1 | 0.54891 | 0.59158 | 0.88222 |
| TE w1 | 0.21605 | 0.28686 | 0.96658 |

93% of TorchAO's wgrad error is deterministic (0.549 of 0.592); for TE it is 75%
(0.216 of 0.287). A deterministic error is the only kind that can survive 382
optimizer steps as a stable offset, and TorchAO has 2.5x more of it.

### M4 -- the `_sr_seed` hypothesis is falsified

TorchAO pins one `_sr_seed` per module for the whole run and refreshes only the
two 32-bit Philox counter words per backward (`nvfp4_grouped_mm.py:273-279`),
where TE pulls seed *and* offset from the CUDA generator on every cast
(`csrc/extensions/cast.cpp:177`). Re-drawing `_sr_seed` before every forward:

| | bias (mean of 16) | per-draw | gain |
| --- | ---: | ---: | ---: |
| fixed seed (shipped) | 0.54891 | 0.59158 | 0.88222 |
| re-drawn every step | 0.54823 | 0.59079 | 0.88188 |

Ratio 1.0012. Pinning the Philox *key* while varying the counter costs nothing
measurable, because Triton's `randint4x` puts the caller's counter words in `c0`
and the per-element index in `c1`, so the streams are already distinct
(`triton/language/random.py:88-106`). This closes the seed hypothesis; it joins
the cross-rank seed collision from the backward audit as real-but-not-the-cause.

## Caveats

- Random Gaussian weights and synthetic gradients at one step, `debugmodel`
  expert dims, 4 experts x 256 rows. The group-count sweep is the load-bearing
  evidence and it is shape-independent in mechanism, but the specific relL2
  values are not.
- Which side of the `F.scaled_grouped_mm` contract is at fault -- TorchAO's
  whole-M swizzle or the kernel's per-group scale addressing -- is not
  established here. Only the disagreement is, and that it is exact at one group.
- The training runs used `pad_multiple=128` groups from the dispatcher; this
  probe uses 256-row groups, both multiples of the 64-element swizzle stride, so
  neither is a misalignment case.

## Next step

Reproduce the M6 group-count sweep as a standalone `F.scaled_grouped_mm` repro
without torchao in the loop -- quantize a known matrix, run it at 1 and 4 groups,
diff against the dequantized reference -- and take it to whichever of torchao or
pytorch owns the columnwise grouped scale layout. Everything downstream (the
0.070 loss gap, the 0.88 gain, the three-GEMM compounding) follows from that one
disagreement, so there is nothing else to chase until it is resolved.

Do not re-run the 200M-token comparison to confirm; it already agrees.
