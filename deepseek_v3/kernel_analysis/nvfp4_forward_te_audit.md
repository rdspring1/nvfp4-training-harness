# NVFP4 GroupedExperts forward-path audit vs TransformerEngine

Companion to `nvfp4_forward_te_audit.py`. Run on GB200 (sm100), torch
`2.14.0a0+gitc451937`, TE `2.19.0.dev0+f1d5f8d` (editable, `/opt/nvidia/TransformerEngine`),
torchao `nvfp4_moe`, torchtitan `nvfp4_moe_converter`.

```
python deepseek_v3/kernel_analysis/nvfp4_forward_te_audit.py --all \
    --model debugmodel --model 16B --model 671B
```
102 checks (12 informational) across DSV3 debugmodel / 16B / 671B MoE shapes.

## Conclusion

**The forward path does not explain the observed training bias.** Every
discrepancy found against TE is at most 1 ULP, affects under 1% of block scale
bytes and under 0.12% of FP4 codes, has no consistent sign across tensors, and
does not survive to the module output: the full NVFP4 `GroupedExperts` forward
is bitwise deterministic and its error against bf16 is unbiased (signed mean
~1e-5 against a relative L2 of ~0.25).

Both of the specific suspects identified before the run were falsified, as was
the padding-tail hypothesis. **The next run should target the backward path**,
which is where all the untested machinery lives.

## What was checked, and what it showed

| Stage | Check | Result |
|---|---|---|
| 0 | TE CUDA kernel vs TE `NVFP4QuantizerRef` | **12/12 bitwise** -- harness calibrated |
| 1 | per-tensor decode scale | bitwise except 671B, 1/4 values, max rel **1.1e-07** |
| 2 | activation rowwise quantize vs TE | row amax **bitwise**; block scales **bitwise everywhere**; codes differ 0.002--0.040% |
| 3a | weight 2D quantize vs TE | codes differ 0.018--0.114%; block scales differ 0--0.879%, always exactly 1 ULP |
| 3b | isolated `div_rn` / pvscale-order probes | see "Suspects" below |
| 3c | attribute the weight divergence | see "Suspects" below |
| 4 | `F.scaled_grouped_mm` vs dequant+bf16 matmul | SQNR **~51 dB**, signed mean ~1e-6 -- GEMM is clean |
| 5 | full `GroupedExperts` forward | **bitwise deterministic**, finite, SQNR ~12 dB vs bf16, signed mean 2.7e-06 (debugmodel) to 1.5e-04 (671B) |
| 6 | padding tail + empty expert | **all clean** |

## The two suspects: both falsified

Going in, the torchao weight quantizer (`quantize_2d_triton.py:73,84`) looked
wrong because it differs from the activation quantizer
(`hadamard_utils.py:248,258-260`) on two counts, and the activation one carries
a comment saying it was written to match TE:

|  | `global_encode_scale` | `pvscale` |
|---|---|---|
| torchao activations | `tl.div_rn(448*6, amax)` | `vmax * (ges * (1/6))` |
| torchao weights | `(448*6) / amax` | `(vmax / 6) * ges` |
| TE ref (`quantization_ref_nvfp4.py:713,748-751`) | `torch.div(448*6, amax)` | `vmax * (ges * (1/6))` |

**(A) `div_rn` vs plain `/`.** Real at the fp32 level -- a Triton kernel running
both forms on identical inputs shows 8--10% of fp32 bit patterns differ, so the
code comment is factually correct. But it does **not** propagate: the resulting
e4m3 scale byte is identical in 0/4096 cases. e4m3 has 3 mantissa bits, and the
cast absorbs the difference. **Dead.**

**(B) pvscale association order.** Also does not propagate on its own (0
differing bytes in the isolated probe).

The weight quantizer **does** diverge from TE in the real kernels (up to 0.879%
of scale bytes), but stage 3c shows the association order is **not** a
sufficient explanation. Recomputing both orders in fp32 PyTorch and matching
against both kernels gives an inconsistent picture across shapes:

| shape | AO kernel == AO order | TE kernel == TE order | AO == TE |
|---|---|---|---|
| debugmodel w1/w3 | 1 differ | 1 differ | 9 differ |
| 16B w1/w3 | **0** | 49 differ | 49 differ |
| 16B w2 | 51 differ | 51 differ | **0** |
| 671B w1/w3 | **0** | **0** | **0** |
| 671B w2 | 108 differ | 108 differ | **0** |

Where the two kernels agree with each other they both deviate from the fp32
reformulation by the same amount, so neither kernel is exactly modelled by a
naive fp32 expression. The honest statement is: **both quantizers occasionally
break e4m3 midpoint ties differently, at 0--0.9% of blocks and always by exactly
1 ULP, and the association order accounts for only part of it.**

An earlier, narrower experiment on two experts of one shape appeared to pin this
entirely on the association order. Widening to all three DSV3 flavors did not
support that.

### Is the 1-ULP weight-scale difference harmful?

No, and not consistently signed. Example of the mechanism, 16B expert 7, block
with `vmax=3.75, amax=5`: the true scale is exactly `336`, the exact midpoint
between e4m3 neighbours `320` and `352`. torchao rounds up, TE rounds down.

Reconstruction error is a wash -- torchao is very slightly better on some
tensors and very slightly worse on others:

| tensor | rel L2, torchao | rel L2, TE |
|---|---|---|
| debugmodel w1/w3 | 1.11391e-01 | 1.11398e-01 (torchao better) |
| 16B w1/w3 | 1.11353e-01 | 1.11351e-01 (TE better) |

The gap is ~1e-5 relative against a baseline NVFP4 weight quantization error of
**11.1%**. It cannot produce a +0.035 loss offset.

## Padding-tail hypothesis: falsified

`torchtitan/components/quantization/nvfp4.py:410-411` folds the dispatcher's
capacity tail into the last expert's group via `offs[-1] = A.shape[0]`. This is
safe, and now verified rather than assumed:

- The tail is **exactly zero**, not uninitialized. `permute.py:204-206` appends a
  zero row and gathers with a `-1` sentinel (`kernels.py:83-85`); PyTorch's
  negative-index wrap makes every pad row a copy of that zero row. Verified
  bitwise for 512- and 384-row tails at all three flavors.
- `x_row_amax[E-1]` is **bitwise identical** whether the tail is folded in or the
  activation is truncated -- zeros cannot raise an amax.
- An expert receiving **zero routed tokens** still presents a 128-row all-zero
  group (`kernels.py:182-187` clamps counts up to the alignment). Its block
  scales stay finite; the `global_amax == 0` guard in the kernels handles it.

Worth noting: the converter's rewrite reproduces the semantics of torchao's own
reference permute (`test/prototype/moe_training/reference_moe.py:128` absorbs the
tail into the last expert), so it is not an ad-hoc workaround.

## Coverage gaps found along the way

1. **`test/prototype/moe_training/test_nvfp4_grouped_mm.py:39` imports
   `_to_nvfp4_then_scaled_grouped_mm`, which does not exist** (the symbol is
   `_to_nvfp4_rht_rs_then_scaled_grouped_mm`, `nvfp4_grouped_mm.py:41`). On SM100
   with torch >= 2.10 the guard at `:37` is live, so the module raises
   `ImportError` at collection and all three hardware fwd/bwd tests are dead.
   One-line fix, not applied here.
2. Nothing pins the zero-tail invariant that `nvfp4.py:410-411` depends on.
   `test_permute.py` compares by SQNR and would not catch a nonzero tail;
   `test_kernels.py:84` filters `-1` indices out rather than asserting on them.
3. `test_group_rht_quantize_row_col_triton.py` asserts only SQNR >= 20 dB and
   1-ULP scale adjacency, which by construction cannot detect a systematic
   1-ULP scale bias.

## Next: the backward path

Forward is clean, deterministic, and unbiased, so the bias is downstream of it.
The backward path is untested and carries the higher-risk machinery:

- **Stochastic rounding** is applied to both grad row and col paths
  (`nvfp4_grouped_mm.py:292`) but not in forward. SR is only unbiased if the RNG
  is right.
- **The SR offsets are drawn from the global CUDA generator on every backward**
  (`nvfp4_grouped_mm.py:273-279`), with `rng_state = cat((sr_seed, col_offset,
  sr_seed ^ 1, row_offset))`. This both consumes the default generator (a
  cross-rank RNG desync risk under FSDP/EP) and makes backward non-reproducible.
  A biased or correlated SR stream would produce exactly the observed symptom: a
  small persistent one-directional loss offset that compounds.
- **RHT cancellation in wgrad** relies on `H H^T = I` across the contraction,
  with RHT applied in 16-blocks along M inside each group. Worth verifying
  directly that `dy_col @ x_col^T` recovers `dy^T x`.
- The weight path uses **no RHT and no SR**, so weight gradients get RTNE
  treatment while activations and grads get SR.

Recommended first experiment: run the same bitwise harness on backward with SR
disabled (`enable_stochastic_rounding=False`) against TE's grad quantization. If
backward is clean without SR and biased with it, the RNG at `:273-279` is the
target.
