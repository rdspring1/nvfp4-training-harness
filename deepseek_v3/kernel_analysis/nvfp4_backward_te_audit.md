# NVFP4 GroupedExperts backward-path audit vs TransformerEngine

Companion to `nvfp4_backward_te_audit.py` and `nvfp4_backward_rank_probe.py`.
Run on GB200 (sm100), torch `2.14.0a0+gitc451937`, TE `2.19.0.dev0+f1d5f8d`
(editable, `/opt/nvidia/TransformerEngine`), torchao `nvfp4_moe`, torchtitan
`nvfp4_moe_converter`.

```
python deepseek_v3/kernel_analysis/nvfp4_backward_te_audit.py --all \
    --model debugmodel --model 16B
torchrun --nproc_per_node=4 deepseek_v3/kernel_analysis/nvfp4_backward_rank_probe.py
```
50 checks (26 informational). Output is reproducible across processes.

## Conclusion

**No kernel defect was found in backward, and none was found in forward.** The
persistent +0.035 loss offset is not attributable to a bug in either path.

Every backward operand matches TE bitwise on block scales and per-tensor amax,
with FP4 code differences of 0.002-0.079% at one code step. Stochastic rounding
is unbiased to ~5e-5 across every FP4 interval, decorrelated, and spatially
white. RHT cancels correctly in wgrad. The isolated wgrad primitive has SQNR
15-17 dB and a magnitude ratio of 1.00.

What the run *did* establish is that the NVFP4 MoE recipe carries large inherent
error that compounds across the three chained grouped GEMMs, and that the
leading hypothesis going in was wrong.

## The leading hypothesis: confirmed as a fact, falsified as a cause

The plan led with cross-rank SR correlation. The 4-rank probe confirms the
mechanism exists exactly as predicted:

```
_sr_seed per rank : [2083742018897791291] x 4   -> IDENTICAL on every rank
backward SR offset: [1404726525]          x 4   -> IDENTICAL on every rank
```

`set_determinism` (`torchtitan/distributed/utils.py:276`) seeds all SPMD ranks
identically, and both `_sr_seed` (`nvfp4.py:392-398`) and the per-step offsets
(`nvfp4_grouped_mm.py:273-279`) are plain `torch.randint` calls the DTensor RNG
tracker never sees. So SR noise really is bit-identical on every data-parallel
rank, and really does add coherently in the all-reduced wgrad.

**But the consequence is negligible.** Making `_sr_seed` per-rank distinct moved
the all-reduced wgrad error from 0.701604 to 0.702724 -- a ratio of 0.9984 where
the hypothesis predicted sqrt(4) = 2.0.

The reason is visible in B6c: the wgrad error is dominated by a deterministic
term that no amount of SR decorrelation touches. The floor set by `x_col`'s
round-to-nearest quantization is relL2 0.095, against a single-draw SR error of
0.167 -- so even perfectly decorrelated, zero-mean SR noise averaged over four
ranks cannot get below 0.095. That floor is identical on every rank regardless
of seed.

Still worth fixing as a latent correctness issue -- it would matter at larger DP
degree or if SR noise ever became dominant -- but it is not the cause here.

## Results

| Stage | Check | Result |
|---|---|---|
| B0 | TE kernel vs TE ref, RHT + columnwise | **10/10 bitwise**; TE sign vector == torchtitan's, verified not assumed |
| B1 | `dy_row` vs TE | scales **bitwise**; codes differ 0.021-0.079% |
| B1 | `weight_t` vs TE 2D on `W.T` | scales **bitwise**; codes differ 0.007-0.069% |
| B2 | `x_col` / `dy_col` vs TE columnwise+RHT | **amax bitwise, scales bitwise**; codes differ 0.002-0.043% |
| B3 | RHT cancellation in wgrad | SQNR 14.7-18.6 dB, signed mean flips sign across groups |
| B4 | SR P(up) vs position, 7 intervals | **unbiased**: mean deviation 5e-6..7e-5, all within +-1.4 sigma |
| B4 | SR on the RHT col path | max deviation -9.3e-4 (1.2 sigma), weighted mean -8.2e-5 |
| B5 | SR decorrelation | all correlations ~2 sigma; `seed` vs `seed^1` streams independent |
| B6b | isolated wgrad primitive | SQNR 15.5-17.4 dB, **magnitude ratio 1.00** |
| B6c | RTNE vs SR vs averaged SR | averaged SR **beats** RTNE and converges to the floor |
| B6 | full module backward | grad_input SQNR ~10-11 dB; w1/w3 SQNR 0.6-2.6 dB (see caveat) |

### B2 is the first test of the RHT/columnwise path in this repo

The forward audit compared only the row outputs. `x_col` and `dy_col` -- the two
wgrad operands, both RHT'd -- had never been checked against anything. They match
TE bitwise on post-RHT amax and on block scales.

### B4: stochastic rounding is unbiased, including on the non-uniform grid

The e2m1 grid `{0, .5, 1, 1.5, 2, 3, 4, 6}` has widths `.5,.5,.5,.5,1,1,2`. A
rounder that implicitly assumed uniform spacing would be biased only in the three
wide intervals. It is not: `[2,3]`, `[3,4]` and `[4,6]` show the same deviations
as the narrow ones, all within +-1.4 sigma at ~3.9M samples per point.

This test was chosen over TE's own criterion deliberately. TE asserts only that
averaged SR RMSE beats round-nearest (`test_nvfp4_sr_quantize.py:281-282`), and
the existing torchao test checks a single midpoint for a 50/50 split
(`test_hadamard_quantize_row_col.py:583`). A biased RNG passes both.

Incidental observation: with a shared RNG stream, the same elements round up at a
given fractional position regardless of which interval they are in -- the
hardware `cvt.rs` decision depends on position, not on absolute spacing. The
stages use per-interval offsets so the seven rows are independent evidence.

### B6c: stochastic rounding is doing its job

SR trades variance for bias. Per draw it is strictly **worse** than
round-to-nearest, because RTNE always picks the nearer grid point. The payoff is
that `E[SR(x)] = x`, so SR error averages away across training steps while RTNE
error is a fixed function of the value and accumulates coherently. Comparing
single-draw SQNR therefore measures the cost of SR and misses its benefit
entirely.

Measured on the isolated wgrad (16B, 1408x2048), relative L2 against the exact
bf16 contraction:

| | relL2 |
|---|---|
| RTNE wgrad | 0.13424 |
| single-draw SR | 0.16696 |
| SR averaged over 64 draws | **0.09700** |
| floor: exact `dy`, RTNE-quantized `x` | 0.09509 |

Averaged SR converges essentially onto the floor (0.097 vs 0.095), while RTNE
sits at 0.134 with a systematic component that never averages away. SR is the
correct choice here and the implementation delivers the property it exists for.

### Each backward GEMM keeps a residual RTNE floor

`grad_output` is the only tensor SR ever touches -- both its row and col
outputs, from the single `enable_stochastic_rounding=True` call at
`nvfp4_grouped_mm.py:292`. The forward call at `:184` passes `False`, and
`triton_group_weight_quantize_2d` has no SR path at all. So the forward GEMM has
zero SR operands and each backward GEMM has exactly one:

```
dgrad = dy_row (SR) @ weight_t^T (RTNE)
wgrad = dy_col (SR) @ x_col^T    (RTNE)
```

Both therefore keep a bias contributed by their *other* operand, which averaging
SR draws can never remove. Measured (16B; debugmodel agrees to 4 decimals, so
this is shape-independent):

| | wgrad | dgrad |
|---|---|---|
| RTNE | 0.13470 | 0.14602 |
| single-draw SR | 0.16710 | 0.17656 |
| SR averaged over 64 | 0.09725 | 0.11294 |
| **floor (the RTNE operand)** | **0.09533** (`x_col`) | **0.11131** (`weight_t`) |

Averaged SR converges to within 1.3-1.6% of each floor, which is what
unbiasedness looks like at the GEMM level.

**The dgrad floor is ~17% higher, and it is entirely the 2D blocking.**
`weight_t` is the only operand in the step that gets none of the three
mitigations: 2D 16x16 block scales, no RHT, no SR. Re-quantizing the same `W.T`
with 1D 1x16 blocks isolates the cause:

| `weight_t` variant | dgrad floor |
|---|---|
| as shipped: 2D 16x16, no RHT | 0.11116 |
| 1D 1x16, no RHT | **0.09511** |
| TE 2D, cross-check | 0.11115 |

1D lands at 0.09511, essentially identical to the wgrad floor of 0.09533 whose
operand *does* carry RHT. So on this data RHT buys nothing and the whole gap is
2D blocking sharing one scale across 256 elements instead of 16.

This matters more than the wgrad number because dgrad propagates backwards
through every layer, so its bias compounds with depth while wgrad's applies once
per weight.

Two caveats before acting on it. 2D weight quantization is deliberate -- it lets
one quantized weight serve both the forward GEMM and dgrad with consistent
scales, and TE's `fp4_quant_fwd_weight` sets `fp4_2d_quantization=True` for the
same reason -- so switching to 1D is a real cost/accuracy trade, not a free win.
And these are random Gaussian weights, which have no outliers for RHT to fix;
on trained weights RHT would likely contribute more than the ~0 measured here.

## Two corrections to claims made during the run

**1. "The wgrad is deterministically attenuated to 39% of its magnitude."** This
came from B6 as first written, which called `.backward()` on each model's own
loss. The two models therefore received *different* upstream gradients, differing
by the forward's ~25% error. After fixing both to take an identical fixed
`grad_output`, and after isolating the primitive in B6b, the wgrad magnitude
ratio is **1.00**, not 0.39. There is no attenuation.

**2. B6's low SQNR is not a backward defect.** Even with a shared `grad_output`,
the two modules' *intermediate* activations differ by the forward error, so B6
measures backward error compounded with forward divergence. That also explains
why averaging 32 backward passes improved SQNR by only 0.2 dB: the dominant term
is deterministic forward divergence, not zero-mean SR noise. B6b, which fixes
`dy` and `x`, is the stage that actually attributes error to backward.

Both are recorded because the B6 numbers are still meaningful as an end-to-end
measure of accumulated error -- they are just not evidence about the backward
kernels.

**3. "Turning SR off improves wgrad SQNR from 15.5 to 17.4 dB."** True as
arithmetic, wrong as framing: it presents SR's expected and intended cost as a
degradation. Per-draw error is the axis SR deliberately gives up. B6c was added
to measure the axis it wins on, and on that axis SR beats RTNE by a wide margin.

## Where the error actually accumulates

| | relative L2 error |
|---|---|
| single NVFP4 weight quantization | ~0.11 |
| isolated wgrad (two NVFP4 operands) | ~0.15 |
| full GroupedExperts forward vs bf16 | ~0.25 |
| module-level w1 wgrad (16B) | ~0.70 |

The MoE block chains three grouped GEMMs (w1, w3, w2) plus silu and an
elementwise product, and the backward re-quantizes each. That compounding, not
any single kernel, is what separates the module-level error from the primitive
error. It is a plausible reason NVFP4 MoE degrades more than the linear-only
NVFP4 Llama3 runs that were previously validated, which have one GEMM per layer.

**Limitation, stated plainly:** these numbers come from random weights and
synthetic gradients at a single step, not from a trained checkpoint. Real
trained weights and real gradient distributions may compound very differently.
Measuring this on a real checkpoint mid-run is the natural next step and this
audit does not substitute for it.

## Coverage gaps

1. Still unfixed from the forward run: the dead import at
   `test/prototype/moe_training/test_nvfp4_grouped_mm.py:39`.
2. `compare_codes` in the audit harness had been fed unpacked nibbles, which made
   the high-nibble half of the comparison trivially equal and doubled the
   denominator. Fixed by adding `compare_nibbles`; the forward report's *counts*
   were correct but its code percentages were half the true value.
3. There is no SR distribution test in torchao for the Triton path at all -- the
   only unbiasedness test is CuTeDSL-only and checks a single midpoint on the row
   path (`test_hadamard_quantize_row_col.py:906`).

## Carried forward, not chased

- The **CuTeDSL columnwise SR path omits `pid_ns` from its RNG base**
  (`_cutedsl_kernels_impl.py:1108-1116`, `:1207-1215`), so for `M > 256` two
  columnwise vectors at the same N-row and same local `u` derive an identical
  `rng_base` and receive bit-identical stochastic rounding. The row path's own
  comment (`:977-982`) warns about exactly this aliasing class. The DSV3 MoE path
  is Triton-only, so this does not affect the training run.
- Triton's SR generates 8 Philox words per 16 values and consumes 4
  (`$11`/`$12` declared but unused, `hadamard_utils.py:169-188`). Waste, not bias.
- The cross-rank seed collision above: real, worth fixing, not the cause.

## Suggested next step

Both audits are clean, so the remaining explanations are recipe-level rather than
kernel-level. In rough order of expected value:

1. **Try 1D block scaling for `weight_t`.** B6c attributes the dgrad bias floor
   almost entirely to 2D 16x16 weight blocking: 0.111 as shipped versus 0.095
   with 1D 1x16. dgrad bias compounds with depth, so this is the highest-leverage
   single knob found in either audit. It costs a second weight quantization and
   diverges from TE's recipe, so measure the loss curve before adopting it.
2. **Ablate the three MoE GEMMs.** Keep w2 in bf16 and quantize only w1/w3 (or
   vice versa) and compare loss curves. If the gap collapses, compounding across
   the chain is confirmed as the mechanism and the fix is recipe selection, not
   a kernel patch.
3. **Measure on a real checkpoint** at step ~50 and ~200, where the divergence
   starts and where it grows, rather than on random weights. This also re-tests
   the RHT contribution, which is ~0 on Gaussian weights but should be larger
   where real outliers exist.
4. Fix the cross-rank `_sr_seed` collision by folding the rank into the seed --
   cheap, correct, and removes a confound from any future measurement.
