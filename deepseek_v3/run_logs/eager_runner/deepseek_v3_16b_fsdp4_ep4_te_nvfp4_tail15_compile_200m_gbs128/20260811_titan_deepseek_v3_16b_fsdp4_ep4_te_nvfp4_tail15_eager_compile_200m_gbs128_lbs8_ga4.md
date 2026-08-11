# TorchTitan DeepSeek V3 16B NVFP4 (TransformerEngine) Eager Compile 200M Token Run

Date: 2026-08-11

Paired with the TorchAO arm in
`../deepseek_v3_16b_fsdp4_ep4_nvfp4_tail15_compile_200m_gbs128/`. The two runs
share every config field except the quantization converter class, so the loss
difference between them isolates the NVFP4 grouped-GEMM backend.

## Command

Run from the repo root:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python deepseek_v3/run_titan.py \
  --precision te-nvfp4 \
  --flavor 16b \
  --compile \
  --expert-parallel-degree 4 \
  --batch-size 8 \
  --global-batch-size 128 \
  --seq-len 4096 \
  --steps 382 \
  --dataset c4 \
  --log-freq 10
```

## Run Shape

- Model: DeepSeek V3 16B via `deepseek_v3_16b_te_nvfp4`
  (`te_moe_overrides/config_registry.py`, selected with `--module te_moe_overrides`)
- Trainer: eager
- Precision path: `TEGroupedExpertsConverter` -- TransformerEngine's NVFP4 grouped
  GEMM at the `GroupedExperts._grouped_mm` seam
- Recipe: `NVFP4BlockScaling` (activations RHT + 1D, weights 2D, grad RHT + SR),
  15% bf16 layer tail, `pad_multiple=128`
- Converted: 21 MoE expert modules (layers 1-21); layers 22-26 stay bf16.
  Verified identical to the TorchAO arm's converted set.
- Dense Linear layers stay bf16 (DSV3 MLA projections are not 128-divisible)
- Compile: TorchTitan compile enabled (`components=["loss"]`, so the model itself
  is not compiled -- same for both arms)
- Parallelism: 4x FSDP, TP 1, EP 4
- Local batch size: 8, global batch size 128, gradient accumulation 4
- Sequence length: 4096; tokens per optimizer step 524,288
- Steps: 382; tokens processed 200,278,016
- Dataset: `c4`

## Result

| Step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 380 | 4.62002 | 14,025 | 56,100 | 253.95 | 152.54 GiB |

Last logged metric is step 380 because `--log-freq 10`; the trainer then
completed step 382. Training time 11:27:06 -> 12:25:41, about 58m 35s.

## Comparison against the TorchAO arm

| step | TorchAO | TE | TE - AO |
| ---: | ---: | ---: | ---: |
| 1 | 12.04406 | 12.02424 | -0.01982 |
| 100 | 6.22219 | 6.20670 | -0.01549 |
| 200 | 5.43987 | 5.38230 | -0.05757 |
| 300 | 4.85781 | 4.78912 | -0.06869 |
| 380 | 4.69620 | 4.62002 | -0.07618 |

The delta is negative at **39 of 39** logged steps -- TE is never worse -- so this
is a systematic difference, not noise around zero. Its shape over the run:

| phase | n | mean delta | min | max |
| --- | ---: | ---: | ---: | ---: |
| steps 1-100 (warmup) | 11 | -0.03665 | -0.16116 | -0.00862 |
| steps 110-220 | 12 | -0.02198 | -0.05757 | -0.00150 |
| steps 230-380 | 16 | **-0.07019** | -0.08392 | -0.06232 |

It is **not** a compounding divergence: after a noisy warmup (a -0.161 transient at
step 20) it settles into a stable plateau of about -0.070 +/- 0.011 from step 230
onward. A bounded persistent offset is the thing to explain, not runaway drift.
See `../training_loss_te_vs_ao_nvfp4_200m_tokens.png` (lower panel).

For scale, the archived bf16 run reaches 4.61289 at step 380. TE lands within
0.007 of bf16; TorchAO sits 0.083 above it.

## Resolution

The -0.070 plateau was a TorchAO defect, not a property of the backend: the
columnwise NVFP4 block scales were blocked over the packed extent rather than per
group, so the expert weight gradient reached the optimizer with gain 0.88 against
bf16 where TE reached 0.97. Root cause in
`../../../kernel_analysis/nvfp4_module_te_vs_ao.md`; fix in torchao `b3c77e59`.

Re-running the TorchAO arm with the fix
(`../deepseek_v3_16b_fsdp4_ep4_nvfp4_tail15_scalefix_compile_200m_gbs128/`) moves
the steps 230-380 plateau from -0.07019 to **+0.00290** and step 380 from -0.07618
to +0.00227. This arm is unchanged and remains the comparator -- the fix is in a
kernel the TE path never calls.

That also settles the seed caveat below. The re-run is same-seed, same-command
against the pre-fix arm, so the collapse is attributable to the one changed
kernel; a seed sweep would have been a weaker instrument than the A/B that
replaced it.

## Notes

- **lbs 8, not 16.** The TE path OOMs at local batch 16 on 184 GiB GB200s: 180.6
  GiB allocated with only 883 MiB free, and re-running with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` cut reserved-but-unallocated
  memory from 10.5 GiB to 467 MiB without fixing it, so the footprint is real and
  not fragmentation. The archived TorchAO run fit at lbs16 (178.81 GiB peak). Both
  arms here use lbs8/ga4 to keep them comparable; global batch, sequence length,
  token count and recipe are unchanged. At lbs8 the two arms use nearly identical
  memory (152.54 vs 153.25 GiB peak).
- **TFLOPs are not comparable between the arms.** `has_quantization`
  (`torchtitan/components/quantization/utils.py`) only knows the float8/mx/nvfp4
  experts caches, so the TE-converted model falls back to bf16 FLOP accounting
  (and reports an MFU where the TorchAO arm reports `N/A`). Loss and TPS are
  unaffected.
- TE costs about 11% throughput at this shape (14,025 vs 15,797 TPS/GPU).
- Single seed, single run per arm. A one-sided sign at 39/39 logged steps is hard
  to explain as seed noise, but seed variance has not been measured and should be
  before the gap is attributed to the backend.
