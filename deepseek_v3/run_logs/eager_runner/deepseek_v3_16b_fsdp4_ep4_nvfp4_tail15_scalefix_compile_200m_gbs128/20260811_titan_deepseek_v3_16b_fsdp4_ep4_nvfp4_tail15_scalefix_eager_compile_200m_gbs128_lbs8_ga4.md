# TorchTitan DeepSeek V3 16B NVFP4 (TorchAO, per-group scale fix) Eager Compile 200M Token Run

Date: 2026-08-11

Validation run for torchao `b3c77e59`, which blocks the columnwise NVFP4 scales
per group instead of over the packed extent. Byte-for-byte the same command,
recipe and seed as the pre-fix arm in
`../deepseek_v3_16b_fsdp4_ep4_nvfp4_tail15_compile_200m_gbs128/`, so this is a
one-variable A/B against it and remains a one-variable comparison against the TE
arm in `../deepseek_v3_16b_fsdp4_ep4_te_nvfp4_tail15_compile_200m_gbs128/`.

Root cause and tensor-level evidence:
`../../../kernel_analysis/nvfp4_module_te_vs_ao.md`.

## Command

Run from the repo root:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python deepseek_v3/run_titan.py \
  --precision nvfp4 \
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

- Model: DeepSeek V3 16B via torchtitan's `deepseek_v3_16b_nvfp4`
- Trainer: eager
- Precision path: `NVFP4GroupedExpertsConverter` -- TorchAO's Triton NVFP4 grouped
  GEMM at the `GroupedExperts._grouped_mm` seam, with torchao at `e6934feb`
  (fix `b3c77e59`)
- Recipe: 15% bf16 layer tail, `pad_multiple=128`; 21 MoE expert modules
  converted (layers 1-21), layers 22-26 bf16
- Parallelism: 4x FSDP, TP 1, EP 4
- Local batch size: 8, global batch size 128, gradient accumulation 4
- Sequence length: 4096; tokens processed 200,278,016 over 382 steps
- Dataset: `c4`

## Result

| Step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 380 | 4.61775 | 15,293 | 61,172 | 276.89 | 153.45 GiB |

Training time 14:15:27 -> 15:08:44, about 53m 17s.

## Comparison

**The backend gap closes.** At step 380: this run 4.61775, TE arm 4.62002
(+0.00227, i.e. TorchAO marginally *lower*), pre-fix TorchAO 4.69620.

Over the plateau where the pre-fix gap had stabilized (steps 230-380, 16 logged
points):

| arm vs TE | mean(TE - AO) | sd | sign |
| --- | ---: | ---: | --- |
| TorchAO pre-fix | **-0.07019** | 0.00555 | 16/16 TE lower |
| TorchAO post-fix | **+0.00290** | 0.00368 | 12/16 TorchAO lower |

Mean `|TE - AO|` over steps 200-380 is 0.00558, inside the plan's 0.01 decision
criterion, with no drift. The residual is smaller than the scatter of either arm
and its sign now favors TorchAO, so what is left is seed noise, not a backend
effect.

Per-step, against both references:

| Step | AO post-fix | TE | AO pre-fix | TE - AO(post) | TE - AO(pre) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 5.39145 | 5.38230 | 5.43987 | -0.00915 | -0.05757 |
| 230 | 5.19229 | 5.20218 | 5.28610 | +0.00989 | -0.08392 |
| 260 | 5.04016 | 5.04177 | 5.10428 | +0.00161 | -0.06251 |
| 290 | 4.90522 | 4.90992 | 4.97224 | +0.00470 | -0.06232 |
| 320 | 4.75167 | 4.75050 | 4.81940 | -0.00117 | -0.06890 |
| 350 | 4.71465 | 4.71617 | 4.78707 | +0.00152 | -0.07090 |
| 380 | 4.61775 | 4.62002 | 4.69620 | +0.00227 | -0.07618 |

This confirms the module-level prediction quantitatively: the fix moved the
expert wgrad gain from 0.882 to 0.966 against TE's 0.967, and the loss offset
that attenuation implied is gone at 200M tokens.

Note the pre-fix arm tracks TE closely for the first ~150 steps and only
separates from ~200 onward. That is the shape a constant gradient attenuation
produces -- an effective learning-rate cut costs nothing while the loss is
falling steeply and shows up once the curve flattens -- and it is why the
200M-token horizon was needed to see it at all.

## Notes

- Same lbs8/ga4 shape as both comparison arms (the TE arm OOMs at lbs16).
- MFU still reports `N/A`; `has_quantization` recognizes the TorchAO experts
  cache but not the TE one, so the TFLOPs columns remain non-comparable across
  arms. TPS is 3% below the pre-fix arm (15,293 vs 15,797) -- the fix adds no
  work (no repack, no extra launch, same allocation), so this is run-to-run
  variation, not a regression, though a repeat would be needed to say so
  tightly.
