# TorchTitan DeepSeek V3 16B NVFP4 (TorchAO) Eager Compile 200M Token Run

Date: 2026-08-11

Control arm for the TransformerEngine comparison in
`../deepseek_v3_16b_fsdp4_ep4_te_nvfp4_tail15_compile_200m_gbs128/`. Re-run under
the current converter code so the two arms differ only in the converter class --
the archived 2026-07-09 NVFP4 run predates both the converter migration and the
bf16-tail recipe, so it is a reference, not a control.

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
  GEMM at the `GroupedExperts._grouped_mm` seam
- Recipe: 15% bf16 layer tail, `pad_multiple=128`; 21 MoE expert modules
  converted (layers 1-21), layers 22-26 bf16
- Parallelism: 4x FSDP, TP 1, EP 4
- Local batch size: 8, global batch size 128, gradient accumulation 4
- Sequence length: 4096; tokens processed 200,278,016 over 382 steps
- Dataset: `c4`

## Result

| Step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 380 | 4.69620 | 15,797 | 63,188 | 286.03 | 153.25 GiB |

Training time 10:34:09 -> 11:26:39, about 52m 30s.

## Comparison

At step 380: this run 4.69620, TE arm 4.62002 (-0.076), archived bf16 4.61289,
archived NVFP4 without the bf16 tail 4.75427.

Two separate effects are visible and should not be conflated:

1. **The bf16 tail helps.** 4.69620 here vs 4.75427 archived. That delta also
   carries the lbs16->lbs8 change and a month of torchtitan, so it is not a clean
   attribution to the tail alone.
2. **The backend matters more than expected.** Against the TE arm -- which *is* a
   clean one-variable comparison -- TorchAO is 0.076 worse at step 380, one-sided
   at every logged step. See the TE run's `.md` for the per-step table.

**Resolved.** Effect 2 was a TorchAO defect: the columnwise NVFP4 block scales
were blocked over the packed extent instead of per group, corrupting the expert
weight gradient. Root cause in
`../../../kernel_analysis/nvfp4_module_te_vs_ao.md`, fix in torchao `b3c77e59`,
re-run in `../deepseek_v3_16b_fsdp4_ep4_nvfp4_tail15_scalefix_compile_200m_gbs128/`,
which lands at 4.61775 -- 0.002 *below* TE. This arm is retained as the pre-fix
reference.

## Notes

- lbs8/ga4 rather than the archived lbs16/ga2 because the TE arm OOMs at lbs16;
  both arms use the same shape so the comparison holds. Same global batch,
  sequence length, token count and recipe as the archived run's intent.
- MFU reports `N/A` here (and a real value on the TE arm) because
  `has_quantization` recognizes the TorchAO experts cache but not the TE one;
  this makes the TFLOPs columns non-comparable across arms.
