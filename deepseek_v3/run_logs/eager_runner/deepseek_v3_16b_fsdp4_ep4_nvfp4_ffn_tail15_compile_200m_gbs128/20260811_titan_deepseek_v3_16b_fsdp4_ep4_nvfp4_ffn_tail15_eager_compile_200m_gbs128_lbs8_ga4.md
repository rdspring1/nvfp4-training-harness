# TorchTitan DeepSeek V3 16B NVFP4 (TorchAO, FFN linears + experts) Eager Compile 200M Token Run

Date: 2026-08-11

Validation run for torchtitan `f5889f85`, which extends `deepseek_v3_16b_nvfp4`
from the routed-expert grouped GEMMs to every convertible FFN Linear in the
leading 85% of layers. Same command, shape and seed as the experts-only arm
(`deepseek_v3_16b_fsdp4_ep4_nvfp4_tail15_scalefix_compile_200m_gbs128`, archived
on branch `nvfp4-te-audit`), so this is a one-variable A/B against it: the FFN
Linears are the only difference.

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
- Precision path: `NVFP4LinearConverter` on the FFN Linears plus
  `NVFP4GroupedExpertsConverter` on the routed experts, with torchtitan at
  `f5889f85` and torchao at `6b062ac5`
- Recipe: 15% bf16 layer tail, `pad_multiple=128`; 21 MoE expert modules
  converted (layers 1-21), layers 22-26 bf16. 63 FFN Linears converted --
  `moe.shared_experts.{w1,w2,w3}` on layers 1-21. Layer 0's dense
  `feed_forward` stays bf16 because `dense_hidden_dim=10944` and
  `10944 % 128 == 64`, which NVFP4 cannot represent.
- Parallelism: 4x FSDP, TP 1, EP 4
- Local batch size: 8, global batch size 128, gradient accumulation 4
- Sequence length: 4096; tokens processed 200,278,016 over 382 steps
- Dataset: `c4`

## Result

| Step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 380 | 4.61204 | 15,217 | 60,868 | 275.52 | 147.59 GiB |

Training time 16:29:59 -> 17:24:15, about 54m 16s.

## Comparison

**The FFN conversion is free in loss and saves 6.1 GiB.** Against the
experts-only arm, over the plateau (steps 200-380, 19 logged points):

| metric | FFN + experts | experts only | delta |
| --- | ---: | ---: | ---: |
| loss @ 380 | 4.61204 | 4.61775 | -0.00571 |
| mean(FFN - experts) over 200-380 | **-0.00355** | -- | sd 0.00819, 14/19 lower |
| mean TPS (step >= 50) | 15,563 | 15,757 | -1.2% |
| mean TFLOPs (step >= 50) | 281.79 | 285.30 | -1.2% |
| peak reserved | **147.59 GiB** | 153.70 GiB | **-6.1 GiB** |

The loss mean is well inside its own scatter, so the plateau difference is seed
noise, not an effect of quantizing the shared experts. Per-step, against both
references:

| Step | FFN + experts | experts only | TE experts | FFN - experts-only |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 5.39320 | 5.39145 | 5.38230 | +0.00175 |
| 230 | 5.19015 | 5.19229 | 5.20218 | -0.00214 |
| 260 | 5.03031 | 5.04016 | 5.04177 | -0.00985 |
| 290 | 4.90643 | 4.90522 | 4.90992 | +0.00121 |
| 320 | 4.74429 | 4.75167 | 4.75050 | -0.00738 |
| 350 | 4.71081 | 4.71465 | 4.71617 | -0.00384 |
| 380 | 4.61204 | 4.61775 | 4.62002 | -0.00571 |

This run also ends marginally below the TE arm, so extending the recipe does not
disturb the TorchAO-vs-TE agreement established by the per-group scale fix.

## Notes

- The 1.2% TPS difference is inside the noise floor of a single run -- the
  experts-only arm's own notes treat a 3% spread between otherwise identical
  runs as run-to-run variation. Read this as "no measurable throughput
  regression", not as a 1.2% cost.
- The 6.1 GiB saving does **not** buy lbs16. Both recipes OOM at lbs16 on the
  same allocation (3.00 GiB at `token_dispatcher.py:650` `combine`) on all four
  ranks, so lbs8/ga4 remains the working shape at this parallelism.
- MFU still reports `N/A` and the TFLOPs column remains non-comparable against
  the TE arm, unchanged from the previous runs.
