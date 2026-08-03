# TorchTitan DeepSeek V3 16B BF16 Eager Compile 200M Token Run

Date: 2026-07-09

## Command

Run from `/home/me/nvfp4/deepseek_v3`:

```bash
python run_titan.py \
  --flavor 16b \
  --compile \
  --expert-parallel-degree 4 \
  --batch-size 16 \
  --global-batch-size 128 \
  --seq-len 4096 \
  --steps 382 \
  --dataset c4 \
  --log-freq 10
```

## Run Shape

- Model: DeepSeek V3 16B via `deepseek_v3_16b`
- Trainer: eager
- Precision path: BF16
- Compile: TorchTitan eager compile enabled
- Parallelism: 4x FSDP, TP 1, EP 4
- Local batch size: 16
- Global batch size: 128
- Gradient accumulation steps: 2
- Sequence length: 4096
- Tokens per optimizer step: 524,288
- Steps requested/completed: 382
- Tokens requested/processed: 200,278,016
- Dataset: `c4`

## Result

The run completed successfully.

Last logged training metric was step 380 because `--log-freq 10`; the trainer then completed step 382.

| Step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 380 | 4.61289 | 18,469 | 73,876 | 334.39 | 180.84 GiB |

TorchTitan logs `tps` as tokens/sec per device. Aggregate TPS above is `tps * 4`.

Average logged throughput from step 10 onward was 19,148.8 TPS/GPU, or 76,595.1 aggregate TPS. Average logged TFLOPs/GPU from step 10 onward was 346.71.

Training time from `Training starts at step 1` to `Training completed` was 43m 44.3s. End-to-end process wall time from the first torchrun log to process group destruction was 44m 04.2s.

## Notes

- Local batch 32 with global batch 128 maps exactly with gradient accumulation 1 but OOMed during the first step while trying to allocate another 6.00 GiB.
- Local batch 16 with global batch 128 maps exactly with gradient accumulation 2, passed the fit check, and completed the 200M-token run.
- The full local-batch-16 run reported five recoverable allocator OOM warnings and up to three CUDA memory allocation retries on one rank, but did not abort.
- The maximum logged peak reserved-memory watermark across the run was 181.82 GiB.
- The console `memory` field is TorchTitan's peak CUDA reserved-memory watermark, not current steady-state allocation.
