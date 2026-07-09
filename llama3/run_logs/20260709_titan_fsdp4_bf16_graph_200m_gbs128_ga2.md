# TorchTitan BF16 GraphTrainer 200M Token Run

Date: 2026-07-09

## Command

Run from `/home/me/nvfp4/third_party/torchtitan`:

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 \
  -m torchtitan.train \
  --module graph_trainer.llama3 \
  --config graph_trainer_llama3_8b \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.data_parallel_shard_degree 4 \
  --training.local_batch_size 16 \
  --training.global_batch_size 128 \
  --training.seq_len 2048 \
  --training.steps 763 \
  --dataloader.dataset c4 \
  --metrics.log_freq 10 \
  --optimizer.param-groups.0.optimizer-kwargs.lr 0.0003 \
  --hf-assets-path ./tests/assets/tokenizer
```

## Run Shape

- Model: Llama 3 8B via `graph_trainer_llama3_8b`
- Precision path: BF16
- Parallelism: 4x FSDP, TP 1
- Local batch size: 16
- Global batch size: 128
- Gradient accumulation steps: 2
- Sequence length: 2048
- Tokens per optimizer step: 262,144
- Steps requested: 763
- Tokens requested/processed: 200,015,872
- Dataset: `c4`

## Result

The run completed successfully.

Last logged training metric was step 760:

| Step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 760 | 1.32800 | 21,488 | 85,952 | 1,036.83 | 137.03 GiB |

TorchTitan logs `tps` as tokens/sec per device. Aggregate TPS above is `tps * 4`.

Training time from `Training starts at step 1` to `Training completed` was 39m 26.7s. End-to-end process wall time from the first torchrun log to process group destruction was 39m 44.3s.

## Notes

- Physical local batch 32/global 128, local batch 31/global 124, local batch 30/global 120, and local batch 28/global 112 all OOMed during first-step graph execution.
- Local batch 16 with `--training.global_batch_size 128` completed the fit check and preserved the requested global batch through gradient accumulation.
- The console `memory` field is TorchTitan's `memory/max_reserved(GiB)`, a peak CUDA reserved-memory watermark, not current steady-state allocation.
