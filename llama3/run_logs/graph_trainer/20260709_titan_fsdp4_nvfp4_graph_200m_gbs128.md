# TorchTitan NVFP4 GraphTrainer 200M Token Run

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
  --training.local_batch_size 32 \
  --training.seq_len 2048 \
  --training.steps 763 \
  --dataloader.dataset c4 \
  --metrics.log_freq 10 \
  --optimizer.param-groups.0.optimizer-kwargs.lr 0.0003 \
  --hf-assets-path ./tests/assets/tokenizer \
  --override.imports torchtitan.overrides.nvfp4_linear
```

## Run Shape

- Model: Llama 3 8B via `graph_trainer_llama3_8b`
- Precision path: NVFP4 override
- Parallelism: 4x FSDP, TP 1
- Local batch size: 32
- Global batch size: 128
- Sequence length: 2048
- Tokens per step: 262,144
- Steps requested: 763
- Tokens requested/processed: 200,015,872
- Dataset: `c4`

## Result

The run completed successfully.

Last logged training metric was step 760:

| Step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 760 | 1.27011 | 28,488 | 113,952 | 1,374.58 | 87.58 GiB |

TorchTitan logs `tps` as tokens/sec per device. Aggregate TPS above is `tps * 4`.

## Notes

- Batch-size fit check at local batch 32 completed successfully before the long run.
