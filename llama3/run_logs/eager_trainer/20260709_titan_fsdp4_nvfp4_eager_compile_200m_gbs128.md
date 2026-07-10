# TorchTitan NVFP4 Eager Trainer Compile 200M Token Run

Date: 2026-07-09

## Command

Run from `/home/me/nvfp4/third_party/torchtitan`:

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 \
  -m torchtitan.train \
  --module llama3 \
  --config llama3_8b \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.data_parallel_shard_degree 4 \
  --training.local_batch_size 32 \
  --training.global_batch_size 128 \
  --training.seq_len 2048 \
  --training.steps 763 \
  --dataloader.dataset c4 \
  --metrics.log_freq 10 \
  --optimizer.param-groups.0.optimizer-kwargs.lr 0.0003 \
  --hf-assets-path ./tests/assets/tokenizer \
  --compile.enable \
  --compile.components model \
  --override.imports torchtitan.overrides.nvfp4_linear
```

## Run Shape

- Model: Llama 3 8B via the eager `Trainer` and `llama3_8b`
- Compilation: each TransformerBlock compiled with `torch.compile`
- Precision path: NVFP4 override
- Parallelism: 4x FSDP, TP 1
- Local batch size: 32
- Global batch size: 128
- Gradient accumulation steps: 1
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
| 760 | 1.27536 | 31,407 | 125,628 | 1,515.40 | 104.95 GiB |

TorchTitan logs `tps` as tokens/sec per device. Aggregate TPS above is `tps * 4`.

Training time from `Training starts at step 1` to `Training completed` was 31m 11.3s. End-to-end process wall time from the first torchrun log to process group destruction was 31m 29.2s.
