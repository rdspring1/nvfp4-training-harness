# TorchTitan BF16 Eager Trainer Compile 200M Token Run

Date: 2026-07-12

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
  --compile.components model
```

## Source State

- Harness commit at launch: `4e16a81349f443e6484c28c4aafb9cbe1d513a1b`
- TorchTitan submodule recorded commit: `ff8b307dbc310e89469399fbd873faad3d1e1001`
- TorchTitan working tree commit at launch: `041ec0170c36ddefab0d4d928c9cdac3ea430a07` (dirty local MXFP8/GraphTrainer compatibility changes retained and not staged)
- TorchAO submodule: `f229086c0aa04c4b36c0c153db268f7e81d851fe`

## Run Shape

- Model: Llama 3 8B via eager `Trainer` and `llama3_8b`
- Compilation: each TransformerBlock compiled with `torch.compile`
- Precision: BF16
- Parallelism: FSDP 4, TP 1
- Local batch size: 32
- Physical global batch size: 128
- Explicit global batch size: 128
- Gradient accumulation steps: 1
- Sequence length: 2048
- Tokens per optimizer step: 262,144
- Steps requested/completed: 763 / 763
- Tokens requested/processed: 200,015,872 / 200,015,872
- Dataset: `c4`

## Result

The run completed successfully with finite loss and emitted `Training completed`. Metrics were logged every 10 steps, so the final logged metric is step 760; successful completion of the configured 763-step loop is confirmed by the completion marker.

| Final logged step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 760 | 1.27381 | 21,919 | 87,676 | 1,057.59 | 174.49 GiB |

Training runtime from `Training starts at step 1` to `Training completed` was 38m 14.5s. End-to-end runtime from the first torchrun log timestamp to process-group destruction was 38m 31.8s.
