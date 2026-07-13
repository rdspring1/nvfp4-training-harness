# TorchTitan MXFP8 GraphTrainer 200M Token Run

Date: 2026-07-12

## Command

Run from `/home/me/nvfp4/third_party/torchtitan`:

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 \
  -m torchtitan.train \
  --module graph_trainer.llama3 \
  --config graph_trainer_llama3_8b_mxfp8 \
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

## Source State

- Harness commit at launch: `2e577e4`
- TorchTitan working tree commit: `e8e3998442f65c743ecfab44f3b7ce9874a31e1d` (branch `nvfp4_linear_ao`)
- TorchAO commit: `00e05f82896e6d6d8991ab2e2a01764c0f1f0dbe` (branch `mxfp8_triton_compat`)

### MXFP8 mechanism (changed from prior benchmark)

This run uses the **committed** `MXFP8LinearConverter` (`torchtitan.components.quantization.mx`), enabled
through the `graph_trainer_llama3_8b_mxfp8` config function, **not** the prior
`torchtitan.overrides.mxfp8_linear` override (which was lost and is unrecoverable). The torchao MXFP8 dim1
cast is set to the **TRITON** kernel because this torchao is a Python-only editable install with no compiled
`_C` extension. The converter log line `Converted Linear layers to MXFP8Linear` confirms the committed path
was active, and CUDA graphs were preserved (all 11 graph passes including the cudagraph pass completed).

**These numbers supersede but are not directly comparable to the prior override-based MXFP8 numbers** — the
quantization mechanism (committed converter + TRITON dim1 cast) differs from the lost override.

## Run Shape

- Model: Llama 3 8B via `GraphTrainer` and `graph_trainer_llama3_8b_mxfp8`
- Graph execution: default memory policy, 11 graph passes, and CUDA graphs
- Precision: MXFP8 committed `MXFP8LinearConverter`, TRITON dim1 cast
- Parallelism: FSDP 4, TP 1
- Local batch size: 16
- Physical global batch size: 64
- Explicit global batch size: 128
- Gradient accumulation steps: 2
- Sequence length: 2048
- Tokens per optimizer step: 262,144
- Steps requested/completed: 763 / 763
- Tokens requested/processed: 200,015,872 / 200,015,872
- Dataset: `c4`

## Result

The run completed successfully with finite loss and emitted `Training completed`. Metrics were logged every
10 steps, so the final logged metric is step 760; successful completion of the configured 763-step loop is
confirmed by the completion marker. All 11 graph passes completed in 31.040 seconds. At `lbs16` the run sits
at 75.72% of memory and logged **0 CUDA allocator retries**.

| Final logged step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 760 | 1.32741 | 25,817 | 103,268 | 1,245.69 | 139.55 GiB (75.72%) |

Training runtime from `Training starts at step 1` to `Training completed` was 34m 1.4s. End-to-end runtime
from the first torchrun log timestamp to process-group destruction was 34m 10.2s.
