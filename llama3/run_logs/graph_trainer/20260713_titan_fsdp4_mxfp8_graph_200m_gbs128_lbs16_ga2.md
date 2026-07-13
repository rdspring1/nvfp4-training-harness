# TorchTitan MXFP8 GraphTrainer 200M Token Run

Date: 2026-07-13

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

- Harness commit at launch: `3038e2d`
- TorchTitan working tree commit: `e8e3998442f65c743ecfab44f3b7ce9874a31e1d` (branch `nvfp4_linear_ao`)
- TorchAO: `0.18.0+gitcb76f29` (installed from local source `/opt/pytorch/ao`)

### MXFP8 mechanism (compiled CUDA dim1 cast — supersedes the prior TRITON run)

This run uses the **committed** `MXFP8LinearConverter` (`torchtitan.components.quantization.mx`),
enabled through the `graph_trainer_llama3_8b_mxfp8` config function. The torchao MXFP8 dim1 cast now
uses the **compiled CUDA** kernel (`torchao::mxfp8_quantize`), provided by the prebuilt `_C_mxfp8`
extension already present in this torchao install — **no rebuild was needed** (`mxfp8_linear.py:66`
dim1 default is `CUDA`). The converter log line `Converted Linear layers to MXFP8Linear` confirms the
committed path was active, and CUDA graphs were preserved (all 11 graph passes including the
cudagraph pass completed).

**These numbers supersede the prior TRITON-dim1 200M MXFP8 numbers** (produced when the compiled CUDA
op was unavailable). The authoritative, same-environment CUDA-vs-TRITON isolate is
`smoke_cuda_triton_ab_results/20260713_cuda_triton_ab/` (commit `3b37eed`): **CUDA is +3.0% graph**
over TRITON at fixed batch on identical torchao `0.18.0`. The raw uplift versus the *previous* 200M
TRITON graph run here is consistent (26,658 vs 25,817, **+3.3%**); the prior run used a different
torchao (`f229086c`), so the fixed-batch A/B remains the clean kernel delta. (The committed converter
and the earlier, lost `torchtitan.overrides.mxfp8_linear` override are distinct mechanisms; only the
committed converter is reproducible.)

## Run Shape

- Model: Llama 3 8B via `GraphTrainer` and `graph_trainer_llama3_8b_mxfp8`
- Graph execution: default memory policy, 11 graph passes, and CUDA graphs
- Precision: MXFP8 committed `MXFP8LinearConverter`, **CUDA dim1 cast**
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

The run completed successfully with finite loss and emitted `Training completed`. Metrics were logged
every 10 steps, so the final logged metric is step 760; successful completion of the configured
763-step loop is confirmed by the completion marker. All 11 graph passes completed in 28.614 seconds.
At `lbs16` the run sits at 75.72% of memory and logged **0 CUDA allocator retries**.

| Final logged step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 760 | 1.33292 | 26,658 | 106,632 | 1,286.28 | 139.55 GiB (75.72%) |

Training runtime from `Training starts at step 1` to `Training completed` was 32m 0.1s. End-to-end
runtime from the first torchrun log timestamp to process-group destruction was 33m 9.3s.
