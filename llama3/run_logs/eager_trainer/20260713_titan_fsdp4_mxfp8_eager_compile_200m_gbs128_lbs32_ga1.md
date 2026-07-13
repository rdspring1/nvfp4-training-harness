# TorchTitan MXFP8 Eager Trainer Compile 200M Token Run

Date: 2026-07-13

## Command

Run from `/home/me/nvfp4/third_party/torchtitan`:

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 \
  -m torchtitan.train \
  --module llama3 \
  --config llama3_8b_mxfp8 \
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

- Harness commit at launch: `3b37eed`
- TorchTitan working tree commit: `e8e3998442f65c743ecfab44f3b7ce9874a31e1d` (branch `nvfp4_linear_ao`)
- TorchAO: `0.18.0+gitcb76f29` (installed from local source `/opt/pytorch/ao`)

### MXFP8 mechanism (compiled CUDA dim1 cast — supersedes the prior TRITON run)

This run uses the **committed** `MXFP8LinearConverter` (`torchtitan.components.quantization.mx`),
enabled through the `llama3_8b_mxfp8` config function. The torchao MXFP8 dim1 cast now uses the
**compiled CUDA** kernel (`torchao::mxfp8_quantize`), provided by the prebuilt `_C_mxfp8` extension
already present in this torchao install — **no rebuild was needed** (the kernel was verified running
on GB200 before the run; `mxfp8_linear.py:66` dim1 default is `CUDA`). The converter log line
`Converted Linear layers to MXFP8Linear` confirms the committed path was active.

**These numbers supersede the prior TRITON-dim1 200M MXFP8 numbers** (which were produced when the
compiled CUDA op was unavailable). The authoritative, same-environment CUDA-vs-TRITON isolate is
`smoke_cuda_triton_ab_results/20260713_cuda_triton_ab/` (commit `3b37eed`): **CUDA is +2.3% eager /
+3.0% graph** over TRITON at fixed batch on identical torchao `0.18.0`. The raw TPS uplift versus the
*previous* 200M TRITON run here is smaller (28,084 vs 27,959, +0.4%) because that run used a
**different torchao** (`f229086c`, branch `mxfp8_triton_compat`); the fixed-batch A/B is the clean
kernel delta, not this cross-torchao comparison. (Both the committed converter and the earlier,
lost `torchtitan.overrides.mxfp8_linear` override are distinct mechanisms; only the committed
converter is reproducible.)

## Run Shape

- Model: Llama 3 8B via eager `Trainer` and `llama3_8b_mxfp8`
- Compilation: model compiled with `torch.compile`
- Precision: MXFP8 committed `MXFP8LinearConverter`, **CUDA dim1 cast**
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

The run completed successfully with finite loss and emitted `Training completed`. Metrics were logged
every 10 steps, so the final logged metric is step 760; successful completion of the configured
763-step loop is confirmed by the completion marker. At `lbs32` MXFP8 eager sits near the memory
ceiling (peak 97.63%); this CUDA run logged **0 CUDA allocator retries** over the run.

| Final logged step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 760 | 1.26707 | 28,084 | 112,336 | 1,355.08 | 179.94 GiB (97.63%) |

Training runtime from `Training starts at step 1` to `Training completed` was 30m 0.8s. End-to-end
runtime from the first torchrun log timestamp to process-group destruction was 30m 12.2s.
