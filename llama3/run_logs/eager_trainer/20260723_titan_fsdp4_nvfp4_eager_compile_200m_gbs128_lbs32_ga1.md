# TorchTitan NVFP4 Eager Trainer Compile 200M Token Run

Date: 2026-07-23

## Command

Run from `/home/me/nvfp4/third_party/torchtitan` (via `run_titan.py multi --only fsdp4
--nvfp4 --batch-size 32 --total-tokens 200000000`):

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 \
  -m torchtitan.train \
  --module llama3 \
  --config llama3_8b_nvfp4 \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.data_parallel_shard_degree 4 \
  --training.local_batch_size 32 \
  --training.seq_len 2048 \
  --training.steps 763 \
  --dataloader.dataset c4 \
  --metrics.log_freq 10 \
  --optimizer.param-groups.0.optimizer-kwargs.lr 0.0003 \
  --hf-assets-path ./tests/assets/tokenizer
```

`torch.compile` is **not** passed on the CLI: the `llama3_8b_nvfp4` config function enables it
(`CompileConfig(enable=True, components=["model"])`), since NVFP4's dynamic quantization needs
compile for competitive perf. Global batch size is not passed either; it defaults to
`local_batch_size × dp_shard = 128` (gradient accumulation 1).

## Source State

- Harness commit at launch: `a3fd17f` (branch `nvfp4-converter-harness`)
- TorchTitan working tree commit: `d9413379` (branch `nvfp4_converter`) — "Replace NVFP4
  override with a converter"
- TorchAO: `0.18.0+gitcb76f29` (installed from local source `/opt/pytorch/ao`)

### NVFP4 mechanism (committed converter — supersedes the prior override run)

This run uses the **committed** `NVFP4LinearConverter`
(`torchtitan.components.quantization.nvfp4`), enabled through the `llama3_8b_nvfp4` config
function (`model_registry("8B", converters=[NVFP4LinearConverter.Config(fqns=["layers"], ...)])`).
`fqns=["layers"]` converts every in-layer Linear (attention + feed_forward) and leaves the
`lm_head` stock, since NVFP4 requires each GEMM dim divisible by 128 and the vocab projection
does not satisfy it. The converter is a pure leaf swap of `Linear -> NVFP4Linear`; the log line
`Converted Linear layers to NVFP4Linear` confirms the committed path was active.

**These numbers supersede the prior override 200M NVFP4 run** (`20260709_titan_fsdp4_nvfp4_eager
_compile_200m_gbs128`, mechanism `torchtitan.overrides.nvfp4_linear`, now removed). The override
added a parent-block sequence-parallel rewrite and fp4-code collectives that only engage under
TP>1; at this **FSDP 4 / TP 1** shape neither mechanism runs TP collectives, so the two are the
same compiled fp4-GEMM path and the numbers are at parity (see comparison below).

## Run Shape

- Model: Llama 3 8B via eager `Trainer` and `llama3_8b_nvfp4`
- Compilation: model compiled with `torch.compile` (enabled by the config)
- Precision: NVFP4 committed `NVFP4LinearConverter` (leaf swap, `fqns=["layers"]`)
- Parallelism: FSDP 4, TP 1
- Local batch size: 32
- Physical global batch size: 128
- Gradient accumulation steps: 1
- Sequence length: 2048
- Tokens per optimizer step: 262,144
- Steps requested/completed: 763 / 763
- Tokens requested/processed: 200,015,872 / 200,015,872
- Dataset: `c4`

## Result

The run completed successfully with finite loss and emitted `Training completed`. Metrics were
logged every 10 steps, so the final logged metric is step 760; successful completion of the
configured 763-step loop is confirmed by the completion marker. NVFP4 sits well below the memory
ceiling at this batch (peak 55.89%); the run logged **0 CUDA allocator retries**.

| Final logged step | Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 760 | 1.26874 | 31,758 | 127,032 | 1,532.34 | 103.00 GiB (55.89%) |

Training runtime from `Training starts at step 1` to `Training completed` was 30m 53.3s.
End-to-end runtime from the first torchrun log timestamp to process-group destruction was 31m 3.8s.

### Converter vs. prior override (both FSDP 4 / TP 1, step 760)

| Metric | Override (`20260709`) | Converter (this run) | Δ |
| --- | ---: | ---: | ---: |
| Final loss | 1.27536 | 1.26874 | −0.007 (comparable) |
| Peak reserved mem | 104.95 GiB | 103.00 GiB | −1.95 GiB (−1.9%) |
| TPS / GPU | 31,407 | 31,758 | +1.1% |
| TFLOPs / GPU | 1,515.40 | 1,532.34 | +1.1% |

Parity as expected for TP 1: the converter is marginally leaner and faster (lower wrapper
overhead from the pure leaf swap). A real divergence would only surface at TP>1, where the
override's fp4 collectives did extra work the converter drops.
