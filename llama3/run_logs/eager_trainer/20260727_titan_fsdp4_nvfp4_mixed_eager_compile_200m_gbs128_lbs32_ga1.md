# TorchTitan NVFP4 (bf16 tail) Eager Trainer Compile 200M Token Run

Date: 2026-07-27

## Recipe

Mixed-precision NVFP4: the leading decoder layers convert to NVFP4 while the last
`ceil(n_layers * 0.15)` layers (plus the lm_head) stay in bf16 — the final layers are the most
precision-sensitive. Selected via the `llama3_8b_nvfp4_mixed` config flavor added in torchtitan
commit `a057292c` ("Add mixed-precision NVFP4 configs (bf16 tail)"); the flavor computes the
converter `fqns` include-list with `nvfp4_bf16_tail_fqns(n_layers, 0.15)`. For Llama 3 8B
(32 layers) this keeps layers 27–31 in bf16 and converts layers 0–26 to NVFP4.

## Command

Run from `/home/me/nvfp4/llama3` (via `run_titan.py multi --only fsdp4 --nvfp4-mixed
--batch-size 32 --total-tokens 200000000`):

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 \
  -m torchtitan.train \
  --module llama3 \
  --config llama3_8b_nvfp4_mixed \
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

`torch.compile` is **not** passed on the CLI: the `llama3_8b_nvfp4_mixed` config function enables it
(`CompileConfig(enable=True, components=["model"])`), since NVFP4's dynamic quantization needs
compile for competitive perf. Global batch size is not passed either; it defaults to
`local_batch_size × dp_shard = 128` (gradient accumulation 1).

## Result

Final (step 760): loss **1.2715**, tps **30,040**, tflops **1449.45**, memory **110.95 GiB (60.2%)**.

Comparison at step 760 with the same-config baselines in this directory:

| Run | Final loss | Tps | Memory |
| --- | --- | --- | --- |
| NVFP4 (bf16 tail, this run) | 1.2715 | 30,040 | 110.95 GiB (60.2%) |
| NVFP4 (full) | 1.2687 | 31,758 | 103.00 GiB (55.9%) |
| MXFP8 | 1.2671 | 28,084 | 179.94 GiB (97.6%) |
| BF16 | 1.2738 | 21,919 | 174.49 GiB (94.7%) |

The bf16 tail costs ~5% throughput and ~8 GiB versus full NVFP4 (the last 5 layers run bf16 GEMMs
and hold bf16 weights), while staying well ahead of MXFP8/BF16 on both. Final loss lands on par
with all three precisions.
