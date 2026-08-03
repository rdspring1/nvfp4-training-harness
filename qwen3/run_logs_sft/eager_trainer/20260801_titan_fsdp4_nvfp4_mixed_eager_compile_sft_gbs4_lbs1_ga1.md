# Qwen3-8B NVFP4-MIXED GSM8K SFT FSDP4 Run

## Command

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 -m torchtitan.train --module qwen3 --config qwen3_8b_nvfp4_mixed --parallelism.data_parallel_shard_degree 4 --parallelism.tensor_parallel_degree 1 --training.local_batch_size 1 --training.global_batch_size 4 --training.steps 180 --metrics.log_freq 10 --metrics.no-enable-wandb --checkpoint.load_only
```

## Run Shape

- Trainer: eager Trainer with model `torch.compile`
- Model/dataset: Qwen3-8B SFT on `openai/gsm8k` (`main/train`)
- Parallelism: FSDP 4, TP 1
- Local/global batch size: 1 / 4 (gradient accumulation 1)
- Sequence length/steps: 2048 / 180
- TorchTitan revision: `5d2e9ff4c14220df011ae7e52f2dc25521479071`
- Harness revision: `637cd8b19f607ccca4213b7039c78d9a47eb8f96`

## Result

- Completion marker: present
- Wall time: 0:03:19.094841

| Final logged step | Loss | TPS / GPU | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: |
| 180 | 0.30534 | 3,711 | 181.95 | 46.55GiB(25.26%) |
