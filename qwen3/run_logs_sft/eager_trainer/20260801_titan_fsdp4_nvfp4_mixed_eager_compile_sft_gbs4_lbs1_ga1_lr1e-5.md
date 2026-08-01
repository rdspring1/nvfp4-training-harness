# Qwen3-8B NVFP4-MIXED GSM8K SFT FSDP4 Run

## Command

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 -m torchtitan.train --module qwen3 --config qwen3_8b_nvfp4_mixed --parallelism.data_parallel_shard_degree 4 --parallelism.tensor_parallel_degree 1 --training.local_batch_size 1 --training.global_batch_size 4 --training.steps 180 --metrics.log_freq 10 --metrics.no-enable-wandb --checkpoint.load_only --optimizer.param-groups.0.optimizer-kwargs.lr 1e-05
```

## Run Shape

- Trainer: eager Trainer with model `torch.compile`
- Model/dataset: Qwen3-8B SFT on `openai/gsm8k` (`main/train`)
- Parallelism: FSDP 4, TP 1
- Local/global batch size: 1 / 4 (gradient accumulation 1)
- Sequence length/steps: 2048 / 180
- TorchTitan revision: `29e59407a0c88868047572fba6a380c35e520295`
- Harness revision: `5f9d112fae1bb87f24d51805fc1c596a20328749`

## Result

- Completion marker: present
- Wall time: 0:05:39.543415

| Final logged step | Loss | TPS / GPU | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: |
| 180 | 0.29542 | 3,547 | 173.93 | 46.55GiB(25.26%) |
