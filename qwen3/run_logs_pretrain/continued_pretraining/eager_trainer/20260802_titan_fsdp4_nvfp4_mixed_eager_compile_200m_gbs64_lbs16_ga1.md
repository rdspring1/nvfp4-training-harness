# Qwen3-8B NVFP4-MIXED C4 Continued Pretraining

## Command

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 -m torchtitan.train --module qwen3 --config qwen3_8b_continue_pretrain_nvfp4_mixed --parallelism.data_parallel_shard_degree 4 --parallelism.tensor_parallel_degree 1 --training.local_batch_size 16 --training.global_batch_size 64 --training.seq_len 2048 --training.steps 1526 --metrics.log_freq 10 --metrics.no-enable-wandb --checkpoint.load_only
```

## Run Shape

- Trainer: eager Trainer with model `torch.compile`
- Initialization: local Qwen3-8B HF checkpoint
- Model/dataset: Qwen3-8B on C4
- Parallelism: FSDP 4, TP 1
- Local/global batch size: 16 / 64 (gradient accumulation 1)
- Sequence length/steps: 2048 / 1526
- Tokens requested: 200,000,000
- Tokens processed: 200,015,872
- TorchTitan revision: `c75cd8152400a24ba72e7524c86b956168b2662f`
- Harness revision: `35bf8f56706abae0d88e23459ffc98797d376b8d`

## Result

- Completion marker: present
- Wall time: 0:36:53.958685

| Final logged step | Loss | TPS / GPU | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: |
| 1520 | 2.50078 | 25,934 | 1 | 82.24GiB(44.63%) |
