# Qwen3-8B MXFP8 C4 Continued Pretraining

## Command

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 -m torchtitan.train --module qwen3 --config qwen3_8b_continue_pretrain_mxfp8 --parallelism.data_parallel_shard_degree 4 --parallelism.tensor_parallel_degree 1 --training.local_batch_size 16 --training.global_batch_size 64 --training.seq_len 2048 --training.steps 1526 --metrics.log_freq 10 --metrics.no-enable-wandb --checkpoint.load_only
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
- TorchTitan revision: `20a66c9a108af41444222169982e15105de4c0e9`
- Harness revision: `413f3e38d947cf959f86800f854644a79a57d0a6`

## Result

- Completion marker: present
- Wall time: 0:30:59.468690

| Final logged step | Loss | TPS / GPU | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: |
| 1520 | 5.03457 | 27,532 | 1 | 112.63GiB(61.12%) |
