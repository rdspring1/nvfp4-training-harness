# Llama 3.1 8B NVFP4-MIXED C4 Continued Pretraining

## Command

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 -m torchtitan.train --module llama3 --config llama3_8b_continue_pretrain_nvfp4_mixed --parallelism.data_parallel_shard_degree 4 --parallelism.tensor_parallel_degree 1 --training.local_batch_size 32 --training.global_batch_size 128 --training.seq_len 2048 --training.steps 2 --metrics.log_freq 1 --metrics.no-enable-wandb --checkpoint.load_only
```

## Run Shape

- Trainer: eager Trainer with model `torch.compile`
- Initialization: local Llama 3.1 8B HF checkpoint
- Dataset: C4
- Parallelism: FSDP 4, TP 1
- Local/global batch size: 32 / 128 (gradient accumulation 1)
- Sequence length/steps: 2048 / 2
- Tokens processed: 524,288
- TorchTitan revision: `3c1999e6c7d6b97f46d591bfd58ccf406b9f85cb`
- Harness revision: `2a015da6b3166c80c48bc5aca862a194d102094d`

## Result

- Completion marker: present
- Step-1 loss gate: passed (< 6.10)
- Wall time: 0:04:29.573172

| Final logged step | Loss | Grad norm | TPS / GPU | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2.43748 | 2.3116 | 28,063 | 1,354.06 | 114.28GiB(62.01%) |
