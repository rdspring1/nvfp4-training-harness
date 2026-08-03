# TorchTitan FSDP4 NVFP4 (bf16 tail) Eager Trainer Compile 200M Token Run

Date: 2026-08-01

## Command

Run from `/home/me/nvfp4/llama3`:

```bash
PYTHONUNBUFFERED=1 python run_titan.py multi --only fsdp4 --nvfp4-mixed \
  --batch-size 32 --total-tokens 200000000
```

This selects `llama3_8b_nvfp4_mixed`, which enables `torch.compile` for the model. The run used
FSDP 4, TP 1, local batch size 32, global batch size 128, gradient accumulation 1, sequence
length 2048, the C4 dataset, and 763 optimizer steps (200,015,872 tokens).

## Source State

- PyTorch: `2.14.0a0+gitd9abf9e` (`d9abf9e1053c`, `/opt/pytorch/pytorch`)
- TorchAO: `0.18.0+gitcb76f29` (`cb76f2943f74`, `/opt/pytorch/ao`)
- TorchTitan: `5d2e9ff4c142` (branch `nvfp4_linear_ao`, version `0.2.2`)

## Result

The run completed all 763 requested steps with finite loss. The final logged metric at step 760:

| Loss | TPS / GPU | Aggregate TPS | TFLOPs / GPU | Peak Reserved Mem |
| ---: | ---: | ---: | ---: | ---: |
| 1.26408 | 30,011 | 120,044 | 1,448.07 | 110.95 GiB (60.20%) |

`Training completed` was logged at 11:40:43 UTC. Training runtime from the step-1 marker was
32m 6.6s; end-to-end runtime from the first torchrun log timestamp to process-group destruction
was 32m 15.9s.

The raw rank-0 log is the sibling `.txt` file. It confirms the NVFP4 converter, FSDP mesh
`dp_shard=4`, global batch size 128, and model compilation.
