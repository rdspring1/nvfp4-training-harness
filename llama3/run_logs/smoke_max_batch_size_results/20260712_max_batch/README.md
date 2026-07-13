# Llama 3 8B Exact Maximum-Batch Smoke Suite

Date: 2026-07-12

## Contract and Source State

- Model: Llama 3 8B
- Hardware/parallelism: 4 NVIDIA GB200 GPUs, FSDP 4, TP 1
- Sequence length: 2048
- Dataset: `c4_test`
- Gradient accumulation: 1
- Acceptance boundary: local batch `N` completes 10 steps with finite loss and `Training completed`; adjacent local batch `N+1` fails from CUDA OOM.
- Harness commit at suite completion: `52528f8`
- TorchTitan submodule recorded commit: `ff8b307dbc310e89469399fbd873faad3d1e1001`
- TorchTitan working tree commit: `041ec0170c36ddefab0d4d928c9cdac3ea430a07` (dirty local MXFP8/GraphTrainer compatibility changes retained and not staged)
- TorchAO submodule: `f229086c0aa04c4b36c0c153db268f7e81d851fe`

## Commands

Each eager run used this command, substituting `N`, `4*N`, and the precision override (none for BF16):

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node 4 --local-ranks-filter 0 \
  -m torchtitan.train --module llama3 --config llama3_8b \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.data_parallel_shard_degree 4 \
  --training.local_batch_size N --training.global_batch_size 4*N \
  --training.seq_len 2048 --training.steps 10 \
  --dataloader.dataset c4_test --metrics.log_freq 10 \
  --optimizer.param-groups.0.optimizer-kwargs.lr 0.0003 \
  --hf-assets-path ./tests/assets/tokenizer \
  --compile.enable --compile.components model \
  [--override.imports torchtitan.overrides.{mxfp8,nvfp4}_linear]
```

Each GraphTrainer run used the same training arguments with `--module graph_trainer.llama3 --config graph_trainer_llama3_8b`, no eager compile flags, the default memory policy, normal graph passes, and CUDA graphs.

The retained MXFP8 boundary logs came from the same-day canonical launcher probes. They used physical global batches `4*N`; TorchTitan selected GA=1 from its default global-batch setting.

## Exact Boundaries

TPS is TorchTitan's per-device value at step 10. Aggregate TPS is TPS/GPU multiplied by four. Peak reserved memory and loss are also the step-10 values from each winning raw log.

| Precision | Trainer | Max local batch | Physical global batch | TPS / GPU | Aggregate TPS | Peak reserved memory | Step-10 loss | Adjacent OOM evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 | eager + `torch.compile` | 36 | 144 | 21,930 | 87,720 | 182.52 GiB | 8.77061 | LBS37 reached step 1, then exited nonzero with CUDA OOM; no completion marker |
| BF16 | GraphTrainer | 23 | 92 | 19,871 | 79,484 | 173.79 GiB | 8.98496 | LBS24 failed before step 1 with CUDA OOM; no completion marker |
| MXFP8 | eager + `torch.compile` | 33 | 132 | 28,833 | 115,332 | 180.10 GiB | 9.40290 | LBS34 reached step 1, then exited nonzero with CUDA OOM; no completion marker |
| MXFP8 | GraphTrainer | 22 | 88 | 23,561 | 94,244 | 171.82 GiB | 7.70225 | LBS23 failed before step 1 with CUDA OOM; no completion marker |
| NVFP4 | eager + `torch.compile` | 67 | 268 | 32,329 | 129,316 | 172.68 GiB | 7.37868 | LBS68 reached step 1, then exited nonzero with CUDA OOM; no completion marker |
| NVFP4 | GraphTrainer | 67 | 268 | 28,205 | 112,820 | 166.83 GiB | 7.26846 | LBS68 failed before step 1 with CUDA OOM; no completion marker |

All six winning logs contain 10 finite-loss steps and `Training completed`. Some winning eager logs contain recoverable allocator retry warnings near the memory ceiling; these did not terminate training. In every adjacent failure log, CUDA OOM is fatal and `Training completed` is absent.

## Raw Boundary Evidence

- `bf16_eager_lbs36_pass.txt` / `bf16_eager_lbs37_oom.txt`
- `bf16_graph_lbs23_pass.txt` / `bf16_graph_lbs24_oom.txt`
- `mxfp8_eager_lbs33_pass.txt` / `mxfp8_eager_lbs34_oom.txt`
- `mxfp8_graph_lbs22_pass.txt` / `mxfp8_graph_lbs23_oom.txt`
- `nvfp4_eager_lbs67_pass.txt` / `nvfp4_eager_lbs68_oom.txt`
- `nvfp4_graph_lbs67_pass.txt` / `nvfp4_graph_lbs68_oom.txt`
