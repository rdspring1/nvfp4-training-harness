# Training Run Reproduction

This note captures how the LLaMA 3 runs were reproduced.

## Context

- Model: LLaMA 3 8B shape from `LLAMA3_8B` in the training scripts.
- Data: streaming WikiText-103 raw train split, tokenized with GPT-2 and packed
  to `seq_len=2048`.
- Optimizer: AdamW with `lr=3e-4`, betas `(0.9, 0.95)`, and weight decay `0.1`.
- Runtime precision: fp32 master weights with bf16 autocast.
- Reference hardware: NVIDIA GB200.
- Single-GPU runs use batch size 4. Multi-GPU runs use 4 GPUs with
  `tp-size=2`, `fsdp-size=2`, and batch size 8 per data-parallel replica.

## Code Used

- AO single GPU: `ao_llama3_train.py`
- AO multi GPU: `ao_llama3_fsdp2_tp_train.py`
- TE single GPU: `te_llama3_train.py`
- TE multi GPU: `te_llama3_fsdp2_tp_train.py`
- Launchers: `run_comparison.py`, `run_triton.py`, `run_te.py`,
  `run_comparison_multi.py`, `run_triton_multi.py`, and `run_te_multi.py`

TorchAO BF16 runs leave `nn.Linear` unquantized. TorchAO NVFP4 runs apply
`torchao.quantization.quantize_()` with `NVFP4TrainingConfig` and the Triton
kernel. TE NVFP4 runs use TransformerEngine modules with `NVFP4BlockScaling`.

## Execution Modes

- Eager is the default: omit `--compile`.
- AO single-GPU `run_comparison.py` requires an explicit compile mode, for
  example `--compile reduce-overhead`.
- The other launchers accept bare `--compile` as `reduce-overhead`.
- Accepted compile modes are `reduce-overhead`, `default`, `max-autotune`, and
  `max-autotune-no-cudagraphs`.
- TE single-GPU `run_te.py` uses the compile option as a compatibility label:
  `reduce-overhead` and `max-autotune` enable CUDA graphs instead of calling
  `torch.compile`.

## Distributed Shapes

The AO NVFP4 and TE NVFP4 multi-GPU launchers support these 4-GPU layouts:

| Launcher name | TP size | FSDP size | Meaning |
| --- | ---: | ---: | --- |
| `tp4` | 4 | 1 | 4x tensor parallel |
| `fsdp4` | 1 | 4 | 4x FSDP |
| `tp2_fsdp2` | 2 | 2 | 2x tensor parallel and 2x FSDP |

Use the shape with `--only`, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_triton_multi.py --only fsdp4 --compile
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_triton_multi.py --only tp4 --compile
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_triton_multi.py --only tp2_fsdp2 --compile

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_te_multi.py --only fsdp4
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_te_multi.py --only tp4
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_te_multi.py --only tp2_fsdp2
```

The BF16 comparison launcher currently wraps the `tp2_fsdp2` baseline.

## Reproduction Commands

The public quick-start commands in `README.md` launch the six runs with the
same 8-hour wall-clock configuration:

```bash
nohup python run_comparison.py --only bf16 --gpu 0 --compile reduce-overhead > run_ao_bf16.log 2>&1 &
nohup python run_triton.py --gpu 1 --compile > run_ao_nvfp4.log 2>&1 &
nohup python run_te.py --only nvfp4 --gpu 2 > run_te_nvfp4.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_comparison_multi.py --compile > run_ao_bf16_multi.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_triton_multi.py --only tp2_fsdp2 --compile > run_ao_nvfp4_multi.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_te_multi.py --only tp2_fsdp2 > run_te_nvfp4_multi.log 2>&1 &
```
