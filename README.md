# NVFP4 Training Harness

Public harness for reproducing LLaMA 3 8B training runs across:

- TorchAO BF16 baseline
- TorchAO NVFP4 training with the Triton kernel
- TransformerEngine native NVFP4 training

The main runs train on streaming WikiText-103 with a GPT-2 tokenizer. The
launchers use `seq_len=2048`, `lr=3e-4`, fp32 master weights, and bf16 autocast.
They write per-run logs to `llama3_results/`.

For reproduction context, see [TRAINING_RUNS.md](TRAINING_RUNS.md).

## Install

Start from a CUDA/PyTorch environment that can see the target GPUs, then install
the harness dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install 'transformer-engine[pytorch]'

git clone https://github.com/rspring/ao.git ../ao
git -C ../ao checkout nvfp4_linear_poc_stack
USE_CPP=0 python -m pip install -e ../ao --no-build-isolation

git clone https://github.com/meta-pytorch/MSLK.git ../MSLK
MSLK_PYTHON_ONLY=1 python -m pip install -e ../MSLK

python -m pip install -r requirements.txt
```

## Run

Run commands from the repo root. Each launcher is configured with an 8-hour
wall clock. Single-GPU commands can run concurrently if you give them different
`--gpu` values.

### Single GPU

```bash
nohup python run_comparison.py --only bf16 --gpu 0 --compile reduce-overhead > run_ao_bf16.log 2>&1 &
nohup python run_triton.py --gpu 1 --compile > run_ao_nvfp4.log 2>&1 &
nohup python run_te.py --only nvfp4 --gpu 2 > run_te_nvfp4.log 2>&1 &
```

### Multi GPU

These commands use 4 visible GPUs with the `tp2_fsdp2` shape.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_comparison_multi.py --compile > run_ao_bf16_multi.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_triton_multi.py --only tp2_fsdp2 --compile > run_ao_nvfp4_multi.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python run_te_multi.py --only tp2_fsdp2 > run_te_nvfp4_multi.log 2>&1 &
```

Monitor launcher output with `tail -f run_*.log`. Per-run training logs are
written under `llama3_results/`.

## Acknowledgments

Thanks to Bruce Zitelli ([@tbqh](https://github.com/tbqh)) for the initial
version of this harness, which trained a LLaMA 3 model with TorchAO BF16 and
TransformerEngine (TE) NVFP4.
