# NVFP4 Training Harness

Public harness for TorchTitan and local NVFP4 training runs across:

- TorchAO BF16 baseline
- TorchAO NVFP4 training with the Triton kernel
- TransformerEngine native NVFP4 training
- TorchTitan Llama3 debug/8B runs
- TorchTitan DeepSeek V3 debugmodel runs

Llama3 launchers live under `llama3/` and write logs to `llama3_results/`.
DeepSeek V3 uses TorchTitan's `deepseek_v3_debugmodel` config and writes logs to
`deepseek_v3_results/`.

For Llama3 reproduction context, see [llama3/TRAINING_RUNS.md](llama3/TRAINING_RUNS.md).

## Install

Start from a CUDA/PyTorch environment that can see the target GPUs, then install
the harness dependencies:

```bash
python -m pip install --upgrade pip
git submodule update --init third_party/torchtitan third_party/torchao
python -m pip install -e third_party/torchtitan
USE_CPP=0 python -m pip install -e third_party/torchao --no-build-isolation

python -m pip install 'transformer-engine[pytorch]'

git clone https://github.com/meta-pytorch/MSLK.git ../MSLK
MSLK_PYTHON_ONLY=1 python -m pip install -e ../MSLK

python -m pip install -r requirements.txt
```

## Run

Run commands from the repo root. Llama3 long-run launchers are configured with
an 8-hour wall clock. Single-GPU commands can run concurrently if you give them
different `--gpu` values.

### TorchTitan Smoke

```bash
python llama3/run_titan.py single --smoke --gpu 0 --nvfp4
python deepseek_v3/run_titan.py --steps 1 --gpu 0
```

### Llama3 Single GPU

```bash
nohup python llama3/run_comparison.py --only bf16 --gpu 0 --compile reduce-overhead > run_ao_bf16.log 2>&1 &
nohup python llama3/run_triton.py --gpu 1 --compile > run_ao_nvfp4.log 2>&1 &
nohup python llama3/run_te.py --only nvfp4 --gpu 2 > run_te_nvfp4.log 2>&1 &
```

### Llama3 Multi GPU

These commands use 4 visible GPUs with the `tp2_fsdp2` shape.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python llama3/run_comparison_multi.py --compile > run_ao_bf16_multi.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python llama3/run_triton_multi.py --only tp2_fsdp2 --compile > run_ao_nvfp4_multi.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python llama3/run_te_multi.py --only tp2_fsdp2 > run_te_nvfp4_multi.log 2>&1 &
```

Monitor launcher output with `tail -f run_*.log`. Per-run training logs are
written under `llama3_results/` and `deepseek_v3_results/`.

## Acknowledgments

Thanks to Bruce Zitelli ([@tbqh](https://github.com/tbqh)) for the initial
version of this harness, which trained a LLaMA 3 model with TorchAO BF16 and
TransformerEngine (TE) NVFP4.
