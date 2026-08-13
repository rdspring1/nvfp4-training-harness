# DeepSeek V3 16B TorchAO NVFP4 Triton replay

## Command

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python deepseek_v3/run_titan.py --precision nvfp4 --flavor 16b --compile --expert-parallel-degree 4 --batch-size 8 --global-batch-size 128 --seq-len 4096 --steps 382 --dataset c4 --log-freq 10`

## Inputs

- Raw replay log: `replay.txt`
- TorchAO: `1a1a843b`; TorchTitan: `f5889f85`; MSLK: `d9a7b37`
- Wallclock: 51m 39s

## Metrics

| Metric | Replay | August TorchAO baseline |
| --- | ---: | ---: |
| loss at step 380 | 4.62037 | 4.61204 |
| mean TPS/GPU, steps 50-380 | 16,521 | 15,563 |
| peak reserved GiB | 146.89 | 147.59 |

Step-380 loss delta (replay - baseline): +0.00833. Plateau mean delta over common logged steps 200-380: +0.00908.

## Interpretation

The overlay (`training_loss_four_series_overlay.png`) includes historical TorchAO, TE, BF16, and this replay. The baseline used TorchAO `6b062ac5`; this replay used `1a1a843b`, so performance and loss comparisons are observational rather than an exact source-revision reproduction.
