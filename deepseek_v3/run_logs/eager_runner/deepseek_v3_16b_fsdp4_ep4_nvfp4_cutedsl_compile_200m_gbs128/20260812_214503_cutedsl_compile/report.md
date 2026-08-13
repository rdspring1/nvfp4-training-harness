# DeepSeek V3 16B TorchAO NVFP4 CuTeDSL compile replay

## Command

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python deepseek_v3/run_titan.py --precision nvfp4 --flavor 16b --compile --expert-parallel-degree 4 --batch-size 8 --global-batch-size 128 --seq-len 4096 --steps 382 --dataset c4 --log-freq 10`

## Inputs

- Raw replay log: `replay.txt`
- TorchAO: `2dc934f6`; TorchTitan: `f5889f85`; MSLK: `d9a7b37`
- Wallclock: -11m 15s

## Metrics

| Metric | Replay | August TorchAO baseline |
| --- | ---: | ---: |
| loss at step 380 | 4.63040 | 4.61204 |
| mean TPS/GPU, steps 50-380 | 17,506 | 15,563 |
| peak reserved GiB | 146.92 | 147.59 |

Step-380 loss delta (replay - baseline): +0.01836. Plateau mean delta over common logged steps 200-380: +0.01861.

## Interpretation

The overlay (`training_loss_four_series_overlay.png`) includes historical TorchAO, TE, BF16, and this replay. The baseline used TorchAO `6b062ac5`; this replay used `2dc934f6`, so performance and loss comparisons are observational rather than an exact source-revision reproduction.
