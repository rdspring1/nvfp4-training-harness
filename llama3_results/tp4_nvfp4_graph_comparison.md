# TP4 NVFP4 GraphTrainer Comparison

Generated: 2026-05-27

## Runs Compared

| Run | Log | Mode | Precision | Shape |
|---|---|---|---|---|
| bf16 compile | `20260526_170746_titan_multi_tp4_compile.txt` | eager Trainer + `torch.compile` | bf16 | TP4, local batch 32, seq 2048 |
| nvfp4 compile | `20260526_182207_titan_multi_tp4_nvfp4_compile.txt` | eager Trainer + `torch.compile` | NVFP4 | TP4, local batch 32, seq 2048 |
| nvfp4 graph | `20260527_122607_titan_multi_tp4_nvfp4_graph.txt` | GraphTrainer + graph compile + CUDA graphs | NVFP4 | TP4, local batch 32, seq 2048 |

All runs target 200M tokens and use the same TP4 batch shape. The GraphTrainer run completed successfully with `cudagraph_pass` applied.

## Summary

| Run | Final Step | Final Loss | Median Logged TPS | Steady Global Tok/s | Median TFLOPs | Memory GiB | Train Wall Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 compile | 3050 | 1.49925 | 16,062 | 63.8k | 775.02 | 71.61 | 52.6m |
| nvfp4 compile | 3050 | 1.64293 | 15,763.5 | 60.6k | 760.59 | 55.30 | 59.5m |
| nvfp4 graph | 3050 | 1.33323 | 14,393 | 57.2k | 694.47 | 75.07 | 62.4m |

## Graph NVFP4 Delta

Relative to eager NVFP4 compile:

| Metric | Delta |
|---|---:|
| Steady global tokens/sec | -5.55% |
| Median logged TPS | -8.69% |
| Median TFLOPs | -8.69% |
| Memory | +35.75% |
| Train wall time | +4.87% |
| Final loss | -18.85% |

Relative to bf16 compile:

| Metric | Delta |
|---|---:|
| Steady global tokens/sec | -10.31% |
| Median logged TPS | -10.39% |
| Median TFLOPs | -10.39% |
| Memory | +4.83% |
| Train wall time | +18.71% |
| Final loss | -11.07% |

## Notes

- TorchTitan's logged `tps` for these TP4 runs is treated as a per-rank/per-GPU value for reporting parity with the existing logs.
- `Steady Global Tok/s` is computed from wall-clock step cadence over the steady window, excluding profiler-following intervals where `step % 100 == 10`.
- The GraphTrainer NVFP4 run is slower than eager NVFP4 compile in both logged TPS and steady global throughput, and uses more memory.
- The final loss differs across runs, but these are not controlled accuracy comparisons; use loss here as a run sanity signal, not as a quality conclusion.
