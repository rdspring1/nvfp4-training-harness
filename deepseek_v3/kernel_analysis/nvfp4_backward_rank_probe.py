#!/usr/bin/env python3
"""Does NVFP4 stochastic-rounding noise decorrelate across data-parallel ranks?

Hypothesis
----------
torchtitan's set_determinism (torchtitan/distributed/utils.py:276) calls
torch.manual_seed(seed) with the SAME seed on all SPMD ranks; only PP ranks get
a distinct seed via distinct_seed_mesh_dims. With pp=1 every DP/EP rank shares
one generator state. Then:

  * _sr_seed is drawn per module in _init_self_buffers
    (torchtitan/components/quantization/nvfp4.py:392-398) with torch.randint,
  * every backward draws col_offset / row_offset from the default CUDA generator
    (nvfp4_grouped_mm.py:273-279).

Both are plain torch.randint calls on local tensors, so the DTensor RNG tracker
never sees them. If they collide across ranks, SR noise is bit-identical
everywhere, and in the all-reduced wgrad it adds coherently instead of averaging
down by 1/sqrt(N_dp) -- raising the gradient noise floor by sqrt(N_dp), which
raises the stationary loss and slows convergence.

This probe answers two questions with measurements rather than inference:
  1. Do _sr_seed and the per-step SR offsets actually collide across ranks?
  2. Does the all-reduced wgrad noise fall by sqrt(N) when the seeds are made
     distinct, and not when they are shared?

Run
---
  torchrun --nproc_per_node=4 nvfp4_backward_rank_probe.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nvfp4_audit_common import (  # noqa: E402
    build_grouped_experts,
    moe_dims,
    require_blackwell,
)

MODEL = "16B"
SEED = 42
N_EXPERTS = 4
ROWS_PER_GROUP = 256


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def main() -> int:
    if (rc := require_blackwell()) is not None:
        return rc

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())

    # Exactly what set_determinism does for an SPMD run with pp=1: one seed,
    # identical on every rank.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    dim, hidden, _ = moe_dims(MODEL)
    log(rank, f"world={world} model={MODEL} dim={dim} hidden={hidden} seed={SEED}")
    log(rank, "\n--- 1. do the SR seeds collide across ranks? ---")

    q = build_grouped_experts(MODEL, device, nvfp4=True, num_experts=N_EXPERTS)
    sr_seed = q._sr_seed.clone()
    gathered = [torch.zeros_like(sr_seed) for _ in range(world)]
    dist.all_gather(gathered, sr_seed)
    seeds = [int(g.item()) for g in gathered]
    seeds_collide = len(set(seeds)) == 1
    log(rank, f"_sr_seed per rank : {seeds}")
    log(rank, f"  -> {'IDENTICAL on every rank' if seeds_collide else 'distinct per rank'}")

    # The exact draw backward makes at nvfp4_grouped_mm.py:273-278.
    off = torch.randint(0, 2**32, (1,), dtype=torch.int64, device=device)
    g2 = [torch.zeros_like(off) for _ in range(world)]
    dist.all_gather(g2, off)
    offs = [int(x.item()) for x in g2]
    offs_collide = len(set(offs)) == 1
    log(rank, f"backward SR offset: {offs}")
    log(rank, f"  -> {'IDENTICAL on every rank' if offs_collide else 'distinct per rank'}")

    log(rank, "\n--- 2. does the all-reduced wgrad noise average down? ---")
    log(rank, "each rank gets different data, as in real DP training")

    ref = build_grouped_experts(MODEL, device, nvfp4=False, num_experts=N_EXPERTS)
    torch.manual_seed(SEED)          # identical weights on every rank
    with torch.no_grad():
        for _, p in ref.named_parameters():
            p.copy_(torch.randn_like(p) * 0.02)
        for n, p in ref.named_parameters():
            getattr(q, n).data = p.data.clone()

    counts = torch.full((N_EXPERTS,), ROWS_PER_GROUP, dtype=torch.int32, device=device)
    M = int(counts.sum())
    gen = torch.Generator(device=device).manual_seed(1000 + rank)   # per-rank data
    x = torch.randn(M, dim, generator=gen, device=device, dtype=torch.bfloat16)
    gout = torch.randn(M, dim, generator=gen, device=device, dtype=torch.bfloat16)

    def allreduced_wgrad(module, distinct_seed: bool):
        if distinct_seed:
            module._sr_seed = torch.tensor(
                [SEED * 1000003 + rank], dtype=torch.int64, device=device
            )
        else:
            module._sr_seed = sr_seed.clone()
        module.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        module(xi, counts).backward(gout)
        g = module.w1_EFD.grad.detach().clone().float()
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        return g / world

    ref.zero_grad(set_to_none=True)
    xr = x.clone().requires_grad_(True)
    ref(xr, counts).backward(gout)
    g_bf16 = ref.w1_EFD.grad.detach().clone().float()
    dist.all_reduce(g_bf16, op=dist.ReduceOp.SUM)
    g_bf16 /= world

    results = {}
    for label, distinct in (("shared _sr_seed (as shipped)", False),
                            ("per-rank distinct _sr_seed", True)):
        # Re-align the per-step offset draws across ranks so the only thing
        # varying between the two cases is _sr_seed itself.
        torch.manual_seed(SEED + 7)
        torch.cuda.manual_seed(SEED + 7)
        g = allreduced_wgrad(q, distinct)
        err = (g - g_bf16).norm() / g_bf16.norm()
        results[label] = float(err)
        log(rank, f"  {label:32s} relative wgrad error {float(err):.6f}")

    a = results["shared _sr_seed (as shipped)"]
    b = results["per-rank distinct _sr_seed"]
    log(rank, f"\nratio shared/distinct = {a / b:.4f}   (sqrt(world) = {world ** 0.5:.4f})")
    log(rank, "if SR noise were the dominant error and it decorrelated with distinct")
    log(rank, "seeds, this ratio would approach sqrt(world); ~1.0 means either the")
    log(rank, "seeds already differ or SR noise is not what dominates the wgrad error.")

    log(rank, "\n--- summary ---")
    log(rank, f"  _sr_seed collides across ranks     : {seeds_collide}")
    log(rank, f"  SR step offset collides across ranks: {offs_collide}")
    log(rank, f"  wgrad noise ratio shared/distinct   : {a / b:.4f} vs sqrt(N)={world ** 0.5:.4f}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
