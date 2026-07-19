"""Training (fwd+bwd) grouped-experts crossover sweep: bf16 vs TorchAO NVFP4 vs TE
NVFP4, at DSV3-671B expert dims (dim=7168, hidden=2048). Runs the 3-GEMM expert
MLP (gate/up/down) forward+backward and sweeps tokens/expert to find where each
NVFP4 backend overtakes bf16. Produces Table 4 in nvfp4_grouped_gemm_crossover.md.

128-aligned token counts -> no TorchAO group-pad waste. One backend per process
for clean peak-memory numbers.

Run from the repo root so `torchtitan`, `torchao`, and `te_moe_overrides` import:
    PYTHONPATH=. python deepseek_v3/nvfp4_grouped_gemm_crossover.py <bf16|torchao|te>
"""

import sys

import torch

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher

DIM = 7168
HIDDEN = 2048
TOP_K = 8
WARMUP = 3
ITERS = 10

EXPERTS = [8]
TOKENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def build(backend, num_experts):
    disp = LocalTokenDispatcher.Config(num_experts=num_experts, top_k=TOP_K)
    kw = dict(dim=DIM, hidden_dim=HIDDEN, num_experts=num_experts, token_dispatcher=disp)
    if backend == "bf16":
        return GroupedExperts.Config(**kw).build().cuda()
    if backend == "torchao":
        from torchtitan.overrides.nvfp4_grouped_experts import NVFP4GroupedExperts

        m = NVFP4GroupedExperts.Config(**kw).build().cuda()
        m._init_self_buffers(buffer_device=torch.device("cuda"))
        return m
    if backend == "te":
        from te_moe_overrides.te_grouped_experts import TEGroupedExperts

        m = TEGroupedExperts.Config(**kw).build().cuda()
        m._ensure_te_state()
        return m
    raise SystemExit(f"unknown backend {backend}")


def bench(m, num_experts, tpe):
    num_tokens = torch.full((num_experts,), tpe, device="cuda", dtype=torch.int64)
    rows = int(num_tokens.sum())
    x = torch.randn(rows, DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def step():
        m.zero_grad(set_to_none=True)
        x.grad = None
        out = m._experts_forward(x, num_tokens)
        out.backward(torch.ones_like(out))

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        step()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / ITERS
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return ms, rows / (ms / 1e3), peak


def main():
    backend = sys.argv[1]
    torch.manual_seed(0)
    print(f"# backend={backend} fwd+bwd dim={DIM} hidden={HIDDEN}")
    print(f"{'experts':>7} {'tok/exp':>7} {'rows':>7} {'ms':>8} {'Mtok/s':>8} {'mem_GiB':>7}")
    for e in EXPERTS:
        m = build(backend, e)
        with torch.no_grad():
            for p in (m.w1_EFD, m.w2_EDF, m.w3_EFD):
                p.copy_(0.02 * torch.randn_like(p))
        for tpe in TOKENS:
            try:
                ms, tok_s, peak = bench(m, e, tpe)
                print(f"{e:>7} {tpe:>7} {e*tpe:>7} {ms:>8.3f} {tok_s/1e6:>8.3f} {peak:>7.2f}")
            except Exception as ex:
                print(f"{e:>7} {tpe:>7} {e*tpe:>7}   FAIL: {type(ex).__name__}: {str(ex)[:50]}")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
