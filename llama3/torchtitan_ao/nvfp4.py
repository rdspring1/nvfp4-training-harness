"""torchao NVFP4 quantization plugin for TorchTitan."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    get_wgrad_sign_vector,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
    nvfp4_linear,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
    NVFP4Linear,
)
from torchao.quantization.quantize_.common.kernel_preference import (
    KernelPreference,
)
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common import Linear
from torchtitan.protocols.module import Module


class TritonNVFP4Linear(NVFP4Linear, Module):
    """NVFP4 Linear backed by torchao's NVFP4Linear (Triton kernels).

    Diamond inheritance (NVFP4Linear + Module) keeps the module tree flat
    and satisfies TorchTitan's Module protocol — no nested nn.Linear child
    that would trip verify_module_protocol().
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for Linear.Config that builds TritonNVFP4Linear."""

    def __init__(self, config: Config):
        super().__init__(
            in_features=config.in_features,
            out_features=config.out_features,
            bias=config.bias,
            kernel_preference=KernelPreference.TRITON,
        )

    def _shard_states(self, parallel_dims: ParallelDims) -> None:
        # NVFP4 buffers are not declared in the llama3 sharding helper. Use an
        # empty NamedPlacement so resolve_mesh([]) -> None -> parent skips them
        # and they stay as local tensors.
        sc = self._sharding_config
        assert sc is not None
        for name in ("_sr_seed", "_rht_sign_vector"):
            sc.state_shardings.setdefault(name, {})
        super()._shard_states(parallel_dims)
        if not parallel_dims.tp_enabled:
            return
        self._tp_mesh = parallel_dims.get_mesh("tp")
        # Detect colwise vs rowwise from the weight's own DTensor placements.
        if isinstance(self.weight, DTensor):
            tp_axis = self.weight.device_mesh.mesh_dim_names.index("tp")
            placement = self.weight.placements[tp_axis]
            if isinstance(placement, Shard) and placement.dim == 1:
                self.tensor_parallel_style = "rowwise"
            else:
                self.tensor_parallel_style = "colwise"
        # Colwise: local matmul yields [M, N_local] -> Shard(-1) on TP.
        # Rowwise: local matmul yields partial sums over K_local -> Partial
        # on TP; torchtitan's _redistribute_outputs reduces Partial -> the
        # declared out_dst_shardings placement (Replicate or Shard(1)).
        self._tp_out_placement: Shard | Replicate | Partial = (
            Partial() if self.tensor_parallel_style == "rowwise" else Shard(-1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.weight, DTensor):
            # FSDP-only or single-GPU: NVFP4Linear handles the local case.
            return super().forward(input)
        # TP path: deliberately bypass torchao's nvfp4_{col,row}_parallel_linear
        # (which run their own cross-TP amax all-reduce + dy all-gather) and
        # quantize against each rank's local amax, letting DTensor's
        # redistribute reduce the Partial output for rowwise. Keeps the graph
        # DTensor-native for graph_trainer / compile; numerics diverge from
        # torchao's TP path by design.
        x_local = input.to_local() if isinstance(input, DTensor) else input
        w_local = self.weight.to_local()
        b_local = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
        out = nvfp4_linear(
            x_local,
            w_local,
            b_local,
            kernel_preference=self.kernel_preference,
            sr_seed=self._sr_seed,
            sign_vector=self.rht_sign_vector,
        )
        return DTensor.from_local(
            out, self._tp_mesh, (self._tp_out_placement,), run_check=False
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # to_empty() leaves _sr_seed and _rht_sign_vector as uninitialized
        # memory on the target device. Re-draw real values here.
        dev = buffer_device or self.weight.device
        self._sr_seed.copy_(
            torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev)
        )
        self._rht_sign_vector.copy_(
            get_wgrad_sign_vector(16, device=dev, dtype=torch.int8)
        )
        # Under TP every rank must agree on the RHT basis. _sr_seed stays
        # per-rank (torchao treats SR seeds as independent across ranks).
        # _tp_mesh is only set when _shard_states ran with TP enabled.
        tp_mesh = getattr(self, "_tp_mesh", None)
        if tp_mesh is not None:
            # group_src=0 (not src=0) so each TP group broadcasts from its own
            # local rank 0 — otherwise ranks whose TP group doesn't contain
            # global rank 0 (e.g. under tp2_fsdp2) raise.
            dist.broadcast(
                self._rht_sign_vector, group=tp_mesh.get_group(), group_src=0
            )
        self._refresh_rht_sign_vector_tuple()
