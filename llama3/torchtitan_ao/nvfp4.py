"""torchao NVFP4 quantization plugin for TorchTitan.

Implements an NVFP4 sequence-parallel block path. TorchTitan owns regular module
sharding and boundary redistribution: weight/bias placement, the parent block's
sequence-parallel input boundary, and the rowwise output contract are all driven
by ShardingConfig. NVFP4 owns quantization semantics: RHT amax, the required TP
MAX all-reduce for amax, stochastic rounding, the RHT sign vector, scaled_mm, and
the fp4 code/scale collectives.

The TP all-gather (colwise) moves NVFP4 codes/scales, NOT bf16 activations -- this
is the whole point of keeping activations sequence-sharded at the parent boundary
instead of letting TorchTitan bf16-all-gather them to TP Replicate. We reuse
TorchAO's column/row-parallel autograd Functions (which all-gather fp4 codes and
reduce-scatter bf16 partials internally) wrapped as thin DTensor<->local bridges.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import nvfp4_linear
from torchao.prototype.moe_training.nvfp4_training.nvfp4_tensor_parallel import (
    nvfp4_col_parallel_linear,
    nvfp4_row_parallel_linear,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
    _make_rht_sign_vector,
    _rht_sign_vector_to_tuple,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

from torchtitan.distributed.parallel_dims import MeshAxisName, ParallelDims
from torchtitan.distributed.spmd_types import spmd_layout_to_dtensor_placements
from torchtitan.models.common import Linear
from torchtitan.protocols.sharding import ShardingConfig

TP = MeshAxisName.TP


def _infer_tp_style(
    sharding_config: ShardingConfig | None,
) -> Literal["colwise", "rowwise"] | None:
    """Infer colwise vs rowwise from the weight's declared TP placement.

    Read from the ShardingConfig (not a runtime DTensor): colwise shards the
    weight output dim (Shard(0)), rowwise shards the input/contraction dim
    (Shard(1)).
    """
    if sharding_config is None:
        return None
    weight_layout = sharding_config.state_shardings.get("weight")
    if weight_layout is None:
        return None
    tp_placement = spmd_layout_to_dtensor_placements(weight_layout).get(TP)
    if isinstance(tp_placement, Shard) and tp_placement.dim == 1:
        return "rowwise"
    return "colwise"


def _nvfp4_colwise_sp(
    x_BLD, w_local, bias, sr_seed, sign_vector, tp_group, world_size
):
    """Colwise NVFP4 over a local sequence shard, returning a feature shard.

    x_BLD is the local SP activation shard [B, L_local, D]. Transpose the
    sequence to dim 0 before flattening so TorchAO's all-gather over the TP group
    (gather dim 0) reconstructs the full sequence in order -- rank-major
    concatenation equals sequence-global ordering -- even for batch > 1. TorchAO
    all-gathers fp4 codes/scales (not bf16) and returns the full-sequence,
    feature-sharded output [m, n_local].
    """
    B, L_local, D = x_BLD.shape
    x_2d = x_BLD.transpose(0, 1).reshape(L_local * B, D)
    out_2d = nvfp4_col_parallel_linear(
        x_2d,
        w_local,
        bias,
        sr_seed=sr_seed,
        tp_group=tp_group,
        world_size=world_size,
        sign_vector=sign_vector,
    )
    n_local = out_2d.shape[-1]
    l_full = out_2d.shape[0] // B
    return out_2d.reshape(l_full, B, n_local).transpose(0, 1)


def _nvfp4_rowwise_sp(
    x_BLD, w_local, bias, sr_seed, sign_vector, tp_group, world_size
):
    """Rowwise NVFP4 over a full-sequence feature shard, returning a seq shard.

    x_BLD is the full-sequence, feature-sharded activation [B, L, K_local].
    TorchAO computes the local partial outer product and reduce-scatters the bf16
    result along the sequence dim internally, returning the SP seq shard
    [m/w, n]. Sequence-first flatten makes the reduce-scatter split on dim 0
    yield clean per-rank sequence shards even for batch > 1. TorchTitan must NOT
    reduce again (the rowwise override declares out_dst=None).
    """
    B, L, K_local = x_BLD.shape
    x_2d = x_BLD.transpose(0, 1).reshape(L * B, K_local)
    out_2d = nvfp4_row_parallel_linear(
        x_2d,
        w_local,
        bias,
        sr_seed=sr_seed,
        tp_group=tp_group,
        world_size=world_size,
        sign_vector=sign_vector,
    )
    n = out_2d.shape[-1]
    l_local = out_2d.shape[0] // B
    return out_2d.reshape(l_local, B, n).transpose(0, 1)


def _swap_tp_placement(dtensor: DTensor, tp_placement) -> tuple:
    """Return (mesh, placements) of ``dtensor`` with the TP axis swapped.

    Activations are replicated over data axes (DP/FSDP) and only the TP axis
    carries the linear's sharding. Reusing the input's own mesh keeps the output
    DTensor on the same mesh (1-D under tp-only, 2-D under tp+fsdp) so it composes
    with the surrounding activations.
    """
    mesh = dtensor.device_mesh
    names = mesh.mesh_dim_names
    assert names is not None and "tp" in names, (
        f"NVFP4 TP path requires a 'tp' mesh axis, got {names}"
    )
    placements = list(dtensor.placements)
    placements[names.index("tp")] = tp_placement
    return mesh, tuple(placements)


class NVFP4Linear(Linear):
    """NVFP4 Linear satisfying TorchTitan's Module protocol.

    Inherits TorchTitan's ``Linear`` (a flat ``nn.Linear`` + ``Module`` leaf), so
    weight/bias are sharded by ``_distribute_states`` from the inherited
    colwise/rowwise ``sharding_config``. NVFP4 runtime buffers start as ``None``
    (skipped by ``_distribute_states``) and are materialized in
    ``_init_self_buffers``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        # colwise -> w1/w3, qkv; rowwise -> w2, wo. Inferred from the weight
        # placement at parallelize() time when None.
        tensor_parallel_style: Literal["colwise", "rowwise"] | None = None

    def __init__(self, config: "NVFP4Linear.Config"):
        super().__init__(config)
        self.kernel_preference = KernelPreference.TRITON
        self.tensor_parallel_style = config.tensor_parallel_style
        self.tp_group: dist.ProcessGroup | None = None
        self._tp_mesh = None
        self.world_size = 1
        # Persistent for native NVFP4 TorchTitan checkpoint/resume. Start as None
        # so _distribute_states() skips distribution; materialized later in
        # _init_self_buffers().
        self.register_buffer("_sr_seed", None, persistent=True)
        self.register_buffer("_rht_sign_vector", None, persistent=True)
        self._rht_sign_vector_tuple: tuple[int, ...] | None = None

    def _refresh_rht_sign_vector_tuple(self) -> None:
        self._rht_sign_vector_tuple = _rht_sign_vector_to_tuple(self._rht_sign_vector)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._refresh_rht_sign_vector_tuple()

    @property
    def rht_sign_vector(self) -> tuple[int, ...]:
        if self._rht_sign_vector_tuple is None:
            self._refresh_rht_sign_vector_tuple()
        if self._rht_sign_vector_tuple is None:
            raise RuntimeError("rht_sign_vector is not materialized")
        return self._rht_sign_vector_tuple

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        # Cache the TP group before super().parallelize() distributes states,
        # mirroring Embedding.parallelize(). The NVFP4 TP path owns the amax
        # all-reduce and fp4 collectives over this group.
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self._tp_mesh = tp_mesh
            self.tp_group = tp_mesh.get_group("tp")
            self.world_size = tp_mesh.size()
            if self.tensor_parallel_style is None:
                self.tensor_parallel_style = _infer_tp_style(self._sharding_config)
        super().parallelize(parallel_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tp_group is None:
            # Single-GPU or FSDP-only: no TP collectives.
            return nvfp4_linear(
                input,
                self.weight,
                self.bias,
                kernel_preference=self.kernel_preference,
                sr_seed=self._sr_seed,
                sign_vector=self.rht_sign_vector,
            )

        w_local = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        bias_local = (
            self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
        )
        # The TorchAO Functions return input gradients already in the input's
        # placement (colwise reduce-scatters dx to the seq shard; rowwise dx is a
        # full-seq feature shard), so the default to_local() grad placement is
        # correct -- no explicit grad_placements needed.
        x_local = input.to_local() if isinstance(input, DTensor) else input

        if self.tensor_parallel_style == "rowwise":
            out = _nvfp4_rowwise_sp(
                x_local, w_local, bias_local, self._sr_seed,
                self.rht_sign_vector, self.tp_group, self.world_size,
            )
            out_tp_placement = Shard(1)  # SP: sequence shard
        else:
            out = _nvfp4_colwise_sp(
                x_local, w_local, bias_local, self._sr_seed,
                self.rht_sign_vector, self.tp_group, self.world_size,
            )
            out_tp_placement = Shard(-1)  # feature shard

        mesh, placements = _swap_tp_placement(input, out_tp_placement)
        return DTensor.from_local(out, mesh, placements, run_check=False)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # Materialize NVFP4 runtime buffers after parallelize() + to_empty().
        dev = buffer_device or self.weight.device
        self._sr_seed = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
        )
        self._rht_sign_vector = _make_rht_sign_vector(None, device=dev)
        # Under TP every rank must agree on the RHT basis. group_src=0 (per-group
        # local rank 0, not global src=0) so TP groups whose ranks exclude global
        # rank 0 (e.g. tp2_fsdp2) still broadcast correctly. _sr_seed stays
        # per-rank -- torchao treats SR seeds as independent across ranks.
        if self.tp_group is not None:
            dist.broadcast(self._rht_sign_vector, group=self.tp_group, group_src=0)
        self._refresh_rht_sign_vector_tuple()
