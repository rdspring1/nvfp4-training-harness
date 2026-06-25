"""Registers torchao NVFP4 overrides for TorchTitan's registry.

The NVFP4 communication optimization all-gathers fp4 codes/scales (not bf16
activations) in the column-parallel forward. That only works if activations stay
sequence-sharded at the attention/FFN block boundary -- but stock TorchTitan
sharding gathers them to TP Replicate in bf16 first (parent ``in_dst -> tp=R``),
destroying the opportunity.

So we override the PARENT FeedForward / GQAttention configs (not the child
Linear configs): each keeps the incoming sequence-parallel input (``in_dst =
in_src``) and replaces its own child linears with NVFP4Linear. This must be
parent-only -- a parent override plus a child Linear override would be an
ancestor/descendant pair, which TorchTitan's override conflict check rejects.
"""

import os

import spmd_types as spmd
import torch
from torch.distributed.tensor import Shard

from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_layout_to_dtensor_placements
from torchtitan.models.common.attention import (
    FusedQKVLinear,
    GQAttention,
    QKVLinear,
)
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.sharding import ShardingConfig

from .nvfp4 import TP, NVFP4Linear

_NVFP4_RECOMPILE_LIMIT = 64

# NVFP4 passes per-module Python sign-vector tuples into the compiled block.
# Under fullgraph=True those constants plus async DTensor collectives can exceed
# Dynamo's default limit before step 1 on Llama3 8B TP runs.
if torch._dynamo.config.recompile_limit < _NVFP4_RECOMPILE_LIMIT:
    torch._dynamo.config.recompile_limit = _NVFP4_RECOMPILE_LIMIT


def _tp_degree() -> int:
    return int(os.environ.get("TORCHTITAN_AO_TP_DEGREE", "1"))


def _linear_ok(cfg: Linear.Config, local_block: int) -> bool:
    # torchao's NVFP4 kernels require both local weight dims divisible by 128.
    # Under TP=N a dim becomes dim // N locally, so require 128 * N on both dims
    # to cover colwise (out sharded) and rowwise (in sharded).
    return cfg.in_features % local_block == 0 and cfg.out_features % local_block == 0


def _is_seq_parallel(layout) -> bool:
    """True if the activation layout shards the sequence (dim 1) on the TP axis.

    Activations are [B, L, D]; sequence is dim 1. The feature shard (colwise
    output) is Shard(-1), so a bare Shard check is not enough -- match dim 1.
    """
    tp_placement = spmd_layout_to_dtensor_placements(layout).get(TP)
    return isinstance(tp_placement, Shard) and tp_placement.dim == 1


def _require_sp(in_src, arg_name: str, block: str) -> None:
    """The NVFP4 TP path needs sequence-sharded input to fp4-all-gather it.

    Under TP without sequence parallelism the parent feeds a TP-replicated
    activation; the column-parallel path would then all-gather a full tensor and
    corrupt numerics. Fail loudly rather than silently miscompute.
    """
    if in_src is None or not _is_seq_parallel(in_src):
        raise ValueError(
            f"NVFP4 {block} override requires sequence parallelism (the '{arg_name}' "
            "input must be sequence-sharded on the TP axis). Set "
            "parallelism.enable_sequence_parallel=True for NVFP4 + TP."
        )


def _with_nvfp4_buffers(state: dict) -> dict:
    # _distribute_states() requires a placement entry for every direct buffer
    # name; the buffers are None at distribution time (skipped) and materialized
    # as replicated local buffers in NVFP4Linear._init_self_buffers().
    s = dict(state)
    s["_sr_seed"] = dense_param_placement(tp=spmd.R)
    s["_rht_sign_vector"] = dense_param_placement(tp=spmd.R)
    return s


def _to_nvfp4_colwise(cfg: Linear.Config) -> NVFP4Linear.Config:
    base = cfg.sharding_config
    sc = ShardingConfig(
        state_shardings=_with_nvfp4_buffers(base.state_shardings),
        in_src_shardings=base.in_src_shardings,
        in_dst_shardings=base.in_dst_shardings,
        out_src_shardings=base.out_src_shardings,
        out_dst_shardings=base.out_dst_shardings,
        local_map=base.local_map,
    )
    return derive(
        cfg, NVFP4Linear.Config, tensor_parallel_style="colwise", sharding_config=sc
    )


def _to_nvfp4_rowwise(cfg: Linear.Config) -> NVFP4Linear.Config:
    base = cfg.sharding_config
    # TorchAO's row-parallel Function reduce-scatters the bf16 output internally
    # and returns the SP sequence shard, so the output is already at out_dst.
    # Declare out_src = post-reduce-scatter SP placement and out_dst = None so
    # TorchTitan does NOT reduce again (no double-reduce).
    sc = ShardingConfig(
        state_shardings=_with_nvfp4_buffers(base.state_shardings),
        in_src_shardings=base.in_src_shardings,
        in_dst_shardings=base.in_dst_shardings,
        out_src_shardings=dense_sequence_parallel_placement(),
        out_dst_shardings=None,
        local_map=base.local_map,
    )
    return derive(
        cfg, NVFP4Linear.Config, tensor_parallel_style="rowwise", sharding_config=sc
    )


def _keep_sp_input(base: ShardingConfig | None) -> ShardingConfig | None:
    """Rewrite a parent block config to keep its SP input (no bf16 gather)."""
    if base is None:
        return None
    return ShardingConfig(
        state_shardings=base.state_shardings,
        in_src_shardings=base.in_src_shardings,
        in_dst_shardings=base.in_src_shardings,  # keep SP; was tp=R (bf16 gather)
        out_src_shardings=base.out_src_shardings,
        out_dst_shardings=base.out_dst_shardings,
        local_map=base.local_map,
    )


@override(
    "nvfp4_feed_forward",
    target=FeedForward.Config,
    description="NVFP4 sequence-parallel FFN block (fp4 all-gather, no bf16 gather).",
)
def nvfp4_feed_forward(cfg: FeedForward.Config) -> FeedForward.Config:
    tp = _tp_degree()
    local_block = 128 * max(tp, 1)
    # All-or-nothing: keeping SP requires every colwise child to be an NVFP4
    # fp4-gatherer; if any linear can't be converted, leave the stock bf16 gather.
    if not all(_linear_ok(c, local_block) for c in (cfg.w1, cfg.w2, cfg.w3)):
        return cfg
    if tp > 1:
        in_src = cfg.sharding_config.in_src_shardings if cfg.sharding_config else None
        _require_sp(in_src and in_src.get("x"), "x", "FFN")
    return derive(
        cfg,
        FeedForward.Config,
        w1=_to_nvfp4_colwise(cfg.w1),
        w3=_to_nvfp4_colwise(cfg.w3),
        w2=_to_nvfp4_rowwise(cfg.w2),
        sharding_config=_keep_sp_input(cfg.sharding_config),
    )


def _qkv_linears(qkv) -> list[Linear.Config] | None:
    if isinstance(qkv, QKVLinear.Config):
        return [qkv.wq, qkv.wkv]
    if isinstance(qkv, FusedQKVLinear.Config):
        return [qkv.wqkv]
    return None


@override(
    "nvfp4_attention",
    target=GQAttention.Config,
    description="NVFP4 sequence-parallel attention block (fp4 all-gather).",
)
def nvfp4_attention(cfg: GQAttention.Config) -> GQAttention.Config:
    tp = _tp_degree()
    local_block = 128 * max(tp, 1)
    qkv_linears = _qkv_linears(cfg.qkv_linear)
    if qkv_linears is None:
        return cfg
    if not all(_linear_ok(c, local_block) for c in (*qkv_linears, cfg.wo)):
        return cfg
    if tp > 1:
        in_src = cfg.sharding_config.in_src_shardings if cfg.sharding_config else None
        _require_sp(in_src and in_src.get("x_BLD"), "x_BLD", "attention")

    qkv = cfg.qkv_linear
    if isinstance(qkv, QKVLinear.Config):
        new_qkv = derive(
            qkv,
            QKVLinear.Config,
            wq=_to_nvfp4_colwise(qkv.wq),
            wkv=_to_nvfp4_colwise(qkv.wkv),
        )
    else:
        new_qkv = derive(
            qkv, FusedQKVLinear.Config, wqkv=_to_nvfp4_colwise(qkv.wqkv)
        )

    return derive(
        cfg,
        GQAttention.Config,
        qkv_linear=new_qkv,
        wo=_to_nvfp4_rowwise(cfg.wo),
        sharding_config=_keep_sp_input(cfg.sharding_config),
    )
