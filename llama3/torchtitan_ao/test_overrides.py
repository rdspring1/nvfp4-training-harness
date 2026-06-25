"""Config-level tests for the NVFP4 sequence-parallel block overrides.

No GPU required: these exercise the override factories and ShardingConfig
rewrites. The end-to-end TP numerics (fp4 collectives) are covered by the GPU
smoke (`run_titan.py multi --smoke --nvfp4`).
"""

import os

import pytest
import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_sequence_parallel_placement,
    rowwise_config,
    set_dense_ffn_sharding,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.nn_modules import Linear

from torchtitan_ao.nvfp4 import NVFP4Linear, _infer_tp_style
from torchtitan_ao.overrides import (
    _is_seq_parallel,
    _to_nvfp4_rowwise,
    nvfp4_feed_forward,
)

DIM, HIDDEN = 512, 1024  # divisible by 128 * tp for tp <= 4


def _ffn_config() -> FeedForward.Config:
    cfg = FeedForward.Config(
        w1=Linear.Config(in_features=DIM, out_features=HIDDEN),
        w2=Linear.Config(in_features=HIDDEN, out_features=DIM),
        w3=Linear.Config(in_features=DIM, out_features=HIDDEN),
    )
    set_dense_ffn_sharding(
        cfg, attn_x_layout=dense_sequence_parallel_placement(), enable_sp=True
    )
    return cfg


@pytest.fixture
def tp4(monkeypatch):
    monkeypatch.setenv("TORCHTITAN_AO_TP_DEGREE", "4")


def test_infer_tp_style():
    assert _infer_tp_style(colwise_config()) == "colwise"
    assert _infer_tp_style(rowwise_config(output_sp=True)) == "rowwise"
    assert _infer_tp_style(None) is None


def test_ffn_keeps_sequence_parallel_input(tp4):
    """Parent FFN must NOT bf16-gather x to Replicate before NVFP4 colwise."""
    cfg = _ffn_config()
    # Stock sharding gathers to Replicate (the bf16 all-gather we want to avoid).
    assert not _is_seq_parallel(cfg.sharding_config.in_dst_shardings["x"])

    out = nvfp4_feed_forward(cfg)
    # After override, in_dst keeps the SP seq-shard (no bf16 gather).
    assert _is_seq_parallel(out.sharding_config.in_dst_shardings["x"])
    assert out.sharding_config.in_dst_shardings["x"] is out.sharding_config.in_src_shardings["x"]


def test_ffn_children_converted(tp4):
    out = nvfp4_feed_forward(_ffn_config())
    assert isinstance(out.w1, NVFP4Linear.Config)
    assert isinstance(out.w2, NVFP4Linear.Config)
    assert isinstance(out.w3, NVFP4Linear.Config)
    assert out.w1.tensor_parallel_style == "colwise"
    assert out.w3.tensor_parallel_style == "colwise"
    assert out.w2.tensor_parallel_style == "rowwise"


def test_buffer_state_shardings_declared(tp4):
    """_distribute_states requires a placement entry for every direct buffer."""
    out = nvfp4_feed_forward(_ffn_config())
    for child in (out.w1, out.w2, out.w3):
        state = child.sharding_config.state_shardings
        assert "_sr_seed" in state
        assert "_rht_sign_vector" in state
        assert "weight" in state  # inherited


def test_rowwise_output_contract():
    """Rowwise reduce-scatters internally: out_src=SP, out_dst=None (no double-reduce)."""
    w2 = Linear.Config(in_features=HIDDEN, out_features=DIM)
    w2.sharding_config = rowwise_config(output_sp=True)
    nvfp4_w2 = _to_nvfp4_rowwise(w2)
    sc = nvfp4_w2.sharding_config
    assert sc.out_dst_shardings is None
    assert _is_seq_parallel(sc.out_src_shardings)


def test_colwise_output_contract_preserved(tp4):
    """Colwise keeps stock out_src (feature shard), out_dst stays None."""
    out = nvfp4_feed_forward(_ffn_config())
    sc = out.w1.sharding_config
    # colwise out_src is Shard(-1) (feature), NOT sequence-parallel.
    assert not _is_seq_parallel(sc.out_src_shardings)
    assert sc.out_dst_shardings is None


def test_tp_without_sp_raises(monkeypatch):
    """NVFP4 TP path requires sequence parallelism; TP-only must fail loudly."""
    monkeypatch.setenv("TORCHTITAN_AO_TP_DEGREE", "4")
    cfg = FeedForward.Config(
        w1=Linear.Config(in_features=DIM, out_features=HIDDEN),
        w2=Linear.Config(in_features=HIDDEN, out_features=DIM),
        w3=Linear.Config(in_features=DIM, out_features=HIDDEN),
    )
    # enable_sp=False feeds a TP-replicated activation (the real non-SP layout).
    set_dense_ffn_sharding(
        cfg, attn_x_layout=dense_activation_placement(tp=spmd.I), enable_sp=False
    )
    with pytest.raises(ValueError, match="sequence parallelism"):
        nvfp4_feed_forward(cfg)


def test_indivisible_dims_not_converted(monkeypatch):
    """Dims not clearing 128*tp leave the stock block (and its bf16 gather)."""
    monkeypatch.setenv("TORCHTITAN_AO_TP_DEGREE", "4")
    cfg = FeedForward.Config(
        w1=Linear.Config(in_features=DIM, out_features=300),  # 300 % 512 != 0
        w2=Linear.Config(in_features=300, out_features=DIM),
        w3=Linear.Config(in_features=DIM, out_features=300),
    )
    set_dense_ffn_sharding(
        cfg, attn_x_layout=dense_sequence_parallel_placement(), enable_sp=True
    )
    out = nvfp4_feed_forward(cfg)
    assert out is cfg  # unchanged
    assert not isinstance(out.w1, NVFP4Linear.Config)
