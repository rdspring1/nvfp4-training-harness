"""NVFP4 grouped experts for TorchTitan DeepSeek V3."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
    _to_nvfp4_then_scaled_grouped_mm,
)
from torchtitan.models.common.moe import GroupedExperts

RHT_SIGN_VECTOR = tuple(1 if i % 2 == 0 else -1 for i in range(16))


class NVFP4GroupedExperts(GroupedExperts):
    """GroupedExperts implementation backed by torchao NVFP4 GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pad_token_groups_for_grouped_mm: bool = True

    def __init__(self, config: Config):
        super().__init__(config)
        self._pad_token_groups = config.pad_token_groups_for_grouped_mm
        self.sr_seed = None

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        self.sr_seed = torch.tensor(
            [1234],
            dtype=torch.int64,
            device=buffer_device or self.w1_EFD.device,
        )

    def _experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        # NVFP4 grouped MM requires non-empty groups, but routing may leave
        # experts idle. This single-GPU debugmodel override filters them here;
        # the broader MXFP8 integration instead installs a padded EP dispatcher.
        # For an EP>1 override, replace both GroupedExperts.Config components:
        # implementation -> NVFP4GroupedExperts.Config and token_dispatcher ->
        # TorchAOTokenDispatcher.Config, which pads empty experts.
        active_experts_E = num_tokens_per_expert_E > 0
        num_tokens_per_expert_E = num_tokens_per_expert_E[active_experts_E]
        w1_EFD = w1_EFD[active_experts_E]
        w2_EDF = w2_EDF[active_experts_E]
        w3_EFD = w3_EFD[active_experts_E]
        offsets_E = torch.cumsum(
            num_tokens_per_expert_E, dim=0, dtype=torch.int32
        )
        gate_RF = _to_nvfp4_then_scaled_grouped_mm(
            x_RD.bfloat16(),
            w1_EFD.bfloat16(),
            RHT_SIGN_VECTOR,
            self.sr_seed,
            offs=offsets_E,
            pad_token_groups_for_grouped_mm=self._pad_token_groups,
        )
        up_RF = _to_nvfp4_then_scaled_grouped_mm(
            x_RD.bfloat16(),
            w3_EFD.bfloat16(),
            RHT_SIGN_VECTOR,
            self.sr_seed,
            offs=offsets_E,
            pad_token_groups_for_grouped_mm=self._pad_token_groups,
        )
        h_RF = F.silu(gate_RF) * up_RF
        return _to_nvfp4_then_scaled_grouped_mm(
            h_RF,
            w2_EDF.bfloat16(),
            RHT_SIGN_VECTOR,
            self.sr_seed,
            offs=offsets_E,
            pad_token_groups_for_grouped_mm=self._pad_token_groups,
        ).type_as(x_RD)
