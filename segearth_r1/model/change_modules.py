import torch
import torch.nn as nn


SUPPORTED_TEMPORAL_FUSION_TYPES = (
    "change_aware",
    "concat_conv",
    "abs_diff",
    "avg",
)


def _group_count(num_channels: int) -> int:
    for num_groups in (32, 16, 8, 4, 2):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class BiTemporalFusionBlock(nn.Module):
    """
    Explicit change-aware fusion for one Swin stage.

    The block keeps both absolute and directional temporal differences, then
    uses a lightweight gate to inject the most discriminative change cues back
    into the fused representation.
    """

    def __init__(self, channels: int):
        super().__init__()
        bottleneck = max(channels // 4, 32)

        self.t1_proj = ConvNormAct(channels, channels, kernel_size=1, padding=0)
        self.t2_proj = ConvNormAct(channels, channels, kernel_size=1, padding=0)

        self.diff_encoder = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=3),
            ConvNormAct(channels, channels, kernel_size=3),
        )

        self.fuse = nn.Sequential(
            ConvNormAct(channels * 4, channels, kernel_size=1, padding=0),
            ConvNormAct(channels, channels, kernel_size=3),
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, bottleneck, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(bottleneck, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.out_norm = nn.GroupNorm(_group_count(channels), channels)
        self.out_act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor) -> torch.Tensor:
        feat_t1 = self.t1_proj(feat_t1)
        feat_t2 = self.t2_proj(feat_t2)

        signed_diff = feat_t2 - feat_t1
        abs_diff = signed_diff.abs()
        diff_feat = self.diff_encoder(torch.cat([signed_diff, abs_diff], dim=1))

        fused = self.fuse(torch.cat([feat_t1, feat_t2, abs_diff, diff_feat], dim=1))
        gate = self.gate(torch.cat([feat_t1, feat_t2], dim=1))
        shortcut = 0.5 * (feat_t1 + feat_t2)

        return self.out_act(self.out_norm(fused + gate * diff_feat + shortcut))


class ConcatConvFusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1, padding=0),
            ConvNormAct(channels, channels, kernel_size=3),
        )
        self.out_norm = nn.GroupNorm(_group_count(channels), channels)
        self.out_act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor) -> torch.Tensor:
        shortcut = 0.5 * (feat_t1 + feat_t2)
        fused = self.fuse(torch.cat([feat_t1, feat_t2], dim=1))
        return self.out_act(self.out_norm(fused + shortcut))


class AbsDiffFusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.diff_encoder = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=3),
            ConvNormAct(channels, channels, kernel_size=3),
        )
        self.merge = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1, padding=0),
            ConvNormAct(channels, channels, kernel_size=3),
        )
        self.out_norm = nn.GroupNorm(_group_count(channels), channels)
        self.out_act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor) -> torch.Tensor:
        shortcut = 0.5 * (feat_t1 + feat_t2)
        diff_feat = self.diff_encoder((feat_t2 - feat_t1).abs())
        fused = self.merge(torch.cat([shortcut, diff_feat], dim=1))
        return self.out_act(self.out_norm(fused))


class AverageFusionBlock(nn.Module):
    def reset_parameters(self):
        return None

    def forward(self, feat_t1: torch.Tensor, feat_t2: torch.Tensor) -> torch.Tensor:
        return 0.5 * (feat_t1 + feat_t2)


def build_temporal_fusion_block(fusion_type: str, channels: int) -> nn.Module:
    if fusion_type == "change_aware":
        return BiTemporalFusionBlock(channels)
    if fusion_type == "concat_conv":
        return ConcatConvFusionBlock(channels)
    if fusion_type == "abs_diff":
        return AbsDiffFusionBlock(channels)
    if fusion_type == "avg":
        return AverageFusionBlock()
    raise ValueError(
        f"Unsupported temporal_fusion_type={fusion_type}. "
        f"Supported: {', '.join(SUPPORTED_TEMPORAL_FUSION_TYPES)}"
    )


class BiTemporalFeatureFusion(nn.Module):
    def __init__(self, channels_per_level, fusion_type: str = "change_aware"):
        super().__init__()
        self.fusion_type = fusion_type
        self.blocks = nn.ModuleDict(
            {
                level_name: build_temporal_fusion_block(fusion_type, level_channels)
                for level_name, level_channels in channels_per_level.items()
            }
        )

    def reset_parameters(self):
        for block in self.blocks.values():
            block.reset_parameters()

    def forward(self, feat_t1_dict, feat_t2_dict):
        if feat_t2_dict is None:
            return feat_t1_dict

        return {
            level_name: self.blocks[level_name](feat_t1_dict[level_name], feat_t2_dict[level_name])
            for level_name in feat_t1_dict
        }
