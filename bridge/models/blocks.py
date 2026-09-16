from __future__ import annotations

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.SiLU(inplace=True),
        )


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class VolumeEncoder3D(nn.Module):
    def __init__(self, latent_dim: int = 256, base_channels: int = 12) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv3d(1, c, 5, stride=2, padding=2, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.SiLU(inplace=True),
            ResidualBlock3D(c),
            ConvNormAct(c, 2 * c, stride=2),
            ResidualBlock3D(2 * c),
            ConvNormAct(2 * c, 4 * c, stride=2),
            ResidualBlock3D(4 * c),
            ConvNormAct(4 * c, 8 * c, stride=2),
            ResidualBlock3D(8 * c),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.project = nn.Sequential(nn.Linear(8 * c, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.project(self.pool(self.features(image)).flatten(1))


class DiagnosticEncoder3D(nn.Module):
    def __init__(self, token_dim: int = 256, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv3d(1, c, 5, stride=2, padding=2, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.SiLU(inplace=True),
            ResidualBlock3D(c),
            ConvNormAct(c, 2 * c, stride=2),
            ResidualBlock3D(2 * c),
            ConvNormAct(2 * c, 4 * c, stride=2),
            ResidualBlock3D(4 * c),
            ConvNormAct(4 * c, 8 * c, stride=2),
            ResidualBlock3D(8 * c),
        )
        self.token_project = nn.Linear(8 * c, token_dim)
        self.global_project = nn.Sequential(nn.Linear(8 * c, token_dim), nn.LayerNorm(token_dim))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(image)
        tokens = self.token_project(feature_map.flatten(2).transpose(1, 2))
        global_feature = self.global_project(feature_map.mean(dim=(2, 3, 4)))
        return tokens, global_feature

