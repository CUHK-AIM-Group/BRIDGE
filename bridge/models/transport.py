from __future__ import annotations

import torch
from torch import nn

from .blocks import VolumeEncoder3D


class ConditionalVectorField(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim + hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([z_t, z_b, self.time_embed(t.to(z_t))], dim=1))


class BrainGutLatentTransport(nn.Module):
    """Stage 1: conditional flow matching from brain to gut latent space."""

    def __init__(self, latent_dim: int = 256, base_channels: int = 12,
                 field_hidden_dim: int = 512, integration_steps: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.integration_steps = integration_steps
        self.brain_encoder = VolumeEncoder3D(latent_dim, base_channels)
        self.gut_encoder = VolumeEncoder3D(latent_dim, base_channels)
        self.vector_field = ConditionalVectorField(latent_dim, field_hidden_dim)
        self.endpoint_norm = nn.LayerNorm(latent_dim)

    def integrate(self, z_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, velocities = z_b, []
        dt = 1.0 / self.integration_steps
        for step in range(self.integration_steps):
            t = z.new_full((z.shape[0], 1), (step + 0.5) * dt)
            velocity = self.vector_field(z, t, z_b)
            z = z + dt * velocity
            velocities.append(velocity)
        return self.endpoint_norm(z), torch.stack(velocities, dim=1)

    def encode_brain(self, brain: torch.Tensor) -> dict[str, torch.Tensor]:
        z_b = self.brain_encoder(brain)
        z_g_hat, velocities = self.integrate(z_b)
        return {"z_b": z_b, "z_g_hat": z_g_hat, "trajectory_velocities": velocities}

    def forward(self, brain: torch.Tensor, gut: torch.Tensor) -> dict[str, torch.Tensor]:
        z_b, z_g = self.brain_encoder(brain), self.gut_encoder(gut)
        t = torch.rand(z_b.shape[0], 1, device=z_b.device, dtype=z_b.dtype)
        z_t = (1.0 - t) * z_b + t * z_g
        z_g_hat, velocities = self.integrate(z_b)
        return {
            "z_b": z_b, "z_g": z_g, "z_g_hat": z_g_hat,
            "target_velocity": z_g - z_b,
            "predicted_velocity": self.vector_field(z_t, t, z_b),
            "trajectory_velocities": velocities,
        }

