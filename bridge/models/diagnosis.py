from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import DiagnosticEncoder3D
from .transport import BrainGutLatentTransport


class EvidenceAnchors(nn.Module):
    def __init__(self, embeddings: torch.Tensor, num_heads: int = 4) -> None:
        super().__init__()
        self.register_buffer("base", embeddings.detach().clone())
        self.delta = nn.Parameter(torch.zeros_like(embeddings))
        self.anchor_to_image = nn.MultiheadAttention(embeddings.shape[1], num_heads, batch_first=True)
        self.image_to_anchor = nn.MultiheadAttention(embeddings.shape[1], num_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(embeddings.shape[1])
        self.norm_x = nn.LayerNorm(embeddings.shape[1])

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        anchors = (self.base + self.delta).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        evidence, _ = self.anchor_to_image(anchors, tokens, tokens)
        conditioned = self.norm_a(anchors + evidence)
        modulation, _ = self.image_to_anchor(tokens, conditioned, conditioned)
        return {"anchors": self.base + self.delta, "conditioned": conditioned,
                "tokens": self.norm_x(tokens + modulation)}


class BrainEvidenceClassifier(nn.Module):
    def __init__(self, anchor_embeddings: torch.Tensor, num_classes: int = 3,
                 token_dim: int = 256, base_channels: int = 16) -> None:
        super().__init__()
        if anchor_embeddings.shape[1] != token_dim:
            raise ValueError("anchor dimension must equal token_dim")
        self.encoder = DiagnosticEncoder3D(token_dim, base_channels)
        self.anchors = EvidenceAnchors(anchor_embeddings)
        self.fusion = nn.Sequential(nn.Linear(3 * token_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(self, brain: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, global_feature = self.encoder(brain)
        evidence = self.anchors(tokens)
        feature = self.fusion(torch.cat([
            global_feature, evidence["tokens"].mean(1), evidence["conditioned"].mean(1)
        ], dim=1))
        return {"brain_logits": self.classifier(feature), "brain_feature": feature, **evidence}


class BridgeDiagnosisModel(nn.Module):
    """Stage 2: evidence-guided, reliability-gated virtual-gut fusion."""

    def __init__(self, brain_model: BrainEvidenceClassifier, transport: BrainGutLatentTransport,
                 num_classes: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.brain_model, self.transport = brain_model, transport
        self.residual_head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, num_classes))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.gate = nn.Sequential(
            nn.Linear(2 * feature_dim + num_classes + 1, feature_dim), nn.LayerNorm(feature_dim),
            nn.SiLU(), nn.Linear(feature_dim, 1), nn.Sigmoid(),
        )

    def freeze_priors(self) -> None:
        for module in (self.brain_model, self.transport):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward(self, brain: torch.Tensor) -> dict[str, torch.Tensor]:
        cerebral = self.brain_model(brain)
        transported = self.transport.encode_brain(brain)
        delta = self.residual_head(transported["z_g_hat"])
        p = F.softmax(cerebral["brain_logits"], dim=1)
        entropy = -(p * p.clamp_min(1e-8).log()).sum(1, keepdim=True)
        alpha = self.gate(torch.cat([
            cerebral["brain_feature"], transported["z_g_hat"], delta, entropy
        ], dim=1))
        return {**cerebral, **transported, "delta_logits": delta, "alpha": alpha,
                "final_logits": cerebral["brain_logits"] + alpha * delta}

    def nonmatched_logits(self, output: dict[str, torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
        return output["brain_logits"] + self.residual_head(output["z_g_hat"][indices])

