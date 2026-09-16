from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_contrastive(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def transport_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    flow = F.mse_loss(output["predicted_velocity"], output["target_velocity"])
    endpoint = F.mse_loss(output["z_g_hat"], output["z_g"])
    cosine = 1.0 - F.cosine_similarity(output["z_g_hat"], output["z_g"], dim=1).mean()
    contrastive = symmetric_contrastive(output["z_g_hat"], output["z_g"])
    energy = output["trajectory_velocities"].pow(2).mean()
    return flow + endpoint + cosine + 0.25 * contrastive + 0.005 * energy


def same_label_mismatch(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(labels.shape[0], device=labels.device)
    mismatch, valid = indices.clone(), torch.zeros_like(labels, dtype=torch.bool)
    for label in labels.unique():
        members = indices[labels == label]
        if members.numel() > 1:
            mismatch[members] = torch.roll(members, 1)
            valid[members] = True
    return mismatch, valid


def fusion_loss(output: dict[str, torch.Tensor], labels: torch.Tensor,
                nonmatched_logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    classification = F.cross_entropy(output["final_logits"], labels)
    if not valid.any():
        return classification + 0.005 * output["delta_logits"].pow(2).mean()
    gather = labels[:, None]
    brain = F.log_softmax(output["brain_logits"], 1).gather(1, gather).squeeze(1)
    matched = F.log_softmax(output["brain_logits"] + output["delta_logits"], 1).gather(1, gather).squeeze(1)
    nonmatched = F.log_softmax(nonmatched_logits, 1).gather(1, gather).squeeze(1)
    contrast = F.relu(0.2 - ((matched - brain) - (nonmatched - brain)))[valid].mean()
    return classification + 0.2 * contrast + 0.005 * output["delta_logits"].pow(2).mean()

