import torch

from bridge.losses import fusion_loss, same_label_mismatch, transport_loss
from bridge.models import BrainEvidenceClassifier, BrainGutLatentTransport, BridgeDiagnosisModel


def test_end_to_end_forward_and_losses() -> None:
    torch.manual_seed(7)
    brain, gut = torch.randn(3, 1, 32, 32, 32), torch.randn(3, 1, 32, 32, 32)
    transport = BrainGutLatentTransport(latent_dim=16, base_channels=2, field_hidden_dim=32, integration_steps=2)
    transported = transport(brain, gut)
    assert torch.isfinite(transport_loss(transported))
    cerebral = BrainEvidenceClassifier(torch.randn(6, 16), num_classes=3, token_dim=16, base_channels=2)
    model = BridgeDiagnosisModel(cerebral, transport, num_classes=3, feature_dim=16)
    output = model(brain)
    labels = torch.tensor([0, 0, 1])
    mismatch, valid = same_label_mismatch(labels)
    loss = fusion_loss(output, labels, model.nonmatched_logits(output, mismatch), valid)
    assert output["final_logits"].shape == (3, 3)
    assert torch.isfinite(loss)

