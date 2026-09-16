from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from bridge.data import LABELS, PETDataset, fold_split, read_manifest
from bridge.losses import fusion_loss, same_label_mismatch
from bridge.models import BrainEvidenceClassifier, BrainGutLatentTransport, BridgeDiagnosisModel
from bridge.utils import device_from_name, load_checkpoint, save_checkpoint, seed_all, write_json


def classification_epoch(model, loader, device, optimizer=None) -> float:
    model.train(optimizer is not None)
    total = 0.0
    for batch in loader:
        image, label = batch["brain"].to(device), batch["label"].to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            loss = F.cross_entropy(model(image)["brain_logits"], label)
            if optimizer:
                loss.backward()
                optimizer.step()
        total += float(loss.detach()) * image.shape[0]
    return total / len(loader.dataset)


def fusion_epoch(model, loader, device, optimizer=None) -> float:
    model.train(optimizer is not None)
    model.brain_model.eval()
    model.transport.eval()
    total = 0.0
    for batch in loader:
        image, label = batch["brain"].to(device), batch["label"].to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(image)
            mismatch, valid = same_label_mismatch(label)
            loss = fusion_loss(output, label, model.nonmatched_logits(output, mismatch), valid)
            if optimizer:
                loss.backward()
                optimizer.step()
        total += float(loss.detach()) * image.shape[0]
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    truth, prediction = [], []
    for batch in loader:
        logits = model(batch["brain"].to(device))["final_logits"]
        truth.extend(batch["label"].tolist())
        prediction.extend(logits.argmax(1).cpu().tolist())
    accuracy = sum(a == b for a, b in zip(truth, prediction)) / len(truth)
    return {"accuracy": accuracy, "subjects": len(truth)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate BRIDGE Stage 2 for CN/MCI/AD diagnosis.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/bridge.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/diagnosis"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_all(config["seed"] + args.fold)
    device = device_from_name(config["device"])
    train, validation, test = fold_split(read_manifest(args.manifest), args.fold, config["folds"])
    make_loader = lambda records, shuffle=False: DataLoader(
        PETDataset(records, tuple(config["shape"])), config["batch_size"], shuffle=shuffle,
        num_workers=config["workers"],
    )
    train_loader, validation_loader, test_loader = (
        make_loader(train, True), make_loader(validation), make_loader(test)
    )
    anchors = load_checkpoint(args.anchors, device)["embeddings"].float().to(device)
    cerebral = BrainEvidenceClassifier(anchors, len(LABELS), anchors.shape[1], config["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(cerebral.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    best, epochs = float("inf"), args.epochs or config["diagnostic_epochs"]
    cerebral_path = args.output / f"fold_{args.fold}" / "cerebral.pt"
    for epoch in range(1, epochs + 1):
        classification_epoch(cerebral, train_loader, device, optimizer)
        loss = classification_epoch(cerebral, validation_loader, device)
        if loss < best:
            best = loss
            save_checkpoint(cerebral_path, model=cerebral.state_dict())
    cerebral.load_state_dict(load_checkpoint(cerebral_path, device)["model"])

    transport_payload = load_checkpoint(args.transport, device)
    transport = BrainGutLatentTransport(**transport_payload["model_config"]).to(device)
    transport.load_state_dict(transport_payload["model"])
    model = BridgeDiagnosisModel(cerebral, transport, len(LABELS), anchors.shape[1]).to(device)
    model.freeze_priors()
    optimizer = torch.optim.AdamW(
        [*model.residual_head.parameters(), *model.gate.parameters()], lr=config["learning_rate"]
    )
    best, epochs = float("inf"), args.epochs or config["fusion_epochs"]
    model_path = args.output / f"fold_{args.fold}" / "bridge.pt"
    for epoch in range(1, epochs + 1):
        fusion_epoch(model, train_loader, device, optimizer)
        loss = fusion_epoch(model, validation_loader, device)
        if loss < best:
            best = loss
            save_checkpoint(model_path, model=model.state_dict())
    model.load_state_dict(load_checkpoint(model_path, device)["model"])
    metrics = evaluate(model, test_loader, device)
    write_json(args.output / f"fold_{args.fold}" / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()

