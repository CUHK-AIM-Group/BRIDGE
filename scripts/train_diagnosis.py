from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from bridge.data import PETDataset, diagnosis_label, fold_split, patient_key, read_manifest
from bridge.losses import fusion_loss, same_label_mismatch
from bridge.models import BrainEvidenceClassifier, BrainGutLatentTransport, BridgeDiagnosisModel
from bridge.utils import device_from_name, load_checkpoint, save_checkpoint, seed_all, write_json


def classification_epoch(model, loader, device, optimizer=None, gradient_clip=5.0) -> float:
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        total += float(loss.detach()) * image.shape[0]
    return total / len(loader.dataset)


def fusion_epoch(model, loader, device, optimizer=None, gradient_clip=5.0) -> float:
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
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], gradient_clip)
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
        if logits.shape[1] != 2:
            raise ValueError("Binary AUC requires two logits for the selected diagnosis task")
        prediction.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
    if set(truth) != {0, 1}:
        raise ValueError("ROC-AUC is undefined: the evaluation set must contain both classes")
    return {"auc": float(roc_auc_score(truth, prediction))}


def auc_summary(values: list[float]) -> dict[str, object]:
    if len(values) != 5 or not np.isfinite(values).all():
        raise ValueError("AUC summary requires five finite fold AUC values")
    return {"fold_auc": values, "mean_auc": float(np.mean(values)),
            "std_auc": float(np.std(values, ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-fold BRIDGE binary diagnosis with ROC-AUC evaluation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/bridge.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/diagnosis"))
    parser.add_argument("--fold", type=int, choices=range(1, 6), help="Omit to run all five folds")
    parser.add_argument("--task", choices=("CN/MCI", "CN/AD"), default="CN/MCI")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--source-cerebral", type=Path,
                        help="Validation-selected source cerebral checkpoint; skip cerebral initialization")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    labels = tuple(args.task.split("/"))
    records = [r for r in read_manifest(args.manifest) if diagnosis_label(r.label) in labels]
    args.output = args.output / args.task.replace("/", "_")
    values = []
    for fold in ([args.fold] if args.fold else range(1, config["folds"] + 1)):
        values.append(train_fold(args, config, records, labels, fold))
    if args.fold is None:
        summary = auc_summary(values)
        write_json(args.output / "auc.json", summary)
        print(f"Five-fold AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")


def train_fold(args, config, records, labels, fold) -> float:
    seed_all(config["seed"] + fold)
    device = device_from_name(config["device"])
    train, validation, test = fold_split(records, fold, config["folds"], config["seed"])
    make_loader = lambda records, shuffle=False: DataLoader(
        PETDataset(records, tuple(config["shape"]), labels=labels), config["batch_size"], shuffle=shuffle,
        num_workers=config["workers"],
    )
    train_loader, validation_loader, test_loader = (
        make_loader(train, True), make_loader(validation), make_loader(test)
    )
    anchors = load_checkpoint(args.anchors, device)["embeddings"].float().to(device)
    cerebral = BrainEvidenceClassifier(anchors, len(labels), anchors.shape[1], config["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(cerebral.parameters(), lr=config["cerebral_learning_rate"],
                                  weight_decay=config["weight_decay"])
    best = float("inf")
    epochs = min(args.epochs or config["diagnostic_epochs"], config["diagnostic_epochs"])
    cerebral_path = args.output / f"fold_{fold}" / "cerebral.pt"
    if args.source_cerebral:
        source = load_checkpoint(args.source_cerebral, device)
        if source.get("labels") != list(labels):
            raise ValueError("Source cerebral checkpoint must store the same ordered binary labels")
        if "patient_ids" not in source:
            raise ValueError("Source cerebral checkpoint must store patient_ids for overlap checking")
        if set(source["patient_ids"]) & {patient_key(r) for r in records}:
            raise ValueError("Source and private cohorts must have disjoint patient IDs")
        cerebral.load_state_dict(source["model"])
        epochs = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        classification_epoch(cerebral, train_loader, device, optimizer, config["gradient_clip"])
        loss = classification_epoch(cerebral, validation_loader, device)
        if loss < best:
            best = loss
            save_checkpoint(cerebral_path, model=cerebral.state_dict(), labels=list(labels),
                            patient_ids=sorted({patient_key(r) for r in train + validation}))
            stale = 0
        else:
            stale += 1
        if stale >= config["early_stopping_patience"]:
            break
    if not args.source_cerebral:
        cerebral.load_state_dict(load_checkpoint(cerebral_path, device)["model"])

    transport_payload = load_checkpoint(args.transport, device)
    transport = BrainGutLatentTransport(**transport_payload["model_config"]).to(device)
    transport.load_state_dict(transport_payload["model"])
    model = BridgeDiagnosisModel(cerebral, transport, len(labels), anchors.shape[1]).to(device)
    model.freeze_priors()
    optimizer = torch.optim.AdamW(
        [*model.residual_head.parameters(), *model.gate.parameters()],
        lr=config["fusion_learning_rate"], weight_decay=config["weight_decay"]
    )
    best = float("inf")
    epochs = min(args.epochs or config["fusion_epochs"], config["fusion_epochs"])
    model_path = args.output / f"fold_{fold}" / "bridge.pt"
    stale = 0
    for epoch in range(1, epochs + 1):
        fusion_epoch(model, train_loader, device, optimizer, config["gradient_clip"])
        loss = fusion_epoch(model, validation_loader, device)
        if loss < best:
            best = loss
            save_checkpoint(model_path, model=model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config["early_stopping_patience"]:
            break
    model.load_state_dict(load_checkpoint(model_path, device)["model"])
    metrics = evaluate(model, test_loader, device)
    write_json(args.output / f"fold_{fold}" / "metrics.json", metrics)
    print(f"Fold {fold}: AUC={metrics['auc']:.4f}")
    return metrics["auc"]


if __name__ == "__main__":
    main()

