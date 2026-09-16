from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import yaml

from bridge.data import PETDataset, read_manifest
from bridge.losses import transport_loss
from bridge.models import BrainGutLatentTransport
from bridge.utils import device_from_name, save_checkpoint, seed_all, write_json


def run_epoch(model, loader, device, optimizer=None) -> float:
    model.train(optimizer is not None)
    total = 0.0
    for batch in loader:
        brain, gut = batch["brain"].to(device), batch["gut"].to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            loss = transport_loss(model(brain, gut))
            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total += float(loss.detach()) * brain.shape[0]
    return total / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BRIDGE Stage 1 brain-to-gut latent transport.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/bridge.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/transport"))
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_all(config["seed"])
    device = device_from_name(config["device"])
    dataset = PETDataset(read_manifest(args.manifest), tuple(config["shape"]), paired=True)
    validation_size = max(1, round(0.2 * len(dataset)))
    train_set, validation_set = random_split(
        dataset, [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    train_loader = DataLoader(train_set, config["batch_size"], shuffle=True, num_workers=config["workers"])
    validation_loader = DataLoader(validation_set, config["batch_size"], num_workers=config["workers"])
    model_config = {
        "latent_dim": config["latent_dim"], "base_channels": config["base_channels"],
        "integration_steps": config["integration_steps"],
    }
    model = BrainGutLatentTransport(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    best, history = float("inf"), []
    epochs = args.epochs or config["transport_epochs"]
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best:
            best = validation_loss
            save_checkpoint(args.output / "transport.pt", model=model.state_dict(), model_config=model_config)
        print(history[-1])
    write_json(args.output / "history.json", history)


if __name__ == "__main__":
    main()

