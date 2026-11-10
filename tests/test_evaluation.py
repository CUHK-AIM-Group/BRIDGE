import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
import yaml

from bridge.data import PETDataset, Record, diagnosis_label, fold_split, patient_key, read_manifest
from bridge.utils import device_from_name
from scripts import train_diagnosis as diagnosis


def records(visits=1, assigned=True):
    return [Record(f"P{i}_V{visit}", f"P{i}_V{visit}.nii.gz", label="CN" if i < 100 else "AD",
                   fold=i % 5 + 1 if assigned else 0, patient_id=f"P{i}")
            for i in range(200) for visit in range(visits)]


@pytest.mark.parametrize("assigned", [True, False])
def test_five_fold_grouped_stratified_70_10_20(assigned):
    data = records(visits=2, assigned=assigned)
    test_patients = []
    for fold in range(1, 6):
        parts = fold_split(data, fold, seed=7)
        assert [len(part) for part in parts] == [280, 40, 80]
        assert parts == fold_split(data, fold, seed=7)
        groups = [{patient_key(r) for r in part} for part in parts]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        for part in parts:
            assert sum(r.label == "CN" for r in part) == len(part) // 2
        test_patients.extend(groups[2])
    assert len(test_patients) == len(set(test_patients)) == 200
    assert fold_split(data, 1, seed=7)[1] != fold_split(data, 1, seed=8)[1]


def test_repeated_subjects_are_grouped_without_new_manifest_column(tmp_path):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "brain_path", "label", "fold"])
        for r in records(visits=2):
            writer.writerow([r.patient_id, r.brain_path, r.label, r.fold])
    parsed = read_manifest(path)
    train, val, test = fold_split(parsed, 1)
    assert [len(part) for part in (train, val, test)] == [280, 40, 80]
    assert not ({r.subject_id for r in val} & {r.subject_id for r in test})


def test_rejects_patient_across_outer_folds():
    data = records(visits=2)
    data[0] = replace(data[0], fold=2)
    with pytest.raises(ValueError, match="one outer fold"):
        fold_split(data, 1)


def test_rejects_same_scan_under_different_patient_ids():
    data = records()
    data[1] = replace(data[1], brain_path=data[0].brain_path)
    with pytest.raises(ValueError, match="leakage"):
        fold_split(data, 1)


def test_rejects_incomplete_or_non_stratified_outer_folds():
    with pytest.raises(ValueError, match="five outer folds"):
        fold_split([replace(r, fold=1) for r in records()], 1)
    with pytest.raises(ValueError, match="every diagnosis"):
        fold_split([replace(r, fold=2) if r.fold == 1 and r.label == "AD" else r
                    for r in records()], 1)


@pytest.mark.parametrize("labels,raw", [(('CN', 'MCI'), ['0', '1', 'CN', 'MCI']),
                                        (('CN', 'AD'), ['0', '2', 'CN', 'AD'])])
def test_binary_label_mapping(monkeypatch, labels, raw):
    monkeypatch.setattr("bridge.data.load_volume", lambda *_: torch.zeros(1, 4, 4, 4))
    dataset = PETDataset([Record(str(i), str(i), label=value) for i, value in enumerate(raw)],
                         (4, 4, 4), labels=labels)
    assert [dataset[i]['label'].item() for i in range(4)] == [0, 1, 0, 1]
    assert [diagnosis_label(value) for value in raw] == [labels[0], labels[1]] * 2


class Predictions(torch.nn.Module):
    def forward(self, image):
        return {"final_logits": image}


def test_auc_uses_softmax_probabilities_not_argmax():
    scores = torch.tensor([.1, .4, .2, .3])
    logits = torch.stack((1 - scores, scores), 1).log()
    assert logits.argmax(1).tolist() == [0, 0, 0, 0]
    loader = DataLoader([{"brain": x, "label": y} for x, y in zip(logits, [0, 1, 0, 1])],
                        batch_size=3)
    assert diagnosis.evaluate(Predictions(), loader, torch.device("cpu")) == {"auc": 1.0}


def test_single_class_auc_fails():
    loader = DataLoader([{"brain": torch.tensor([1., 2.]), "label": 0}])
    with pytest.raises(ValueError, match="both classes"):
        diagnosis.evaluate(Predictions(), loader, torch.device("cpu"))


def test_five_fold_summary():
    values = [.85, .87, .88, .86, .89]
    result = diagnosis.auc_summary(values)
    assert set(result) == {"fold_auc", "mean_auc", "std_auc"}
    assert result["mean_auc"] == pytest.approx(np.mean(values))
    assert result["std_auc"] == pytest.approx(np.std(values, ddof=1))
    with pytest.raises(ValueError, match="five"):
        diagnosis.auc_summary(values[:4])


@pytest.mark.parametrize("source_init", [False, True])
def test_validation_only_checkpoint_selection_and_early_stopping(tmp_path, monkeypatch, source_init):
    config = yaml.safe_load(Path("configs/demo.yaml").read_text())
    config.update(diagnostic_epochs=10, fusion_epochs=10, early_stopping_patience=2)
    anchors = tmp_path / "anchors.pt"
    transport_path = tmp_path / "transport.pt"
    torch.save({"embeddings": torch.randn(6, 16)}, anchors)
    model_config = {"latent_dim": 16, "base_channels": 2, "integration_steps": 2}
    transport = diagnosis.BrainGutLatentTransport(**model_config)
    torch.save({"model": transport.state_dict(), "model_config": model_config}, transport_path)
    args = SimpleNamespace(anchors=anchors, transport=transport_path, output=tmp_path,
                           epochs=None, source_cerebral=None)
    if source_init:
        source_path = tmp_path / "source.pt"
        cerebral = diagnosis.BrainEvidenceClassifier(torch.randn(6, 16), 2, 16, 2)
        torch.save({"model": cerebral.state_dict(), "labels": ["CN", "AD"],
                    "patient_ids": ["EXTERNAL_SOURCE"]}, source_path)
        args.source_cerebral = source_path
    data = records()
    train, val, test = fold_split(data, 1)
    seen = {"cerebral_train": 0, "cerebral_val": 0, "fusion_train": 0, "fusion_val": 0}

    def epoch(stage):
        def run(model, loader, device, optimizer=None, gradient_clip=5.):
            expected = train if optimizer is not None else val
            assert loader.dataset.records == expected
            seen[stage + ("_train" if optimizer is not None else "_val")] += 1
            if optimizer is not None:
                expected_lr = config["cerebral_learning_rate" if stage == "cerebral" else "fusion_learning_rate"]
                assert optimizer.param_groups[0]["lr"] == expected_lr
                assert optimizer.param_groups[0]["weight_decay"] == config["weight_decay"]
                assert gradient_clip == config["gradient_clip"]
            return 1.
        return run

    monkeypatch.setattr(diagnosis, "classification_epoch", epoch("cerebral"))
    monkeypatch.setattr(diagnosis, "fusion_epoch", epoch("fusion"))

    def evaluate(model, loader, device):
        assert loader.dataset.records == test
        assert seen == {"cerebral_train": 0 if source_init else 3,
                        "cerebral_val": 0 if source_init else 3,
                        "fusion_train": 3, "fusion_val": 3}
        assert all(not p.requires_grad for p in model.brain_model.parameters())
        assert all(not p.requires_grad for p in model.transport.parameters())
        return {"auc": .75}

    monkeypatch.setattr(diagnosis, "evaluate", evaluate)
    assert diagnosis.train_fold(args, config, data, ("CN", "AD"), 1) == .75
    if source_init:
        payload = torch.load(args.source_cerebral, weights_only=False)
        payload["patient_ids"] = [patient_key(data[0])]
        torch.save(payload, args.source_cerebral)
        with pytest.raises(ValueError, match="disjoint patient"):
            diagnosis.train_fold(args, config, data, ("CN", "AD"), 1)


def test_configs_and_cpu_fallback(monkeypatch):
    config = yaml.safe_load(Path("configs/bridge.yaml").read_text())
    assert [config[key] for key in ("batch_size", "weight_decay", "gradient_clip",
                                   "diagnostic_epochs", "fusion_epochs", "early_stopping_patience",
                                   "cerebral_learning_rate", "fusion_learning_rate", "integration_steps")] == [
                                       8, 1e-4, 5., 60, 70, 15, 2e-4, 5e-5, 8]
    assert config["transport_epochs"] == 120
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device_from_name("cuda").type == "cpu"
