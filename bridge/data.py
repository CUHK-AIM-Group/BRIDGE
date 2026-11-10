from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset


LABELS = {"CN": 0, "MCI": 1, "AD": 2}


@dataclass(frozen=True)
class Record:
    subject_id: str
    brain_path: str
    gut_path: str = ""
    label: str = ""
    fold: int = 0
    patient_id: str = ""


def diagnosis_label(label: str) -> str:
    value = str(label).strip().upper()
    value = {str(index): name for name, index in LABELS.items()}.get(value, value)
    if value not in LABELS:
        raise ValueError(f"Unknown diagnosis label: {label!r}")
    return value


def patient_key(record: Record) -> str:
    return record.patient_id or record.subject_id


def read_manifest(path: str | Path) -> list[Record]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = [Record(
        subject_id=row["subject_id"], brain_path=row["brain_path"],
        gut_path=row.get("gut_path", ""), label=row.get("label", ""),
        fold=int(row.get("fold", 0) or 0),
        patient_id=row.get("patient_id", "").strip(),
    ) for row in rows]
    if any(not r.subject_id.strip() for r in records):
        raise ValueError("subject_id must not be empty")
    return records


def load_volume(path: str | Path, shape: tuple[int, int, int]) -> torch.Tensor:
    volume = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    foreground = volume > 0
    if foreground.sum() >= 32:
        low, high = np.percentile(volume[foreground], [0.5, 99.5])
        volume = np.clip(volume, low, high)
        values = volume[foreground]
        scale = max(float(np.percentile(values, 75) - np.percentile(values, 25)), 1e-6)
        volume = (volume - float(np.median(values))) / scale
        volume[~foreground] = 0
    volume = volume.astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.transpose(volume, (2, 1, 0)).copy())[None, None]
    if tuple(tensor.shape[-3:]) != shape:
        tensor = F.interpolate(tensor, size=shape, mode="trilinear", align_corners=False)
    return tensor[0]


class PETDataset(Dataset):
    def __init__(self, records: list[Record], shape: tuple[int, int, int], paired: bool = False,
                 labels: tuple[str, ...] | None = None) -> None:
        self.records, self.shape, self.paired = records, shape, paired
        self.labels = labels or tuple(LABELS)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        item: dict[str, object] = {"brain": load_volume(record.brain_path, self.shape),
                                  "subject_id": record.subject_id}
        if self.paired:
            if not record.gut_path:
                raise ValueError(f"missing gut_path for {record.subject_id}")
            item["gut"] = load_volume(record.gut_path, self.shape)
        else:
            item["label"] = torch.tensor(self.labels.index(diagnosis_label(record.label)), dtype=torch.long)
        return item


def fold_split(records: list[Record], test_fold: int, folds: int = 5,
               seed: int = 2026) -> tuple[list[Record], ...]:
    """Fixed outer test fold; diagnosis-stratified patient-level 7:1 inner split."""
    if folds != 5 or not 1 <= test_fold <= folds:
        raise ValueError("Diagnosis evaluation requires five folds numbered 1..5")
    groups: dict[str, list[Record]] = {}
    owners: dict[str, str] = {}
    subject_owners: dict[str, str] = {}
    for record in records:
        key = patient_key(record)
        groups.setdefault(key, []).append(record)
        # The same scan or subject must not be assigned to two different patients.
        for mapping, value in ((owners, str(Path(record.brain_path).resolve())),
                               (subject_owners, record.subject_id)):
            if value in mapping and mapping[value] != key:
                raise ValueError("subject/scan leakage: inconsistent patient_id")
            mapping[value] = key
    keys = sorted(groups)
    # Longitudinal diagnoses may change. Stratify by the patient's majority diagnosis
    # (ties use the diagnosis code), while keeping every examination together.
    labels = [min(Counter(diagnosis_label(r.label) for r in groups[key]).items(),
                  key=lambda item: (-item[1], LABELS[item[0]]))[0] for key in keys]
    assigned = {}
    if all(r.fold == 0 for r in records):
        if min(Counter(labels).values(), default=0) < folds:
            raise ValueError("Need at least five patients per diagnosis for five-fold stratification")
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for number, (_, indices) in enumerate(splitter.split(keys, labels), 1):
            assigned.update({keys[i]: number for i in indices})
    else:
        for key, visits in groups.items():
            values = {r.fold for r in visits}
            if len(values) != 1 or not values.issubset(set(range(1, folds + 1))):
                raise ValueError("All examinations of a patient must share one outer fold in 1..5")
            assigned[key] = values.pop()
        if set(assigned.values()) != set(range(1, folds + 1)):
            raise ValueError("Manifest must contain all five outer folds")
    remaining = [i for i, key in enumerate(keys) if assigned[key] != test_fold]
    try:
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + test_fold)
        train_idx, validation_idx = next(inner.split(remaining, [labels[i] for i in remaining]))
    except ValueError as exc:
        raise ValueError(f"Not enough patients for stratified 70/10/20 splitting: {exc}") from exc
    train_keys = {keys[remaining[i]] for i in train_idx}
    validation_keys = {keys[remaining[i]] for i in validation_idx}
    test_keys = {key for key in keys if assigned[key] == test_fold}
    if train_keys & validation_keys or train_keys & test_keys or validation_keys & test_keys:
        raise ValueError("patient leakage detected")
    train = [r for r in records if patient_key(r) in train_keys]
    validation = [r for r in records if patient_key(r) in validation_keys]
    test = [r for r in records if patient_key(r) in test_keys]
    expected = {diagnosis_label(r.label) for r in records}
    if any({diagnosis_label(r.label) for r in part} != expected for part in (train, validation, test)):
        raise ValueError("Each partition must contain every diagnosis; check stratified outer folds")
    return train, validation, test
