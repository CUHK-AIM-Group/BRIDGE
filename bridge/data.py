from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


LABELS = {"CN": 0, "MCI": 1, "AD": 2}


@dataclass(frozen=True)
class Record:
    subject_id: str
    brain_path: str
    gut_path: str = ""
    label: str = ""
    fold: int = 0


def read_manifest(path: str | Path) -> list[Record]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = [Record(
        subject_id=row["subject_id"], brain_path=row["brain_path"],
        gut_path=row.get("gut_path", ""), label=row.get("label", ""),
        fold=int(row.get("fold", 0) or 0),
    ) for row in rows]
    if len({r.subject_id for r in records}) != len(records):
        raise ValueError("subject_id must be unique")
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
    def __init__(self, records: list[Record], shape: tuple[int, int, int], paired: bool = False) -> None:
        self.records, self.shape, self.paired = records, shape, paired

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
            item["label"] = torch.tensor(LABELS[record.label], dtype=torch.long)
        return item


def fold_split(records: list[Record], test_fold: int, folds: int = 5) -> tuple[list[Record], ...]:
    validation_fold = test_fold % folds + 1
    train = [r for r in records if r.fold not in {test_fold, validation_fold}]
    validation = [r for r in records if r.fold == validation_fold]
    test = [r for r in records if r.fold == test_fold]
    if {r.subject_id for r in train} & ({r.subject_id for r in validation} | {r.subject_id for r in test}):
        raise ValueError("subject leakage detected")
    return train, validation, test
