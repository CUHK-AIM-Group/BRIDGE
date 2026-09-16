from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "brain_path", "gut_path", "label", "fold"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path("data/demo").resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    paired, diagnosis = [], []
    for index in range(15):
        label = ["CN", "MCI", "AD"][index % 3]
        brain = rng.normal(index % 3 * 0.15, 1, size=(32, 32, 32)).astype(np.float32)
        gut = (0.7 * brain + rng.normal(0, 0.3, brain.shape)).astype(np.float32)
        brain_path, gut_path = root / f"P{index:03d}_brain.nii.gz", root / f"P{index:03d}_gut.nii.gz"
        nib.save(nib.Nifti1Image(brain, np.eye(4)), brain_path)
        nib.save(nib.Nifti1Image(gut, np.eye(4)), gut_path)
        paired.append({"subject_id": f"PAIR_{index:03d}", "brain_path": brain_path, "gut_path": gut_path,
                       "label": "", "fold": 0})
        diagnosis.append({"subject_id": f"SUBJECT_{index:03d}", "brain_path": brain_path, "gut_path": "",
                          "label": label, "fold": index % 5 + 1})
    write_manifest(root / "paired.csv", paired)
    write_manifest(root / "diagnosis.csv", diagnosis)
    torch.save({"embeddings": torch.randn(6, 16, generator=torch.Generator().manual_seed(2026))},
               root / "anchors.pt")
    print(root)


if __name__ == "__main__":
    main()
