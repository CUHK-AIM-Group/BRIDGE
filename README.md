# BRIDGE

BRIDGE is a two-stage framework for Alzheimer's disease diagnosis from brain PET. It first learns a
brain-to-gut latent transport from paired whole-body PET and then performs evidence-guided, reliability-gated
virtual-gut fusion using brain-only PET. The public code supports CN, MCI, and AD classification while keeping
the deployment input brain-only.

![BRIDGE architecture](assets/architecture.jpg)

## Overview

**Stage 1 — Brain-to-Gut Latent Transport.** Organ-specific 3D encoders map paired brain and gut PET into a
shared latent space. A conditional vector field is trained with flow matching and an endpoint constraint to
predict a virtual-gut representation from the brain representation.

**Stage 2 — Evidence-Guided Virtual-Gut Fusion.** A diagnostic encoder is modulated by a reviewed disease-
evidence anchor bank. The frozen Stage 1 transport produces virtual-gut evidence from brain PET. A residual
head and confidence gate add only reliable virtual-gut information to the cerebral diagnostic logits.

At inference, BRIDGE requires only a brain PET volume. The offline evidence workflow is separate from model
training and never receives patient images or labels.

Repository layout:

```text
bridge/                 model, loss, data, and utility code
scripts/                Stage 1 and Stage 2 entry points
configs/                full and CPU demonstration configurations
examples/               anonymous manifest and anchor schemas
tests/                  end-to-end tensor smoke test
tools/                   pre-release sensitive-information scanner
assets/architecture.jpg method overview
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Idea89560041/BRIDGE.git
cd BRIDGE
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the CPU smoke test:

```bash
python -m pytest -q
```

## Data preparation

BRIDGE expects preprocessed NIfTI volumes in a common analysis space. Create two CSV manifests following
`examples/paired_manifest.csv` and `examples/diagnosis_manifest.csv`.

- The paired manifest contains `subject_id`, `brain_path`, and `gut_path` for Stage 1.
- The diagnosis manifest contains `subject_id`, `brain_path`, `label`, and a fixed outer `fold` for Stage 2.
- Labels are `CN`, `MCI`, or `AD`; subject identifiers must be pseudonymous and unique.
- Paths in the examples are placeholders. Do not commit real manifests or images.

Stage 2 also expects a reviewed PyTorch anchor artifact containing an `embeddings` tensor of shape `[K, D]`.
The embedding dimension `D` must match `latent_dim` in the selected configuration.

## Training and Evaluation

### Minimal reproducibility check

The following creates synthetic NIfTI volumes and runs one CPU epoch for both stages. Synthetic outputs are
written under ignored `data/` and `runs/` directories.

```bash
python scripts/make_demo_data.py
python scripts/train_transport.py \
  --manifest data/demo/paired.csv \
  --config configs/demo.yaml \
  --output runs/demo_transport
python scripts/train_diagnosis.py \
  --manifest data/demo/diagnosis.csv \
  --anchors data/demo/anchors.pt \
  --transport runs/demo_transport/transport.pt \
  --config configs/demo.yaml \
  --fold 1 \
  --output runs/demo_diagnosis
```

### Full experiment

Train Stage 1 once using paired whole-body PET:

```bash
python scripts/train_transport.py \
  --manifest /path/to/paired_manifest.csv \
  --config configs/bridge.yaml \
  --output runs/transport
```

Then train and evaluate Stage 2 independently for each fixed outer fold:

```bash
for FOLD in 1 2 3 4 5; do
  python scripts/train_diagnosis.py \
    --manifest /path/to/diagnosis_manifest.csv \
    --anchors /path/to/reviewed_anchor_embeddings.pt \
    --transport runs/transport/transport.pt \
    --config configs/bridge.yaml \
    --fold ${FOLD} \
    --output runs/diagnosis
done
```

Each fold writes selected checkpoints and `metrics.json`. Use validation data for model selection and evaluate
the outer test fold only after selection. Keep fold assignments fixed across all reported experiments. See
`RESULTS.md` for the reporting structure and `DATA_AND_PRIVACY.md` for release requirements.

## License

This code is released under the MIT License. Dataset licenses and clinical governance requirements remain with
their respective providers and institutions.
