# BRIDGE

BRIDGE learns brain-to-gut latent transport from paired PET, then uses reliability-gated virtual-gut evidence for brain-only diagnosis. The released binary evaluation supports **CN/MCI** (MCI positive) and **CN/AD** (AD positive).

![BRIDGE architecture](assets/architecture.jpg)

## Installation

Use Python 3.10+ and a PyTorch build appropriate for your CUDA environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

`device: auto` selects CUDA when available; CPU-only execution is supported. The scripts do not require mixed precision.

## Data input

Use preprocessed NIfTI volumes and CSV manifests following `examples/`:

- Paired transport data: `subject_id,brain_path,gut_path`.
- Diagnosis data: `subject_id,brain_path,label,fold`. Labels may be `CN/MCI/AD` or their original codes `0/1/2`. The selected binary task remaps its labels to `0/1` before training and evaluation.
- Fixed outer folds must be numbered `1..5`, diagnosis-stratified, and shared by every examination of a patient. If all folds are missing/zero, five patient-level stratified folds are generated with the configured seed.
- For repeated examinations, use the same `subject_id`, or add an optional `patient_id` column linking scan-specific subject IDs. Patient identity cannot be inferred from arbitrary scan names. Conflicting patient/fold assignments fail explicitly.
- The reviewed anchor artifact must contain an `embeddings` tensor `[K, latent_dim]`.

Each outer fold is held out for testing (~20%). The remaining patients are split by diagnosis into training and validation at 7:1, giving ~70/10/20 overall. Entire patient groups stay together; longitudinal diagnoses are stratified by patient-majority diagnosis (ties follow diagnosis-code order). Ratios are approximate for small cohorts or unequal examination counts. Cohorts too small to stratify, or fixed folds missing a class, fail explicitly. Use identical manifests and seeds across compared methods.

## Training and AUC evaluation

Run from the repository root. Stage 1 transport keeps its existing 120-epoch budget; do not confuse this with cerebral initialization or private fusion adaptation:

```bash
python scripts/train_transport.py --manifest /path/to/paired.csv --config configs/bridge.yaml --output runs/transport
```

Run all five diagnosis folds (omit `--fold`):

```bash
python scripts/train_diagnosis.py --manifest /path/to/diagnosis.csv --anchors /path/to/anchors.pt --transport runs/transport/transport.pt --config configs/bridge.yaml --task CN/MCI --output runs/diagnosis
```

Use `--task CN/AD` for CN/AD, or `--fold 1` for a single fold. Each fold uses validation loss for checkpoint selection and early stopping, then evaluates the outer test set once. ROC-AUC uses `softmax(final_logits)[:, 1]`, never argmax labels. A single-class test set raises an error. No test data are used for model selection.

The full configuration uses batch size 8, weight decay `1e-4`, clipping at `5.0`, 60 cerebral-initialization epochs at `2e-4`, and at most 70 fusion-adaptation epochs at `5e-5`, with validation patience 15. Transport uses eight integration steps. These parameters are read from YAML; demo epoch/shape settings remain small. `--epochs` can shorten, but not increase, the configured diagnosis budgets.

For a separate source-to-private experiment, first obtain a source cerebral checkpoint selected using **source validation data only**, then pass `--source-cerebral /path/to/cerebral.pt` when running the private manifest. Checkpoints saved by this script contain matching ordered binary `labels` and the `patient_ids` used for training/validation. Private runs reject overlapping source/private patient IDs, skip cerebral initialization, and preserve the existing cerebral/transport freezing policy during residual/gate adaptation. Use consistent patient identifiers across cohorts. Without this option, cerebral initialization uses the current fold's training data: it is a same-cohort run, not a separate-source transfer experiment. This release does not add private brain-head adaptation or change the fusion loss.

Per-fold `metrics.json` contains only `auc`. A complete five-fold invocation writes `runs/diagnosis/CN_MCI/auc.json` (or `CN_AD/auc.json`) with `fold_auc`, `mean_auc`, and `std_auc`. The standard deviation uses `ddof=1`. Single-fold runs do not combine potentially stale results from separate runs. These are measured predictions, not the paper's reported results.

## CPU demo

```bash
python scripts/make_demo_data.py
python scripts/train_transport.py --manifest data/demo/paired.csv --config configs/demo.yaml --output runs/demo_transport
python scripts/train_diagnosis.py --manifest data/demo/diagnosis.csv --anchors data/demo/anchors.pt --transport runs/demo_transport/transport.pt --config configs/demo.yaml --task CN/MCI --output runs/demo_diagnosis
```

The demo creates synthetic data under ignored `data/` and runs one epoch per phase, including held-out inference. Its AUC is a software check, not a clinical performance estimate. `--task CN/AD` runs the other mapping.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhu2026bridge,
  title={BRIDGE: Learning Brain-Gut Metabolic Evidence for Alzheimer's Disease Diagnosis},
  author={Zhu, Pengli and Chang, Ruxi and Yao, Tianliang and Jiang, Yu and Liu, Shengyuan and Lin, Yiyang and Duan, Xiaoyi and Wu, Shuang and Yuan, Yixuan and Alzheimer's Disease Neuroimaging Initiative and others},
  journal={arXiv preprint arXiv:2610.39520},
  year={2026}
}
```

## License

MIT. Dataset access and permissions remain the responsibility of the data providers and users.
