# Data and privacy

This repository contains source code, synthetic examples, and schema-only manifests. It does not contain
clinical images, subject-level metadata, trained weights, experiment outputs, credentials, or private paths.

Users are responsible for obtaining each dataset under its original terms and for preparing PET volumes in a
common analysis space. Manifests must use pseudonymous `subject_id` values and local paths. Never commit real
manifests, images, linkage keys, or output folders.

The evidence-anchor workflow is offline: patient images and labels must never be sent to a language model.
Only reviewed scientific descriptors may be embedded into the versioned anchor tensor consumed by training.

Before a public release, run:

```bash
python tools/scan_sensitive.py
```

