# Result reporting

Run each outer fold independently and report subject-level out-of-fold results. The recommended table is:

| Task | Model | AUROC | Balanced accuracy | Macro F1 | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|
| CN vs MCI vs AD | BRIDGE | — | — | — | n/a | n/a |
| CN vs MCI | BRIDGE | — | — | — | — | — |
| CN vs AD | BRIDGE | — | — | — | — | — |
| MCI vs AD | BRIDGE | — | — | — | — | — |

Use the subject as the independent unit. Keep outer-test subjects out of normalization fitting, model selection,
calibration, evidence-anchor construction, and Stage 1 adaptation. Report confidence intervals and the exact fold
assignment alongside the final table.

