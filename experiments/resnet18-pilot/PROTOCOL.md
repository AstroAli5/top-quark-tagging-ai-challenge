# Named ResNet18 variation: frozen pilot protocol

Specified on 30 September 2026 before examining the new predictions. This is
an optional Project 238 variation, separate from the full-source CNN result.

| Choice | Fixed value |
| --- | --- |
| Source | Verified official HDF5 train, validation and test files |
| Selection | First 10,000 train / 2,000 validation / 10,000 test rows |
| Representation | Identical 32×32 single-channel transverse-momentum images |
| Preparation | Shared MATLAB-hosted Python → Parquet → tall → imageDatastore route |
| Models | Existing compact CNN; named `resnet18(Weights='none')` |
| ResNet adaptation | Grayscale input and first convolution; global average pool; two-class head; eight residual blocks retained |
| Initialization | Random weights; no ImageNet weights or support-package download |
| Training | Adam, learning rate 0.001, batch 64, three epochs, seed 101, CPU |
| Selection | Best validation loss, with validation once per epoch |
| Evaluation | Fixed 0.5 threshold accuracy and tie-aware ROC AUC |
| Checks | Source IDs/labels, MAT/CSV agreement, checkpoint hash, independent rank AUC and accuracy |

Run `run_resnet18_study` at the repository root. Both models reuse the same
prepared images; existing fitted outputs are protected. The unchanged
`run_project238` default remains the compact network.

The size is a CPU feasibility budget. It is not selected using test scores.
This controls the input representation, selected rows and training settings;
the architectures still differ in parameter count and compute. One seed and
three epochs do not establish superiority, convergence, or equal hyperparameter
optimization. No comparison with the larger historical study is presented as
an architecture-only result. Do not tune this pilot on its test scores.

Provenance and trained models are retained in the `resnet18-pilot` workflow
artifact for 90 days. Reports and independent verification are published after
successful execution; there is no result claim before then.

Reference: [MathWorks R2024a ResNet18 documentation](https://www.mathworks.com/help/releases/R2024a/deeplearning/ref/resnet18.html).
