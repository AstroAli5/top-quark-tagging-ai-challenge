# Project status — 28 September 2026

The author supplied a review of their **MATLAB Challenge Project 238** submission.
It was not accepted at that time. The review identifies missing MATLAB big-data
steps and other repairs; it invites discussion but does not promise acceptance
following changes. [Specific requirements and gaps](PROJECT238.md).

Earlier documentation incorrectly stated that no submission existed and focused
on the closed 2025 AI Challenge, a different event. The author submitted to the
Project Hub; the assistant did not make that submission or contact the reviewers.

## Verified research

| Study | Scope and evidence |
| --- | --- |
| Original comparison | [50k training, three models, three seeds](../experiments/official-study/); all 404,000 official test jets |
| Additional seeds | [Five-seed CNN/GraphSAGE report](../experiments/seed-extension/); reference remains three seeds |
| Larger subset | [100k training, three core seeds](../experiments/scaling/); all official test jets; measured MATLAB peak RAM 3.68–3.73 GiB |
| Final-model explanations | Both [50k](../experiments/explanations/) and [100k](../experiments/explanations-100k/) checkpoint sets; each verifies 51 perturbation measurements and six checkpoint hashes |
| Controlled graph edges | [Three-seed comparison](../experiments/graph-edges/); edges improve clean AUC, zero edges improve AUC at the strongest tested smearing |
| Qiskit pilot | [Four-qubit CPU simulator](../experiments/quantum-pilot/); the classical RBF SVM performed better |

These results were checked against saved predictions and provenance. Historical
results remain unchanged. The current check suite contains 20 Python tests
(including four optional quantum checks) and 18 MATLAB tests.

## Verified response to the review

The new [MATLAB data route](../experiments/project238-demo/) executes MATLAB-hosted
Python HDF5 → Parquet conversion, tall image creation, folder-labelled
imageDatastore training and test evaluation. Its small real-data run selected
2,000 training / 500 validation / 1,000 test jets. All test predictions, row IDs,
source labels, metrics and the saved model hash passed independent checks.

The repaired route also passed a [100k training / 1k validation / 1k test,
one-epoch check](../experiments/project238-full/scale-check/): 73.28 seconds for
image preparation and 124.94 seconds for training. Every saved test prediction
was independently checked. This execution probe is separate from the earlier
three-seed 100k research study.

The original CNN checkpoint and a 256-jet verification sample are included in
Git. `verify_results` needs no training or data download. CI also restores the
other two seed-101 models from their existing public artifacts; all three had
zero score difference on the latest verification run. Their permanent Git
copies still await upload approval.

`summarize_matlab` now computes and exports the original study's statistics and
[summary figure](../experiments/official-study/matlab_study_summary.png) in MATLAB.
Those tables were independently checked. Prediction execution settings, local
noise/explanation random streams, release checks, JVM handling and vectorized
image creation are covered by the passing MATLAB suite.

## What still remains

The [first full-source attempt](../experiments/project238-full/) reached all-row
Parquet conversion in 185.72 seconds, then exceeded its 305-minute limit during
image preparation/validation. It did not reach training and produced no full-data
accuracy. A cached file-list repair and finer timing checkpoints now pass the
100k execution probe; opening and validating those training filenames took 3.3
seconds. This supports a retry but does not prove full-source completion.

Full-source training has not completed. The old MAT-v5 array limit is avoided
by the new disk-backed route, but that alone is not evidence of a successful
large fit. A full-source run must be measured, not inferred from this demo.

Broad architecture comparisons, GPU and quantum-device experiments are still
unrun. ResNet18 and FPGA are optional brief extensions. The existing graph-edge
control is a narrower experiment, and the Qiskit result is a CPU simulation.

The author still needs to understand and explain the work, with AI assistance
acknowledged. Reviewer acceptance or permission to revise the rejected
submission has not been confirmed; the assistant has not contacted them.

[Requirements map](PROJECT238.md) · [Walkthrough](WALKTHROUGH.md) ·
[Results](RESULTS.md) · [MATLAB figure gallery](FIGURES.md) ·
[AI assistance and attribution](AI_ASSISTANCE.md)
