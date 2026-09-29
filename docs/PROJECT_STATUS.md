# Project status — 29 September 2026

The author supplied a review of their **MATLAB Challenge Project 238** submission.
It was not accepted at that time. The review identifies missing MATLAB big-data
steps and other repairs; it invites discussion but does not promise acceptance
following changes. [Specific requirements and gaps](PROJECT238.md) and
[one-by-one email/submission checklist](SUBMISSION_CHECKLIST.md).

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
(including four optional quantum checks) and 19 MATLAB tests, all passing in
[the reviewer-update test run](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36425844079).

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

The [second full-source attempt](../experiments/project238-full/) then completed
one epoch on all 1,211,000 training jets and tested all 404,000 test jets. All
12,110 training iterations completed; accuracy is 90.2267% and AUC 0.9739505904.
Saved predictions, source labels, row IDs and model hash passed independent
checks. MATLAB generated the ROC/confusion-matrix figure in the central gallery.

The original seed-101 CNN, GraphSAGE and reference checkpoints and a 256-jet
verification sample are included in Git. `run_submission` verifies all three
checkpoints and recreates the MATLAB study summary without training or a data
download. The published files match the original manifest hashes. CI uses the
included files directly, with no dependency on expiring checkpoint artifacts.
The author approved permanent publication and the PR #9 merge on 29 September.

`summarize_matlab` now computes and exports the original study's means, sample
SDs, Student-t intervals, matched-seed accuracy/AUC differences and
[summary figure](../experiments/official-study/matlab_study_summary.png) in MATLAB.
Those tables were independently checked against SciPy/Pandas; the maximum
absolute difference was 4.45e-16. Prediction execution settings, local
noise/explanation random streams, release checks, JVM handling and vectorized
image creation are covered by the passing MATLAB suite.

## What still remains

Full-source execution is now verified for one epoch and one seed. A longer fit
with convergence checks and repeated full-data seeds remains unrun, including
the configuration's 12-epoch default. The full-source CNN has no explanation
study yet; existing explanations cover the original 50k/100k checkpoints.
Its model and per-jet predictions are retained in the public workflow artifact,
which expires on 27 December 2026; hashes and compact evidence are retained in Git.

The original failed attempt remains documented. The successful repaired run
measured 20.61 minutes for image preparation, 35.32 minutes for training and
4.76 GiB MATLAB peak memory, plus a separately measured 0.47 GiB Python peak.
These are observations under one runner's conditions, not minimum requirements.

Broad architecture comparisons, GPU and quantum-device experiments are still
unrun. ResNet18 and FPGA are optional brief extensions. The existing graph-edge
control is a narrower experiment, and the Qiskit result is a CPU simulation.

The author still needs to understand and explain the work, with AI assistance
acknowledged. Reviewer acceptance or permission to revise the rejected
submission has not been confirmed; the assistant has not contacted them.

[Requirements map](PROJECT238.md) · [Walkthrough](WALKTHROUGH.md) ·
[Results](RESULTS.md) · [MATLAB figure gallery](FIGURES.md) ·
[AI assistance and attribution](AI_ASSISTANCE.md)
