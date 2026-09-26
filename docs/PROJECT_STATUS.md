# Project status — 26 September 2026

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
results remain unchanged. The current check suite contains 17 Python tests
(including four optional quantum checks) and 16 MATLAB tests.

## What prevents calling the submission finished

The submitted version does not implement the required MATLAB-hosted Parquet,
tall-array and folder-labelled imageDatastore route. It also needs accessible
checkpoints, MATLAB reporting, measured justification of its training scope,
and the smaller engineering fixes listed in [Project 238](PROJECT238.md).
Completing research extensions does not satisfy these missing requirements.

Full-source training, a broad architecture study, GPU and quantum-device runs
remain unrun. The [earlier resource assessment](../experiments/full-data-assessment/)
describes limits of the existing MAT route; it does not establish a minimum
hardware purchase or demonstrate a working big-data route.

[Walkthrough](WALKTHROUGH.md) · [Results](RESULTS.md) · [MATLAB figure gallery](FIGURES.md) · [AI assistance and attribution](AI_ASSISTANCE.md)
