# Project 238 revision response

**Prepared draft — not sent.** Reply in the original review thread using the
account used for registration. The official instructions also require that same
email for the project submission form. Confirm the preferred revision route
with the team if the form does not clearly distinguish a revised submission.
The repository changes must be published on the linked branch before sending.

**Subject:** Project 238 — revised MATLAB solution and request for reevaluation

Dear MathWorks Capstone Projects team,

Thank you for the detailed feedback on my Top Quark Detection with Deep Learning
and Big Data submission. The revised repository addresses the required MATLAB
workflow and reproducibility points, with a point-by-point evidence checklist:

https://github.com/AstroAli5/top-quark-tagging-ai-challenge

https://github.com/AstroAli5/top-quark-tagging-ai-challenge/blob/main/docs/SUBMISSION_CHECKLIST.md

The main revisions are:

- MATLAB now calls Python through `pyrun` for bounded HDF5-to-Parquet conversion.
  `parquetDatastore` and tall preprocessing create floating-point jet images;
  folder-labelled `imageDatastore` objects feed `trainnet` and `minibatchpredict`.
- A verified CPU run completed one epoch on all 1,211,000 official training jets,
  with 10,000 validation jets and all 404,000 official test jets. Independently
  checked accuracy is 90.2267%, with ROC AUC 0.9739505904. This is a one-epoch,
  single-seed result, not a claim of convergence.
- The requested seed-101 CNN, GraphSAGE and reference checkpoints and a small
  official sample are included in Git. `run_submission` checks their restored
  predictions and generates the MATLAB study summary without training or a
  dataset download. The complete command was measured at 14.19 seconds on the
  verified runner, excluding MATLAB startup.
- MATLAB now produces the study statistics, matched-seed differences and summary
  plots. The documented ROC toolbox path, vectorized image accumulation,
  execution-environment settings, local random streams, JVM requirement and
  compatibility fixes are covered by the verification records.
- The seven-step Project 238 implementation table is directly in the README.
  The optional named ResNet18 variation has also been run against a compact CNN
  on the same images and fixed training budget. Its small, single-seed scope and
  results are reported separately. A Deep Learning HDL Toolbox processor latency
  estimate also completed; no generated HDL or FPGA deployment is claimed.

The README links the protocols, MATLAB figures, saved evidence and limitations.
The quick-check sample is explicitly distinguished from the full-test scores
and from the original three-seed comparison.

The repository discloses substantial AI assistance with implementation,
verification and documentation. I understand that the author remains responsible
for understanding, explaining and justifying the submitted work; the automated
checks are not a substitute for that requirement.

Could you please reevaluate the revised solution and confirm whether I should
update the project submission form or use this thread for the revision? Please
let me know if a required point remains unresolved.

Best regards,
Ali Mohamed

## Before delivery

The draft does not state that the author has completed a code defense, that the
organizers accepted the revision, or that a new stage is guaranteed. Review the
[practice route](WALKTHROUGH.md#prepare-to-explain-the-revision-to-a-reviewer)
and use the original registered account. Record a form receipt or reviewer reply
after delivery; until then, the revision remains unsubmitted.

[Official submission instructions](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/wiki/Submission-Instructions-%26-Project-Repository-Guidelines)
· [Official AI guidelines](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/wiki/Generative-AI-Guidelines)
