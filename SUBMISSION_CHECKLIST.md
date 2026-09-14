# Submission checklist

This project relates to MathWorks **Project #238**, not #193.
The [official project discussion](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/discussions/74)
identifies the project. Consult the
[Challenge Project Hub](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub)
for current participation instructions. No deadline or judging-score claim is
assumed here.

## Repository work

- [x] Public repository with an MIT license.
- [x] English README, setup commands, references, and concepts guide.
- [x] HDF5-to-MAT converter with input validation and provenance.
- [x] Portable entry point and all seven pipeline stages.
- [x] CNN probabilities and saved class order; tie-aware ROC/AUC.
- [x] Sparse CPU graph batching for training, validation, and inference.
- [x] Regression tests and a synthetic end-to-end workflow.
- [x] Historical result discrepancies documented without replacing measurements.

## Evidence required before calling the scientific project complete

- [ ] Confirm the latest [GitHub test run](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions) passed.
- [ ] Run the corrected pipeline on real data, retraining both models.
- [ ] Preserve the exact code commit, data checksum, run metadata, and split counts.
- [ ] Inspect saved predictions and verify the clean/zero-noise outputs agree.
- [ ] Publish fresh matching CSVs and plots with an accurate results discussion.
- [ ] Distinguish the internal holdout from the official dataset test partition.
- [ ] Repeat with additional seeds and report uncertainty before claiming robust superiority.
- [ ] Explain the noise simplification, unequal training budgets, and limits of feature permutation.

## Author's submission tasks

- [ ] Read the current
  [Generative AI Guidelines](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/wiki/Generative-AI-Guidelines)
  and understand the code and results.
- [ ] Confirm current eligibility, deadline, and sign-up/submission forms from official sources.
- [ ] Review the AI-assistance disclosure and all attribution/license requirements.
- [ ] Prepare a walkthrough if requested: research question, representations,
  clean results, noise curves, and limitations.
- [ ] Complete any required forms and record actual submission confirmation.

A completed repository or passing synthetic test does not itself mean that a
real-data experiment or challenge submission has been completed.
