# Verified five-seed core study

Completed and independently checked on 23 September 2026. The
[additional-seeds run](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35841193310)
adds CNN and GraphSAGE seeds 404 and 505 to the original 101/202/303 fits.
Training uses 50,000 official training jets, 10,000 validation jets, and
12 epochs; clean evaluation covers all 404,000 official test jets.

| Model | Accuracy, mean ± seed SD | AUC, mean ± seed SD |
| --- | ---: | ---: |
| CNN | 91.063% ± 0.040 percentage points | 0.96899 ± 0.00065 |
| GraphSAGE | 86.583% ± 0.361 percentage points | 0.92707 ± 0.00092 |

The core finding persists: CNN has the higher clean AUC, while GraphSAGE has
the higher mean AUC at 20% and 35% synthetic smearing. On the same first 10,000
test jets, mean AUC at 35% smearing is 0.67142 for CNN and 0.73644 for GraphSAGE.
Repeated noise draws are averaged within each fitted model before computing
training-seed SD. Five seeds are still a modest study on one fixed test set.

![Clean accuracy across five seeds](clean_accuracy.png)
![Paired noise sensitivity across five seeds](noise_auc.png)

## Evidence and compatibility

- [Protocol](PROTOCOL.md), [complete report](report.json), [clean summary](clean_summary.csv),
  [noise summary](noise_summary.csv), [per-seed clean](per_seed_clean.csv),
  [per-seed noise](per_seed_noise.csv), and [paired model AUC](paired_auc.csv).
- [Artifact IDs, hashes and expiry dates](artifacts.json). The new archive's
  downloaded SHA-256 matches GitHub's recorded digest. It contains four new
  checkpoints, saved predictions and configurations. Committed tables and
  figures remain available after the downloadable archives expire.
- Every clean prediction and all 150 noise-metric rows were checked against
  the saved probabilities, including row order, labels, zero-noise agreement
  and float32 rounding bounds.
- `report.json → compatibility.runs` records **each seed's** source commits,
  workflow and implementation fingerprints. The top-level `code_commit` and
  `workflow_run` identify the first original run, not all five runs.
- Source files, partition selections, configuration, MATLAB version and Git
  blob identities match the reviewed baseline. The original and core-only run
  controllers were reviewed separately. Regenerated MAT headers have different
  timestamps and file hashes; the original source checksums and row choices match.

The original [three-model study](../official-study/) remains unchanged. The
ResNeXt-SE reference still has three seeds. The existing explanation study
covers the original three core seeds, not the two additional fits.

## Reproduce the summary

Download the three `research-core-seed-*` artifacts from the
[original core evaluation](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35095505566)
and `additional-core-seeds` from the new run. Arrange exactly five folders as
`audit/seed-101/results/`, through `audit/seed-505/results/`, each with its saved
metadata, clean/noise metrics and prediction MAT files. Keep models separately
if only recalculating the summary.

Use a full Git clone. The compatibility check reads the immutable training and
evaluation commits, including the original PR merge snapshots:

```bash
git fetch origin 0f7ae539735f1aa5ee113c84bf36a17f4f366e2c
git fetch origin d3b5eb286c87e7f8a6d8cb7f40a72b196ad13177
git fetch origin 9c767326cdfab32dc9bb6cdc6ff43c341493b97b
python -m pip install -r requirements.txt matplotlib
python scripts/summarize_experiment.py --input audit --output results/seed-extension --five-seed-core
```

The manual **Additional core seeds** workflow reproduces seeds 404/505 using
the pinned implementation. New training can vary with hardware or library
behavior; recalculating saved predictions reproduces the recorded measurements.
