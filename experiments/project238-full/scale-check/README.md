# Verified 100k datastore scaling check

The repaired MATLAB route completed on 28 September 2026:
[workflow 36388169653, attempt 2](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36388169653/attempts/2).
The first job attempt failed inside the MATLAB installer before project code ran.

This execution probe uses the first **100,000 training / 1,000 validation / 1,000
test jets** from the separate official partitions, seed 101, CPU, one epoch and
batch size 100. The log confirms all 1,000 training iterations completed. It is
separate from the earlier three-seed, 12-epoch 100k study with 404,000 test jets.

| Measurement | Observed value |
| --- | ---: |
| Official-source verification and Parquet preparation | 17.75 s |
| All selected image creation and coverage checks | 73.28 s |
| Training image write alone, rounded log timing | 66.5 s |
| Open and validate 100,000 training image filenames, rounded log timing | 3.3 s |
| CNN training | 124.94 s |
| Test inference and metrics | 1.01 s |
| MATLAB process peak resident memory | 2.00 GiB |
| Separate Python process peak resident memory | 0.48 GiB |
| Selected-test accuracy | 90.80% |
| Selected-test AUC | 0.966324 |

The process peaks were measured separately and must not be summed as an exact
combined peak. Raw HDF5 files were restored from the workflow cache; preparation
time includes source verification and conversion, not a fresh dataset download.

The cached file-list repair passes this larger execution check. The old full
attempt used a different dataset size and lacked detailed timing, so these
measurements do not establish a before/after speedup factor or prove that the
full-source run will finish. They justify a separately recorded full retry.

![MATLAB-exported ROC and confusion matrix](matlab_evaluation.png)

The workflow checked every selected official test label and row ID, MAT/CSV score
agreement, checkpoint hash and metrics. A separate local calculation used SciPy
average ranks for AUC and matched all labels to the previously verified official
test chunk. The confusion matrix, with true classes as rows and predicted classes
as columns in background/signal order, is `[[437,63],[29,471]]`.

[Per-jet scores](test_predictions.csv) · [MATLAB report](report.json) ·
[Workflow verification](verification.json) · [Independent local verification](local_verification.json) ·
[Artifact identity and log measurements](artifact.json).
The trained checkpoint and MAT predictions remain in the linked public workflow
artifact, whose expiry is recorded. This is a single-seed feasibility result,
not evidence of convergence, full-data training or reviewer acceptance.
