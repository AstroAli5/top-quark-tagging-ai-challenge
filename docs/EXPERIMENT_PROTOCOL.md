# Official-partition study: protocol frozen before evaluation

The primary question is how the compact CNN and particle-graph GraphSAGE change
under synthetic momentum smearing. A credited ResNeXt-SE adaptation provides an
additional image baseline. All three models see the same jets, labels, official
partitions, and perturbations. This compares complete pipelines; architecture and
representation effects are not isolated by this experiment.

## Data and budgets

Use the [Top Quark Tagging Reference Dataset](https://doi.org/10.5281/zenodo.2603256),
by Kasieczka, Plehn, Thompson, and Russel, released under CC-BY-4.0. The publisher
provides separate training, validation, and test files. Preserve that assignment.

| Setting | Frozen choice |
| --- | --- |
| Training | First 50,000 rows of official train.h5 |
| Validation | First 10,000 rows of official val.h5 |
| Clean testing | Every row of official test.h5 (published count: 400,000) |
| Training seeds | 101, 202, 303; same fixed data across runs |
| Epoch budget | 12 per model; best validation-loss checkpoint |
| CNN | Existing 32-pixel compact CNN, Adam at 0.001, batch 64 |
| GraphSAGE | Existing three-layer mean aggregator, hidden width 32, Adam at 0.01, batch 32 |
| Reference | 37-pixel aligned 12-channel images, four radial features, widths 32/64/128, four groups, Adam at 0.005, batch 64 |
| Reference schedule | Learning rate multiplied by 0.3 every four epochs; L2 0.0001 |
| Noise study | First 10,000 official test rows, identical for every model/run |
| Smearing strengths | 0, 0.05, 0.10, 0.20, 0.35 |
| Noise realizations | Independent streams seeded 7, 17, 27; shared across models/training seeds |
| Classification threshold | Fixed P(top) >= 0.5 |

The official training set contains 1.2 million rows and validation contains 400,000.
This is a **training-subset study with a full official test evaluation**, not a
full-data state-of-the-art benchmark. Equal epochs do not mean equal compute;
training seconds are reported. No hyperparameter search or test-guided retuning is
part of this frozen run. The previous 2,000-jet demonstration used an internal
holdout from train.h5 and is separate from this evaluation.

## Leakage and reproducibility controls

The downloader checks the publisher's MD5 and byte count for each partition.
The conversion manifest records SHA-256 values, row intervals, and prepared-file
checksums. `fitting.mat` contains only training and validation rows with explicit
partition labels. The test set is read only after all three checkpoints are fixed.
Training-only normalization is saved with the reference model. The baseline CNN
also learns its normalization during training. No test labels select checkpoints.

Test chunks must cover consecutive, unique source rows. The evaluator aborts if a
test jet cannot be processed, rather than silently removing it. Per-jet
probabilities, labels, source rows, configuration, code commit, and runtime are
saved. Summarization independently recomputes AUC and accuracy from predictions
and rejects mismatched data, code versions, or missing seeds.

Noise multiplies each momentum component by 1 + sigma Z and recomputes energy from
the original nonnegative mass-squared estimate. A separate random stream makes
noise independent of inference batching. Each nonzero strength reuses the same
standard-normal draws at its corresponding realization seed. Clean predictions
on the same 10,000 jets provide the zero-noise baseline.

## Reporting and uncertainty

Report all three training runs, their mean and sample standard deviation, and
95% Student-t intervals across those three runs. These intervals measure seed
variation on the fixed test data. With only three runs, the intervals can be wide;
they do not establish universal architecture superiority.

Also report fixed-model test-sampling intervals: Wilson intervals for accuracy
and normal intervals using tie-aware positive/negative placement values for AUC.
For paired AUC differences, preserve the shared test jets and covariance between
models. Average placement differences across the fitted training runs to obtain
the conditional test interval for the mean difference. Training-seed and
test-sampling intervals are reported separately, not combined into one claim.
Method context: [DeLong et al. (1988)](https://pubmed.ncbi.nlm.nih.gov/3203132/)
and [SciPy's Student-t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html).

Average the three noise realizations within each training seed before reporting
between-training-seed variation. Nine combinations are not treated as nine
independent training runs. Compare noisy results with the zero-noise results on
the same 10,000 jets, not with clean results on all 400,000 jets.

## Limits

The data are simulated. Smearing is a stress test, not a calibrated detector
response; no hardware deployment claim follows. The reference keeps 35
constituents while the original models retain up to 200, so it has a different
information budget. Specific improvements cannot be attributed to attention,
alignment, or radial features without separate ablations. The question here is
whether the observed clean/noisy model ranking persists across repeated runs.
