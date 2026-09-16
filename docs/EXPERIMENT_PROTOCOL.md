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
| Clean testing | Every row of official test.h5 (404,000 rows in the verified file) |
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

The publisher describes approximately 1.2 million training and 400,000 validation
and test jets. The verified files contain 1,211,000 training, 403,000 validation,
and 404,000 test rows. The frozen choice is to use every test row; the exact count
comes from the source file, not a rounded description.
This is a **training-subset study with a full official test evaluation**, not a
full-data state-of-the-art benchmark. Equal epochs do not mean equal compute;
training seconds are reported. No hyperparameter search or test-guided retuning is
part of this frozen run. The previous 2,000-jet demonstration used an internal
holdout from train.h5 and is separate from this evaluation.

## Leakage and reproducibility controls

The downloader checks the publisher's MD5 and byte count for each partition.
The conversion manifest records SHA-256 values, row intervals, and prepared-file
checksums. `fitting.mat` contains only training and validation rows with explicit
partition labels. Each model's test set is read only after its checkpoint is fixed.
All architecture, training, and evaluation choices are fixed before any test result
is inspected.
Training-only normalization is saved with the reference model. The baseline CNN
also learns its normalization during training. No test labels select checkpoints.

Test chunks must cover consecutive, unique source rows. The evaluator aborts if a
test jet cannot be processed, rather than silently removing it. Per-jet
probabilities, labels, source rows, configuration, code commit, and runtime are
saved. Summarization independently recomputes AUC and accuracy from predictions
and rejects mismatched data, code versions, or missing seeds.

The final analysis also checks every noise metric against its saved predictions,
verifies zero-noise identity on the same sample, and rejects missing or duplicated
noise rows. Stored float32 scores can round near ties and the classification
threshold; adjacent representable values bound that rounding when checking the
original full-precision metrics. The analysis script's SHA-256 is saved in its report.

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
the same 10,000 jets, not with clean results on all 404,000 jets.

## Limits

### Execution recovery, 16 September 2026

The original three-hour jobs completed all CNN and GraphSAGE training, then timed
out during reference training. The completed core checkpoints are restored from
run 35042290356 together with its original prepared data. The recovery verifies
the fitting-file checksum and seed/settings before evaluating those checkpoints.
Their training times are reconstructed from stage start/save log timestamps.

The reference runs separately with a six-hour job allowance, epoch checkpoint
files, and visible training progress. Its cached datastore now reads one full
mini-batch per call, following [MathWorks' performance guidance](https://www.mathworks.com/help/deeplearning/ug/optimize-datastores-performance.html).
These are execution changes: the data, seeds, architecture, optimizer settings,
12-epoch budget, and validation-based selection are unchanged. An interrupted
reference is not counted as a completed trained model. Core and reference outputs
are joined only after checks of source hashes, settings, labels, and source rows.
The completed core analysis can be inspected independently of the slower reference.

A full input audit found three one-particle and five two-particle test jets, with
no empty jets among all 404,000 rows. An initially over-strict reader rejected
those eight rows. The corrected reader retains every nonempty jet: the existing
graph implementation already uses min(k,n-1) neighbors and supports isolated
nodes. No model weights or test-set rows are changed. Recovery evaluates saved
checkpoints; summaries record both training and evaluation revisions. If model
families were evaluated at different commits, combination requires an empty Git
diff for their shared model, representation, and evaluation implementation.

The data are simulated. Smearing is a stress test, not a calibrated detector
response; no hardware deployment claim follows. The reference keeps 35
constituents while the original models retain up to 200, so it has a different
information budget. Specific improvements cannot be attributed to attention,
alignment, or radial features without separate ablations. The question here is
whether the observed clean/noisy model ranking persists across repeated runs.
