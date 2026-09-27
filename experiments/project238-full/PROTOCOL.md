# Full-source MATLAB feasibility run

Protocol recorded before the full-source run on 27 September 2026. This run
measures whether the verified datastore route can complete all official training
and test rows on the available CPU runner. It is separate from the original
50k/100k research comparisons and from the one-epoch 2,000-jet integration demo.

## Fixed selections and fit

- Data: checksum-verified Zenodo record 2603256; official partitions stay separate.
- Training: all 1,211,000 official training rows, source order preserved in files.
- Validation: first 10,000 rows of the official validation partition.
- Test: all 404,000 official test rows, untouched during fitting/model selection.
- Seed 101, CPU, one epoch, Adam learning rate 0.001, batch size 100. The batch
  size divides 1,211,000 exactly, so no training rows form a discarded partial
  minibatch. Training filenames are shuffled by MATLAB at the epoch boundary.
- Compact 32x32 grayscale CNN and train-only z-score input normalization from
  `trainProject238`; best validation network. No test-dependent model choice.
- Bounded HDF5-to-Parquet reads and tall image transforms use 2,000-row chunks.
  Extracting particle columns once per block avoids repeated table-index work;
  existing MATLAB pixel round-trip tests must still pass.

## Execution and evidence

The workflow first runs the small real-data demonstration, then attempts the
full run. The full job has a 330-minute limit and the MATLAB step a 305-minute
limit, leaving time to retain completed outputs if the step times out. It uses
an ordinary GitHub-hosted CPU runner; no paid GPU service or quantum hardware.

Record runner CPU count, available memory and disk; stage times; separate
MATLAB/Python process peak resident memory; source and checkpoint hashes;
per-test-jet row IDs, labels and probabilities; AUC and accuracy. Independently
verify all test rows and labels against the official HDF5 file and recalculate
AUC and accuracy from the saved probabilities. Preserve failed/partial runs as
such; a partial run must never be counted as completed full-data training.

Only the selected final CNN, training information, predictions and compact
resource/provenance outputs are retained as artifacts. Raw HDF5, Parquet and
image collections remain temporary. Epoch network checkpoints are included;
exact optimizer/RNG resume is not implemented.

## Interpretation limits

One epoch and one seed test execution and resource use. They do not demonstrate
convergence, optimized accuracy, seed uncertainty, a controlled architecture
comparison, GPU acceleration, quantum advantage or competition acceptance.
If execution succeeds, it establishes full-source training coverage for this
one-epoch CNN run only. The repository's default 12-epoch full-data configuration
remains unrun unless separately executed and verified. If execution fails or
exceeds the budget, retain the actual cause and completed stage measurements as
a concrete resource limitation; do not substitute a small-demo extrapolation
for an observed full-data result.
