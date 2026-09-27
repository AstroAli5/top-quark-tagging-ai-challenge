# Verified MATLAB datastore demonstration

This is an execution check of the Project 238 data route, separate from the
larger CNN/GraphSAGE research study. MATLAB R2024a calls Python with `pyrun` in
`OutOfProcess` mode, verifies the official HDF5 files, writes bounded Parquet
chunks, transforms a `tall` table into lossless single-precision TIFFs, and trains
a CNN using folder-labelled image datastores.

Run [36280538500](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36280538500)
passed on 26 September 2026. The selected official prefixes contain 2,000
training, 500 validation and 1,000 test jets; seed 101, one epoch, CPU. Batch size
64 gives 31 complete training minibatches; `trainnet` discards the final partial
training minibatch. All 1,000 selected test rows are evaluated.

| Measurement | Observed value |
| --- | ---: |
| Test accuracy | 85.4% |
| Test AUC | 0.915692 |
| HDF5 download, verification and Parquet preparation | 81.06 s |
| Tall image creation, including coverage checks | 31.26 s |
| CNN training | 10.36 s |
| Test inference and metrics | 1.03 s |
| MATLAB process peak resident memory | 2.04 GiB |
| Separate Python process peak resident memory | 0.97 GiB |

The two process peaks were not measured simultaneously and must not be reported
as an exact combined peak. Small-run timings include startup costs and are not
reliable full-dataset estimates.

![MATLAB-exported ROC and confusion matrix](matlab_evaluation.png)

The workflow independently checked MAT/CSV score agreement, official source-file
hash and test labels, all 1,000 source row IDs, checkpoint hash, accuracy and AUC.
A second local calculation used SciPy average ranks for AUC and matched labels
against the previously verified official test chunk. See [verification](verification.json),
[per-jet scores](test_predictions.csv), [run report](report.json) and
[archive identity](artifact.json). The original checkpoint is in the linked
workflow artifact; its expiry is recorded.

This result proves the data route executes. It is not a full-training or
full-test result, is not a controlled comparison with the prior pipelines, and
does not demonstrate convergence or a competition placement. Reproduce with
the short `run_project238` command in the main README.
