# Quick verification without retraining

The exact **seed 101, 50,000-training-jet CNN, GraphSAGE and ResNeXt-SE reference**
checkpoints from the [original study](../experiments/official-study/) and a
256-jet sample are included in this repository (under 1 MB total). A normal clone
downloads all four files without Git LFS, sign-in or expiring artifact links.

In MATLAB R2024a+ with Deep Learning Toolbox and the JVM enabled:

```matlab
run_submission
```

The command checks hashes, loads all three checkpoints, runs the first
**256 official test jets**, and compares every probability with the saved
full-study predictions. It prints each model's sample accuracy and AUC. These
are sample scores, not the full 404,000-test-jet or three-seed mean scores.

The check allows at most `64*eps(single)` (about 0.00000763) probability
difference, matching the previously audited explanation study's floating-point
budget. It also requires unchanged class decisions and AUC agreement within
0.000001, and prints the observed differences separately for every model.

It then recreates the recorded full-study summary tables, paired seed
comparisons, and accuracy/AUC/noise figure in MATLAB. The command prints its
elapsed time; runtime depends on the machine. It needs no Python, new data
download or fitting. CI checks the included files directly.

Use `verify_results("all")` for just the three-model restoration check or
`summarize_matlab` for just the study report. `verify_results` without an argument
retains the smaller CNN-only check.

## File identities and provenance

[The manifest](manifest.json) records each file's exact SHA-256, size and scope.
The two newly published files are byte-for-byte copies of the original verified
artifacts; no model was retrained or replaced:

- CNN and GraphSAGE: `research-core-seed-101` in [run 35095505566](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35095505566).
- Reference: `research-reference-seed-101` in [run 35148856141](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35148856141).

The original artifacts, which also hold the other seeds and full predictions,
currently expire on 15 December 2026. The four files committed here remain in
Git after that date. See the study's artifact records for the larger archives.

The sample is from the [Top Quark Tagging Reference Dataset](https://doi.org/10.5281/zenodo.2603256)
by Kasieczka, Plehn, Thompson and Russel (CC BY 4.0), selected and converted to
MATLAB format. The code's MIT license does not replace the dataset license.
