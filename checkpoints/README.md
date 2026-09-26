# Quick verification without retraining

The exact **seed 101, 50,000-training-jet CNN** checkpoint from the
[original study](../experiments/official-study/) and a 256-jet sample are included
in the repository (under 0.5 MB total). A normal clone downloads them without
Git LFS or expiring artifact links.

In MATLAB R2024a+ with Deep Learning Toolbox and the JVM enabled:

```matlab
verify_results
summarize_matlab
```

The first command checks hashes, loads the CNN checkpoint, runs the first
**256 official test jets**, and compares every probability with the saved
full-study predictions. It prints the sample accuracy and AUC. These are sample
scores, not the full 404,000-test-jet or three-seed mean scores.
The second command recreates the full study summary tables and figure in MATLAB.

[File identities and scope](manifest.json). The GraphSAGE and reference weights, and the other seeds, remain available
through the original study's artifact records. The sample is from the
[Top Quark Tagging Reference Dataset](https://doi.org/10.5281/zenodo.2603256)
by Kasieczka, Plehn, Thompson and Russel (CC BY 4.0), selected and converted to
MATLAB format. The code's MIT license does not replace the dataset license.

## The other two seed-101 models

Their permanent Git uploads are pending approval. They are already present in
the original public repository's Actions artifacts:

- GraphSAGE: `research-core-seed-101` in [run 35095505566](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35095505566).
- Reference: `research-reference-seed-101` in [run 35148856141](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35148856141).

The artifact records currently expire on 15 December 2026 and downloading them
may require GitHub sign-in. To verify all three models locally, place their
`models/graphsage_model.mat` and `models/winner_reference.mat` files in
`checkpoints/seed-101/`, then run `verify_results("all")`. The included manifest
checks exact hashes. CI reads these existing artifacts for verification; it
does not publish the two pending files into Git.
