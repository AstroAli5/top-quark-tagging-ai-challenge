# Learning from the 2025 winner

MathWorks names **Adit Shah** as its 2025 AI Challenge first-place winner for a
top-quark-tagging project. The official announcement links to his
[public repository](https://github.com/adit-smoak/Top-Quark-Tagging-Using-Deep-CNN).
Sources: [2025 winners](https://www.mathworks.com/academia/students/competitions/student-challenge/ai-challenge/2025-winners.html)
and the [organizer's announcement](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/discussions/142).

His approach combines aligned, multichannel jet images with radial statistics,
a ResNeXt-style CNN, and squeeze-and-excitation attention. His README reports
90.87% test accuracy for a 90,000-sample configuration. This is an **author-reported
result**, not a result reproduced by this repository. The author's
[deployment note](https://github.com/adit-smoak/Top-Quark-Tagging-Using-Deep-CNN/blob/main/deploy/NOTE.txt)
says the FPGA scripts were untested without access to a board.

## What this repository adds

The new reference is independently written, credits Shah's project, and applies
those ideas to Ali Mohamed's existing synthetic-noise research question. It does
not include the winner's code, trained weights, report, or images. It is a compact
adaptation, not an exact reproduction or a claim of matching his score.

| Component | Implementation here |
| --- | --- |
| Constituent selection | Highest-pT 35 valid particles, sorted explicitly |
| Coordinates | Pseudorapidity and wrapped azimuth relative to the hardest particle |
| Alignment | Continuous rotation toward the second-hardest particle, then an energy-based reflection applied to one jet |
| Image | 37 by 37 pixels spanning -1.6 to 1.6 in both aligned coordinates |
| Channels | Count; sums of pT, absolute pz, and E; E/pT variance, skewness, and kurtosis; mean E; a radius-weighted momentum proxy |
| Radial features | Skewness and Pearson kurtosis of E and pT sums in square rings, before dataset normalization |
| Normalization | Signed log compression of image channels; channel/global statistics fitted only on training jets |
| Network | Three grouped residual blocks of widths 32, 64, 128, four groups, channel attention, pooled image/global-feature fusion, softmax |
| Optimization | Adam, 12 epochs, batch 64, initial learning rate 0.005, factor 0.3 every four epochs, L2 0.0001 |
| Comparison | Same train/validation/test indices and paired noisy four-vectors as the original CNN and GraphSAGE |

The radius-weighted momentum channel is a geometric proxy, not physical angular
momentum. Pixel moments use actual occupied constituents; padding zeros do not
enter their calculation. Single-particle and constant samples have zero
standardized moments by convention. Empty jets produce finite zero features;
the shared data pipeline excludes jets with fewer than three valid constituents.

Images are generated on demand through a datastore. This avoids storing a full
12-channel image tensor, although the original pipeline still holds jets and
baseline representations in memory. Sorting, continuous alignment, moment
conventions, log compression, normalization, network size, split protocol, and
training subset can all differ from the winning implementation. These choices
must be considered when interpreting any accuracy difference.

## Run all three models

Prepare data using the main README, then run:

```matlab
cfg = projectConfig;
run_all(cfg)
run_winner_comparison(cfg)
```

If `run_all(cfg)` has already completed with the same data, skip that call.
`run_winner_comparison` trains the reference and evaluates all three models.
`evaluate_winner_comparison(cfg)` repeats only the evaluation using saved models
and their training-time preprocessing. Dataset IDs reject stale checkpoints.

New outputs in `results/` are `winner_comparison.csv`, `winner_robustness.csv`,
`winner_predictions.mat`, `winner_roc.png`, `winner_robustness.png`, and
`winner_metadata.json`. The reference checkpoint is `models/winner_reference.mat`.
The metadata records configurations for all three models and actual split counts.
The original historical artifacts in the repository root remain historical.

For a first real-data feasibility run:

```bash
python scripts/download_dataset.py
python scripts/convert_dataset.py --input data/train.h5 --output data/jets_real_2000.mat --max-jets 2000
```

```matlab
run_small_benchmark
```

This selects 2,000 rows from the official **training** file, makes the internal
70/15/15 split, and trains each model for three epochs. It establishes whether
the implementation runs on actual data; it does not establish a competitive
score. Equal epochs do not imply equal compute. The download is about 1.04 GB,
even when only 2,000 rows are converted; the published checksum is verified.

## What remains for a submission-quality scientific claim

Record actual metrics and the code commit; do not copy the winner's score into
your results. Use the official dataset partitions for an official benchmark,
tune only on validation data, repeat several training/noise seeds, quantify
uncertainty, and report runtime and training budget. Ablations should separately
test alignment, added channels, radial features, and attention to establish which
changes help. The synthetic smearing experiment cannot validate detector or
FPGA deployment.

## Architecture references

- Xie et al., [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431), CVPR 2017.
- Hu et al., [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507), CVPR 2018 / TPAMI.
- MathWorks, [Train Network on Image and Feature Data](https://www.mathworks.com/help/deeplearning/ug/train-network-on-image-and-feature-data.html).
- Dataset: [Top Quark Tagging Reference Dataset, record 2603256](https://doi.org/10.5281/zenodo.2603256).
