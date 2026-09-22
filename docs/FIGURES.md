# Figure gallery

Every figure below is linked to its experiment. A small demonstration and a
large evaluation can give different rankings; their pictures are not interchangeable.

## Final classification and noise study — verified

**Training:** 50,000 official training jets; 10,000 validation jets; three
training seeds per model; 12 epochs per fit.
**Plot renderer:** Python/Matplotlib, using verified MATLAB predictions.
These are research charts, not screenshots of the MATLAB desktop.

### Clean classification

All **404,000 official test jets**. Bars show mean accuracy and training-seed SD.

![Clean accuracy from the larger study](../experiments/official-study/clean_accuracy.png)

### Response to synthetic noise

The same **10,000 official test jets** at every noise level. Noise repeats are
averaged inside each training seed before calculating the plotted mean and SD.

![Noise sensitivity from the larger study](../experiments/official-study/noise_auc.png)

[Measured tables and uncertainty](RESULTS.md) ·
[Saved evidence](../experiments/official-study/)

## Explanation of the final models — pending verification

The new `explainSavedModels` analysis reuses the frozen CNN and GraphSAGE
checkpoints from all three training seeds. It plans to use the first 10,000
official test jets, three feature-shuffle streams, and five radial masks.
It does not train or select new models.

The first real-data execution stopped at its clean-prediction consistency guard.
A correction that preserves the original inference batch boundaries is prepared;
new final-model figures are not yet verified or published. The small-run
explanation figures below must not be presented as final-model explanations.

## Small demonstration — actual MATLAB exports

**Data:** the first 2,000 rows of the official training file, split into
1,400 training, 300 validation, and **300 internal test jets**.
**Training:** one seed per model, three epochs. **Renderer:** MATLAB R2024a.

All six original exports and their result tables are now saved in
[the small-run evidence folder](../experiments/small-benchmark/), so the figures
remain available after the original workflow artifact expires.

| File | What it shows |
| --- | --- |
| [roc_baseline.png](../experiments/small-benchmark/roc_baseline.png) | CNN and GraphSAGE ROC curves |
| [robustness_curves.png](../experiments/small-benchmark/robustness_curves.png) | Their accuracy and AUC under noise |
| [graphsage_feature_importance.png](../experiments/small-benchmark/graphsage_feature_importance.png) | AUC changes after graph features are shuffled |
| [cnn_radial_occlusion.png](../experiments/small-benchmark/cnn_radial_occlusion.png) | AUC when the outer image region is masked |
| [winner_roc.png](../experiments/small-benchmark/winner_roc.png) | Three-model ROC comparison |
| [winner_robustness.png](../experiments/small-benchmark/winner_robustness.png) | Three-model noise comparison |

<details>
<summary>View the six small-run MATLAB figures</summary>

![Small-run CNN and GraphSAGE ROC](../experiments/small-benchmark/roc_baseline.png)
![Small-run noise response](../experiments/small-benchmark/robustness_curves.png)
![Small-run graph feature sensitivity](../experiments/small-benchmark/graphsage_feature_importance.png)
![Small-run image occlusion](../experiments/small-benchmark/cnn_radial_occlusion.png)
![Small-run three-model ROC](../experiments/small-benchmark/winner_roc.png)
![Small-run three-model noise response](../experiments/small-benchmark/winner_robustness.png)

</details>

The explanation plots are sensitivity measurements. Shuffling correlated
features or masking image regions does not establish a causal physics explanation.
The [original run](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/34936617829)
also holds the small trained models and predictions; that artifact currently
expires on 15 October 2026. The committed pictures and tables do not expire.

## Original figures — historical, unchanged

These four MATLAB exports were preserved from the original project. Its CNN
probability mapping and ROC implementation had unresolved issues, and its README
disagreed with its CSVs. These are a historical record, not corrected benchmark evidence.

| Clean ROC | Noise response |
| --- | --- |
| ![Original ROC](../archive/original-results/roc_baseline.png) | ![Original robustness](../archive/original-results/robustness_curves.png) |

| Graph feature shuffling | Image radial occlusion |
| --- | --- |
| ![Original graph sensitivity](../archive/original-results/graphsage_feature_importance.png) | ![Original image sensitivity](../archive/original-results/cnn_radial_occlusion.png) |

[Original files and caveats](../archive/original-results/README.md)

MATLAB training-progress windows were disabled for automated execution.
The available images are exported research plots, not recordings of those windows.
