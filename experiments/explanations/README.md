# Explanations of the final core models

This is a completed **post-hoc sensitivity analysis** of the original final
CNN/GraphSAGE checkpoints. It is not additional training or a new clean benchmark.

- Training: 50,000 official training jets; 10,000 official validation jets.
- Checkpoints: seeds 101, 202, and 303 from [the completed core evaluation](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35095505566).
- Explanation sample: first 10,000 official test jets, source rows 0–9,999.
- Graph: four node features shuffled independently across all constituent nodes;
  three streams (11, 21, 31); original edges fixed.
- CNN: retained center-to-corner radius fractions 1, 0.75, 0.5, 0.35, and 0.2;
  pixels outside each disk set to zero; learned normalization fixed.
- Renderer: MATLAB R2024a. Error bars are training-seed sample SD.

[Successful execution](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35709427585)
and `artifact.json` identify the exact output archive, digest, and code revisions.
`metadata.json` records original checkpoint hashes and evaluation provenance.

![Graph feature sensitivity](graphsage_feature_importance.png)
![CNN radial occlusion](cnn_radial_occlusion.png)

## Read the measurements

`permutation_per_repeat.csv` holds all 36 graph measurements;
`occlusion_per_seed.csv` holds all 15 CNN measurements. The two summary tables
average shuffles inside each training seed, then report means and SDs across
three trained models. The shuffles are not independent training runs.

Graph AUC drops most after the angular-coordinate shuffles in this setting.
The CNN's AUC is nearly unchanged at retained radius 0.5, but falls to about
0.76774 at radius 0.2. These are sensitivities to artificial changes, not causal
physics conclusions. Radius is not retained image area. No test result was used
to tune or select model weights in this follow-up.

## Verification and numerical recovery

The first attempt used different inference batch boundaries and stopped at a
strict probability guard. Matching the original test chunks gave exact scores
for two seeds. The third retained a maximum float32-scale probability difference
of 3.159046173095703e-06 across executions. No clean classification or AUC changed.

The final guard requires probability differences below 64 times float32 epsilon
(0.0000076294), identical classifications, and AUC drift no greater than 0.000001.
It also checks test identities, labels, source partition hashes, and model data
identities. This tolerance is an explicit numerical check, not a claim of
bit-for-bit reproducibility across all hardware. Metadata reports the actual
drift for every seed; observed AUC drift is zero for both models in all three.

`verification.json` records the independent Python average-rank AUC check of
every perturbation measurement and summary, the identity mask, the clean-score
comparison, and all six original checkpoint hashes. No model weights changed.

## Reproduce

Download the `final-model-explanations` artifact from run **35709427585** into
`results/explanations`. Download the three `research-core-seed-*` artifacts from
run **35095505566** into separate folders under `downloads`.

```bash
python -m pip install -r requirements.txt
python scripts/verify_explanations.py --input results/explanations --core-input downloads
```

To recalculate the predictions in MATLAB, prepare the source data with
`python scripts/prepare_official.py`, then run:

```matlab
setupProject
explainSavedModels({'downloads/research-core-seed-101', ...
    'downloads/research-core-seed-202','downloads/research-core-seed-303'}, ...
    'data/official','results/new-explanations')
```

Alternatively, run the **Explain saved final models** workflow manually.
It can rebuild expired prepared input data from the public source. Recreated
MAT headers can change file hashes; the source partition hashes, row intervals,
current chunk hashes, and frozen clean predictions are still checked.

The explanation prediction archive expires on 21 December 2026; the source core
checkpoint artifacts expire on 15 December 2026. Save them before those dates to
reuse these exact fits later. Figures and compact evidence here remain in Git.
Data: [Top Quark Tagging Reference Dataset](https://doi.org/10.5281/zenodo.2603256),
Kasieczka, Plehn, Thompson, and Russel; CC BY 4.0.
