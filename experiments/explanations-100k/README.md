# Verified explanations of the 100k models

[MATLAB run](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36158770932),
25 September 2026. This reuses the six frozen CNN/GraphSAGE checkpoints trained
on 100,000 jets with seeds 101/202/303. Explanations use the first 10,000 official
test jets, three feature-permutation streams, and five radial masks. No model
was retrained or selected from these results.

| Graph feature shuffled, fixed edges | Mean AUC drop | Training-seed SD |
| --- | ---: | ---: |
| deltaEta | 0.11071 | 0.01045 |
| deltaPhi | 0.13822 | 0.00607 |
| log(pT) | 0.10069 | 0.01814 |
| log(E) | 0.01220 | 0.00195 |

The angular-feature shuffles have the largest mean effects here. Shuffling
breaks correlations and keeps the graph edges fixed, so these are sensitivity
measurements rather than causal feature importances.

![100k GraphSAGE feature shuffling, exported by MATLAB](graphsage_feature_importance.png)

| CNN radius fraction retained | Mean AUC | Training-seed SD |
| --- | ---: | ---: |
| 1.00 | 0.97137 | 0.00156 |
| 0.75 | 0.97137 | 0.00156 |
| 0.50 | 0.97122 | 0.00156 |
| 0.35 | 0.94918 | 0.00157 |
| 0.20 | 0.74756 | 0.01049 |

Half the center-to-corner radius preserves almost all clean AUC. Stronger
central cropping reduces it substantially. Radius fractions are not area
fractions; normalization stays fixed to the trained model.

![100k CNN radial occlusion, exported by MATLAB](cnn_radial_occlusion.png)

Both PNGs are **actual MATLAB R2024a exports**. Error bars show SD across the
three fitted training seeds after averaging permutation repeats within each
fit. These are not nine independently trained models.

## Verification and reproduction

All 36 permutation and 15 occlusion measurements were independently recomputed
with an average-rank AUC calculation. Six checkpoint hashes match the original
100k artifacts. Restored clean probabilities matched the saved source predictions
**exactly**, with zero changed decisions. The locally recomputed verification
record equals the workflow record.

[Protocol](PROTOCOL.md) · [feature summary](feature_summary.csv) ·
[permutation repeats](permutation_per_repeat.csv) ·
[occlusion summary](occlusion_summary.csv) · [per-seed occlusion](occlusion_per_seed.csv) ·
[metadata and checkpoint provenance](metadata.json) · [verification](verification.json) ·
[artifact digests](artifacts.json).

The **Explain 100k models and test graph edges** workflow downloads the source
checkpoints from run 35842565638 and prepares only the required test prefix
from the verified public source. To reproduce locally, download the three
`scaling-100000-seed-*` artifacts and arrange their saved `models/` and `results/`
folders under `downloads/research-core-seed-101/`, `-202/`, and `-303/`.

```bash
python scripts/prepare_explanation_sample.py --metadata downloads/research-core-seed-101/results/metadata.json --output data/explanation-100k
```

In MATLAB:

```matlab
setupProject;
explainSavedModels({'downloads/research-core-seed-101','downloads/research-core-seed-202','downloads/research-core-seed-303'},'data/explanation-100k','results/explanations-100k');
```

Then independently verify:

```bash
python scripts/verify_explanations.py --input results/explanations-100k --core-input downloads
```

The explanation artifact stores predictions as well as tables/plots. Its
recorded expiry is 24 December 2026; committed figures and tables remain
available afterward. The [original 50k explanation report](../explanations/)
remains unchanged and describes different trained checkpoints.
