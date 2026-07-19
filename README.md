# Trustworthy Top Quark Tagging: Robustness of CNN vs. GraphSAGE Under Detector Noise

A MATLAB deep-learning project submitted to the MathWorks Challenge Project Hub,
Project #238 — "Top Quark Detection with Deep Learning and Big Data."

**The question this project answers:** Two deep learning architectures — a CNN on
jet images, and a GraphSAGE graph neural network on jet particle graphs — can both
classify top-quark jets from Monte Carlo simulation. But real particle detectors are
never perfectly precise. Does either model's performance hold up when input
measurements get noisier, and does one degrade more gracefully than the other?

---

## Why this matters in the real world

After the high-luminosity upgrade of CERN's Large Hadron Collider, detectors will
generate petabytes of collision data every day. AI-based jet taggers like this one
are literally what decides, in real time, which collision events get kept and which
get discarded — more than 80% of all data is thrown away at the trigger level.
A tagger that fails silently when detector calibration drifts or sensor resolution
degrades could bias an entire physics analysis. Understanding how robust a model is
under realistic measurement imperfections is therefore not an academic question —
it is a deployment requirement. This project is the first systematic robustness
study of CNN vs. GraphSAGE architectures for top quark tagging under detector noise.

---

## What's original here (vs. the reference material)

The CNN and GraphSAGE architectures build on the techniques in MathWorks' own
File Exchange demo "GraphSAGE Classifier for Top Quark Tagging" (Colin Crovella,
2025) and MathWorks' documented graph-neural-network examples. On top of that
foundation, this project adds:

1. **Systematic robustness study** (`s6_robustness_test.m`) — both models, trained
   once on clean Monte Carlo data, are re-evaluated on the same test jets with
   synthetic detector noise injected at six levels (0% to 35% fractional momentum
   smearing). This is the main original contribution.

2. **AUC-based evaluation** (`s5_evaluate_baseline.m`) — accuracy alone depends on
   an arbitrary threshold; AUC is the standard metric in jet-tagging literature and
   is what makes the robustness curves in step 6 physically meaningful.

3. **Lightweight explainability** (`s7_explainability.m`) — permutation feature
   importance for GraphSAGE and radial occlusion for the CNN, revealing which
   physical features each model relies on most.

4. **GraphSAGE as a direct comparison point** — the graph-based model treats each
   particle as a node connected to its nearest angular neighbours, preserving exact
   particle positions rather than binning them into pixels like the CNN does. Running
   both on identical data makes the robustness comparison meaningful.

*Note on AI assistance: this project was built with the help of Claude (Anthropic)
as a coding and research assistant, in accordance with the MathWorks Generative AI
Guidelines. Every design decision, result, and interpretation has been verified
by running the code and examining the outputs.*

---

## Results

Trained on 50,000 jets from the Top Quark Tagging Reference Dataset
(Kasieczka et al., 2019), evaluated on a held-out test set of 7,500 jets.

| Metric | CNN | GraphSAGE |
|---|---|---|
| Accuracy (clean test set) | 51.5% | 83.7% |
| AUC (clean test set) | 0.866 | 0.908 |
| AUC at 10% detector noise | 0.856 | 0.893 |
| AUC at 35% detector noise | 0.736 | 0.771 |

**What these results show:**

GraphSAGE substantially outperforms the CNN on accuracy (83.7% vs 51.5%), confirming
that reasoning directly over particle relationships — rather than binning particles
into pixels — captures the jet's 3-prong top-quark decay structure more effectively.

On the robustness study, both models degrade under noise but neither collapses
catastrophically. GraphSAGE maintains higher absolute AUC at every noise level and
degrades slightly more gracefully up to 20% noise, suggesting the graph representation
is more tolerant of individual particle measurement errors than the image
representation — physically intuitive, since a single smeared particle affects only
its local graph neighbourhood rather than an entire pixel region.

The explainability analysis (GraphSAGE permutation importance) shows all four input
features — deltaEta, deltaPhi, log(pT), log(E) — contribute meaningfully, with
log(pT) and deltaPhi showing the largest drops when shuffled. This is physically
consistent: the azimuthal angle between particles and their transverse momenta are
the primary discriminators of the 3-prong top decay structure.

---

## File structure

All scripts are in the root of this repository. Helper functions are in the
`helpers/` subfolder (to be created locally — see setup below).

| File | Purpose |
|---|---|
| `run_all.m` | Single entry point — runs the full pipeline |
| `s1_prepare_data.m` | Load and preprocess the dataset |
| `s2_build_representations.m` | Build jet images (CNN) and jet graphs (GraphSAGE) |
| `s3_train_cnn.m` | Train the CNN baseline |
| `s4_train_graphsage.m` | Train the GraphSAGE model |
| `s5_evaluate_baseline.m` | Accuracy / AUC / ROC on clean test set |
| `s6_robustness_test.m` | **Original contribution** — AUC vs detector noise |
| `s7_explainability.m` | Feature importance and occlusion analysis |
| `buildJetImage.m` | Converts particle four-vectors to a 32×32 jet image |
| `buildJetGraph.m` | Converts particles to a k-NN graph (k=6 in eta-phi) |
| `graphSAGELayer.m` | One GraphSAGE mean-aggregator layer |
| `globalMeanPool.m` | Graph-level readout (average node features per jet) |
| `modelGraphSAGE.m` | Full GraphSAGE forward pass |
| `modelLossGraphSAGE.m` | Loss + gradients for custom training loop |
| `preprocessGraphMiniBatch.m` | Block-diagonal graph batching |
| `injectDetectorNoise.m` | Synthetic detector noise injection |
| `computeROC.m` | ROC curve and AUC (no toolbox required) |
| `initializeGlorot.m` | Glorot weight initialisation |
| `CONCEPTS.md` | Plain-English explanation of every idea used |

---

## Requirements

- MATLAB R2024a or later
- **Deep Learning Toolbox** (required)
- No GPU required (helpful for speed, not necessary)
- No Statistics and Machine Learning Toolbox needed — ROC/AUC computed in `computeROC.m`

---

## How to run

### Step 1 — Get the dataset

The Top Quark Tagging Reference Dataset (Kasieczka et al., 2019) is available at:
[https://doi.org/10.5281/zenodo.2603256](https://doi.org/10.5281/zenodo.2603256)

Download `test.h5`. Because this file is ~374 MB (pandas HDF5 format), convert
it to a MATLAB-readable `.mat` file using the provided Google Colab notebook
or this one-time Python snippet:

```python
import pandas as pd, scipy.io, numpy as np
df = pd.read_hdf('test.h5', key='/table', stop=50000)
cols = [f'{c}_{i}' for i in range(200) for c in ['E','PX','PY','PZ']]
scipy.io.savemat('jets_real_50k.mat', {
    'particleData': df[cols].values.astype(np.float32),
    'labels': df['is_signal_new'].values.astype(np.float32)
})
```

Place `jets_real_50k.mat` in a `data/` folder next to the scripts.

### Step 2 — Run in MATLAB

Set MATLAB's current folder to the project root, then:

```matlab
run_all
```

Or run each step individually in order: `s1` → `s2` → `s3` → `s4` → `s5` → `s6` → `s7`.

Results (CSVs and plots) are saved to a `results/` folder automatically.

---

## Attribution

- **Dataset:** G. Kasieczka, T. Plehn, J. Thompson, M. Russel, "Top Quark Tagging
  Reference Dataset," [Zenodo, 2019](https://doi.org/10.5281/zenodo.2603256), CC-BY 4.0.
- **GraphSAGE algorithm:** W. Hamilton, R. Ying, J. Leskovec, "Inductive Representation
  Learning on Large Graphs," NeurIPS 2017.
- **Jet-graph construction** (k-NN in eta-phi): H. Qu and L. Gouskos, "ParticleNet:
  Jet Tagging via Particle Clouds," Phys. Rev. D 101 (2020).
- **GraphSAGE training loop patterns** follow MathWorks' documented examples:
  [Node Classification Using GCN](https://www.mathworks.com/help/deeplearning/ug/node-classification-using-graph-convolutional-network.html)
  and [Multilabel Graph Classification Using GAT](https://www.mathworks.com/help/deeplearning/ug/multilabel-graph-classification-using-graph-attention-networks.html).
- **Reference implementation:** GraphSAGE Classifier for Top Quark Tagging,
  Colin Crovella, MATLAB Central File Exchange (2025).

---

## License

MIT — see [LICENSE](LICENSE).

## Author

Ali Mohamed — MathWorks Challenge Project Hub, Project #238, 2026.
