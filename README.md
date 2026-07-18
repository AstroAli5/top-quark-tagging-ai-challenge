# Trustworthy Top Quark Tagging: Robustness of CNN vs. GraphSAGE Under Detector Noise

A MATLAB deep-learning project for the MathWorks AI Challenge / Challenge
Project Hub Project #193 ("Top Quark Detection with Deep Learning and Big
Data"), built on the same dataset and dual CNN/GraphSAGE architecture as
MathWorks' own **GraphSAGE Classifier for Top Quark Tagging** demo
(File Exchange #181442, Colin Crovella, 2025).

**The question this project answers:** two deep learning architectures —
a CNN on jet images, and a GraphSAGE graph neural network on jet
particle graphs — can both classify top-quark jets from Monte Carlo
simulation. But real particle detectors are never perfectly precise. Does
either model's performance hold up when the input measurements get
noisier, and does one degrade more gracefully than the other?

## Scope note (read this first)

This started as a much bigger plan — a full literature review, both
explainability and robustness studies at full depth, and a 6-week
research-style timeline. It was deliberately rightsized down to **one
well-executed original contribution (robustness to detector noise) with
lightweight explainability as a bonus**, because the challenge's actual
100-point rubric (25 each for real-world applicability, novelty, code/doc
quality, and depth) rewards a focused, well-executed, well-explained
project — not literature-review volume or research-paper polish. Same
core idea, sized to what's actually finishable. See
`SUBMISSION_CHECKLIST.md`.

## What's original here (vs. the reference material)

The CNN and GraphSAGE architectures are based on the same well-established
techniques MathWorks' own File Exchange demo uses for this exact problem
(see [Attribution](#attribution)). On top of that shared foundation, this
project adds:

1. **A systematic robustness study** (`s6_robustness_test.m`) — both
   models, trained once on clean data, are re-evaluated on the same test
   jets with synthetic detector noise injected at six levels (0% to 35%).
   This is the main original contribution: it's a genuinely open question
   the reference material doesn't address, and it's directly relevant to
   real-world deployment (detectors are never perfectly precise).
2. **Quantitative evaluation with AUC, not just accuracy**
   (`s5_evaluate_baseline.m`) — accuracy alone depends on an arbitrary
   0.5 threshold; AUC is the standard way this field actually reports
   tagger performance, and it's what makes the robustness curves in step
   6 meaningful.
3. **Lightweight explainability** (`s7_explainability.m`, stretch) —
   permutation feature importance for GraphSAGE, and radial occlusion for
   the CNN. Simple, honest, and interpretable, rather than a heavyweight
   dedicated GNN-explainability framework that wasn't realistic to build
   correctly in the available time.
4. **An independent implementation.** I (working with Claude) could not
   access the actual source code of the File Exchange submission — it's
   behind a MathWorks account download wall — so this is a from-scratch
   implementation of the same idea, grounded in MathWorks' own *published,
   documented* graph-neural-network examples (see Attribution), not a
   copy of unseen code.

## Repository structure

```
top-quark-tagging-ai-challenge/
├── README.md                    <- you are here
├── LICENSE                      <- MIT
├── SUBMISSION_CHECKLIST.md      <- maps this repo to the Challenge rules
├── docs/
│   └── CONCEPTS.md              <- plain-English explanation of every idea used
├── src/
│   ├── s1_prepare_data.m        <- STEP 1: load a subset of the dataset (needs manual download)
│   ├── s2_build_representations.m <- STEP 2: build jet images + jet graphs, train/val/test split
│   ├── s3_train_cnn.m           <- STEP 3: train the CNN baseline
│   ├── s4_train_graphsage.m     <- STEP 4: train the GraphSAGE baseline (custom training loop)
│   ├── s5_evaluate_baseline.m   <- STEP 5: accuracy/AUC/ROC on clean test data
│   ├── s6_robustness_test.m     <- STEP 6: ORIGINAL CONTRIBUTION — accuracy/AUC vs detector noise
│   ├── s7_explainability.m      <- STEP 7: stretch — feature importance / occlusion
│   └── helpers/
│       ├── initializeGlorot.m
│       ├── buildJetImage.m
│       ├── buildJetGraph.m
│       ├── graphSAGELayer.m
│       ├── globalMeanPool.m
│       ├── modelGraphSAGE.m
│       ├── modelLossGraphSAGE.m
│       ├── preprocessGraphMiniBatch.m
│       ├── injectDetectorNoise.m
│       └── computeROC.m
├── data/       <- created by s1/s2 (dataset + your manual download goes here)
├── models/     <- created by s3/s4 (trained networks)
└── results/    <- created by s5/s6/s7 (CSVs + plots)
```

## Requirements

- MATLAB (a recent version; uses `trainnet` for the CNN and a custom
  `dlfeval`/`adamupdate` training loop for GraphSAGE)
- Toolboxes: **Deep Learning Toolbox**. (No Statistics and Machine
  Learning Toolbox needed — ROC/AUC is computed directly in
  `helpers/computeROC.m`.)
- **A Python 3 installation with pandas**, visible to MATLAB via `pyenv`
  — needed only for `s1_prepare_data.m`, to read this dataset's
  pandas-HDF5 file format. See the note in that script for why.
- No GPU required at this project's scale (helps with speed, not
  necessary)
- If you don't have MATLAB yet: MathWorks offers free student licenses —
  see [Get MATLAB](https://www.mathworks.com/academia/students.html) or
  check if your university already provides one.

## How to run it, in order

1. **Download the data manually first.** Go to
   [doi.org/10.5281/zenodo.2603256](https://doi.org/10.5281/zenodo.2603256)
   (Kasieczka, Plehn, Thompson & Russel, 2019 — CC-BY 4.0) and download
   `test.h5` (~600 MB). Put it in this project's `data/` folder. (A
   Kaggle mirror also exists if you prefer:
   [kaggle.com/datasets/algomorgomor/top-quark-tagging-reference-dataset](https://www.kaggle.com/datasets/algomorgomor/top-quark-tagging-reference-dataset).)
2. Open MATLAB, set your **Current Folder** to `src/`, then run each
   script in order:
   - `s1_prepare_data` — loads a subset (8,000 jets by default) via
     Python/pandas. **This is the step most likely to need debugging
     together** — see the honesty note in that script.
   - `s2_build_representations` — builds jet images and jet graphs,
     splits into train/val/test.
   - `s3_train_cnn` and `s4_train_graphsage` — train both models (can run
     in either order).
   - `s5_evaluate_baseline` — accuracy/AUC on clean test data.
   - `s6_robustness_test` — the main original contribution.
   - `s7_explainability` — stretch goal.

## Results

*Fill this in with your own numbers after running the scripts above —
don't guess or estimate them.*

| Metric | CNN | GraphSAGE |
|---|---|---|
| Accuracy (clean test set) | 51.5% | 83.7% |
| AUC (clean test set) | 0.866 | 0.908 |
| AUC at 10% detector noise | 0.856 | 0.893 |
| AUC at 35% detector noise | 0.736 | 0.771 |

A few sentences on what you noticed — did one model degrade faster than
the other? Did the feature-importance results match physical intuition?
— will do more for your score than the numbers alone. The judging
criteria explicitly reward depth of understanding, not just a working
script.

## A note on how this was built

I have not been able to run any of this code myself — I don't have
MATLAB, a matching Python/pandas environment, or this dataset available
where I work. The core CNN and custom graph-training-loop patterns are
grounded in MathWorks' own published, verified documentation (see
Attribution below), but this is meaningfully more complex code than a
typical first MATLAB project (a graph neural network plus a
research-data pipeline spanning two languages), so expect a real
debugging pass — especially in `s1_prepare_data.m` and
`s4_train_graphsage.m` — rather than everything running clean on the
first try. That's normal, not a sign anything is fundamentally wrong.

## Attribution

This project's architecture choices follow MathWorks' own published work
on this exact problem:

- **GraphSAGE Classifier for Top Quark Tagging**, Colin Crovella, MATLAB
  Central File Exchange (2025), the demo this project is directly
  inspired by (its source code was not accessible to me — see above).
- The GraphSAGE custom-training-loop mechanics (block-diagonal batching,
  `minibatchqueue`, `dlfeval`/`adamupdate`, graph-level pooling readout)
  follow the patterns in MathWorks' documented examples [Node
  Classification Using Graph Convolutional
  Network](https://www.mathworks.com/help/deeplearning/ug/node-classification-using-graph-convolutional-network.html)
  and [Multilabel Graph Classification Using Graph Attention
  Networks](https://www.mathworks.com/help/deeplearning/ug/multilabel-graph-classification-using-graph-attention-networks.html).
- GraphSAGE algorithm: W. Hamilton, R. Ying, J. Leskovec, "Inductive
  Representation Learning on Large Graphs," NeurIPS 2017.
- Graph/particle-cloud construction (k-NN in eta-phi space) and jet-image
  construction follow standard conventions in the jet-tagging literature,
  e.g. H. Qu and L. Gouskos, "ParticleNet: Jet Tagging via Particle
  Clouds," Phys. Rev. D 101 (2020).
- **Dataset:** G. Kasieczka, T. Plehn, J. Thompson, M. Russel, "Top Quark
  Tagging Reference Dataset,"
  [Zenodo, 2019](https://doi.org/10.5281/zenodo.2603256), CC-BY 4.0.

## License

MIT — see [LICENSE](LICENSE).

## Author

Ali — submitted to the MathWorks AI Challenge, 2026.
