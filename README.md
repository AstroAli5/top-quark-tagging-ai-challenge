# Top Quark Tagging: CNN and GraphSAGE Under Synthetic Detector Noise

[![Tests](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/workflows/tests.yml/badge.svg)](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/workflows/tests.yml)

A MATLAB research project by **Ali Mohamed**, developed for MathWorks
[Challenge Project #238](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/discussions/74).

**Research question:** how does classification performance change when the same
held-out jets are represented as images or particle graphs and their measured
momenta are perturbed?

The repository implements data preparation, training, clean evaluation, a
controlled robustness experiment, and two sensitivity analyses. **The corrected
pipeline needs a fresh real-data training run before its scientific conclusions
can be reported.** Existing outputs are preserved as historical artifacts; see
[Results status](#results-status).

## What the project does

| Stage | Entry point | Output |
| --- | --- | --- |
| 1 | `s1_prepare_data` | Validated constituent four-vectors and source metadata |
| 2 | `s2_build_representations` | Images, sparse graphs, one shared stratified split |
| 3 | `s3_train_cnn` | Compact CNN with softmax probabilities and saved class order |
| 4 | `s4_train_graphsage` | Three GraphSAGE layers with a sigmoid classifier |
| 5 | `s5_evaluate_baseline` | Accuracy, ROC/AUC, and per-jet probabilities |
| 6 | `s6_robustness_test` | Paired noise experiment using the same perturbed jets |
| 7 | `s7_explainability` | Feature-permutation and radial-occlusion CSVs and plots |

All MATLAB files are in the repository root. There is no need to move helpers
into another folder or edit an absolute MATLAB Drive path.

## Requirements

- MATLAB **R2024a or later** and **Deep Learning Toolbox**.
- CPU execution is the default. GraphSAGE uses double-precision sparse CPU
  batches; a GPU and Parallel Computing Toolbox are not required.
- Python 3.11 and [requirements.txt](requirements.txt) for one-time data conversion.
- Enough memory and disk for the chosen subset. Images and per-jet representations
  are held in memory; graph inference is batched. Start with a smaller subset on
  a laptop.

The CNN uses MathWorks'
[`trainnet`](https://www.mathworks.com/help/deeplearning/ref/trainnet.html) and
[`minibatchpredict`](https://www.mathworks.com/help/deeplearning/ref/minibatchpredict.html).
ROC/AUC is implemented locally; Statistics and Machine Learning Toolbox is not required.

## Run the project

### 1. Clone and prepare the data

```bash
git clone https://github.com/AstroAli5/top-quark-tagging-ai-challenge.git
cd top-quark-tagging-ai-challenge
python -m pip install -r requirements.txt
```

Download `train.h5` from the
[Top Quark Tagging Reference Dataset](https://doi.org/10.5281/zenodo.2603256)
and place it in `data/`. Then:

```bash
python scripts/convert_dataset.py --input data/train.h5 --output data/jets_real_50k.mat --max-jets 50000
```

The converter selects the first requested rows in source order, explicitly orders
the 800 columns as `E_0, PX_0, PY_0, PZ_0, ..., PZ_199`, checks finite values and
binary labels, and saves column-vector labels. The MAT file includes the source
SHA-256, row-selection description, and conversion package versions. Existing
output files require `--force` to replace.

Conversion uses
[pandas HDF reading](https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html)
and [SciPy MAT export](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html).
An existing correctly formatted `jets_real_50k.mat` also works, although older files
may lack source provenance.

**Evaluation protocol:** this compact experiment makes its own 70%/15%/15%
train/validation/test split within the selected input subset. With 50,000 usable
jets, approximately **35,000 train the models**, rather than all 50,000.
Do not describe this internal holdout as evaluation on the dataset's official test
partition. Earlier instructions used `test.h5` as the input and then split it;
such runs are not comparable to the official benchmark protocol. Use the source
training file for new internal-holdout experiments.

### 2. Run in MATLAB

Open the cloned folder in MATLAB:

```matlab
run_all
```

The pipeline creates `data/`, `models/`, and `results/` under the project root
and saves plots without opening training windows. It does not change the current
folder or clear the caller's workspace.

To customize a run:

```matlab
cfg = projectConfig;
cfg.cnnEpochs = 15;
cfg.graphEpochs = 3;
cfg.graphBatchSize = 32; % reduce the graph working set
run_all(cfg)
```

The seven stages can also be called individually with the same configuration:

```matlab
cfg = projectConfig;
s5_evaluate_baseline(cfg) % requires matching prepared data and trained models
s6_robustness_test(cfg)
s7_explainability(cfg)
```

Every fresh step-1 import receives a dataset ID. The models and representations
must share that ID, so old checkpoints cannot silently be evaluated against a
different split. After changing input data, rebuild and retrain with `run_all`.
Existing generated files in the selected output directories are overwritten.

## Experiment design and limits

The CNN receives 32-by-32 images of summed transverse momentum, log-compressed
after binning around a pT-weighted angular center. GraphSAGE receives
`deltaEta, deltaPhi, log(pT), log(E)` for each constituent, with a symmetrized
six-nearest-neighbor graph in eta/phi. Three neighbor-mean layers use ReLU and
nodewise L2 normalization, followed by mean pooling and a binary classifier.
This is a small GraphSAGE baseline, **not ParticleNet**.

Both models use identical jets and splits, fixed random seeds, and validation
loss to select checkpoints. Their default training budgets differ (15 CNN epochs,
3 GraphSAGE epochs), so this is not an equal-compute architecture benchmark.

For each noise level, every momentum component is independently multiplied by
`1 + sigma * Z`, with standard-normal `Z`. Energy is recomputed using a
nonnegative estimate of the original squared mass. Zero noise leaves inputs
unchanged. The same random draws are scaled across nonzero noise levels and both
models receive the same noisy jets; graphs and images are rebuilt each time.

This is a **synthetic stress test**, not a calibrated detector-resolution model.
It does not establish detector deployment readiness. Results from one training
seed and one noise realization do not establish statistical significance or a
universal architecture ranking. Multiple seeds, uncertainty estimates, a tuned
CNN baseline, and official dataset partitions are needed for stronger claims.

Feature permutation keeps graph edges fixed while shuffling a node-feature
column. This can create inconsistent feature/geometry combinations. Radial
occlusion masks outer image pixels. Both measure perturbation sensitivity, not
causal importance or proof that a network learned a particular decay structure.
See [CONCEPTS.md](CONCEPTS.md) for the intuition.

## Results status

The committed historical CSVs contain the following values:

| Historical metric | CNN | GraphSAGE |
| --- | ---: | ---: |
| Clean accuracy | 50.29% | 83.57% |
| Clean AUC | 0.1961 | 0.9088 |
| AUC at 10% synthetic smearing | 0.1904 | 0.8971 |
| AUC at 35% synthetic smearing | 0.2562 | 0.7693 |

Sources: [baseline_comparison.csv](baseline_comparison.csv) and
[robustness_results.csv](robustness_results.csv). These files have been preserved
unchanged. The earlier README's CNN AUC of 0.866 and several other numbers did not
match them.

**These are not validated results for the corrected code.** The old CNN lacked
a softmax output layer, inference did not explicitly control class/layout mapping,
and ROC handling could depend on label order when scores were tied.
Those defects are now addressed; corrected metrics require retraining and
reevaluation. A reversed score or a manually edited CSV is not a substitute.

The original [ROC](roc_baseline.png), [robustness](robustness_curves.png),
[feature importance](graphsage_feature_importance.png), and
[occlusion](cnn_radial_occlusion.png) images are historical too. They are not
presented as evidence for the corrected pipeline.

Fresh outputs are written to `results/`:

- `baseline_comparison.csv`, `baseline_predictions.mat`, and `roc_baseline.png`
- `robustness_results.csv` and `robustness_curves.png`
- `graphsage_feature_importance.csv` and its PNG
- `cnn_radial_occlusion.csv` and its PNG
- `run_metadata.json` and `run_metadata.mat`: configuration, environment, dataset
  provenance, and actual split counts

Keep a complete set from the same run, along with the code commit used to produce
it. Large data, checkpoints, and generated output directories are ignored by Git.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

```matlab
results = runtests('tests');
assertSuccess(results);
```

The GitHub [test workflow](.github/workflows/tests.yml) runs conversion tests and
MATLAB R2024a tests, including a 40-jet synthetic run through all seven stages.
Checks cover tied-score AUC, wrapped angles, mass preservation, disjoint splits,
graph gradients, inference batch boundaries, and agreement between the clean
baseline and zero-noise evaluation. Synthetic metrics are not physics results.

## References and attribution

- Kasieczka et al., [Top Quark Tagging Reference Dataset](https://doi.org/10.5281/zenodo.2603256).
  Follow the dataset's own license and attribution terms; it is not bundled here.
- Hamilton, Ying, and Leskovec,
  [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), NeurIPS 2017.
- Qu and Gouskos, [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570),
  Physical Review D 101 (2020), for particle-cloud context.
- Colin Crovella,
  [GraphSAGE Classifier for Top Quark Tagging](https://www.mathworks.com/matlabcentral/fileexchange/181442-graphsage-classifier-for-top-quark-tagging),
  the reference demo that inspired this comparison.
- MathWorks,
  [Node Classification Using GCN](https://www.mathworks.com/help/deeplearning/ug/node-classification-using-graph-convolutional-network.html)
  and
  [Multilabel Graph Classification Using GAT](https://www.mathworks.com/help/deeplearning/ug/multilabel-graph-classification-using-graph-attention-networks.html),
  for sparse batching and custom-loop patterns.

**AI assistance:** Claude assisted earlier development; OpenAI Codex assisted the
repository audit, corrections, documentation, and tests. Automated checks validate
software behavior, not the scientific claims. The author should review and
understand the implementation and verify real-data outputs before submission.

[Submission checklist](SUBMISSION_CHECKLIST.md) · [MIT license](LICENSE) ·
[Citation metadata](CITATION.cff)
