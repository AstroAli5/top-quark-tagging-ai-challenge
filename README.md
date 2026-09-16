# Top-quark tagging under synthetic detector noise

[![Tests](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/workflows/tests.yml/badge.svg)](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/workflows/tests.yml)

A MATLAB research project by **Ali Mohamed**.

**Question:** how do an image CNN and a particle GraphSAGE model respond when the
same jets have their measured momenta perturbed?

The central project is the CNN/GraphSAGE comparison. An optional, independently
implemented [ResNeXt-SE reference](docs/WINNER_COMPARISON.md) explores ideas from
Adit Shah's 2025 winning project, with attribution.

**Start here:** [Results](docs/RESULTS.md) · [Student walkthrough](docs/WALKTHROUGH.md) ·
[Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) · [Project status](docs/PROJECT_STATUS.md)

## Run the core project

Requirements: **MATLAB R2024a+**, **Deep Learning Toolbox**, and **Python 3.11**.
CPU execution is supported; a GPU is not required.

```bash
git clone https://github.com/AstroAli5/top-quark-tagging-ai-challenge.git
cd top-quark-tagging-ai-challenge
python -m pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/convert_dataset.py --input data/train.h5 --output data/jets_real_50k.mat --max-jets 50000
```

Open this folder in MATLAB:

```matlab
cfg = projectConfig;
run_all(cfg)
% Optional third model, using the same prepared data:
run_winner_comparison(cfg)
```

`run_all` prepares jets, builds representations, trains both models, evaluates
clean and noisy inputs, and creates sensitivity plots. It writes to `data/`,
`models/`, and `results/`; repeating it replaces those generated files.

This introductory run uses a **70%/15%/15% split inside the selected training
subset**. It is useful for learning the pipeline. It does not use the official
test partition. For a quick 2,000-jet example, see the
[small benchmark](docs/WINNER_COMPARISON.md#run-all-three-models).

## Reproduce the larger study

The [frozen protocol](docs/EXPERIMENT_PROTOCOL.md) uses **50,000 official training
jets, 10,000 official validation jets, the full official test partition, and
three training seeds**. Each of the three models trains for 12 epochs.
This remains a training-subset study; it does not train on all 1.2 million jets.

```bash
python scripts/prepare_official.py
```

Then in MATLAB:

```matlab
run_experiment(101)
run_experiment(202)
run_experiment(303)
```

Finally:

```bash
python -m pip install matplotlib
python scripts/summarize_experiment.py --input runs --output results/official-study
```

The downloader verifies all three source files (about 1.73 GB total). Allow
additional disk for prepared data and models. The larger reference model caches
about 4 GB of input images; use a machine with at least 16 GB RAM.
Existing experiment models are preserved: choose a new output folder to rerun.
The [research workflow](.github/workflows/research.yml) runs the same study in MATLAB
on GitHub Actions in separate core/reference jobs and retains results and
checkpoints as downloadable artifacts. The core summary does not wait for the
slower reference; the combined summary verifies their shared data before joining them.

## Find your way around

| Location | Purpose |
| --- | --- |
| Root commands | `run_all`, `run_experiment`, `run_small_benchmark`, `run_winner_comparison`, `projectConfig` |
| `src/pipeline/`, `src/core/` | Seven stages and shared image/graph helpers |
| `src/reference/` | Optional ResNeXt-SE comparison |
| `src/experiment/` | Official test evaluation in bounded chunks |
| `scripts/`, `notebooks/` | Download, conversion, summaries, and Colab preparation |
| `docs/` | Explanation, results, protocol, and attribution |
| `archive/original-results/` | Unchanged historical outputs, separated from current evidence |

Root commands call `setupProject` automatically. Call it first when exploring
implementation functions directly.

## Checks and interpretation

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

```matlab
results = runtests('tests');
assertSuccess(results);
```

Tests check data boundaries, score mapping, tied-score AUC, graph batching,
noise pairing, and small end-to-end runs. Their synthetic scores are not
physics results. Synthetic momentum smearing measures sensitivity to one
perturbation; it is not a calibrated detector simulation. Read the
[results discussion](docs/RESULTS.md) before making architecture comparisons.

Data: [Top Quark Tagging Reference Dataset](https://doi.org/10.5281/zenodo.2603256)
(CC BY 4.0). Background: [GraphSAGE](https://arxiv.org/abs/1706.02216) and
[Colin Crovella's MATLAB reference demo](https://www.mathworks.com/matlabcentral/fileexchange/181442-graphsage-classifier-for-top-quark-tagging).
See [concepts](docs/CONCEPTS.md), [reference-model attribution](docs/WINNER_COMPARISON.md),
[AI assistance](docs/AI_ASSISTANCE.md), and [MATLAB / Colab / Qiskit](docs/PLATFORMS.md).

[MIT license](LICENSE) · [Citation metadata](CITATION.cff)
