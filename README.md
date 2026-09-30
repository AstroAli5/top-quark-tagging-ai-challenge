# Top-Quark Tagging

**Deep learning, particle graphs, and reproducible evaluation in MATLAB.**

[![Tests](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/workflows/tests.yml)

By **Ali Mohamed** · Developed for [MathWorks Challenge Project 238](docs/PROJECT238.md)

This project classifies simulated particle jets as top-quark signal or background
and investigates how image and graph models respond to synthetic momentum noise.
It combines a MATLAB pipeline that processes the entire official training set
with repeated-seed experiments, model explanations, and independently checked results.

[Results](#results) · [Quick start](#quick-start) · [Experiment portfolio](#experiment-portfolio) · [MATLAB figures](docs/FIGURES.md) · [Student walkthrough](docs/WALKTHROUGH.md)

## Results

### Full training dataset, verified on CPU

| Training jets | Test jets | ROC AUC | Accuracy |
| ---: | ---: | ---: | ---: |
| **1,211,000** | **404,000** | **0.97395** | **90.23%** |

The datastore CNN completed **one epoch, one seed and all 12,110 training
iterations**, using 10,000 official validation jets. Training took **35.32 minutes**;
MATLAB peaked at **4.76 GiB RAM**, with a separate **0.47 GiB Python peak**.
These are measured CPU-runner results; preprocessing and evaluation add runtime.

Every saved test prediction was checked against official source rows and labels.
A separate calculation reproduced the accuracy and AUC.
[Full report, resource measurements and reproduction](experiments/project238-full/).

<details>
<summary>View the MATLAB ROC curve and confusion matrix</summary>

![Full-training-data CNN: ROC AUC 0.97395 and accuracy 90.23% on 404,000 official test jets](experiments/project238-full/attempt2/matlab_evaluation.png)

</details>

### Clean performance and noise sensitivity

A separate study compares **CNN, GraphSAGE and a ResNeXt-SE reference** across
three training seeds. Each fit uses **50,000 training jets, 10,000 validation jets
and 12 epochs**. Clean scores cover all 404,000 official test jets; noise scores
use the same 10,000 test jets across models and perturbations.

| Model | Clean accuracy | Clean ROC AUC | ROC AUC at 35% smearing |
| --- | ---: | ---: | ---: |
| CNN | 91.06% | 0.96929 | 0.66130 |
| GraphSAGE | 86.58% | 0.92769 | **0.73751** |
| ResNeXt-SE reference | **91.59%** | **0.97159** | 0.60460 |

Values are means across three trained seeds. Noise realizations are averaged
within each seed. [Uncertainty and paired comparisons](docs/RESULTS.md).

**The central finding:** the image models perform better on clean jets, while
GraphSAGE has the highest AUC at 20% and 35% synthetic smearing in each training
run. This reveals a trade-off between clean performance and sensitivity to the
tested perturbations.

![MATLAB study summary: clean accuracy, clean AUC, and noise sensitivity across three training seeds](experiments/official-study/matlab_study_summary.png)

*Exported from MATLAB; error bars show training-seed standard deviation.
[Measured tables and independent verification](experiments/official-study/).*

## Quick start

**Tested environment:** MATLAB R2024a with Deep Learning Toolbox and the standard
JVM-enabled session. Statistics and Machine Learning Toolbox enables `rocmetrics`;
a tested tie-aware fallback supports environments without it.

Clone or download the repository, open its root folder in MATLAB, and run:

```matlab
run_submission
```

The command verifies the checksums of **three included seed-101 models**, restores
their predictions on **256 official test jets**, and generates MATLAB study tables
and figures in `results/matlab-summary/`. The models and sample total **under 1 MB**;
no Python, new data download or training is needed for this check.

The complete command ran in **14.19 seconds**, excluding MATLAB startup, with
**zero prediction differences for all three models** on the
[verified CPU runner](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36529745936/job/109280516402).
Sample restoration covers 256 jets; the accompanying study report summarizes the
recorded full-test measurements across three seeds.

[Checkpoint identities and provenance](checkpoints/) ·
[Reviewer checklist](docs/SUBMISSION_CHECKLIST.md)

## Experiment portfolio

The project extends beyond a single classification score. Each study retains its
protocol, measured results and verification evidence.

| Study | Completed work | Evidence |
| --- | --- | --- |
| Repeated training | Five seeds per core model at 50k training jets; evaluates whether the clean/noise trade-off persists | [Five-seed study](experiments/seed-extension/) |
| Larger training subset | Three core seeds at 100k jets; mean accuracy **91.63% CNN / 87.40% GraphSAGE**, evaluated on all 404k test jets | [Scaling study](experiments/scaling/) |
| Model explanations | Feature shuffling and radial image occlusion for both the 50k and 100k core checkpoints; **51 independently checked perturbation measurements per study** | [50k explanations](experiments/explanations/) · [100k explanations](experiments/explanations-100k/) |
| Graph-neighbor control | Matched three-seed comparison of six neighbors versus zero; edges improve clean AUC, while zero edges perform better at the strongest tested smearing | [Graph-edge study](experiments/graph-edges/) |
| Named ResNet18 variation | Same images and fixed training settings as the compact CNN; 10k training jets, three epochs and one seed, with independently checked predictions | [Matched pilot](experiments/resnet18-pilot/) |
| Quantum/classical pilot | Four-qubit Qiskit CPU simulation against matched linear/RBF baselines; the classical RBF model performs better | [Kernel comparison](experiments/quantum-pilot/) |

## Data pipeline and reproduction

The required MATLAB workflow preserves the official train/validation/test
partitions throughout:

1. MATLAB calls Python through `pyrun` to verify HDF5 sources and convert bounded
   blocks to Parquet.
2. `parquetDatastore` and tall arrays transform jets into floating-point images.
3. Folder-labelled `imageDatastore` objects feed `trainnet` and `minibatchpredict`.
4. Evaluation saves probabilities, source row IDs, labels, configuration and hashes
   for independent checks.

### Seven-step Project 238 implementation

| Project step | Implementation and evidence |
| --- | --- |
| 1. Learn MATLAB deep learning | [Student walkthrough](docs/WALKTHROUGH.md), model explanations and runnable examples; the author must be able to explain the code |
| 2. Study the real-time top-quark example | [Official project brief and example](docs/PROJECT238.md); this CPU study does not claim real-time hardware deployment |
| 3. Obtain the public dataset | [Official downloader](scripts/prepare_parquet.py), verified source checksums and separate train/validation/test manifests |
| 4. Call Python from MATLAB to convert HDF5 to Parquet | [`run_project238`](run_project238.m) uses `pyrun` with bounded row blocks |
| 5. Use Parquet datastore and tall preprocessing | [`parquetJetsToImages`](src/bigdata/parquetJetsToImages.m) writes lossless floating-point jet images in bounded blocks |
| 6. Train a CNN in MATLAB | [`trainProject238`](src/bigdata/trainProject238.m) uses `trainnet`; [one complete epoch on 1,211,000 jets](experiments/project238-full/) is verified |
| 7. Test using labelled image datastores | [`jetImageDatastore`](src/bigdata/jetImageDatastore.m) and `minibatchpredict`; saved scores cover all 404,000 official test rows and pass [independent checks](scripts/verify_project238.py) |

[Project 238 requirements and review](docs/PROJECT238.md) ·
[MATLAB, Colab and Qiskit setup](docs/PLATFORMS.md)

<details>
<summary><strong>Run a small real-data training demonstration</strong></summary>

Install Python 3.11 and the preparation dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure MATLAB's `pyenv` to use that Python installation with
`ExecutionMode="OutOfProcess"`. Restart MATLAB first if Python is already loaded
in-process; out-of-process execution isolates the HDF5 libraries.

```matlab
cfg = project238Config;
cfg.trainCount = 2000;
cfg.validationCount = 500;
cfg.testCount = 1000;
cfg.epochs = 1;
cfg.dataDir = fullfile(cfg.rootDir,'data','project238_demo');
cfg.outputDir = fullfile(cfg.rootDir,'runs','project238_demo');
run_project238(cfg)
```

[Verified demonstration](experiments/project238-demo/). Use new data/output folders
for another run. The source files total about 1.73 GB; prepared data require
additional disk space.

The unmodified `project238Config` selects all 1,211,000 training jets, 10,000
validation jets and all 404,000 test jets for **12 epochs**. The verified full-data
run used **one epoch and batch size 100**; follow its
[exact reproduction command](experiments/project238-full/#reproduce-the-one-epoch-run).
The 12-epoch full-data default remains unrun.

</details>

<details>
<summary><strong>Reproduce the three-model research study</strong></summary>

After the Python/MATLAB setup above, prepare the official partitions:

```bash
python scripts/prepare_official.py
```

Train and evaluate the three predefined seeds in MATLAB:

```matlab
run_experiment(101)
run_experiment(202)
run_experiment(303)
```

Recalculate the combined report:

```bash
python -m pip install matplotlib
python scripts/summarize_experiment.py --input runs --output results/official-study
```

The [frozen protocol](docs/EXPERIMENT_PROTOCOL.md) defines the 50k/10k training and
validation selections, full test evaluation and 12-epoch budget. The reference
caches about 4 GB of images; allow at least 16 GB system RAM. Its recorded training
cost was approximately 149 CPU minutes per seed, compared with 9.4 for CNN and
5.7 for GraphSAGE; runner conditions and timing methods vary.

The [research workflow](.github/workflows/research.yml) defaults to CNN and
GraphSAGE; select `include_reference` to run the third model. The MATLAB commands
above reproduce all three. Existing fitted outputs are protected from overwriting.

</details>

## Verification and research scope

The [verified implementation](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36701925222)
passed **20 MATLAB tests and 20 Python tests**, including the optional quantum
checks. Tests cover data boundaries, probability mapping, tied-score AUC, graph
batching, random-state preservation and execution of the data pipeline.
Saved research predictions receive separate metric, source-row and checksum checks.

Interpret the results within their measured scope:

- The full-training-data CNN has **one epoch and one seed**. Longer full-data fits
  and convergence studies remain future work.
- The repeated-seed studies use 50k or 100k training subsets. Different input
  representations and preprocessing make these comparisons of complete pipelines.
- Synthetic smearing is a controlled stress test; explanation plots measure model
  sensitivity. Neither establishes a calibrated detector response or causal physics.
- GPU, FPGA deployment and quantum-hardware experiments remain unrun. The
  [HDL processor assessment](experiments/hdl-assessment/) is a latency estimate,
  not a physical measurement. The Qiskit result is a CPU simulation with a
  stronger classical baseline.

The original Project 238 submission was not accepted. The
[submission checklist](docs/SUBMISSION_CHECKLIST.md) records the completed
technical revisions, review history and remaining submission steps. A
[revision response](docs/REVIEW_RESPONSE.md) and
[reviewer practice guide](docs/WALKTHROUGH.md#prepare-to-explain-the-revision-to-a-reviewer)
are prepared; the revised submission has not been sent.

## Documentation and attribution

| Resource | Purpose |
| --- | --- |
| [Student walkthrough](docs/WALKTHROUGH.md) | Understand the models, code and research decisions |
| [Results and uncertainty](docs/RESULTS.md) | Interpret the measurements and confidence intervals |
| [Figure gallery](docs/FIGURES.md) | Browse MATLAB exports and other labelled research plots |
| [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) | Reproduce the data selections, seeds and evaluation |
| [Project status](docs/PROJECT_STATUS.md) | Distinguish completed work from future experiments |

Dataset: [Top Quark Tagging Reference Dataset](https://doi.org/10.5281/zenodo.2603256)
by Kasieczka, Plehn, Thompson and Russel, **CC BY 4.0**.
Methods and references: [GraphSAGE](https://arxiv.org/abs/1706.02216),
[Colin Crovella's MATLAB example](https://www.mathworks.com/matlabcentral/fileexchange/181442-graphsage-classifier-for-top-quark-tagging),
and the [independently implemented ResNeXt-SE adaptation and its attribution](docs/WINNER_COMPARISON.md).

[MIT code license](LICENSE) · [Citation metadata](CITATION.cff) · [AI-assistance disclosure](docs/AI_ASSISTANCE.md)
