# Verified 100,000-jet training study

The [three-seed MATLAB run](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35842565638)
completed and its saved predictions were independently checked on 23 September
2026. CNN and GraphSAGE use 100,000 training jets, the same 10,000 validation
jets, seeds 101/202/303, 12 epochs, and all 404,000 official test jets.

| Model | Accuracy, mean ± seed SD | AUC, mean ± seed SD |
| --- | ---: | ---: |
| CNN | 91.633% ± 0.117 percentage points | 0.97181 ± 0.00152 |
| GraphSAGE | 87.400% ± 0.265 percentage points | 0.93460 ± 0.00122 |

## Matched comparison with 50,000 training jets

This comparison uses the **same three seed labels at both sizes**, not the
five-seed extension as the baseline. Test rows and labels, model implementations,
configuration and MATLAB version were checked. The changed preparation script
adds size/output controls and resource planning; the source row selection and
particle conversion logic were preserved.

| Model | 50k accuracy | 100k accuracy | Mean gain | 50k AUC | 100k AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 91.059% | 91.633% | +0.574 percentage points | 0.96929 | 0.97181 |
| GraphSAGE | 86.579% | 87.400% | +0.822 percentage points | 0.92769 | 0.93460 |

![Matched three-seed comparison of training sizes](training_size_comparison.png)

The paired 95% seed interval for the accuracy change is +0.197 to +0.950
percentage points for CNN and −0.986 to +2.630 for GraphSAGE. The latter includes
zero despite a positive mean. AUC-change intervals are +0.00056 to +0.00450
and +0.00275 to +0.01106, respectively. These intervals use only three paired
seed labels and hold the test sample fixed.

Equal epochs also mean **more optimizer updates** at 100k. This measures the
combined change in data and training budget. It does not isolate a pure data
effect, compare equal compute, establish an architecture advantage, or form a
complete learning curve. The original test results were known before this
follow-up was designed. No 100k reference-model or explanation study was run.

## Noise and resources

The paired noise analysis uses the same first 10,000 test jets and three noise
draws per fitted model. GraphSAGE retains the higher mean AUC at 20% and 35%
smearing. More training data did not improve every measurement: GraphSAGE's
mean AUC at 35% decreased from **0.73751 to 0.72382**, while CNN changed from
0.66130 to 0.66496. This remains sensitivity to synthetic component noise.

![Full-test accuracy at 100k](clean_accuracy.png)
![Paired noise sensitivity at 100k](noise_auc.png)

| Seed | MATLAB process peak RAM | MATLAB experiment wall time |
| --- | ---: | ---: |
| 101 | 3.733 GiB | 23.77 minutes |
| 202 | 3.717 GiB | 37.87 minutes |
| 303 | 3.681 GiB | 38.63 minutes |

Linux `VmHWM` measures the MATLAB process, not the entire runner. Wall time
includes training and evaluation but excludes source downloads and workflow
setup. Different shared runners limit speed comparisons. The conservative
planning estimate was 12.33 GiB; it is neither measured demand nor an exact
upper bound. See [raw measurements and the plan](resources.json).

## Evidence and reproduction

[Protocol](PROTOCOL.md) · [report](report.json) · [clean summary](clean_summary.csv) ·
[noise summary](noise_summary.csv) · [per-seed clean](per_seed_clean.csv) ·
[per-seed noise](per_seed_noise.csv) · [paired model AUC](paired_auc.csv) ·
[paired size differences](training_size_paired.csv) ·
[size-change intervals](training_size_summary.csv) ·
[comparison verification](training_size_comparison.json) ·
[artifact hashes](artifacts.json).

Every saved clean prediction and all 90 noise-metric rows passed verification.
The three downloaded archive SHA-256 values match GitHub's recorded digests.
Their models and predictions are retained until the recorded artifact expiry;
the committed measurements and figures remain available afterward.

Download the three `scaling-100000-seed-*` artifacts from the run. Arrange their
`results/` folders under `scaling-audit/seed-101/`, `seed-202/`, and `seed-303/`.
Use the original core results arranged as in the
[five-seed reproduction instructions](../seed-extension/README.md#reproduce-the-summary).
From a full Git clone with the original commits available:

```bash
git fetch origin 0dec3b285672c5e6f8eb3bc89c1d646505387642
python -m pip install -r requirements.txt matplotlib
python scripts/summarize_experiment.py --input scaling-audit --output results/scaling --regenerated-partitions
python scripts/compare_training_sizes.py --baseline-report experiments/official-study/report.json --larger-report results/scaling/report.json --baseline-runs audit --larger-runs scaling-audit --output results/scaling
```

Separate jobs regenerated MAT files, whose timestamp headers differ. The
summarizer checks identical source hashes, selections, settings and code before
accepting those differences. [Scaling instructions](../../docs/SCALING.md)
explain how to train a new run. Full 1,211,000-row training still requires
streamed/HDF5 input; it is not a completed result.
