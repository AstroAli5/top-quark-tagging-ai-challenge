# Measured four-qubit simulator comparison

Completed on 23 September 2026. This optional Python experiment uses **Qiskit
2.5.2** to simulate a four-qubit quantum feature map, followed by a classical
support-vector classifier. It does not use a quantum computer.

All three models receive the same four derived jet features, training-fitted
scaling, fitting samples, validation search budget and fixed test sample.
Each of three sampling repeats uses **512 training** and **256 validation**
jets. The same **2,000 official test jets** are used throughout. Every sample
is balanced by class. See [the recorded protocol](PROTOCOL.md).

| Model | Mean test AUC ± sampling SD | Mean accuracy |
| --- | --- | --- |
| Linear SVM | 0.91549 ± 0.00165 | 88.37% |
| RBF SVM | **0.94508 ± 0.00306** | **88.63%** |
| Simulated quantum-kernel SVM | 0.88045 ± 0.00140 | 82.80% |

![Matched kernel comparison](kernel_comparison.png)

The RBF baseline performed better in this pilot. The quantum method supplies
an additional measured comparison, not evidence of quantum advantage. Three
overlapping fitting samples provide limited information about variability.
Do not compare these numbers directly with the full-input MATLAB models:
features, training size and test selection differ.

## Evidence

- [Full report](report.json): selected rows, validation trials, kernel checks,
  package versions, data hashes, clipping rates and timing.
- [Per-jet test decision scores](test_scores.csv): every score used in the report.
- [Summary table](summary.csv).
- [Independent metric verification](verification.json): all nine model/repeat
  combinations checked using an average-rank AUC calculation.

Decision scores are not calibrated probabilities. Timing is measured CPU wall
time including model selection and evaluation; it is not quantum-device time.
The simulation is exact and noiseless, with no shot or device-noise experiment.

## Reproduce

Use Python 3.12 in a separate environment. Prepare the original 50k/10k fitting
data and 5,000-row test chunks with `scripts/prepare_official.py` first.

```bash
python -m pip install -r requirements-quantum.txt
python scripts/quantum_pilot.py --data data/official --output results/quantum-pilot
python scripts/quantum_pilot.py --output experiments/quantum-pilot --verify-only
```

The run command requires a new output directory. Rebuilding prepared MAT files
changes their timestamped headers; the run records the new hashes while using
the same source files and row selections.
