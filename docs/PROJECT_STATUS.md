# Project status

This is an independent student research repository. It does not claim a prize,
registration, or competition submission. The [event page](https://www.mathworks.com/academia/students/competitions/student-challenge/ai-challenge.html) checked on 16 September
2026 describes the closed 2025 MathWorks AI Challenge; the research can be read
and reproduced independently of that event.

## Implemented

- A short entry-point README and a student walkthrough.
- The original seven-stage CNN/GraphSAGE experiment.
- A separately organized, attributed ResNeXt-SE reference.
- Download verification, source provenance, and explicit official partitions.
- Three training seeds with fixed choices and validation-only checkpoint selection.
- Full test evaluation in chunks, plus paired noise experiments.
- Prediction-based metric verification and separate measures of seed variability
  and finite-test uncertainty.
- Python and MATLAB integration checks, including a guard against duplicate
  implementation files shadowing the current code.
- Historical files preserved without changing their values.
- A central figure gallery, including all six small-run MATLAB exports saved in Git.
- A five-file reading path and an opt-in switch for costly reference training.
- Final-model feature permutation and radial occlusion without retraining.

## Measurement status

The small real-data demonstration and larger official-partition study are
complete. All nine model fits finished their 12-epoch budgets, and every model
was evaluated on all 404,000 official test jets. The paired noise study is also
complete. Saved predictions were checked locally, and the complete recalculated
report matches the workflow report exactly.

See [Results](RESULTS.md) for the measured values and
[the evidence folder](../experiments/official-study/) for tables, figures,
uncertainty, source hashes, and reproduction commands. All 10 Python and
15 MATLAB tests passed. Historical outputs remain unchanged.

On 22 September, [the final-model explanation run](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35709427585)
also completed. It evaluates all three frozen CNN/GraphSAGE training seeds on
10,000 official test jets. All 36 permutation and 15 occlusion measurements were
independently verified, along with all six checkpoint hashes. Both new figures,
the per-seed tables, and the verification record are preserved in
[the explanation evidence folder](../experiments/explanations/).

## Scientific scope

This compares the specific implementations and budgets documented in the
[protocol](EXPERIMENT_PROTOCOL.md). It does not establish universal superiority,
state-of-the-art performance, or detector deployment readiness.

Training on the full source training set, extensive hyperparameter searches,
architecture ablations, and a calibrated detector model are possible follow-up
studies. They are outside the present fixed experiment.

## Follow-ups on 23 September 2026

- The [four-qubit Qiskit pilot](../experiments/quantum-pilot/) is complete. All
  nine model/repeat metric pairs were independently recalculated. The classical
  RBF SVM outperformed the simulated quantum kernel on the matched pilot inputs.
- [Seeds 404 and 505](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35841193310)
  are running for the core models. A combined five-seed result is pending verification.
- The [100,000-training-jet study](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35842565638)
  is running for three core-model seeds. Its results and measured memory are pending.
- [Configurable data sizes and memory controls](SCALING.md) are implemented and tested.
- [Competition options and requirements](COMPETITIONS.md) were checked. No entry
  has been made; eligibility and student-authorship requirements remain relevant.

[Next experiments](NEXT_EXPERIMENTS.md) distinguishes these follow-ups from
unrun controlled ablations, a complete learning curve, full-data training,
GPU validation, and quantum-device/noise experiments.

The [walkthrough](WALKTHROUGH.md) explains the project in presentation order.
The author should review and understand the implementation and
[AI-assistance disclosure](AI_ASSISTANCE.md).
