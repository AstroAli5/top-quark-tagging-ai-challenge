# Project status

This is an independent student research repository. It does not claim a prize,
registration, or competition submission. The event page checked on 16 September
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

## Measurement status

The small real-data run completed. The larger official-partition study is being
run and checked; see [Results](RESULTS.md) for measured values and run links.
No pending run is counted as a completed result.

## Scientific scope

This compares the specific implementations and budgets documented in the
[protocol](EXPERIMENT_PROTOCOL.md). It does not establish universal superiority,
state-of-the-art performance, or detector deployment readiness.

Training on the full source training set, extensive hyperparameter searches,
architecture ablations, and a calibrated detector model are possible follow-up
studies. They are outside the present fixed experiment.

The [walkthrough](WALKTHROUGH.md) explains the project in presentation order.
The author should review and understand the implementation and
[AI-assistance disclosure](AI_ASSISTANCE.md).
