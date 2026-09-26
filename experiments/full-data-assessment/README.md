# Full-data and hardware assessment

Checked 25 September 2026. This is an engineering assessment, **not another
completed training experiment**. The measured core study uses up to 100,000
of the 1,211,000 official training rows.

The current combined fitting array would contain **3,907,200,000 bytes** for
1,211,000 training and 10,000 validation jets. It exceeds the current MAT-v5
array limit before the trainer loads representations. The deliberately
conservative planner returns 126.76 GiB, assuming 200 constituents for every
jet and coexisting arrays. This is **not measured demand or a purchase
recommendation**. Measured MATLAB-process peak RAM at 100k was 3.68–3.73 GiB;
linear memory/runtime extrapolation is not validated.

**Review update, 26 September:** this assessment describes the previous MAT route.
It does not satisfy the Project 238 requirement for a demonstrated MATLAB
Parquet/tall/imageDatastore workflow. That work is now the priority in
[the review map](../../docs/PROJECT238.md).

## What a full-data implementation needs

1. Store fitting data and image/graph representations in bounded chunks or
   HDF5. Iterate minibatches without loading every jet into memory.
2. Compute any training normalization from training data only. Specify whether
   shuffle order is global or chunk-based; changing it changes the protocol.
3. Save optimizer moments, step count, random-number state and best-validation
   checkpoint so interrupted training resumes without silently restarting.
4. Verify streamed versus in-memory inputs/updates on a small fixture; verify
   row coverage, partition separation and resume equivalence. Measure actual
   memory and time before choosing a full-scale budget.
5. Record a separate protocol and execute all declared seeds. Do not label a
   subset, one-epoch trial or a preparation-only run as the finished full study.

[MathWorks' big-data documentation](https://www.mathworks.com/help/deeplearning/ug/deep-learning-with-big-data.html)
describes minibatch/datastore training for data that exceed memory. A GPU can
help runtime but is not a substitute for correct streamed input.
[GitHub's hosted-job limit](https://docs.github.com/en/actions/reference/limits)
also makes recoverable checkpoints useful for long runs.

## Hardware access

The verified execution environment is MATLAB R2024a on GitHub's CPU runners.
No usable connected GPU or quantum-device service was found in this session.
No GPU or quantum-hardware job was submitted, no billing was enabled, and no
hardware performance claim is made.

[GitHub documents GPU configurations among its larger runners](https://docs.github.com/en/actions/reference/runners/larger-runners),
which require separate provisioning and billing. Those runners were not enabled
for this work. A GPU GraphSAGE implementation would also need supported sparse
operations, data/parameter placement and numerical validation.

The existing quantum result is a small exact CPU simulation, where the
classical RBF baseline performed better. Device execution would answer a
different question and would need an appropriate authorized service, a circuit
and shot budget, calibration records and matched evaluation. It is optional
research, not a requirement to call the measured classical study complete.

[Machine-readable assessment](assessment.json) · [Measured 100k evidence](../scaling/)
