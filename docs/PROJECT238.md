# MATLAB Challenge Project 238: requirements and review

The target is [Top Quark Detection with Deep Learning and Big Data](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/blob/main/projects/Top%20Quark%20Detection%20with%20Deep%20Learning%20and%20Big%20Data/README.md),
project number 238. Checked 26 September 2026, brief blob
`3079eb94b50b86d20f32c7d360128f982fcca431`.

The author supplied a review email on 26 September: the current submission was
not accepted. The corrections below track its technical requests. This is not
an acceptance claim, a new submission, or a promise that revisions will qualify.
The original seven research stages are not the seven steps in this brief.

| Brief step | Current evidence or gap |
| --- | --- |
| 1. Learn MATLAB deep learning | [Student walkthrough](WALKTHROUGH.md), toolbox examples linked in the brief |
| 2. Study the real-time top-quark example | Brief links the MathWorks example; current implementation is separately attributed |
| 3. Obtain the public dataset | Verified official HDF5 downloader, checksums and split manifests |
| 4. MATLAB calls Python for HDF5 → Parquet | Implemented in `run_project238` and `prepare_parquet.py`; validation in progress |
| 5. Parquet datastore and tall preprocessing into images | Implemented with a pure tall block transform and lossless single-precision TIFFs; validation in progress |
| 6. Train a CNN in MATLAB | Prior subset studies verified; new `trainProject238` consumes image datastores |
| 7. Test with folder-labelled image datastores | New folder-labelled datastore validates source row coverage; official-data execution pending |

Additional requested repairs: measured training-size justification, publicly
retrievable checkpoints and a quick verification command, MATLAB summary figures
and statistics, prediction execution settings, isolated random streams, documented
JVM requirements, and two small compatibility/lint fixes.

ResNet18 and FPGA/HDL are variations or advanced extensions in the brief. They
must be labelled as such; the base workflow should be completed first. GPU and
quantum-device experiments are not prerequisites for the base Project 238 steps.

The earlier explanation and graph-control studies are scientifically useful,
but do not resolve these MATLAB workflow requirements. Their results remain in
[experiments](../experiments/) with the original protocols and provenance.

## Repairs being validated

- `verify_results` uses the included seed-101 CNN checkpoint and 256 official
  test jets, checking every restored score. The other two models are available
  through the original Actions artifacts; their permanent Git copies await approval. It does not substitute sample scores for
  full-test or three-seed means.
- `summarize_matlab` computes the original study mean/SD, Student-t intervals,
  and noise curves in MATLAB. Noise repeats are averaged within training seed.
- `computeROC` now uses toolbox `rocmetrics`; the existing pairwise/ties tests
  remain the numerical acceptance criteria.
- Prediction calls respect the configured execution environment. GPU execution
  remains untested; the GraphSAGE trainer remains a CPU implementation.
- The original sensitivity stages use local random streams. Image accumulation
  is vectorized. Hashing has an explicit JVM requirement.

`run_project238` defaults to all official training rows. The short demonstration
uses 2,000/500/1,000 train/validation/test rows and one epoch; it proves execution,
not a finished full-scale study. No full-data result is claimed before an actual
run completes and its predictions are independently checked. Epoch model
checkpoints are saved, but exact optimizer/RNG training resume is not implemented.

## Participation and authorship

The [Project Hub wiki](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/wiki)
describes ongoing project reviews and welcomes individual participants. The
supplied email rejects the present solution; it does not state a personal
disqualification. Neither source explicitly guarantees reconsideration of this
specific submission. That remains for the review team to confirm.

The [AI guidelines](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/wiki/Generative-AI-Guidelines)
permit assistance with disclosure, testing and the author's own understanding.
Assistant-run verification does not establish that the author can explain the
solution. See [the disclosure](AI_ASSISTANCE.md) and [walkthrough](WALKTHROUGH.md).
