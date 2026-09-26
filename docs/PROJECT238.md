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
| 4. MATLAB calls Python for HDF5 → Parquet | Missing in the submitted version; standalone Python → MAT is insufficient |
| 5. Parquet datastore and tall preprocessing into images | Missing in the submitted version; fitting representations are materialized in RAM |
| 6. Train a CNN in MATLAB | Implemented and measured at 50k and 100k, but requires the datastore input route |
| 7. Test with folder-labelled image datastores | Full official test predictions exist; the required imageDatastore route is missing |

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
