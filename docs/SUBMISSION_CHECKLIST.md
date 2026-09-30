# Project 238: response to the reviewer and submission readiness

Checked against the three screenshots of the author's 26 September 2026 review
email and the current official instructions on 28 September, with validation
completed on 30 September. This is a technical
readiness audit, not organizer acceptance or a claim of student understanding.
The email itself is not republished here.

**Repository package:** [PR #9](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/pull/9)
tracks the verified full-source result, reviewer entry/reporting updates and all
three included seed-101 checkpoints. The author approved permanent publication
of the GraphSAGE/reference files and the merge on 29 September. The PR page
records publication/merge status. [PR #11](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/pull/11)
adds the inline README map, verified optional studies, response draft and practice
guide. No revised submission has been sent here.

## Each point in the email

| Reviewer request | Status | Implementation and evidence |
| --- | --- | --- |
| MATLAB-hosted HDF5 → Parquet ingestion | Satisfied | [`run_project238`](../run_project238.m) calls Python with `pyrun`; [`prepare_parquet.py`](../scripts/prepare_parquet.py) reads bounded HDF5 blocks and verifies official source hashes. Python runs out of process to isolate HDF5 libraries. |
| `parquetDatastore` + tall preprocessing, without fitting every image in RAM | Satisfied | [`parquetJetsToImages`](../src/bigdata/parquetJetsToImages.m) transforms bounded tall blocks and writes floating-point TIFFs. The full run completed with a measured 4.76 GiB MATLAB peak and separate 0.47 GiB Python peak. |
| Folder-labelled `imageDatastore` for training and testing | Satisfied | [`jetImageDatastore`](../src/bigdata/jetImageDatastore.m) reads signal/background labels, verifies row coverage, and supplies [`trainProject238`](../src/bigdata/trainProject238.m). `trainnet` and `minibatchpredict` consume datastores directly. |
| At least one study using the full training dataset, or a measured subset justification | Satisfied | [Verified full-source study](../experiments/project238-full/): all 1,211,000 training jets, 10,000 validation jets, all 404,000 test jets, one epoch, seed 101; all 12,110 training iterations completed. |
| Seed-101 CNN, GraphSAGE and reference weights, plus a quick verification command | Satisfied | All three original model files and the 256-jet sample are included in Git, with hashes checked against the original manifest. [`run_submission`](../run_submission.m) checks every restored probability for all three models, then recreates the MATLAB study report. CI uses included files directly. See [checkpoint instructions](../checkpoints/). |
| Use `rocmetrics` and retain the tie-handling test | Satisfied with documented compatibility fallback | [`computeROC`](../src/core/computeROC.m) uses `rocmetrics` in the tested toolbox environment and a tested rank-based fallback otherwise. Pairwise/tie checks remain. Optional bootstrap analysis is not claimed. |
| MATLAB statistics, paired seed differences and headline figures | Satisfied and independently verified | [`summarize_matlab`](../summarize_matlab.m) computes means, sample SD, Student-t intervals, matched-seed accuracy/AUC differences, and noise means within each fitted seed. The updated figure includes clean accuracy, clean AUC and noise AUC. Independent SciPy/Pandas recalculation agrees within 4.45e-16; [verification record](../experiments/official-study/matlab_summary_verification.json). |
| Named `resnet18` variation | Optional pilot executed and independently checked | [`run_resnet18_study`](../run_resnet18_study.m) uses the named architecture with random weights, a grayscale input and two-class head. Both CNNs use the same 10k/2k/10k rows and three-epoch budget; [protocol and measured results](../experiments/resnet18-pilot/). The custom ResNeXt-SE reference remains a separate model. |
| FPGA/HDL deployment | Processor estimate completed; physical deployment unrun; optional in the email | [`run_hdl_assessment`](../run_hdl_assessment.m) successfully estimates the included CNN with Deep Learning HDL Toolbox and the Xilinx support package. [Evidence](../experiments/hdl-assessment/) distinguishes estimator output from hardware measurement; no generated HDL, synthesis or FPGA execution is claimed. |
| Vectorize image accumulation | Satisfied | [`buildJetImage`](../src/core/buildJetImage.m) uses `accumarray`; pixel round-trip tests pass. |
| Honor `cfg.executionEnvironment` in CNN and reference inference | Satisfied in code; GPU execution unrun | [`predictCNN`](../src/core/predictCNN.m), [`predictWinnerReference`](../src/reference/predictWinnerReference.m), and their callers pass the configured setting. CPU behavior is tested. |
| Avoid changing the caller's random state in stages 6 and 7 | Satisfied | [`s6_robustness_test`](../src/pipeline/s6_robustness_test.m) and [`s7_explainability`](../src/pipeline/s7_explainability.m) use local `RandStream` instances; preservation is covered by MATLAB tests. |
| Remove Java hashing or document the JVM requirement | Satisfied through the offered documentation option | The main README states the JVM requirement beside MATLAB R2024a; hashing gives a clear error under `-nojvm`. JVM-free operation is not claimed. |
| Fix the release check and class-index length check | Satisfied | `run_all` uses `isMATLABReleaseOlderThan`; `predictWinnerReference` uses `isscalar`. |
| Align the repository with Project 238 and acknowledge the actual submission | Satisfied | README identifies Project 238 and contains the [seven-step implementation table](../README.md#seven-step-project-238-implementation) directly; [requirements detail](PROJECT238.md), [status](PROJECT_STATUS.md), and [submission context](COMPETITIONS.md) acknowledge that the author submitted and received a rejection review. The assistant has not submitted or contacted reviewers. |

## Quick reviewer route

On a clean MATLAB R2024a+ installation with Deep Learning Toolbox and its normal
JVM, clone/download the repository and run this at the repository root:

```matlab
run_submission
```

This loads the included original CNN, GraphSAGE and reference, checks their
checksums and every restored probability on 256 official test jets, then recreates the recorded study's
MATLAB tables and figures. It needs no Python, new dataset download or retraining.
Outputs are written to `results/matlab-summary/`, and the command prints its
elapsed time. The complete three-model command completed in **14.19 seconds** in
[a fresh MATLAB process](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36529745936/job/109280516402),
excluding startup, with zero restored-score differences. Runtime depends on the machine.

The [optional-variation test run](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36701925222)
passed **20 MATLAB tests and 20 Python tests**, including the quantum checks and
new ResNet18 shape/residual-block/probability check. The
[datastore demonstration](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36701925280)
also passed. The unchanged full-source training was not rerun for this update.

The 256-jet score is **not** the 404,000-jet score or the three-seed mean.
The 91.06% / 0.96929 headline is reproduced by summarizing the recorded per-seed
full-test measurements in MATLAB; independently rescoring the entire study is a
separate, larger verification. The new full-source one-epoch CNN is also separate
from those original three models. Its verified accuracy is 90.2267%, AUC
0.9739505904. The original model and new full-source model must not be conflated.

## Official repository and authorship requirements

| Requirement | Current evidence or action |
| --- | --- |
| Public repository with MIT or BSD 2-Clause license | Public repository and root MIT license confirmed. |
| Clear main entry point, setup and dependencies | `run_submission` is the quick reviewer entry; `run_project238` is the documented datastore training route. MATLAB/JVM/Python/toolbox requirements are in the README. |
| Small input sample and expected outputs | The included sample is `checkpoints/quick_test.mat`, with source rows, reference probabilities and hashes. Figures, full reports and protocols are linked. |
| Trained models can be loaded without fitting | All three requested seed-101 models are included in Git and checked by `run_submission`, without downloading old artifacts or training. |
| Executable, tested MATLAB solution | Full-source execution passed; the reviewer-entry/reporting suite contains 20 MATLAB and 20 Python checks. The current entry checks all three included models. |
| Explain and acknowledge AI assistance | [Disclosure](AI_ASSISTANCE.md) is present. The author must personally understand and explain the solution; assistant-run tests cannot certify that. Use the [walkthrough and reviewer practice route](WALKTHROUGH.md#prepare-to-explain-the-revision-to-a-reviewer). |
| Submit through the Project 238 form using the original registration email | The official project page links the form. No revised form has been submitted here. The original registration details must be supplied by the author. |

## Remaining actions before resubmitting

1. The author reviews the walkthrough, runs the quick command, and can explain
   the input representation, split isolation, normalization, AUC, uncertainty,
   data-volume choice, AI assistance and remaining limitations.
2. The [prepared revision response](REVIEW_RESPONSE.md) has not been sent. Use the original review thread to clarify their preferred reconsideration
   route if needed; the published instructions use the project submission form
   and the same email used at registration. The rejection email invites discussion
   but does not explicitly guarantee reconsideration or acceptance.

A full 12-epoch/multi-seed study, full-source-model explanations, GPU, quantum
hardware and FPGA deployment are not claimed as completed. The email asks for at least one
full-training study; the verified one-epoch result provides that execution
evidence. Additional work cannot guarantee an acceptance decision.

## Official references checked 28 September 2026

- [Project 238 description and registration/submission links](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/blob/main/projects/Top%20Quark%20Detection%20with%20Deep%20Learning%20and%20Big%20Data/README.md)
- [Submission instructions and repository guidelines](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/wiki/Submission-Instructions-%26-Project-Repository-Guidelines)
- [Generative-AI guidelines](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/wiki/Generative-AI-Guidelines)
- [Project Hub participation information](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/wiki)
- [MATLAB R2024a `rocmetrics` documentation](https://www.mathworks.com/help/releases/R2024a/stats/rocmetrics.html)
