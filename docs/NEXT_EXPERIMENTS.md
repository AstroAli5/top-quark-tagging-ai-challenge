# Next experiments and what they would establish

This roadmap distinguishes completed follow-ups from open research questions.
The original 50,000-training-jet, three-model study remains unchanged. The
[five-seed core extension](../experiments/seed-extension/),
[100,000-jet study](../experiments/scaling/), and
[Qiskit simulator pilot](../experiments/quantum-pilot/) completed and passed
prediction-based verification on 23 September 2026. Each has its own protocol
and output directory. See [Project status](PROJECT_STATUS.md).

| Issue | Current action | What remains |
| --- | --- | --- |
| Too much code to navigate | A short reading path and one figure gallery | Optional code remains available without entering the main learning path |
| Expensive reference | Research workflow defaults to the two core models; reference requires opting in | Same reference architecture still costs more when selected |
| Missing final-model explanation plots | Completed and independently verified; [figures and evidence](../experiments/explanations/) | Broader perturbations or more test jets would be separate extensions |
| Only three training seeds | Completed a separate five-seed CNN/GraphSAGE report | The reference still has three seeds; five core seeds remain modest |
| Only 50,000 training jets | Completed 100k training with three matched seed labels and measured memory | Complete learning curve, equal-compute controls, and streamed full-data training |
| Different representations | Describe results as comparisons of complete pipelines | Controlled ablations and matched constituent selections |
| No quantum model | Completed a separate four-qubit simulator pilot with matched classical baselines | No quantum-device, noise, scalability or advantage result |
| No competition entry | Keep the repository reproducible and attribution clear | Select an open, suitable competition and check its actual requirements |

## 1. Completed: explanations without training again

The analysis below completed on 22 September 2026. See
[the measured report](../experiments/explanations/README.md). These explanations
are specific to the original 50k-trained models; the additional seeds and 100k
checkpoints have no explanation study yet.

Use the final CNN/GraphSAGE checkpoints for seeds 101, 202, and 303. Keep weights,
normalization, class mapping, and test rows fixed. First reproduce the saved clean
probabilities; then shuffle each graph feature with fixed edges using streams
11, 21, and 31. Average shuffles within a training seed before summarizing across
the three fitted models. Evaluate five retained image-radius fractions:
1, 0.75, 0.5, 0.35, and 0.2. Radius is a fraction of the center-to-corner distance,
not the fraction of image area retained.

Use 10,000 official test jets, matching the noise sample. This explains the
50,000-jet-trained models on a stated evaluation subset; it does not claim to
run every perturbation on all 404,000 test jets. Feature shuffling and occlusion
can create unusual inputs, so describe the outcomes as sensitivity, not causality.

## 2. Completed: two additional core-model seeds

CNN/GraphSAGE seeds 404 and 505 used the same 50,000/10,000 fitting partitions,
hyperparameters and 12-epoch budget. All five seeds passed code/data compatibility
and saved-prediction checks. The separate [five-seed report](../experiments/seed-extension/)
preserves the original study and includes both new seeds, including lower AUCs.
The reference remains a three-seed result; its two extra fits were not run.

## 3. Completed at 100k; broader learning curve remains open

The [100k study](../experiments/scaling/) completed for core seeds 101/202/303.
The matched 50k/100k comparison improves mean clean accuracy by 0.574 percentage
points for CNN and 0.822 for GraphSAGE. More data did not improve every noise
measurement. MATLAB-process peak RAM was 3.68–3.73 GiB; this is a process
measurement under these runner conditions, not a universal hardware requirement.

An extended learning curve would compare nested 10,000, 25,000, 50,000, and 100,000 official training selections
using fixed validation data and repeated seeds. Make budget rules explicit:
equal epochs also increase optimizer updates when the dataset grows. Use
validation performance for development decisions and keep final testing separate.
Record any new design as a follow-up informed by the existing studies, not as an
experiment preregistered before the original test results were seen.

The 10k/25k points and an equal-update control have not been run. The current
trainer materializes representations in memory. Full training on
1,211,000 jets requires a memory/runtime assessment and likely streamed input;
simply increasing the row count is not a verified implementation of that study.

## 4. Isolate specific choices with ablations

Start with one change at a time inside a model family: for example, the reference
with/without channel attention, alignment, or radial features. Keep data,
training seeds, optimizer rules, and checkpoint selection identical. A matched
top-35-particle selection across the pipelines can test constituent-budget effects,
but still does not isolate architecture from representation or discretization.

These controls answer narrower, defensible questions. More compute alone cannot
remove confounding between architecture and preprocessing.

## 5. Keep quantum work and competition entry as deliberate choices

The [completed optional quantum pilot](../experiments/quantum-pilot/) uses a
specified four-feature encoding, exact simulator and matched classical controls.
The RBF SVM performed better. This is a small noiseless simulator comparison;
quantum hardware, device noise and scaling are separate research questions.

Competition entry depends on an organizer's current eligibility, dates, required
format, and authorship rules. Repository completion does not create a submission
or a prize. No entry has been made.
See [the current options and requirements](COMPETITIONS.md).
