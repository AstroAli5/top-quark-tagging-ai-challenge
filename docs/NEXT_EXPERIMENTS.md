# Next experiments and what they would establish

This is a research roadmap, not a list of completed results. The verified
classification/noise study remains the fixed 50,000-training-jet, three-seed
experiment. New runs must have their own protocol and output directory.

| Issue | Current action | What remains |
| --- | --- | --- |
| Too much code to navigate | A short reading path and one figure gallery | Optional code remains available without entering the main learning path |
| Expensive reference | Research workflow defaults to the two core models; reference requires opting in | Same reference architecture still costs more when selected |
| Missing final-model explanation plots | Completed and independently verified; [figures and evidence](../experiments/explanations/) | Broader perturbations or more test jets would be separate extensions |
| Only three training seeds | Preserve the recorded three-seed study | Add two core-model seeds under a separately recorded extension |
| Only 50,000 training jets | Clearly state the training subset | Measure a learning curve before committing to full-data training |
| Different representations | Describe results as comparisons of complete pipelines | Controlled ablations and matched constituent selections |
| No quantum model | Keep the central physics question focused | A quantum extension needs its own hypothesis and fair classical baseline |
| No competition entry | Keep the repository reproducible and attribution clear | Select an open, suitable competition and check its actual requirements |

## 1. Completed: explanations without training again

The analysis below completed on 22 September 2026. See
[the measured report](../experiments/explanations/README.md). The remaining
sections describe work that has not been run.

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

## 2. Add training seeds before adding another model

An affordable extension is two additional CNN/GraphSAGE seeds, 404 and 505,
with the same 50,000/10,000 fitting partitions, hyperparameters, and 12-epoch
budget. Decide the analysis before running it. Retain the original three-seed
report, and publish a separate five-seed core report that verifies data and code
compatibility. Do not present the reference as a five-seed result unless its
two additional fits are actually completed too.

The recorded core training times suggest about 30 CPU minutes for the four new
fits, excluding preparation and evaluation. This is a rough planning estimate,
not a runtime promise. Two extra reference fits would cost roughly five hours
of training under the old runner conditions, so they are a lower priority.

## 3. Measure the value of more training data

Compare nested 10,000, 25,000, 50,000, and 100,000 official training selections
using fixed validation data and repeated seeds. Make budget rules explicit:
equal epochs also increase optimizer updates when the dataset grows. Use
validation performance for development decisions and keep final testing separate.
Record the design as a follow-up informed by the existing study, not as an
experiment preregistered before the original test results were seen.

The current trainer materializes representations in memory. Full training on
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

Qiskit is not necessary to complete this classical comparison. A quantum study
would need an explicitly defined encoding, simulator/device budget, noise model,
and classical comparison on the same inputs. Installing a quantum library would
not establish a quantum advantage or strengthen the existing evidence by itself.

Competition entry depends on an organizer's current eligibility, dates, required
format, and authorship rules. Repository completion does not create a submission
or a prize. No entry has been made.
