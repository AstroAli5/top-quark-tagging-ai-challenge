# A student walkthrough

You can understand the central project without reading the optional reference
model or every helper. Start with the [figure gallery](FIGURES.md), then read
the sections below. When you want the code, follow this short path:

| Question | File to read |
| --- | --- |
| What happens in what order? | [`run_all.m`](../run_all.m) |
| How do particles become images and graphs? | [`s2_build_representations.m`](../src/pipeline/s2_build_representations.m) |
| How does the image model learn? | [`s3_train_cnn.m`](../src/pipeline/s3_train_cnn.m) |
| How does the graph model learn? | [`s4_train_graphsage.m`](../src/pipeline/s4_train_graphsage.m) |
| How is the same noise applied to both models? | [`s6_robustness_test.m`](../src/pipeline/s6_robustness_test.m) |

Follow a helper only when its role is unclear. The official evaluation and
reference folders are extensions to this reading path.

## 1. The question

A jet is a collection of particles. Here, each particle has energy and three
momentum components. The task is to predict whether the jet came from a top quark
(label 1) or the background process (label 0).

The question is: **how do image and graph classifiers change when the same jet
measurements are perturbed?**

## 2. Two views of the same jet

The CNN receives an image. Nearby particles fall into angular pixels, and the
pixel values describe their summed transverse momentum.

GraphSAGE receives particles as nodes connected to nearby nodes. It combines
each particle's features with its neighbors' features, then pools the particle
representations into one jet prediction.

The optional ResNeXt-SE reference uses a richer image representation and radial
features. It is an additional comparison, not a replacement for the central
CNN/GraphSAGE question. See [the reference explanation](WINNER_COMPARISON.md).

## 3. Keep learning and evaluation separate

Training jets update the weights. Validation jets select the checkpoint.
Test jets measure performance after those choices are fixed.

The introductory `run_all` makes a split within one training subset.
The larger `run_experiment` uses the dataset publisher's separate train,
validation, and test files. Its 50,000 training jets are a subset of the
1.2 million available; its clean evaluation covers the full official test file.

## 4. Read the two main scores

**Accuracy** is the fraction of correct classifications at the fixed probability
threshold of 0.5.

**AUC** measures ranking: how often does a signal jet receive a higher score than
a background jet? Ties receive half credit. An AUC of 0.5 indicates chance-level
ranking; an AUC of 1 means perfect ranking on that test set.

Both scores depend on the selected data and trained model. Compare the models
within the same experiment. Different datasets or training budgets do not give
a fair before-and-after comparison.

## 5. Read the noise experiment

The code perturbs momentum components with random multiplicative noise and
recomputes energy. All models receive the same perturbed jets.

The larger run uses the first 10,000 official test jets for this more expensive
experiment. Its zero-noise point uses those same 10,000 jets, so it can differ
slightly from a score measured on the full clean test file.

Three training seeds measure variation from training. Three noise realizations
measure variation from the perturbation. They are different sources of
variation, so the analysis does not count them as nine independent models.

This is a synthetic sensitivity study, not a realistic simulation of every
detector effect.

## 6. Explain the result

Use the [results page](RESULTS.md) to answer:

1. Which model scores highest on the clean test set under this budget?
2. How much does each model's AUC change on the same jets as noise increases?
3. Are the differences consistent across training seeds, and how wide is the
   uncertainty?

A larger model need not be best under every budget or perturbation. Report what
was measured, including weak or unstable results.

For more detail, read [Concepts](CONCEPTS.md) and the
[fixed protocol](EXPERIMENT_PROTOCOL.md). To run the project, return to the
[README](../README.md).

## The big-data route added after review

Start with `run_submission`: it restores all three included seed-101 models on a
small official sample and recreates the MATLAB study report, so you can check
predictions before attempting another training run.

The new training route has four data-handling steps:

1. `run_project238` calls Python from MATLAB. HDF5 rows are read in small blocks
   and written to Parquet, preserving official train/validation/test separation.
2. `parquetJetsToImages` creates a tall table. MATLAB evaluates one block at a
   time and converts each jet to a grayscale image with `buildJetImage`.
3. Images are stored as lossless floating-point TIFFs in background/signal
   folders. `imageDatastore` gets each label from its folder. No image is
   quantized to an 8-bit display picture.
4. `trainProject238` reads minibatches from those files, selects its CNN using
   validation loss, and evaluates the untouched test partition. Only small
   prediction/label arrays, not every jet image, are kept for the final metrics.

Normalization is fitted using training images only. Source row IDs stay in the
filenames so missing, duplicate or reordered test rows can be detected. The
[verified full-source CPU run](../experiments/project238-full/) completed one
epoch on all 1,211,000 training jets and evaluated all 404,000 test jets. It proves
the data route works at that size; one epoch and one seed do not prove convergence
or statistical consistency across training runs.

You should be able to explain why the validation partition selects the model,
why the final test partition cannot be used for tuning, what ROC AUC measures,
and why our noise model is only a simplified sensitivity experiment. The
[AI disclosure](AI_ASSISTANCE.md) records the assistance in these changes.

## Prepare to explain the revision to a reviewer

Use this as a practice route, not a script to memorize. If you cannot explain a
step in your own words, return to that file before representing it as understood.

| Show or explain | What you should be able to demonstrate |
| --- | --- |
| Start with `run_submission` | Explain that it restores three saved models on 256 jets and regenerates recorded study summaries. It does not retrain or recompute all 404,000 predictions. |
| Open `buildJetImage` | Start from `(E, px, py, pz)`. Explain transverse momentum, relative angular coordinates, summed momentum per pixel, `log1p`, and why the TIFFs retain floating-point values. |
| Trace `run_project238` | Point to `pyrun`, bounded HDF5 conversion, Parquet, the tall transform, labelled image files, `trainnet`, and `minibatchpredict`. Explain how minibatches bound RAM use. |
| Follow one source row | Its official partition stays fixed and its ordinal appears in the image filename and prediction table. Show the coverage and label checks in `jetImageDatastore` and `verify_project238.py`. |
| Explain model selection | Weights and input normalization use training data. Validation loss selects the checkpoint. Test labels are used only for final metrics; changing hyperparameters after seeing those scores would need a fresh evaluation plan. |
| Distinguish the studies | The 1,211,000-jet CNN is one epoch and one seed. The original three-model study uses 50k training jets, three seeds and 12 epochs. Their scores answer different questions. |
| Interpret uncertainty | A seed is a separate training run. Noise realizations within a trained model are averaged first. Seed intervals summarize modest observed training variation; they are not a guarantee for new detector data. |
| Explain the optional ResNet18 pilot | It uses the named MathWorks residual architecture with random weights, adapted to grayscale images and two classes. Both CNNs share the selected data and training settings. This is not ImageNet transfer learning. |
| State what you contributed and where AI helped | Use the disclosure accurately. Explain the code you submit, including fixes, choices, negative findings and limits; do not describe assistant-run work as unaided implementation. |

Practice these questions without the notes, then check your answer:

1. **Why can accuracy and AUC disagree?** Accuracy uses one threshold; AUC
   measures ranking across thresholds. Neither establishes good calibration.
2. **Why do 404k test jets not mean we trained on every available jet?** Training
   and test sizes are independent choices. The repository reports both explicitly.
3. **Why not call the reference a reproduction of the winner?** Its architecture,
   inputs and budget differ. It is an attributed independent adaptation.
4. **Does the strongest noise AUC make GraphSAGE universally better?** No. It
   performs worse on clean jets in the original study; the perturbation is synthetic.
5. **Does a processor latency estimate prove real-time FPGA performance?** No.
   Synthesis, timing closure, board execution, transfers and end-to-end validation
   would still be needed.

A useful final exercise is to change only the small demonstration's output
folder, run it, and locate its configuration, prediction rows, report and figure.
Explain why existing fitted outputs are protected. This guide cannot certify
the author's understanding; that requires the author's own work and discussion.
