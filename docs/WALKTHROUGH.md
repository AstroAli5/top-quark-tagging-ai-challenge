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

Start with `verify_results`: it restores saved models on a small official sample,
so you can check predictions before attempting another training run.

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
