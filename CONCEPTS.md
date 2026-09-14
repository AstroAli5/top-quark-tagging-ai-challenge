# The ideas behind the project

## What is being classified?

The input is one simulated particle jet: a collection of constituent
four-vectors, each recording energy and three momentum components.
The target is binary: label 1 means a top-quark jet and label 0 means background.
[ParticleNet](https://arxiv.org/abs/1902.08570) describes the particle-cloud view
of this classification problem; the data source is the
[Top Quark Tagging Reference Dataset](https://doi.org/10.5281/zenodo.2603256).

## Why two representations?

An image bins particles into angular cells and adds their transverse momentum.
This pipeline then applies log compression. Binning loses within-cell positions,
and the finite image window excludes particles outside it.

A graph stores four features per particle and connects nearby particles.
It retains finer position information, but its feature choice, graph construction,
and pooling also constrain what the model can learn. A graph's performance must
be measured; its representation alone does not prove that it is better.

Here the angular origin is a pT-weighted mean pseudorapidity and a circular mean
azimuth. This convenient centering convention is not identical to obtaining a
jet axis from a summed four-vector.

## How does GraphSAGE make a prediction?

For each node, average the features of its neighbors, concatenate that average
with the node's own features, and apply learned weights, ReLU, and L2
normalization. Repeat three times. Average the resulting node embeddings within
each jet and apply a sigmoid classifier.

The basic aggregation idea comes from
[Hamilton, Ying, and Leskovec](https://arxiv.org/abs/1706.02216).
This implementation uses all neighbors in its fixed k-nearest-neighbor graph.
The sparse adjacency is block diagonal during batching, so messages cannot cross
from one jet into another. Pooling uses each jet's node count to retain that boundary.

## Why does the CNN need softmax?

Its final two raw outputs are logits, which are not probabilities.
Softmax turns them into nonnegative class scores that sum to one.
The saved category order identifies which column represents signal.
Inference explicitly requests batch-by-class output, rather than guessing the
array layout. See MathWorks'
[training example](https://www.mathworks.com/help/deeplearning/ref/trainnet.html)
and [prediction documentation](https://www.mathworks.com/help/deeplearning/ref/minibatchpredict.html).

## Why report accuracy and AUC?

Accuracy measures decisions at a specified threshold; this project uses
P(signal) >= 0.5. AUC instead measures ranking across thresholds.
The empirical AUC equals the fraction of signal/background pairs where signal
gets the higher score, plus half credit for ties. Thus every score being equal
must give AUC 0.5, independent of the order of labels.

The [ROC implementation](computeROC.m) groups equal scores before adding a curve
point. It rejects a one-class evaluation because its ROC is undefined.

## What does the noise experiment establish?

Each momentum component receives independent multiplicative Gaussian smearing.
Energy is recomputed from the smeared momentum and a nonnegative original
mass-squared estimate. Both models see the same perturbed particles, and both
representations are rebuilt.

This checks sensitivity to a particular mathematical perturbation.
It does not model all detector effects: calibration, momentum, angle, particle
type, and detector region can matter in ways this simple model omits.
At large smearing, component sign changes can occur. The experiment is not a
detector validation or a deployment claim.

## What can explainability tell us?

For GraphSAGE, shuffle one feature across test-set nodes while keeping edges fixed.
AUC change measures reliance under that intervention. Correlated features may
compensate for one another, and shuffled angles can disagree with the stored graph.

For the CNN, remove pixels outside a circle centered on the image.
Radius fractions are measured relative to the center-to-corner distance;
they are not fractions of area or retained momentum.

Neither experiment identifies a causal physical mechanism. One permutation also
does not provide an uncertainty estimate. Repeat perturbations and training runs
before interpreting small differences.

## When is a result credible?

Both models must use the same held-out jets, predictions must have the right
class mapping, and no test information may guide training or checkpoint selection.
Preserve source checksums, splits, configuration, code revision, and predictions.
A synthetic integration test checks software execution; only a real-data run can
answer the research question. The [README](README.md#results-status) distinguishes
archived numbers from results that still need to be measured.
