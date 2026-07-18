# The ideas behind this project, in plain English

You should be able to explain every one of these in your own words — to a
judge, in your video, or if anyone asks. That's the actual goal, not just
having code that runs.

## 1. What is a "jet," and what are we actually classifying?

When particles collide at high energy, some of the resulting particles
fly outward in a tight, narrow spray called a jet. A top quark is
unstable and decays almost instantly; if it decays hadronically and has
enough momentum ("boosted"), all of its decay products land inside a
single jet. The question this project answers, jet by jet: *does this
spray of particles look like it came from a top quark decay, or from
ordinary background QCD processes (light quarks/gluons)?* This is a
binary classification problem: one jet in, one probability out.

Each jet is described by its constituent particles — up to 200 of them —
each with a measured energy and momentum (E, px, py, pz). That's the raw
input to both models.

## 2. Two different ways to represent the same jet

- **As an image (for the CNN):** bin the particles into a 2D grid based
  on their angular position relative to the jet's central axis
  (pseudorapidity eta and azimuthal angle phi), with each pixel holding
  the summed momentum of particles landing there. This turns a jet into
  something that looks like a small grayscale photo, so an ordinary image
  classifier can be applied to it.
- **As a graph (for GraphSAGE):** treat each particle as a graph node,
  and connect each particle to its nearest neighbors in angular distance.
  Unlike the image, nothing is binned or blurred together — every
  particle keeps its own exact position and momentum, and the network
  reasons about actual relationships between specific particles instead
  of pixel intensities.

Neither representation is strictly "correct" — they're different lossy
compressions of the same underlying physics, which is exactly why
comparing the two models trained on them is informative.

## 3. What does GraphSAGE actually do, layer by layer?

GraphSAGE builds up each particle's representation by repeatedly looking
at its neighbors:

1. For every particle, average the feature vectors of its connected
   neighbors.
2. Concatenate that neighbor-average with the particle's own current
   feature vector.
3. Pass the concatenated vector through a learned linear layer and a
   ReLU nonlinearity to get the particle's *updated* feature vector.

Stack this 3 times, and a particle's final representation has been
shaped by information from neighbors-of-neighbors-of-neighbors — a
progressively wider view of the jet's structure. Finally, **average every
particle's final representation together** to get one vector describing
the whole jet (this step is called a "readout" or "pooling" operation),
and pass that through one more small layer to get a single top-quark
probability.

This project uses GraphSAGE's *mean* aggregator — the simplest of a
handful of options in the original paper (Hamilton, Ying & Leskovec,
2017) — specifically because it's easy to reason about and to explain:
"a particle's new representation blends what it already knew with the
average of what its neighbors knew."

## 4. Why compare against a CNN at all?

Because a number in isolation ("GraphSAGE gets 93% accuracy") is much
less informative than a number in context ("GraphSAGE and the CNN get
similar accuracy on clean data, but GraphSAGE holds up better under
noise"). The CNN is also simpler and much better understood, which makes
it a fair, credible reference point — if GraphSAGE can't beat or at least
match it, that's important to report honestly, not hide.

## 5. Why inject artificial detector noise at all — isn't the data already realistic?

The training data is Monte Carlo *simulation*: PYTHIA8 generates the
underlying physics, and Delphes simulates an idealized detector response.
Real detectors have imperfections beyond what any fast simulation
captures, and those imperfections can change over a detector's lifetime,
differ between detector regions, and so on. A model that only ever sees
clean simulated data has never had to prove it can cope with any of that.
Injecting controlled, synthetic noise and watching what happens to
accuracy is a simple, honest way to ask "how much would this degrade in
a less-than-ideal measurement, and does that failure happen gracefully or
suddenly?" — a question the reference material this project builds on
never asks.

The noise model here (independently scaling each momentum component by a
random factor, then recomputing energy to keep the particle physical) is
a deliberate simplification, not a real detector simulation — real
resolution effects depend on particle energy, type, and detector region
in ways this doesn't capture. That's stated plainly in the README rather
than dressed up as more rigorous than it is.

## 6. What does AUC mean, and why use it instead of just accuracy?

Accuracy depends on a single decision threshold (here, 0.5), which can
be misleading if the two classes aren't perfectly balanced or if you
care about a specific operating point (e.g., "how much background can we
reject while keeping 50% of the signal?" — a standard way results are
reported in this field). AUC (area under the ROC curve) summarizes
performance across *every possible threshold* at once: it's the
probability that the model ranks a randomly chosen signal jet above a
randomly chosen background jet. An AUC of 1.0 is a perfect ranking; 0.5
is random guessing. Reporting both accuracy and AUC gives a fuller,
harder-to-cherry-pick picture than either alone.

## 7. What does "permutation feature importance" actually show?

Take one input feature (say, deltaEta) and shuffle its values randomly
across every jet in the test set, while leaving every other feature
untouched. This destroys any real relationship between that feature and
the outcome, while keeping the feature's overall distribution the same.
Re-run the trained model and see how much AUC drops. A feature the model
depends on heavily will cause a big drop when shuffled; a feature the
model barely uses will barely matter. It's a simple, model-agnostic way
to get a rough sense of "what is this network actually paying attention
to?" without needing to open up its internals — the trade-off is that it
only tells you about individual features in isolation, not about
interactions between them.
