# Controlled graph-neighbor ablation

Specified 25 September 2026, before these ablation results were measured, and
after the baseline study was known. This is a follow-up, not a blind benchmark.

Compare the completed 100k GraphSAGE fits with the **same network trained using
zero graph edges** (`kNeighbors=0`). Keep seeds 101/202/303, all four node features,
particle selections, 100,000 training rows, 10,000 validation rows, layer widths,
initialization, optimizer, batch size, 12 epochs and validation-loss checkpoint
selection identical. Reuse the unmodified training and evaluation implementation.

With no edges, the neighbor aggregate is zero. The self-feature pathway,
normalization, mean pooling and classifier remain active. The nominal parameter
shapes are unchanged, but neighbor weights have no effect and receive zero
gradients. This isolates access to neighbor messages within this implementation;
it does not isolate CNN versus graph architecture or equalize effective capacity.

Evaluate all 404,000 official test rows, plus the same paired noise design on
10,000 test jets with streams 7/17/27 and strengths 0/.05/.10/.20/.35. Publish all
three fits, clean and noise metrics, paired per-seed differences and 95% Student-t
seed intervals. These intervals hold the test sample fixed and use only three
seed labels. Do not select the variant based on its test results.

Verify unchanged trainer/evaluator Git blobs, matching source hashes and
configuration except paths, provenance, selected model and `kNeighbors`. Check
every metric against saved probabilities and match the baseline test rows/labels.
This study makes no prior prediction that either variant will win.
