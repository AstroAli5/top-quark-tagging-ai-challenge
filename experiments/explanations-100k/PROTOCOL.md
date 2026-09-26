# Explanations of the 100k checkpoints

Specified 25 September 2026, after the clean/noise results were known.

Reuse all six frozen CNN/GraphSAGE checkpoints for seeds 101/202/303 from run
35842565638. Evaluate the first 10,000 official test jets. Use exactly the
existing explanation procedure: graph-feature permutations with fixed edges
and streams 11/21/31; CNN radial masks retaining fractions 1, .75, .5, .35, .2
of the center-to-corner radius. Average permutation repeats within each fit,
then report mean and sample SD across the three training seeds.

Require checkpoint hashes, rows and labels to match the source artifacts, and
restore clean predictions before proceeding. Recompute every reported AUC from
saved probabilities. No fitting, hyperparameter selection or checkpoint
selection occurs. These perturbations measure sensitivity and can break feature
correlations; they do not establish causality or detector realism.

The explanation preparer downloads the checksum-verified test source only and
materializes the first two test chunks. Its manifest retains the source study's
boundaries and explicitly records that only an explanation prefix exists.
It does not prepare training data or claim another full-test evaluation.
