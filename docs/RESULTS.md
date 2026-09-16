# Results and evidence

## Larger official-partition study

The three-seed study is running from commit
`0f7ae539735f1aa5ee113c84bf36a17f4f366e2c`.
[Execution record](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35042290356).

The fixed design uses 50,000 training jets, 10,000 validation jets, and the full
official test partition, with training seeds 101, 202, and 303 and 12 epochs per
model. Noise evaluation uses the same first 10,000 test jets for all models and
three perturbation seeds. See the [protocol](EXPERIMENT_PROTOCOL.md).

There are no verified larger-study scores yet. An earlier attempt stopped at a
partition check before training because a duplicate root script shadowed the
implementation. That duplication was removed and is now covered by a regression
check. No metrics from that failed attempt are reported.

## Verified small real-data run

[Completed run](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/34936617829),
commit `ec0e3402c1c5130c2d00e975b941b64cd3308d1f`.

This used the first 2,000 rows of the official **training** file, split into
1,400 training, 300 validation, and 300 internal test jets. All three models
trained for three epochs. These measurements verify execution on real data;
the small holdout and single training seed limit scientific conclusions.

| Model | Accuracy | Clean AUC | AUC at 10% smearing | AUC at 20% smearing |
| --- | ---: | ---: | ---: | ---: |
| CNN | 83.33% | 0.93973 | 0.9223 | 0.8643 |
| GraphSAGE | 63.00% | 0.66275 | 0.6294 | 0.6227 |
| ResNeXt-SE reference | 86.33% | 0.94658 | 0.9103 | 0.8399 |

Values above were read from the completed MATLAB logs and saved result tables.
Clean accuracy and AUC were independently recomputed from all 300 saved prediction
rows in Python and matched the reported values. The run's artifact
contains its models, predictions, configuration, result tables, and plots.
The reference has the highest clean score in this small run; the CNN has the
higher AUC at the two shown nonzero noise levels. This is not a general model
ranking or an exact reproduction of the 2025 winner.

## Historical outputs

The [original CSVs and images](../archive/original-results/) are preserved
unchanged. Their clean values were CNN accuracy 50.29%, AUC 0.1961, and GraphSAGE
accuracy 83.57%, AUC 0.9088. They came from earlier code with unresolved
probability mapping and ROC issues.

Those files are historical records, not validated evidence for the corrected
code. The old and new runs also use different budgets and data selections, so
their numbers do not support a fair before-and-after accuracy claim.
