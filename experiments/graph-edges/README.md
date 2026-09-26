# What the graph edges contribute

Specified on 25 September and independently verified on 26 September 2026.
[Protocol](PROTOCOL.md) · [completed MATLAB run](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36158770932) · [verification and intervals](report.json) · [artifact identities](artifacts.json)

The same GraphSAGE implementation trained on 100,000 jets, with 10,000 official
validation jets and all 404,000 official test jets. Seeds 101, 202 and 303 each
ran 12 epochs. The control sets the neighbor count to zero; node features,
trainer, optimizer, initialization rules and validation selection are unchanged.
Source fingerprints, official row selections, saved checkpoint configurations,
and every clean/noise metric were checked before comparison.

| Model | Mean accuracy ± seed SD | Mean AUC ± seed SD |
| --- | ---: | ---: |
| GraphSAGE, six neighbors | 87.40% ± 0.26 percentage points | 0.93460 ± 0.00122 |
| Same network, no neighbors | 83.77% ± 0.19 percentage points | 0.89371 ± 0.00061 |

Removing messages reduces clean AUC in all three runs. The paired mean AUC
change (zero minus six neighbors) is -0.04088. The 95% Student-t interval across
three seed pairs is recorded in the verification report. These intervals
describe seed variability on one fixed test set, not all sources of uncertainty.

![Clean and noisy graph-edge comparison](graph_edge_comparison.png)

This chart was rendered in Python from verified MATLAB predictions. At 35%
synthetic component smearing, the zero-neighbor model has higher AUC: its mean
paired advantage is 0.03156, with a seed interval [0.00596, 0.05716]. At 20% the
interval includes zero. Graph edges therefore help the clean result in this
experiment but are not uniformly better across the tested distortions.

The noise analysis uses 10,000 fixed test jets and three noise streams,
averaged within each training seed. See [per-seed clean scores](per_seed_clean.csv),
[per-repeat noise scores](per_seed_noise.csv) and [paired noise intervals](noise_paired.csv).

This is a message-passing ablation. With zero edges, the neighbor weights receive
zero gradients, so effective capacity changes even though parameter shapes are
identical. It does not isolate CNN-versus-graph architecture, establish a universal
model ranking, or validate a detector simulation. No tuning followed test outcomes.

## Reproduce the verification

Download the baseline artifacts listed in `../scaling/artifacts.json` and the
three artifacts listed here. Arrange each under `baseline/seed-SEED/{models,results}`
and `variant/seed-SEED/{models,results}`. Use a clone with the recorded source
commits available (do not use a shallow clone).

```bash
python scripts/compare_graph_edges.py --baseline baseline --variant variant --output results/graph-edges
```

The raw prediction artifacts currently expire on 24 December 2026. The tables,
figure, checksums and verification report here are retained in Git.
