# Verified official-partition study

Read [Results](../../docs/RESULTS.md) for the interpretation and
[the frozen protocol](../../docs/EXPERIMENT_PROTOCOL.md) for the design.
This folder preserves the measured report, all per-seed metric tables,
uncertainty summaries, figures, and artifact checksums in Git.

The experiment trains on 50,000 official training jets, selects checkpoints on
10,000 official validation jets, and evaluates every one of the 404,000 official
test jets. Each model has three training seeds and a 12-epoch budget. It is a
training-subset study, not a full-training-data leaderboard result.

## Evidence

- [Core evaluation and summary](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35095505566)
- [Reference evaluation and combined summary](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35148856141)
- [Core training](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35042290356)
- [Reference training](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/35094350524)

The two training workflows stopped after saving completed models: the first
timed out in its additional reference stage, and the second encountered the
old test-reader restriction. The evaluation workflows reuse the finished
checkpoints. Training and evaluation revisions are recorded separately in
`report.json`; failed or incomplete fits are not counted as results.

`clean_summary.csv` contains means, seed SDs, and 95% seed intervals.
`per_seed_clean.csv` also includes fixed-model test-sampling intervals.
`noise_summary.csv` averages noise draws within each fitted model before
calculating between-seed variation. `paired_auc.csv` accounts for shared test
events when comparing clean AUCs. `artifacts.json` identifies the source archives
and their SHA-256 digests.

## Recalculate from saved predictions

Download the three `research-core-seed-*` artifacts from the core evaluation
and the three `research-reference-seed-*` artifacts from the reference evaluation.
Keep each artifact's own directory under a new `downloads/` folder. With the
GitHub CLI, from a full clone of this repository:

```bash
gh run download 35095505566 --repo AstroAli5/top-quark-tagging-ai-challenge --pattern 'research-core-seed-*' --dir downloads
gh run download 35148856141 --repo AstroAli5/top-quark-tagging-ai-challenge --pattern 'research-reference-seed-*' --dir downloads
python -m pip install -r requirements.txt matplotlib
python scripts/combine_study_runs.py --input downloads --output runs/recheck
python scripts/summarize_experiment.py --input runs/recheck --output results/rechecked
```

The combiner verifies common data, settings, labels, row IDs, and unchanged shared
evaluation code. The summary recalculates every clean metric, checks noise scores
within float32 rounding bounds, and rejects incomplete runs. The analysis script's
SHA-256 is recorded in the report. A separate local calculation using average
score ranks also verifies clean AUC, independently of the summarizer's algorithm.

Workflow artifacts retain checkpoints and per-jet predictions for 90 days
(through 15 December 2026 for these runs). The compact evidence in this folder
remains in Git. After artifact expiry, use the [research workflow](../../.github/workflows/research.yml)
or the README's commands to train and evaluate fresh runs. Timing depends on the
runner; recovered core training times are estimates from log timestamps.
