# Two additional core-model seeds

This follow-up was specified on 23 September 2026, after the original
three-seed results were known. It is not a new blind benchmark.

- Add training seeds **404 and 505** for CNN and GraphSAGE only.
- Keep the first 50,000 official training rows, first 10,000 validation rows,
  and all 404,000 official test rows, without reassignment across partitions.
- Preserve the original 12 epochs, model settings, best-validation checkpoint
  selection, noise strengths, noise streams, and 10,000-jet noise sample.
- Regenerate expired prepared data from the checksum-verified public files.
  MAT file headers contain timestamps, so regenerated file hashes may differ;
  compare source hashes, row selections, labels and preparation code separately.
- Before combining runs, verify training/evaluation implementation compatibility,
  configuration equality except seeds/paths/provenance, and saved probabilities.
- Report every new seed, including unfavorable results. Summarize all five
  core seeds separately from the preserved three-model study. The reference
  remains a **three-seed** result.
- Report mean, sample SD and uncertainty, with repeated noise draws averaged
  within each trained model. Five seeds remain a modest study.

The workflow runs once when this experiment's PR is opened and can also be
started manually. Recorded outcomes and the verified report will be added only
after the runs finish.
