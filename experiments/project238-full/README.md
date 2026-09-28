# Full-source feasibility: first attempt incomplete

The [predeclared one-epoch run](PROTOCOL.md) did **not** reach training.
The MATLAB step hit its 305-minute limit on 27 September 2026. No full-source
model, test accuracy or AUC resulted. The previous verified 50k/100k studies
remain valid and separate.

## Measured evidence

All 1,211,000 training rows, 10,000 validation rows and 404,000 test rows were
converted to checksum-identified Parquet in **185.72 seconds**. MATLAB reported
the training partition's tall evaluation complete after **18 minutes 48 seconds**.
The first image partition did not finish coverage validation before the timeout,
about 283 minutes after that message. Training never started.

The runner had four CPUs, about 15.61 GiB total memory and 78.55 GiB free disk
before execution. The failure is an observed time-budget limit for this
implementation, not proof of a minimum RAM requirement or a need to buy a GPU.

[Structured record](attempt1.json) · [Last saved stage](attempt1_stage_times.json) ·
[Workflow and logs](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36281531930).
The original diagnostics archive is identified by ID, SHA-256 and expiry in the
record. The original logs do not distinguish final write bookkeeping from
image-datastore construction and validation.

## Repair under validation

The row-validation loop previously requested `ds.Files` for every jet. The
repair retrieves this complete datastore property once, then indexes the local
list. Repeated property access is a suspected scaling bottleneck; the original
logs alone do not establish its share of the delay.

New timings separate image-writing return, datastore opening, row parsing and
completed validation. A separately labelled 100,000-training-row, one-epoch
check must pass before another full-source attempt. Its validation and test
prefixes each contain 1,000 rows. These execution probes do not establish model
convergence, multi-seed uncertainty or reviewer acceptance.
