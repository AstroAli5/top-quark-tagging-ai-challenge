# Preparing a larger, separately recorded study

The data preparer now accepts explicit training/validation/test sizes, chunk
size, output directory and a memory-planning budget. Defaults preserve the
original 50,000/10,000 fitting protocol and full official test partition.

```bash
# No download or output files: inspect the estimate first.
python scripts/prepare_official.py --train-count 100000 --plan-only

# Reuse verified raw HDF5 downloads, but preserve previous prepared outputs.
python scripts/prepare_official.py --train-count 100000 --output-dir data/scaling --memory-budget-gib 14
python scripts/verify_prepared.py --data-dir data/scaling
```

In MATLAB:

```matlab
setupProject;
runScaledExperiment(101,'data/scaling','runs/scaling_101');
```

Or use **Actions → Larger training subset → Run workflow**. Choose 10k, 25k,
50k or 100k rows; the workflow evaluates core-model seeds 101/202/303. Each run
has separate outputs. Available options do not imply that every size was run.

The planning model conservatively assumes 200 constituents per jet, coexisting
arrays and a runtime allowance. It estimates **7.18 GiB for 50k + 10k fitting**
and **12.33 GiB for 100k + 10k**. These are estimates, not measured requirements
or exact upper bounds. Real graphs are often smaller. The optional reference
model is excluded. Actual Linux MATLAB process peak resident memory is saved
to `resources.json` after a successful scaled run.

The original MAT-v5 array route cannot hold the full training matrix. The new
`run_project238` route uses Parquet chunks, tall transforms and disk-backed
image datastores instead. Its [verified full-source run](../experiments/project238-full/)
completed all 1,211,000 training and 404,000 test rows on 28 September: one epoch,
seed 101, batch size 100, CPU. Training took 35.32 minutes after 20.61 minutes of
image preparation. MATLAB peak memory was 4.76 GiB; the separate Python process
peaked at 0.47 GiB. These peaks are not an exact combined value.
Longer full-data fits and repeated seeds remain unrun; this compact CNN is a
separate pipeline from the earlier core-model studies.

The default core path remains CPU-based. No GPU acceleration result has been
measured. A future GPU path must move both data and custom GraphSAGE parameters
to supported GPU operations and verify CPU/GPU agreement on a small fixture.
[MathWorks documents the required Parallel Computing Toolbox and GPU data handling](https://www.mathworks.com/help/deeplearning/ug/run-custom-training-loops-on-gpu-and-in-parallel.html).

The current larger-subset experiment is described in
[its separate protocol](../experiments/scaling/PROTOCOL.md). It completed on
23 September 2026: [the verified 100k report](../experiments/scaling/) records
all three seeds, the matched 50k comparison, and measured MATLAB-process peak
RAM of **3.68–3.73 GiB**. The MATLAB experiment took 23.77–38.63 minutes per
seed, excluding downloads/setup. Those measurements do not establish a minimum
RAM requirement for every environment or for the optional reference model.
