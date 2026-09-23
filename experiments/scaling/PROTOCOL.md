# Larger-subset follow-up

Specified on 23 September 2026 before this follow-up runs. The original test
results were already known. This is not a blind evaluation or a complete
four-point learning curve.

- Fit CNN and GraphSAGE using the first **100,000 official training jets**,
  the same first **10,000 validation jets**, and seeds **101, 202, 303**.
- Keep the original architectures, preprocessing, batch sizes, learning rates,
  12 epochs, and best-validation checkpoint selection.
- Equal epochs mean more optimizer updates at 100,000 than at 50,000 jets;
  this is a combined data-and-training-budget comparison, not equal compute.
- Evaluate all 404,000 official test rows and the same 10,000-jet noise subset.
  Do not select the data size, epochs, or hyperparameters from these test results.
- Use source checksums and disjoint official partitions. Publish a separate
  three-seed report; preserve the original 50,000-jet results.
- Record a conservative memory plan and Linux MATLAB-process peak resident
  memory (VmHWM). These differ: an estimate is not a hardware measurement.
- Full 1,211,000-row training is outside this experiment. The present fitting
  array would exceed the MAT-v5 size limit and requires streamed/HDF5 input.

The workflow also supports explicitly requested 10k/25k/50k follow-ups. Those
settings are available controls, not completed experiments. GPU execution has
not been validated; this workflow uses the existing CPU training path.
