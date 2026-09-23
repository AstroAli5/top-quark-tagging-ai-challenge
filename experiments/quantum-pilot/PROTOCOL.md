# Four-qubit simulator pilot

Specified on 23 September 2026 before running this pilot. The earlier classical
study's test results were already known. This is a follow-up, not a blind benchmark.

**Question:** On the same small, four-feature top-tagging problem, how does a
simulated quantum fidelity kernel compare with linear and RBF SVM kernels?

- Use checksum-verified prepared data from Zenodo 2603256 (CC-BY-4.0).
- Select 512 training jets and 256 validation jets, balanced within each class,
  from their original 50,000/10,000 fitting pools. Repeat the sampling with
  seeds 17, 29 and 43. These are **sampling repeats**, not stochastic quantum fits.
- Use the same fixed, class-balanced 2,000 test jets sampled with seed 73 from
  the first 5,000 official test rows. Never move rows between source partitions.
- Four derived features: jet mass / jet transverse momentum, pT-weighted radial
  width (girth), sqrt(sum constituent pT squared) / sum constituent pT, and
  log(1 + valid constituent count). Use positive-energy, positive-pT constituents.
- Fit MinMax scaling into [0, pi] on each training sample only. Clip subsequent
  validation/test inputs with this frozen transform. All models see these same
  four values; report the clipping rate.
- Quantum map: Qiskit `zz_feature_map`, four qubits, two repetitions, linear
  entanglement. Exact noiseless CPU statevector simulation; no device, shots,
  error mitigation, or quantum speedup claim.
- Kernel: squared absolute inner product of encoded statevectors. The SVM
  optimizer is classical. Check kernel symmetry, diagonal and positive
  semidefiniteness before fitting.
- Baselines: linear SVM and RBF SVM (`gamma='scale'`). Every model tries C in
  [0.1, 1, 10, 100]; choose highest validation AUC, breaking ties by smaller C.
  Do not refit or alter the protocol after looking at test results.
- Report test AUC and threshold-zero accuracy, validation selections, per-repeat
  scores, wall times, packages, data hashes, and means/sample SD across repeats.
  Independently recalculate metrics from saved decision scores. No calibration
  claim: decision scores are not probabilities.

The test sample and feature restriction differ from the full MATLAB study.
These results cannot establish an advantage over its CNN or GraphSAGE models.

References:
- [Dataset](https://zenodo.org/records/2603256)
- [Qiskit ZZ feature map](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.zz_feature_map)
- [Qiskit Statevector](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector)
