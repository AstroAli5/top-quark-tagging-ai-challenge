# HDL processor assessment — estimate only

On 30 September 2026, MATLAB R2024a Deep Learning HDL Toolbox successfully
estimated the included seed-101 CNN using `dlhdl.ProcessorConfig` and
`estimatePerformance`. This is the original 50k-trained compact CNN, not the
full-source CNN or the ResNet18 pilot.

| Processor-model setting or output | Value |
| --- | --- |
| Target configuration | Xilinx Zynq UltraScale+ MPSoC ZCU102 Evaluation Kit |
| Assumed clock | 200 MHz |
| Arithmetic | Single precision |
| Frames modelled | 1 |
| Estimated network latency | 90,629 cycles / 0.453145 ms |
| Reciprocal estimated rate | Approximately 2,206.8 frames/s |
| Generated HDL / physical board execution | Neither performed |

These numbers are **estimator outputs, not measured hardware performance**.
The assumed target frequency has not been verified through synthesis or timing
closure. Host preprocessing, transfers, shared-memory contention and software
layers must not be counted as validated by this estimate. In particular, the
tool reported softmax and its output adapter as software layers. No real-time
detector, synthesis, bitstream, quantization-accuracy or hardware-equivalence
claim follows from this result.

[Exported performance table](estimated_performance.csv) · [Run report](report.json) ·
[Arithmetic and checkpoint-identity check](verification.json) ·
[Successful job](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36701925315/job/109843142706)

The check confirms that layer cycles sum to the reported total, cycles divided
by the assumed clock reproduce latency, and its reciprocal reproduces the
displayed rate. The checkpoint SHA-256 matches the permanently included model.
It does not validate the accuracy of the hardware performance model.

## Reproduce and extend

Install MATLAB R2024a, Deep Learning Toolbox, Deep Learning HDL Toolbox and
**Deep Learning HDL Toolbox Support Package for Xilinx FPGA and SoC Devices**.
Run `run_hdl_assessment` from the repository root. The
[workflow](../../.github/workflows/optional-review.yml) pins these dependencies.

The [first attempt](attempt1/report.json) stopped because the target support
package was missing. Adding it resolved that error; the failed attempt remains
recorded. The successful code head was
`4e611b76bd7fa9cfe7106b2775fa3750282c141c`, tested through merge commit
`9f77de9dfadbc3a10c979b4262e57c94d13f8709`. The
[processor object artifact](https://github.com/AstroAli5/top-quark-tagging-ai-challenge/actions/runs/36701925315/artifacts/11090786130)
expires on 29 December 2026; the compact reports stay in Git.

Completing the optional hardware extension requires an available supported
FPGA board, the applicable licensed development products and vendor tools, a
supported deployment configuration, and comparisons of predictions and measured
end-to-end latency against the MATLAB reference. This environment has no board;
those steps remain unrun.

References: [R2024a performance estimator](https://www.mathworks.com/help/releases/R2024a/deep-learning-hdl/ref/dlhdl.processorconfig.estimateperformance.html)
and [network performance-estimation guide](https://www.mathworks.com/help/releases/R2024a/deep-learning-hdl/ug/estimate-performance-of-deep-learning-network.html).
