# MATLAB, Google Colab, and Qiskit

| Platform | What this repository supports | Access needed |
| --- | --- | --- |
| MATLAB | All three models, synthetic-noise evaluation, tests, and plots | MATLAB R2024a+ and Deep Learning Toolbox; local, MATLAB Online, or the configured GitHub Actions runner |
| Google Colab | A Python notebook to download, verify, and convert the official training data into a MATLAB input file | Open the notebook in your own Colab session |
| Qiskit | A possible separate quantum-computing experiment; not a dependency of these classical models | Local SDK for simulation; your own IBM Quantum credentials for cloud hardware |

## MATLAB

The [test workflow](../.github/workflows/tests.yml) runs MATLAB R2024a with Deep
Learning Toolbox. The manual [small benchmark workflow](../.github/workflows/benchmark.yml)
downloads the official training file and runs the 2,000-jet, three-epoch example.
Open the repository's **Actions** tab, select **Small real-data benchmark**, and
choose **Run workflow**. Successful runs provide model and result artifacts,
including metadata. Check the run's commit before comparing results.

GitHub Actions is a bounded CPU execution route, not an interactive MATLAB
desktop. The [official-partition workflow](../.github/workflows/research.yml) also runs the
larger three-seed study. Its reference jobs allow up to six hours; use your own
MATLAB environment for experiments requiring more time.
MATLAB Online also requires signing in to your MathWorks account and having the
appropriate product access. No account credentials are included in this project.

## Colab

[Open data-preparation notebook](https://colab.research.google.com/github/AstroAli5/top-quark-tagging-ai-challenge/blob/main/notebooks/prepare_data_colab.ipynb).

The notebook downloads approximately 1.04 GB, verifies the source checksum,
converts a chosen subset, and lets you download the resulting MAT file. Move the
MAT file to the repository's `data/` folder in MATLAB, then run the commands shown
in the notebook. It contains no fabricated outputs or pre-executed training.

A standard Colab Python runtime does not supply MATLAB plus Deep Learning
Toolbox, so this notebook prepares data; it does not execute the MATLAB models.
Compute availability and session limits vary; see the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Qiskit

Qiskit is a Python quantum-computing SDK, not a MATLAB execution service. See
IBM's [installation guide](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit).
The models here use classical neural networks. Adding a quantum classifier would
be a separate research comparison requiring a small, explicit feature encoding,
classical controls, and its own evaluation; it would not by itself improve the
current score or reproduce the 2025 winner.

Opening a notebook or preparing SDK code does not grant access to your Colab,
MathWorks, or IBM Quantum account. Keep account authentication in the platform's
own sign-in flow.
