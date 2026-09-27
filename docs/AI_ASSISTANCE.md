# Authorship and AI assistance

Ali Mohamed supplied the original repository and its research question:
comparing CNN and GraphSAGE top-quark tagging under synthetic detector noise.
Earlier repository documentation credited Claude with development assistance.

OpenAI Codex assisted with the code audit, bug fixes, tests, workflow automation,
data provenance, the independently implemented ResNeXt-SE reference, the larger
experiment, and repository organization and explanation.

The September 2026 extensions also used Codex to specify and implement the
additional-seed and larger-subset workflows, memory controls, source-compatibility
checks, and four-qubit Qiskit simulator pilot. Codex launched the additional
MATLAB studies through GitHub Actions, executed the simulator, and independently
recalculated the reported metrics and matched training-size comparison. These contributions are not
being represented as unaided student work. Review [competition-specific
requirements](COMPETITIONS.md) before using the repository in an entry.

On 25 September, Codex also specified and launched the 100k checkpoint
explanations and controlled graph-neighbor ablation, implemented the test-prefix
preparer and comparison verifier, and prepared the full-data/hardware assessment.
The recorded explanation and graph-control results were independently checked
from saved scores. After the author supplied the 26 September review, Codex
corrected its earlier focus on the wrong competition context and prioritized
the actual Project 238 MATLAB big-data requirements.

Measured scores must come from the linked execution records and saved
predictions. A published winner's score is not a measurement of this code.
Passing software tests does not establish the scientific conclusion.

## Attribution

The reference comparison credits Adit Shah's 2025 project and the ResNeXt and
squeeze-and-excitation papers in [WINNER_COMPARISON.md](WINNER_COMPARISON.md).
It is an independent implementation with explicit differences, not an exact
reproduction of the winning submission.

The original graph comparison was inspired by
[Colin Crovella's MATLAB demo](https://www.mathworks.com/matlabcentral/fileexchange/181442-graphsage-classifier-for-top-quark-tagging)
and the [GraphSAGE paper](https://arxiv.org/abs/1706.02216).
The [dataset](https://doi.org/10.5281/zenodo.2603256) has its own CC BY 4.0 license;
the repository's MIT license applies to its code.

The author remains responsible for reviewing and understanding the code,
attribution, measurements, and claims. This disclosure does not assert that
author review or a competition submission has occurred.

## Project Hub requirements checked on 26 September

The [official AI guidelines](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/wiki/Generative-AI-Guidelines)
allow assistance but require the author to understand, explain, verify and
acknowledge the work. Tests executed by an assistant do not demonstrate the
author's own understanding. No such understanding or organizer acceptance is
claimed here. The Parquet/tall/datastore workflow, quick checkpoint verifier,
MATLAB reporting and associated tests also received substantial Codex assistance.
