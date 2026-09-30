# Turning this repository into published research

Written for the repository owner. Everything below refers to measurements already
committed here; the numbers were re-derived from `experiments/*/report.json` rather
than copied from prose.

## 1. What is already publishable

Three things in this repository clear the bar for a peer-reviewed venue.

**The neighbor ablation is the strongest result, and it is not the one the README
leads with.** `experiments/graph-edges/` trains the same GraphSAGE with six angular
neighbors and with none, over three matched seeds, on 100k training jets, evaluated
on all 404,000 official test jets. Architecture, preprocessing, parameter count and
input representation are held fixed; only access to neighbor messages changes. The
paired difference (k=0 minus k=6) crosses zero:

| Smearing | Mean AUC difference | Seed CI |
| ---: | ---: | --- |
| 0.00 | -0.0447 | [-0.0486, -0.0407] |
| 0.10 | -0.0382 | [-0.0392, -0.0371] |
| 0.20 | -0.0122 | [-0.0269, +0.0024] |
| 0.35 | **+0.0316** | [+0.0060, +0.0572] |

Edges help on clean jets and hurt at the strongest tested smearing, with seed
intervals that exclude zero at both ends. That is a controlled claim about one
mechanism, which is what a referee wants.

**The full-partition run is a credible systems result.** One epoch, one seed, all
1,211,000 official training jets and all 404,000 test jets on a CPU runner: ROC AUC
0.97395, accuracy 90.23%, 35.32 minutes of training, 4.76 GiB peak MATLAB memory.
Every test prediction was checked against official source rows. For a
reproducibility or tooling venue this carries a paper on its own.

**The negative quantum result is worth reporting.** A four-qubit fidelity-kernel SVM
at AUC 0.880 against a matched classical RBF SVM at 0.945, on identical features and
splits over 2,000 balanced test jets (Qiskit statevector on CPU, no shot noise).
Matched-baseline negative results in quantum machine learning are under-published
and easy to referee — but report the 2,000-jet denominator and the absence of a
shot-noise model, or the first referee will find both.

## 2. What a referee will push on first

### The three-model table is confounded

CNN, GraphSAGE and the ResNeXt-SE reference differ in architecture *and* input
representation *and* preprocessing at once. "GraphSAGE is more robust to smearing"
cannot be attributed to the graph from that table. Worse, it sits awkwardly beside
your own ablation: if neighbor messages *hurt* at sigma = 0.35, then whatever makes
GraphSAGE degrade more gently than the CNN is not the graph structure. It is more
likely the node features — (dEta, dPhi, log pT, log E) per particle survive smearing
differently from a 32x32 binned pT image, where smearing moves energy across pixel
boundaries.

Say this explicitly rather than waiting to be asked. Then test it: build a
particle-cloud model with the same node features and no edges at all (you already
have k=0), and a CNN trained on the same constituents. The comparison you can
actually defend is representation-versus-representation at matched capacity.

### The noise model is not a detector

`injectDetectorNoise` multiplies each momentum component by (1 + sigma * N(0,1)) and
recomputes energy at fixed mass. Real calorimeter and tracker resolution depends on
energy, particle type and detector region, and sigma = 0.35 per component is far
outside any real working point. The README already flags this. For publication that
is not enough — a referee in hep-ph will stop reading at a 35% flat smear.

Two options, in increasing order of cost:

1. Keep the synthetic smear, but reframe the paper as a *stress test of
   representation sensitivity*, not a detector study. Drop "detector" from the
   framing entirely and justify the sigma grid as a sweep, not a working point.
2. Parameterize sigma(E) from a published resolution curve (Delphes CMS/ATLAS cards
   are the usual reference and are freely available) and add one calibrated working
   point alongside the sweep. This is the version that gets into a physics venue.

Option 1 is honest and cheap. Take it for a first paper.

### Absolute performance is below the field

Kasieczka et al., *The Machine Learning Landscape of Top Taggers*
(SciPost Phys. 7, 014 (2019), arXiv:1902.09914), benchmarks this exact dataset;
the leading taggers there sit near AUC 0.98. Your best is 0.9716 at 50k jets and
0.97395 on the full partition at one epoch. Pull the exact per-model numbers from
that paper's comparison table and put them in your own table.

This is fine — as long as the paper is not claiming performance. A robustness or
ablation paper is *allowed* to be below the state of the art, provided it says so on
page one and shows that the gap does not drive the effect. Quietly omitting the
comparison is what gets a paper desk-rejected.

### Smaller things worth pre-empting

- Three to five seeds is thin. Your seed SDs are small enough to carry it, but say
  how many and show the per-seed points, not just error bars. `per_seed_clean.csv`
  is already in every study folder.
- Noise metrics use 10,000 test jets while clean metrics use 404,000. State both
  denominators in every table caption.
- The 50k to 100k comparison holds epochs fixed, so it also doubles the optimizer
  updates. `experiments/scaling/` already says this; keep that sentence in the paper.

## 3. Where to send it

Roughly in order of how much work stands between here and acceptance.

| Venue | Fit | Notes |
| --- | --- | --- |
| **ML4PS workshop at NeurIPS** | Best first target | ~4 pages, welcomes ablations and negative results, non-archival so a journal paper later is not blocked |
| **EPJ Web of Conferences** (ACAT, CHEP proceedings) | Strong | Requires attending or presenting; very receptive to reproducible-tooling work |
| **Machine Learning: Science and Technology** (IOP) | Strong, more work | Full journal paper; wants the calibrated noise model |
| **SciPost Physics Codebases** | Good for the pipeline itself | Open refereeing, explicitly values reproducibility infrastructure |
| **JOSS** | For the software only | Needs the software framed as research-enabling, not as a study |
| **arXiv** (hep-ph or cs.LG) | Do this regardless | Free, immediate, citable, and standard practice in this field |

Post to arXiv first, then submit. Nothing here is under embargo, and the timestamp
protects priority on the ablation.

## 4. Making the artifact citable

The CI evidence currently points at GitHub Actions run URLs. Those expire — logs and
artifacts are retained for a limited window, and a referee two years from now will
find dead links. Move anything load-bearing into the repository, then archive it:

1. Connect the repository to Zenodo (Zenodo, then GitHub, then enable the repo), and
   cut a GitHub release. Zenodo mints a DOI for that exact commit.
2. Add the DOI to `CITATION.cff` as an `identifiers:` entry, and add `version:` and
   `date-released:`. The file currently has neither, so citations cannot pin a
   version.
3. Add the DOI badge to the README next to the Tests badge.
4. Cite the dataset separately (10.5281/zenodo.2603256, CC BY 4.0) — you already do
   this in the README, and it must appear in the paper's reference list too.
5. Keep `docs/AI_ASSISTANCE.md` current. Most venues now require an AI-assistance
   statement, and having one already written is an advantage rather than a liability.

## 5. A paper skeleton mapped onto this repository

For a four-page workshop submission:

| Section | Source |
| --- | --- |
| Dataset and partitions | `docs/EXPERIMENT_PROTOCOL.md`, `scripts/prepare_official.py` |
| Models | `src/core/`, `src/reference/`, `docs/CONCEPTS.md` |
| Perturbation definition | `src/core/injectDetectorNoise.m` — give the formula in full |
| Main result (ablation) | `experiments/graph-edges/` |
| Supporting comparison | `experiments/seed-extension/` (five seeds) |
| Scale check | `experiments/scaling/` |
| Reproducibility | `experiments/project238-full/`, `run_submission` |
| Limitations | `README.md`, section "Verification and research scope" — reuse it nearly verbatim |

Lead with the ablation. Put the three-model table in an appendix with its
confounding stated, or leave it out.

## 6. Before you submit

- [ ] Every figure regenerated from committed data by a script in this repository
- [ ] Every number in the paper traceable to a `report.json`, with the path in the caption
- [ ] Clean and noise sample sizes stated separately in every table
- [ ] Seed count stated in every error bar caption
- [ ] Published-benchmark comparison table included, not omitted
- [ ] The confound in the three-model comparison stated in the main text
- [ ] "Detector" removed from any claim the noise model does not support
- [ ] Zenodo DOI minted and in `CITATION.cff`
- [ ] Dataset licence (CC BY 4.0) and attribution in the reference list
- [ ] AI-assistance statement matching the venue's policy
- [ ] arXiv preprint posted

## 7. The single highest-value experiment left

Run the k=0 / k=6 ablation at the 35% working point with more seeds. At sigma = 0.35
the seed SD is 0.0103 across three seeds — about six times the 0.0016 at sigma = 0 —
and the interval [+0.0060, +0.0572] clears zero by a margin roughly a fifth of the
mean. That one interval carries the paper's central claim. Ten seeds would cost
roughly three times the current run and would turn the strongest sentence in the
paper from suggestive into solid.

Everything else on this list is writing. That one is measurement.
