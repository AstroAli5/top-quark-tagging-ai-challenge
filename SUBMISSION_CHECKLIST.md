# Submission Checklist

Maps this repository to the MathWorks AI Challenge / Challenge Project
Hub rules. Check these off before you submit.

## Repository requirements
- [x] License file — MIT — `LICENSE` included
- [x] README.md explaining the solution, with hyperlinks — included
- [x] Written in English
- [ ] Solution actually runs end-to-end (s1 → s7) — **you need to do
  this; I can't run MATLAB or this Python/pandas pipeline myself**
- [ ] Public GitHub repository created and this code pushed to it
- [ ] Results section in README filled in with your real numbers

## Before you submit
- [ ] Read the [Generative AI
  Guidelines](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/wiki/Generative-AI-Guidelines) —
  submissions with unverified or misunderstood AI-generated work are
  rejected. Make sure you can explain every script before you submit it,
  especially s4 (the custom GraphSAGE training loop) and s6 (the
  robustness test) — those are the parts that carry this submission.
- [ ] Complete the project sign-up form linked from the [Project #193
  page on the Challenge Hub](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/blob/main/megatrends/Artificial%20Intelligence.md)
- [ ] Submit your repo link via that project's submission form, and check
  the box agreeing to the terms and conditions
- [ ] **Confirm live: current submission deadline.** The official
  challenge page showed a first-edition deadline that had already passed
  when I first researched this, with only indirect evidence of a second
  edition. Check [the live
  page](https://www.mathworks.com/academia/students/competitions/student-challenge/ai-challenge.html)
  yourself, or email capstones@mathworks.com, before finalizing your plan
  — this project is heavier than a typical entry, so timeline risk
  matters more here, not less.

## Bonus points / tiebreakers
- [ ] Record a short YouTube walkthrough — earns extra judging points and
  is the first tiebreaker. Good structure: (1) the question — does
  robustness differ between architectures, (2) show the ROC comparison
  from s5, (3) show the noise-degradation curves from s6 — this is your
  strongest visual, (4) if time allows, the feature-importance plot from
  s7.

## Judging criteria (100 pts) — self-check before submitting
- [ ] **Real-world applicability (25):** can you explain, unprompted, why
  detector-noise robustness matters for a tagger that would actually be
  deployed? (docs/CONCEPTS.md section 5 has the argument — make it your
  own.)
- [ ] **Novelty (25):** the "What's original here" section in the
  README — the robustness study is the load-bearing piece; make sure you
  can defend it as a genuine gap in the reference material, not just an
  add-on.
- [ ] **Code & documentation quality (25):** comments are in place; fill
  in the Results table; confirm the code actually runs clean end-to-end
  before you submit, not just "it looked right."
- [ ] **Depth (25):** could you explain, from memory, why GraphSAGE
  averages neighbor features instead of just using each particle
  independently? Why AUC instead of just accuracy? docs/CONCEPTS.md is
  there so you can — re-read it until you don't need it.
