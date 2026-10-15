# Phase 7 protocol (fixed before running)

*Written 2026-10-15, committed and pushed before any Phase 7 run.*

## Question

Phase 5 found that SCBO fails when it never sees a stable circuit: the constraint score gives every unstable circuit the same value (−60). An **exploratory** variant, `scbo_g`, ranks unstable circuits by how far their worst pole lies in the right half-plane:

> score = −60 − 60·clip(largest relative real part of a right-half-plane pole, 0, 1)

In the Phase 5 data, `scbo_g` beat plain SCBO on the best feasible design in both latent spaces (p = 0.049 and 0.036, uncorrected). The variant was designed after seeing those data. This phase tests the hypothesis on **fresh runs**.

**Hypothesis:** graded-stability SCBO finds better feasible circuits than plain SCBO, within the same budget.

## Design

- **Fresh data.** Seeds 21–40 give new shared initial designs (`rng = 1000 + seed`). None of the Phase 5/6 runs (seeds 1–20) are reused.
- **Latent spaces:** all three trained in earlier phases: CktGNN, A2 (`SubgDVAEpos`), P (`SubgDVAEprop`). Run k uses the VAE from seed ((k−1) mod 3) + 1.
- **Methods:**
  - `scbo_g`: exactly as committed in `8192605`;
  - `scbo`: exactly as in Phase 5;
  - `cbo` and `rs`: as references.

  20 runs × 4 methods × 3 spaces = 240 runs, written to `<DATA>/results/bo7/`.
- **Problem, budget, evaluation:** unchanged from Phases 4–6. Maximise UGF subject to: functional, open- and closed-loop stable, PM ≥ 60°, gain ≥ 60 dB. 200 simulations per run, the first 20 shared within a seed.
- **No code changes** to `latent_bo.py`. Only the analysis script gains this phase's test configuration.

## Analysis

- **Primary endpoint:** best feasible UGF after 200 simulations (0 if none found).
- **Primary comparisons:** `scbo_g` vs `scbo` in each of the 3 latent spaces. Paired Wilcoxon signed-rank test, two-sided; pairs share the initial design. Holm correction across the 3. **Confirmed** if the Holm-adjusted p < 0.05 in at least one space and the direction favours `scbo_g`; the per-space results are reported either way.
- **Secondary**, no correction, reported as descriptive:
  - runs that find a feasible circuit;
  - share of feasible simulations;
  - distinct feasible circuits per run;
  - `scbo_g` vs `cbo` and vs `rs`;
  - `scbo` vs `cbo` and `cbo` vs `rs`, as a replication of Phase 5 on fresh seeds.

Reported regardless of outcome.
