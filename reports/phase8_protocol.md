# Phase 8 protocol (fixed before computing)

*Written 2026-10-15. Committed and pushed before the analysis script was run on any data.*

## Why

Phase 7 showed that the pre-registered optimisation endpoint, the **best feasible UGF after 200 simulations**, is unreliable at 20 runs per cell. The same method in the same latent space gave significantly different medians on two seed sets (e.g. cBO in CktGNN's space, 6.9 vs 1.6 GHz, p = 0.010), and several conclusions flipped.

A final-best endpoint depends on a single lucky or unlucky circuit. An area-under-the-curve endpoint averages over the whole run.

## Honest status of the data

This analysis uses **existing** runs only: seeds 1–20 (`<DATA>/results/bo5`, Phases 5–6) and seeds 21–40 (`<DATA>/results/bo7`, Phase 7). No new optimisation is run.

The final-best results and the convergence plots (medians of the best-so-far curves) of these runs have already been seen. **This is therefore a pre-specified analysis of partly unblinded data, not a blind test.** The new endpoint values and the reliability check below had not been computed when this file was committed.

## Endpoints

- **Primary: AUC**, the mean of the best-feasible-UGF-so-far curve over simulations 21–200, in GHz. The value is 0 before the first feasible circuit. Simulations 1–20 are excluded because they are the shared initial design, identical for all methods within a seed.
  - AUC rewards finding good circuits *early* and keeping them, and penalises runs that find nothing.
  - It equals the final best when a run finds its best circuit at simulation 21.
- **Secondary:**
  - **Time to first feasible:** the simulation index of the first feasible circuit, censored at 201 if none was found.
  - **Final best:** best feasible UGF at simulation 200, the earlier endpoint, kept for comparison.

## Cells

40 seeds each, pooling seeds 1–40:
- CktGNN and A2 spaces: `rs`, `cbo`, `scbo`, `scbo_g`;
- P space: `rs`, `cbo`, `scbo`.

20-seed cells (`fom_bo` seeds 1–20; P-space `scbo_g` seeds 21–40) are reported descriptively only.

## Part 1: reliability of the endpoint (the methodological question)

The two seed sets, 1–20 and 21–40, are compared for each endpoint (AUC, final best):

1. **Cross-set agreement of cell medians:** Spearman correlation between the set-A and set-B medians of the 11 cells.
2. **Sign agreement of method comparisons:** for every within-space pair of methods available in both sets (6 pairs in CktGNN, 6 in A2, 3 in P: 15 comparisons), check whether the median paired difference has the same sign in both sets.
3. **Cross-set shifts:** the number of cells whose set-A and set-B distributions differ at p < 0.05 (Mann–Whitney).

**Expectation (stated in advance):** AUC agrees better across seed sets than final best on all three measures. This is reported whatever the outcome; if AUC is not more reliable, it is not adopted.

## Part 2: method comparisons with AUC (40 paired seeds)

- **Primary family:** 6 comparisons, paired Wilcoxon signed-rank, two-sided, Holm-corrected across the 6.
  - `cbo` vs `rs` in each of the 3 spaces: does constrained BO beat random search?
  - `scbo` vs `cbo` in each of the 3 spaces: does the trust-region version beat global constrained BO?
- **Secondary**, uncorrected and descriptive:
  - `scbo_g` vs `scbo` and vs `cbo` (CktGNN, A2);
  - the same comparisons for time to first feasible and final best;
  - P vs A2 per method (Mann–Whitney, 40 vs 40).

## Implementation

`scripts/bo_auc.py` reads the trace CSVs and computes everything above. It writes:
- `<DATA>/results/summary/bo_auc_runs.csv`, `bo_auc_reliability.csv`, `bo_auc_tests.csv`;
- `reports/figures/bo_auc.png`.
