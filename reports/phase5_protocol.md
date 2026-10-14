# Phase 5 protocol (fixed before running)

*Written 2026-10-14, committed before any Phase 5 optimisation run or A3 result existed.*

Phase 5 has two parts. Both close items left open by Phases 3 and 4.

## Part A: ablation A3 (separates the last ablation step)

Phase 3's last step, A2 → CktGNN, changed three things at once: the device-value pathway, slots in the messages, and slots in the edge scorer.

- **A3** (`SubgDVAEslot`) = A2 plus CktGNN's slot use in the structure decoder. Messages come from `[state, slot one-hot]` instead of D-VAE's `[state, vertex id]`, and the edge scorer sees `[state_i, slot_i, state_new, slot_new]`. Layer shapes are CktGNN's; A3 has 2,510,767 parameters (CktGNN 2.51M).
- Then **A2 → A3** isolates slot-aware structure decoding, and **A3 → CktGNN** isolates the slot-indexed value pathway alone.
- Same protocol as Phase 3: 3 seeds, 300 epochs, patience 10, the paper's SGP regression, the authors' decoder metrics, and circuit-level evaluation of 10,000 generated circuits. Welch t-tests per step, as in Phase 3.
- `tests/test_ablation_models.py` checks A1–A3 for batch independence (closed-form padding) and that A3's slots reach messages and edges.

## Part B: trust-region constrained BO, 20 runs per cell

**Question:** does a trust-region constrained optimiser (SCBO) fix constrained BO's bimodality, and does it beat random search on the best feasible design?

- **Design problem, evaluation, budget, initial design:** unchanged from Phase 4 (maximise UGF subject to functional, open- and closed-loop stable, PM ≥ 60°, gain ≥ 60 dB; 200 simulations, the first 20 shared within a run).
- **New method, `scbo`** (Eriksson & Poloczek, AISTATS 2021). ARD Matérn-5/2 GPs on log10(UGF) (functional points) and on Phase 4's constraint score, with constrained Thompson sampling: among candidates sampled feasible, take the highest sampled objective; if none, the least violated.
  - The trust region is centred on the best point: the best feasible, otherwise the least infeasible.
  - Unit cube = standardised latent clipped to ±3. L_init 0.8, L_min 2⁻⁷, L_max 1.6.
  - 2,000 candidates, each perturbing every dimension with probability 20/d (at least one).
  - **Budget adaptation:** success tolerance 3 and failure tolerance 10. The paper's max(4, d) = 66 would never shrink the region within 180 steps.
  - On collapse, the region restarts at L_init around the global best.
- **Amendment (2026-10-14, before any full SCBO run had finished):** GP hyperparameters are re-optimised every 5 steps, starting from the previous optimum. In between, the GPs are conditioned on the new data with the last hyperparameters. This is a purely computational change: fitting two 66-dimensional ARD GPs at every step made one run take over 30 minutes.
- **Re-runs:** RS, FoM-BO and cBO are re-run unchanged, except that the method's random stream now uses a stable hash. Phase 4 used Python's per-process salted `hash()`, so its runs could not be re-run bit-for-bit.
- **Runs:** 20 seeds × 4 methods × 2 latent spaces (CktGNN, A2) = 160 runs, written to `<DATA>/results/bo5/`. Phase 4's traces stay untouched in `<DATA>/results/bo/`.
- **Primary endpoint:** best feasible UGF after 200 simulations (0 if none found).
- **Primary comparisons**, 4 in total, using a paired Wilcoxon signed-rank test (pairing = shared initial design) with Holm correction across the 4:
  - SCBO vs cBO, in each latent space;
  - SCBO vs random search, in each latent space.
- **Secondary:**
  - runs that find a feasible circuit;
  - share of feasible simulations;
  - bimodality, measured as the share of runs with more than 20% feasible simulations;
  - distinct feasible circuits per run;
  - the Phase 4 comparisons (cBO vs RS and FoM-BO), re-estimated with 20 runs.
- **Reported regardless of outcome.** No setting is tuned after seeing Phase 5 results. Any later change is labelled exploratory.
