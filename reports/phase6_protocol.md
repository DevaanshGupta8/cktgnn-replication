# Phase 6 protocol (fixed before running)

*Written 2026-10-14, committed before the model was trained.*

## Question

Phase 4 found that **the latent space matters more than the optimiser**: every method did better in A2's latent space than in CktGNN's. A VAE trained only to reconstruct graphs has no reason to place circuits that work near each other.

Does a latent space trained **jointly with a predictor of circuit properties** make stability-constrained optimisation easier? This is open-work item 3; the authors' own `--predictor` option in `main.py` is broken.

## Model

**P** (`SubgDVAEprop`) is A2 (`SubgDVAEpos`) plus a small MLP on the latent mean μ (66 → 128 → 7). It is trained jointly from scratch:

> loss = A2's VAE loss + λ · property loss, with **λ = 1**

The property loss is summed over the batch, like the VAE loss. Its targets come from the open simulator on the real circuits (`scripts/circuit_labels.py`):

- **4 binary flags**, scored with binary cross-entropy: functional, open-loop stable, closed-loop stable, meets the Phase 4 spec.
- **3 regressions**, scored with squared error, for functional circuits only: log10 UGF, DC gain (dB), lag phase margin. Each is standardised with training-set statistics and clipped to ±3 SD.

Only training-split labels enter the training loss; test labels are used only to score the predictor.

**Training protocol:** the same as A2. 3 seeds, 300 epochs, lr 1e-4, patience 10, batch 64, GPU training, CPU evaluation.

## Measurements

1. **Cost to the generative model:** reconstruction accuracy, valid/novel rates and the paper's SGP latent regression, P vs A2. Welch t-test, 3 vs 3 seeds.
2. **Predictor quality:** test AUC for the 4 flags; test RMSE (in SD units) for the 3 regressions.
3. **Generated circuits:** circuit-level evaluation of 10,000 prior samples (`eval_circuits.py`), P vs A2.
4. **Optimisation:** the Phase 5 design unchanged in P's latent space. 4 methods (RS, FoM-BO, cBO, SCBO) × 20 seeds; run k uses the VAE from seed ((k−1) mod 3) + 1.
   - **Primary endpoint:** best feasible UGF after 200 simulations, P vs A2, for **cBO** and for **SCBO**. Mann–Whitney U (unpaired: different latent spaces), Holm-corrected across the 2.
   - **Secondary:**
     - share of feasible simulations and runs that find a feasible circuit, all four methods;
     - P vs A2 for RS and FoM-BO;
     - within P, the Phase 5 comparisons.

## What would count as a result

- **Positive:** cBO/SCBO find better feasible circuits, or find them more reliably, in P's space than in A2's, without a large loss in reconstruction or validity.
- **Negative but informative:** the predictor is accurate but optimisation does not improve. That would mean predictability is not the bottleneck, and point to the decoder or the search instead.

Reported regardless of outcome. λ and the architecture are not tuned on the optimisation results. Any later variant (e.g. other λ) is labelled exploratory.
