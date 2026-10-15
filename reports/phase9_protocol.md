# Phase 9 protocol (fixed before running)

*Written 2026-10-15. Committed and pushed before any Phase 9 computation.*

## Goal

Improve the generative model where it is weakest: the circuits it generates. Over 10,000 prior samples, the best model so far (P, `SubgDVAEprop`) generates:

| Generated circuits | P | Real test circuits |
|---|---:|---:|
| Stable | 4.6% | 11.6% |
| Meets spec (stable, PM ≥ 60°, gain ≥ 60 dB) | 1.7% | 2.8% |
| Device values in range | 82% | 100% |

Topology reconstruction is 0.24, vs CktGNN's 0.38.

## Part A: predictor-guided generation (no training)

P has a jointly trained predictor of "meets spec" (test AUC 0.94) that has never been used to steer generation.

- **Method.** For each P seed (1–3):
  1. Draw a pool of 20,000 latents z ~ N(μ_train, σ_train), as in the circuit evaluation of earlier phases.
  2. Score each with the predictor's "meets spec" output.
  3. **Guided:** keep the top 5% (1,000 latents).
  4. **Control:** 1,000 latents chosen uniformly at random from the same pool.
  5. Decode each latent 10 times (stochastic decoding, as before) and simulate all 10,000 circuits per arm.
- **Primary endpoint:** the share of generated circuits that meet the spec.
- **Primary claim:** guidance helps if **(i)** guided > control in all 3 seeds **and (ii)** a bootstrap 95% CI (resampling latents, pooled over seeds, 2,000 resamples) of the guided − control difference excludes 0.
- **Secondary**, descriptive:
  - stable, functional, values-in-range and novel shares;
  - number of unique functional topologies and unique spec-meeting topologies (diversity cost of guidance);
  - a dose–response curve: keep the top 100% / 50% / 20% / 5% / 1% of the pool, subsample 1,000 latents from each, decode 10× each.
- **Secondary: guidance for models without a predictor.** For A2 and CktGNN, fit a logistic regression (sklearn defaults, C = 1, standardised inputs) on the training latents with the simulator's "meets spec" labels for the training circuits. Use it in place of P's predictor with the same pool / top-5% / control design.

## Part B: normalised device-value inputs (retraining)

In A2, A3 and P, raw device values (integers 1–101) enter the encoder and decoder GRUs alongside 0/1 one-hot vectors. Phase 5 found that CktGNN, whose decoder state does not carry raw values, reconstructs topology about twice as well.

- **Model Pn** (`SubgDVAEpropn`): P with the device values fed to the GRUs as (v − 51) / 29.2. This is the mean and standard deviation of a uniform 1–101, used for all three values.
  - The value-prediction head and its loss are **unchanged**: the head still predicts raw values with the same MSE weight, so only the input scaling differs.
  - Everything else is P's protocol: λ = 1, 3 seeds, 300 epochs, GPU training, CPU evaluation, the SGP regression and circuit-level evaluation.
- **Primary endpoints**, Pn vs P, Welch t-test over 3 seeds:
  - topology reconstruction;
  - share of generated circuits with all device values in range.
- **Secondary:** every other metric of Phases 3 and 6, including SGP regression, predictor AUC, stable and meets-spec shares. If Part A succeeds, guided generation with Pn is reported too.

Reported regardless of outcome. Settings are not tuned on the results; any later variant is labelled exploratory.
