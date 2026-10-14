# CktGNN replication — Phase 6: a property-aware latent space

**Date:** 2026-10-15. The protocol was fixed and committed before the model was trained (`reports/phase6_protocol.md`, commit `302d22d`).

**Question:** Phase 4 found that the latent space matters more than the optimiser. Does a VAE trained **jointly with a predictor of circuit properties** give a latent space in which stability-constrained optimisation works better? This is the authors' own idea (`--predictor` in `main.py`), which is broken in their code.

**Model P** (`SubgDVAEprop`) is A2 (`SubgDVAEpos`) plus a small MLP on the latent mean (66 → 128 → 7), trained jointly from scratch with weight λ = 1. The predictor's targets come from the open simulator on the real training circuits:
- 4 flags: functional, open-loop stable, closed-loop stable, meets spec;
- 3 standardised regressions: log10 UGF, DC gain, phase margin.

Everything else is A2's protocol: 3 seeds, 300 epochs, GPU training, CPU evaluation.

---

## 1. Headline findings

1. **The pre-registered optimisation hypothesis is not supported at α = 0.05.** [E] In P's latent space, constrained BO's median best feasible design is **17.6 GHz vs 6.6 GHz** in A2's (means 17.2 vs 11.9), the highest median of all 12 method × space cells. But the difference is not significant over 20 vs 20 runs (Mann–Whitney p = 0.14, Holm p = 0.28). SCBO does not differ (5.4 vs 4.4 GHz, p = 0.72).

2. **The predictor costs nothing; reconstruction improves.** [E] P reconstructs topologies better than A2 (0.239 vs 0.173, p = 0.046). Validity (97.6% vs 98.9%), novelty (91% vs 93%) and every circuit-level rate of generated circuits are statistically unchanged. One exception is borderline: generated circuits with all device values in range fall from 93% to 82% (p = 0.06).

3. **The latent space becomes much more informative, including on the paper's own metric.** [E] With the paper's SGP protocol and labels:
   - gain RMSE falls from 0.678 to **0.298** (Pearson r 0.73 → 0.95);
   - BW/FoM RMSE falls from 0.603 to **0.475** (r 0.80 → 0.88);
   - all p ≤ 0.0004.

   This beats every model in Phases 2–5, including CktGNN (FoM 0.631). The paper's headline latent-space metric can thus be improved far more by a cheap auxiliary loss than by any architectural choice we ablated. The predictor is trained on simulator labels that correlate with the paper's Cadence labels; this is supervision, not leakage of test data.

4. **The predictor is accurate on held-out circuits.** [E] On the 1,000 test circuits:

   | Target | Test score |
   |---|---|
   | Meets spec | AUC 0.943 |
   | Functional | AUC 0.973 |
   | Open-loop stable | AUC 0.896 |
   | Closed-loop stable | AUC 0.862 |
   | log10 UGF | RMSE 0.46 SD |
   | Gain | RMSE 0.27 SD |
   | Phase margin | RMSE 0.73 SD |

5. **Feasible circuits are a little easier to find in P's space.** [E] Secondary measures:
   - **Random search:** 1.9% of simulations are feasible vs 1.2% in A2's space (p = 0.001).
   - **Earlier first feasible circuit**, median simulations needed:

     | Method | P's space | A2's space |
     |---|---:|---:|
     | Random search | 33 | 85 |
     | cBO | 30 | 49 |
     | SCBO | 23 | 38 |

   - **Best FoM-BO circuit:** stable in 2/20 runs vs 0/20.

   The effect on the *best* design is in the same direction but not significant at this sample size.

---

## 2. Results

### Generative model (mean ± std, 3 seeds; Welch t-test)
| Metric | A2 | **P** | p |
|---|---:|---:|---:|
| SGP gain RMSE ↓ / r | 0.678 / 0.726 | **0.298 / 0.954** | < 0.001 |
| SGP BW RMSE ↓ / r | 0.603 / 0.802 | **0.477 / 0.881** | < 0.001 |
| SGP PM RMSE ↓ / r | 1.010 / 0.222 | 0.974 / 0.344 | 0.008 |
| SGP FoM RMSE ↓ / r | 0.603 / 0.803 | **0.475 / 0.883** | < 0.001 |
| Topology reconstruction | 0.173 ± 0.031 | **0.239 ± 0.020** | 0.046 |
| Valid circuits (authors' check) | 98.9% | 97.6% | 0.18 |
| Novel | 92.7% | 91.3% | 0.35 |
| Generated: device values in range | 93.1% | 81.7% | 0.06 |
| Generated: functional | 77.6% | 74.7% | 0.25 |
| Generated: stable (pole check) | 4.4% | 4.6% | 0.71 |
| Generated: meets spec | 1.2% | 1.7% | 0.23 |

### Optimisation (20 runs per cell, 200 simulations; A2 numbers from Phase 5)
| Method | Space | Runs finding a feasible circuit | Median best feasible UGF (GHz) | Mean ± std | Simulations feasible | Distinct feasible / run | First feasible (median simulations) |
|---|---|---:|---:|---:|---:|---:|---:|
| Random search | A2 | 20/20 | 11.32 | 10.6 ± 6.4 | 1.2% | 2.3 | 85 |
| Random search | **P** | 20/20 | 15.03 | 13.3 ± 7.7 | **1.9%** | 3.8 | 33 |
| FoM-BO (paper) | A2 | 13/20 | 4.13 | 7.4 ± 8.4 | 1.9% | 3.9 | 78 |
| FoM-BO (paper) | **P** | 12/20 | 2.38 | 10.9 ± 13.4 | 6.4% | 12.9 | 11 |
| cBO | A2 | 18/20 | 6.63 | 11.9 ± 12.1 | 13.9% | 27.8 | 49 |
| cBO | **P** | 19/20 | **17.64** | **17.2 ± 11.5** | 13.0% | 26.1 | 30 |
| SCBO | A2 | 15/20 | 4.40 | 8.2 ± 9.0 | 16.6% | 33.1 | 38 |
| SCBO | **P** | 17/20 | 5.41 | 9.3 ± 9.9 | 17.9% | 35.5 | 23 |

### Pre-registered tests (P vs A2, Mann–Whitney U, unpaired)
| Method | Metric | P | A2 | p | Holm p |
|---|---|---:|---:|---:|---:|
| **cBO** (primary) | median best feasible UGF | 17.64 | 6.63 | 0.14 | 0.28 |
| **SCBO** (primary) | median best feasible UGF | 5.41 | 4.40 | 0.72 | 0.72 |
| Random search | median best feasible UGF | 15.03 | 11.32 | 0.19 | – |
| Random search | share feasible | 1.5% | 1.0% | **0.001** | – |
| FoM-BO | median best feasible UGF | 2.38 | 4.13 | 0.67 | – |
| cBO | share feasible | 4.3% | 10.8% | 0.84 | – |

Within P's space (paired Wilcoxon): cBO beats SCBO (17.6 vs 5.4 GHz, p = 0.044) and FoM-BO (17.6 vs 2.4, p = 0.048), and does not differ significantly from random search (p = 0.28).

All tests: `results/summary/bo5_P_vs_A2_tests.csv`, `bo5_tests.csv`, `ablation_tests.csv`. The convergence and feasibility figures in `reports/phase5_report.md` (`figures/bo5_*.png`) include P's space as their third panel.

---

## 3. Interpretation

- **Negative but informative, as the protocol anticipated.** The predictor learns stability and feasibility well (AUC 0.86–0.94), and the latent space becomes far more predictive. Yet the best design found by a GP-based optimiser improves only modestly and not significantly. So **predictability is not the main bottleneck.** The remaining limits are:
  - the decoder: only 4.6% of prior samples are stable, versus 11.6% of real circuits;
  - the 200-simulation budget;
  - high run-to-run variance in "best design" endpoints (standard deviations of 8–12 GHz around medians of 5–18 GHz).
- **For the reproducibility story:** the paper's main latent-space metric (SGP RMSE on FoM) moves more under a cheap auxiliary loss (0.603 → 0.475) than under any of CktGNN's architectural components (best variant 0.577). That metric says little about whether a latent space is useful for optimisation.
- **For future work:**
  - use the trained predictor directly, e.g. as a prior mean or feasibility filter for the BO, instead of only shaping the space;
  - increase the number of runs, or use a lower-variance endpoint such as the area under the best-so-far curve;
  - test whether the advantage holds at larger budgets.

## 4. Limitations

- **One weight, λ = 1, not tuned.** A larger λ could organise the space more strongly at some cost in reconstruction.
- **Simulator labels, not the paper's Cadence labels**, train the predictor; the SGP evaluation uses the paper's labels. The two agree on ~96% of circuits (Phase 1).
- **Behavioural circuits and one specification**, as in Phases 4–5.

## 5. Reproduce

```bash
.venv/Scripts/python.exe scripts/circuit_labels.py
```
```bash
DEVICE=cuda bash scripts/launch_job.sh SubgDVAEprop 1 --epochs 300 --patience 10
```
(seeds 1–3), then for each run `sgp_regression.py --run ...` and `eval_circuits.py --run ... --model SubgDVAEprop`, then:
```bash
PAR=3 OUT=D:/CktGNN_data/results/bo5 bash scripts/run_bo_jobs.sh JOBFILE
```
(`SubgDVAEprop` × `rs`/`fom_bo`/`cbo`/`scbo` × seeds 1–20), and finally:
```bash
.venv/Scripts/python.exe scripts/aggregate_results.py
```
```bash
.venv/Scripts/python.exe scripts/plot_ablation.py
```
```bash
.venv/Scripts/python.exe scripts/plot_bo.py --dir bo5
```
