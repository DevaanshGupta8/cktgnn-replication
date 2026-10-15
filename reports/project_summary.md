# Project summary: CktGNN claims, what this project verified, and the status of the proposed improvements

*Last updated: 2026-10-15, after Phases 1–9.*

This page summarises the whole project against the paper's claims and the improvements proposed for it. Each row points to the phase report with the evidence.

**Paper:** Dong et al., *CktGNN: Circuit Graph Neural Network for Electronic Design Automation*, ICLR 2023 ([arXiv 2308.16406](https://arxiv.org/abs/2308.16406), [code](https://github.com/zehao-dong/CktGNN)).

**Legend:**
- ✅ confirmed / done
- ⚠️ checked, but the finding differs from the claim, or only partly done
- ❌ not done

---

## 1. Problem the paper addresses

Analog circuit design is largely manual. Engineers choose the components, their connections (topology) and their values (sizing) from a very large design space. CktGNN aims to generate and optimise circuits automatically: it learns a continuous latent space of circuit graphs with a variational autoencoder (VAE), then searches that space with Bayesian optimisation (BO).

## 2. What the paper proposes, and what the released code does

| Claim | Status | Finding | Evidence |
|---|---|---|---|
| Circuits are represented as graphs | ✅ | Two graph formats are used: a device graph (D-VAE baseline, ≤ 16 nodes) and a graph of subgraph blocks (CktGNN, ≤ 8 nodes). | Phases 1–2 |
| Graph nodes are transistors, resistors and capacitors | ⚠️ | Nodes are R, C and **gm**, a voltage-controlled current source modelling an amplifier stage behaviourally. No transistors anywhere. | Phase 1 §1 |
| Level-1 blocks such as current mirrors or cascode stages | ⚠️ | The 26 block types are combinations of R, C and gm, not transistor-level sub-circuits. | Phase 1 §2.2 |
| A two-level GNN: inner GNN on blocks, outer GNN on the block graph | ⚠️ | **The released encoder has no inner GNN.** The inner-GNN class is dead code that would crash if called. The released model is a D-VAE on the block graph plus an undocumented position-slot prior and a separate device-value pathway. | Phase 1, finding 1 |
| The two-level design explains CktGNN's advantages | ⚠️ | Ablations attribute each advantage to a different component (table below). None comes from two-level message passing. | Phases 3 and 5 |
| Performance is described by gain, bandwidth, phase margin and FoM | ⚠️ | The FoM is **~99% bandwidth** (correlation 0.9996). Phase margin is barely predictable from any latent space (r ≈ 0.2–0.3). | Phases 1–2 |

**What actually explains CktGNN's results** (ablation chain D-VAE → A1 → A2 → A3 → CktGNN, 3 seeds each):

| Component | Effect |
|---|---|
| Block-graph representation | Smoother latent space for bandwidth/FoM (the paper's headline metric); worse for gain |
| Slot prior (undocumented in the paper) | More readable and functional generated circuits |
| Slot-aware messages and edge scorer | Slightly better latent regression; nothing else measurable |
| Slot-indexed device-value pathway | Doubles topology reconstruction; halves the share of physically valid device values |

## 3. Dataset (Open Circuit Benchmark)

| Claim | Status | Finding | Evidence |
|---|---|---|---|
| 10,000 op-amp circuits with simulated performance | ✅ | Ckt-Bench-101 has 10,000 circuits (9,000 train, 1,000 test); Ckt-Bench-301 has 50,000 more. | Phase 1 |
| Labels are reliable | ⚠️ | The released dataset is not the paper's dataset. 7.9% of rows carry placeholder labels from failed simulations, although all are marked valid. | Phase 1, finding 6 |
| Circuits are working amplifiers | ⚠️ | An exact pole-based check finds **63% open-loop unstable**. Only 10% are stable as a unity-gain buffer, and 3% meet a standard spec (stable, PM ≥ 60°, gain ≥ 60 dB). | Phase 4 |
| Our simulator agrees with the authors' Cadence labels | ✅ | The open simulator (`scripts/ocb_sim.py`) matches gain and bandwidth on ~96% of circuits. | Phase 1 |

## 4. Results the paper reports

| Claim | Status | Finding | Evidence |
|---|---|---|---|
| Better performance prediction than earlier GNNs | ⚠️ | The authors' released-data numbers reproduce. CktGNN beats D-VAE on bandwidth/FoM but loses on gain. In the authors' own Appendix G, DAGNN beats CktGNN on gain and phase margin. A simpler model (A1) and the property-aware model P beat CktGNN on FoM. | Phases 2, 3, 6 |
| Better generation (reconstruction) | ⚠️ | Reproduced (0.38 vs 0.15). The gain comes entirely from the device-value pathway, which also makes **half of the generated circuits physically invalid** (device values out of range). | Phases 2, 5 |
| ~98% valid circuits vs ~75% for previous methods | ⚠️ | These are the paper's numbers. On the released data we measured 96% vs 44%, with code checks that differ between the two models. Simulated, only **~80% of generated circuits work at all and ~3% are stable**; for real circuits it is 93% and 12%. | Phases 2, 3 |
| Error bars | ⚠️ | The paper's ± comes from re-seeding the regression on one trained model. Retraining with different seeds gives a spread about 5–20× larger, often larger than the model differences in the paper's Table 1. | Phase 2 |
| BO finds better circuits | ⚠️ | Optimising the paper's FoM selects **unstable** circuits: the circuit each FoM-BO run ranks best is stable in at most 2 of 20 runs per latent space (2 of 9 in Phase 4). Best-design comparisons from 10–20 runs (the paper uses 10) do not replicate across seed sets. | Phases 4, 7, 8 |

## 5. Limitations

| Limitation | Status | Finding |
|---|---|---|
| Only operational amplifiers | ✅ confirmed | Narrower still: behavioural 2–3-stage op-amps, 26 block types, ≤ 8 blocks. |
| No real-world conditions (temperature, voltage, process variation, layout parasitics) | ✅ confirmed | Narrower still: not even transistor-level; every stage is an ideal gm with a fixed output parasitic. |
| Optimises a single performance score | ✅ confirmed | Worse than a single score: the FoM ignores stability, so the optimum is an unstable circuit (Phase 4). |
| Code does not match the paper | Found here | No inner GNN, an undocumented slot prior; the subgraph-basis theorem is never exercised (Phase 1). |
| Statistical claims are weak | Found here | Error bars too small (Phase 2); 10-run BO comparisons are unreliable (Phases 7–8). |

## 6. Proposed improvements: status

| Improvement | Status | What was done | Evidence |
|---|---|---|---|
| **1. Expand the dataset** (ADCs, filters, RF, mixed-signal) | ❌ not done | Needs a new circuit generator and simulator for each class. The two-level idea could carry over (e.g. biquad sections as blocks of a filter), but nothing was built. | Phase 1 §2 |
| **2. Multi-objective optimisation** (power / area / speed trade-offs) | ⚠️ partly done | **Done:** constrained optimisation. Maximise speed (unity-gain frequency) subject to open- and closed-loop stability, PM ≥ 60° and gain ≥ 60 dB; four methods (random search, the paper's FoM-BO, constrained BO, trust-region SCBO), 40 runs per cell. Constrained methods propose spec-meeting circuits about 7–40× more often than random search, while FoM-BO almost never yields a stable best circuit. **Not done:** power and area (the behavioural circuits have neither) and reinforcement learning (we used Bayesian optimisation). | Phases 4–8 |
| **3. Real chip conditions** (PVT variation, parasitics) | ❌ not done | **Done (a first step):** an open AC simulator with an exact pole-based stability check, used in every evaluation. **Not done:** transistor-level validation (planned: ngspice with the open SKY130 PDK) and any variation-aware training. | Phases 1, 4; HANDOVER open work |

**Further work beyond the original proposals:**
- **Better generation** (Phase 9):
  - Selecting latent points with P's jointly trained predictor raises the share of generated circuits that meet the spec 1.6× (2.62% vs 1.62%, confirmed) and increases the variety of working designs. A classifier fitted afterwards does nothing.
  - Normalising device-value inputs (Pn) improves reconstruction (0.24 → 0.32) and physical device values (82% → 95%), not significantly with 3 seeds, but costs latent-space quality.
- **Property-aware latent space** (Phase 6). A VAE trained jointly with a stability/performance predictor gave the best latent-space regression of all models (FoM RMSE 0.475 vs 0.631 for CktGNN) and predicts "meets spec" with AUC 0.94. It did not measurably improve optimisation.
- **Reliability of optimisation results** (Phases 7–8):
  - how *often* a method proposes valid circuits is a stable property (cross-seed rank correlation 0.92);
  - how good its *best* circuit is, within 200 simulations, is not (0.3).
- **Reproducibility:** multi-seed reproduction, a batched training path that is 5–9× faster and numerically equivalent, deterministic optimisation runs, pre-registered protocols for Phases 5–8, and public trained models (release `models-v1`).

## 7. One-paragraph conclusion

CktGNN shows that representing op-amps as graphs of predefined blocks gives a useful latent space for prediction and generation. The released code, however, is not the two-level GNN the paper describes. Its advantages come from the block representation, an undocumented slot prior and a device-value pathway that makes half of the generated circuits physically invalid. The benchmark's objective is essentially bandwidth and ignores stability, so optimising it produces unstable circuits. Adding stability and basic specs as constraints fixes this reliably. Whether any optimiser finds *better* circuits cannot be settled with the 10–20 runs typical of such studies. Extending the approach to other circuit classes, to power/area trade-offs and to transistor-level, variation-aware evaluation remains open.
