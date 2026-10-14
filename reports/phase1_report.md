# CktGNN replication and extension — Phase 1 report

**Paper:** Dong, Cao, Zhang, Tao, Chen, Zhang. *CktGNN: Circuit Graph Neural Network for Electronic Design Automation.* ICLR 2023. [OpenReview PDF](https://openreview.net/pdf?id=NE2911Kq1sp) · [arXiv 2308.16406](https://arxiv.org/abs/2308.16406) (v1 31 Aug 2023; v2 9 Feb 2024).
**Versions used:** the local copy used for this analysis (19 pages, read in full, Appendices A–F; not redistributed here) is the ICLR camera-ready text, identical to the OpenReview PDF. A line-by-line text diff against **arXiv v2** shows identical content apart from the arXiv stamp, one typo, and a **new Appendix G, "Recent update of OCB database"** (p. 19–20). Appendix G contains Tables 3–5 for the released Ckt-Bench-101/301; they match `OCB/OCB_performance_report.pdf`. Both are analyzed below. The OpenReview reviews could not be read yet, because the forum sits behind a browser-verification check.
**Official code:** https://github.com/zehao-dong/CktGNN, cloned to [`external/CktGNN`](../external/CktGNN) at commit `416cd03` (2023-09-11).
**Date:** 2026-10-13. **Scope:** the six "How to Begin" outputs: paper breakdown, methodology, reproduction checklist, missing information and risks, critical evaluation, and a ranked shortlist of improvements. The model has not been modified. The experiments here are audits and smoke tests that establish what a baseline reproduction requires.

**Data location:** datasets, checkpoints and outputs live in `D:\CktGNN_data` (see `scripts/paths.py`). Paths written as `results/...` below are relative to that folder.

### Evidence labels used throughout

| Tag | Meaning |
|---|---|
| **[P]** | Paper-reported (section, table or appendix is cited) |
| **[C]** | Code-verified against the official repository (`file:line` in `external/CktGNN`) |
| **[E]** | Measured in this project (script and output file are cited); fully reproducible |
| **[X]** | Externally sourced (link given) |
| **[H]** | Hypothesis or proposal; not yet tested |

---

## 0. Executive summary: the twelve findings that matter most

1. **The released CktGNN encoder has no inner GNN.** [C] The `CktGNN` class (`layers/models_ig.py:256–708`) never instantiates `subc_GNN`. The GRU input for each subgraph node is a one-hot of the *subgraph type* concatenated with a one-hot of a *position slot* (`models_ig.py:391–396`). Device values (R, C, gm) bypass message passing entirely: they are written into a fixed 27-dimensional, position-indexed vector, passed through a 2-layer MLP, and concatenated to the final graph state (`models_ig.py:489–500`). `subc_GNN` is dead code that would crash if called (`inverse_adj` is defined with two arguments at L123 and called with one at L191). Equation (3) and Section 3.2 describe something else: an undirected inner GNN that produces `h_v'` per subgraph.
2. **CktGNN receives an undocumented domain prior that no baseline receives.** [C] The position slot (`pos ∈ {0..7}`, e.g. "stage 2" or "feedback slot 5") is fed into the encoder GRU, into every message, into the edge predictor, and is predicted by a separate decoder head (`add_pos`, `models_ig.py:309–313`). The paper never mentions it and never ablates it.
3. **The ordered subgraph basis, the graphlizer *f* and Theorem 3.2 are never exercised by the code.** [C] OCB circuits are *generated directly in subgraph space* (`OCB/src/amp_generator.py`). No function in the repository decomposes a device-level circuit into the basis, so the paper's statement that "0 of 10000 circuits have different transformed graphs when we reverse the order of C and R" (App. A) cannot be checked with the released code.
4. **The released dataset is not the paper's dataset, and the authors' own results on it partly reverse Table 1.** [C][P: arXiv v2 App. G] The README says Ckt-Bench-101 "eliminates invalid circuits in the datasets used in our ICLR 2023 paper and replace[s] them". arXiv v2 Appendix G, Tables 3–4 (also `OCB/OCB_performance_report.pdf`, "10 runs with different random seeds") report on that release: **DAGNN beats CktGNN on Gain (RMSE 0.409 vs 0.682) and PM (0.947 vs 0.993)**, and DAGNN has more valid DAGs (98.81% vs 93.10%). CktGNN wins on BW, FoM, valid circuits and reconstruction. The appendix text nevertheless states that "CktGNN still achieve[s] state-of-the-art performance in these tasks", which its own Table 3 contradicts for Gain and PM.
5. **"FoM" is almost entirely bandwidth.** [E, `results/audit/audit_summary.json`] On Ckt-Bench-101, corr(FoM, BW) = **0.9996** and the `10·BW` term accounts for **98.8%** of FoM variance (gain ≈ 0.015%, PM ≈ 0.06%). On Ckt-Bench-301, corr(FoM, BW) = 0.998. Optimizing "FoM" therefore means maximizing the unity-gain frequency.
6. **7.9% of Ckt-Bench-101 rows carry placeholder labels from failed simulations, despite the paper's statement that they were fixed.** [E, `results/simulator/label_diagnostics.json`; P: arXiv v2 App. G] Appendix G says that failed simulations, previously stored as −inf, were "solved" and that Ckt-Bench-101 now holds "10000 successfully simulated circuits". In fact 785 of 10,000 rows have gain = 0 dB, BW ≈ 0 and the shifted-zero PM, exactly the values `performance_readout` writes for a failed simulation, yet every row is marked `valid = 1`. In 769 of the 785, our simulator finds no 0 dB down-crossing anywhere in 1 Hz–100 GHz, so Ocean had no unity-gain frequency to report.
7. **An open-source simulator reproduces the commercial-simulator labels.** [E] The labels came from Cadence Spectre and Ocean. I re-implemented the netlist semantics of `graph_to_fom.py` in a 150-line modified-nodal-analysis AC solver ([`scripts/ocb_sim.py`](../scripts/ocb_sim.py)). Of the 9,215 non-placeholder circuits:
   - **8,820 (95.7%) match BW within 1% and match gain exactly**, up to the stored offset;
   - another 149 match one of several 0 dB crossings, so Ocean simply picked a different crossing;
   - about 2.7% remain unexplained and are concentrated in feedback-direction subgraphs at slot 6;
   - the PM label equals the phase at the unity-gain frequency modulo 360° in 99.9% of matched rows.
   The solver takes about 0.6 ms per circuit on one CPU core, which makes closed-loop decode → simulate → BO feasible without Cadence.
8. **The PM label is not a stability measure, and FoM-optimal circuits are often marginal or unstable.** [E] The label is a wrapped phase whose numeric value jumps at a branch cut. It correlates −0.12 with a lag-based margin (180° minus the phase lag accumulated between DC and the UGF). Under that margin, **36% of matched circuits are below 45°, rising to 55% of the top-10% FoM circuits and 66% of the top-1%**. This margin is only indicative; a proper closed-loop pole analysis is planned (§6).
9. **Simple non-GNN predictors beat every predictive number reported for the GNN encoders.** [E, `results/audit/simple_baselines.csv`] A random forest on hand-made features, using the paper's exact 9000/1000 split and target standardization, achieves:

   | Target | Random forest RMSE / r | Paper CktGNN RMSE / r |
   |---|---|---|
   | Gain | 0.315 / 0.948 | 0.607 / 0.791 |
   | BW | 0.452 / 0.895 | 0.873 / 0.479 |
   | PM | 0.860 / 0.599 | 0.973 / 0.217 |
   | FoM | 0.451 / 0.896 | 0.854 / 0.491 |

   The protocols differ: the paper fits a sparse GP on frozen, unsupervised VAE latents, which measures latent smoothness. Even so, this sets the scale of what "good" means on this benchmark.
10. **Several evaluation choices inflate or confound the reported comparisons.** [C][P]
    - The ± in Table 1 reflects re-seeding the SGP on *one* trained encoder (`sgp_regression.py:261–317`), not retraining the VAE.
    - The code's "valid circuit" test differs from the paper's criterion. It does not check that the decoded graph can be built into a netlist: duplicate or missing stage positions pass.
    - Reconstruction accuracy compares different graph representations across models: CktGNN's ≤8-node, 26-type subgraph DAG versus the baselines' ≤16-node, 10-type device DAG. It also ignores device values and positions.
    - Prior samples are rescaled to the training-latent mean and std (`utils.py:233–264`), contradicting App. E ("standard normal").
    - Table 2 reports bit-identical BO FoMs for different encoders (PACE = DAGNN = 33.274162; GCN = GIN = 31.624473; NGNN = Graphormer = 32.282656).
11. **Reproduction blockers.** [C]
    - The sparse GP depends on Theano.
    - BO depends on `pybnn`.
    - DAGNN uses APIs that no longer exist in PyG (`torch_geometric.nn.glob`) and a hard `torch_sparse` import.
    - **PACE, GCN, GIN, NGNN, Graphormer and Pathformer are not in the repository.**
    - The Cadence flow is replaced by the validated open simulator.
    - There is no code that converts a decoded graph back into a simulator netlist.
12. **Prior art the paper does not cite.** [X] Lu, Lei, Yang, Zeng, Shang, *"Topology Optimization of Operational Amplifier in Continuous Space via Graph Embedding"*, DATE 2022 ([proceedings](https://past.date-conference.com/proceedings-archive/2022/html/0422.html)), already embeds behavioral op-amp DAGs with a variational graph autoencoder and runs BO in the latent space. This weakens the claim that CktGNN "pioneers" learning-based joint topology and parameter optimization (§6). The paper cites a different Lu et al. 2021 DAC paper.

**Bottom line.** The idea (a basis-constrained, two-level DAG encoder plus a VAE and BO for op-amp topology) is sound and the benchmark is useful. However, the released implementation is a D-VAE with a position prior and a device-value MLP, not the two-level GNN the paper describes, and the benchmark's objective and labels have physics problems that matter for any "better circuit" claim. These are the most valuable starting points for the project (§7): first an honest multi-seed baseline with ablations, then a stability-aware, simulator-in-the-loop optimization study.

---

## 1. Comprehensive paper breakdown (Phase 1)

### 1.1 The problem the authors address
Given a set of op-amp specifications, find both a **circuit topology** (which sub-circuits exist and how they connect) and **device parameters** (R, C, gm) that maximize performance. The authors treat this as optimization over directed acyclic graphs (DAGs) with node features, by analogy to neural architecture search (NAS): learn a continuous latent space of circuits with a VAE, then search it with BO [P §1, §3.2, App. A].

They also address a second problem: no public benchmark exists for learning-based analog topology design. Their answer is the Open Circuit Benchmark (OCB) [P §4].

### 1.2 Why it matters in EDA
Analog design is still largely manual. New topologies are "manually invented by human experts … within several weeks or months" [P §2.2], and technology scaling invalidates old designs [P §1]. Digital design is synthesized automatically; analog is not.

### 1.3 Limits of traditional analog design and automated sizing
- **Sizing-only automation.** BO, geometric programming, genetic algorithms, supervised learning and RL methods (GCN-RL, AutoCkt-style) optimize device parameters for a *fixed* topology [P §2.2].
- **Topology synthesis by genetic operators** (e.g. Das & Vemuri 2007) produces mostly non-functional circuits because practical constraints are not built into generation [P §2.2].
- **Expert dependence.** Topology choices such as compensation schemes and feed-forward paths encode tacit knowledge.

### 1.4 Why joint topology and parameter optimization is hard
- **Mixed discrete–continuous space.** Topology is combinatorial; parameters are continuous and span decades.
- **Strong coupling.** The best parameters depend on the topology, and topologies can only be compared after their parameters are optimized. This is the bilevel structure familiar from NAS.
- **Expensive, non-smooth objectives.** Each evaluation is a SPICE simulation; specifications trade off (gain vs bandwidth vs stability) and can be discontinuous, e.g. whether a unity-gain crossing exists at all.
- **Validity.** Most random graphs are not functional amplifiers.

### 1.5 Why graphs suit analog circuits
Circuits are naturally graphs: devices and nets, with connectivity as edges. Signal flow from input to output gives a natural direction, and after the paper's polarity trick for feedback a DAG. Message passing can follow the signal flow (D-VAE style), so the encoder "computes" with the circuit rather than only recognizing structure [P §2.1, §5.2].

### 1.6 Previous approaches and their shortcomings, as the paper argues
| Family | Representative | Shortcoming claimed in the paper |
|---|---|---|
| GNN-based sizing | GCN-RL (Wang 2020), Circuit-GNN (Zhang 2019a), Cao 2022a/b | Fixed topology only [P §1, §2.2] |
| GRU-based DAG encoders | D-VAE (Zhang 2019b), DAGNN (Thost & Chen 2021) | "Shallow layers", so not enough contextual information [P §1, §3.2] |
| Transformer DAG encoder | PACE (Dong 2022) | Encodes structure, not computation [P §1, §5.2] |
| Undirected / higher-order GNNs | GCN, GIN, NGNN, Graphormer | Not designed for DAG computation; NGNN not applicable to DAGs [P §3.1] |
| Topology synthesis by GA | Das & Vemuri 2007 | Non-functional outputs [P §2.2] |

### 1.7 The research gap that motivated CktGNN
(i) No learning-based method optimizes topology and parameters together (the paper's claim; see the novelty caveat in finding 12). (ii) Existing DAG encoders were considered either too shallow or structure-only. (iii) No public benchmark existed.

### 1.8 Claimed contributions and their status after verification
| # | Claimed contribution [P §1] | Status after code and data inspection |
|---|---|---|
| 1 | A two-level GNN with a pre-designed subgraph basis that increases expressiveness and reduces the design space | **Partially reflected in code.** The *basis* (26 subgraph types) is used as the node vocabulary, which does shrink the graphs. The *inner GNN* is absent from the encoder [C]. The design-space reduction comes from generating data directly in subgraph space plus the position prior [C]. |
| 2 | OCB, the first open circuit benchmark, with generation and simulation code | **Released, with caveats.** It differs from the paper's dataset; 7.9% placeholder labels; FoM ≈ BW; Cadence-only simulation; no decode → netlist bridge [C][E]. |
| 3 | CktGNN beats GNN baselines and produces op-amps competitive with human designs | **Not yet reproducible.** Most baselines are missing; the authors' own report on the released data shows mixed results; no human-designed reference circuit or specification is given in the paper [C][P]. |

### 1.9 What the paper solves, and what remains open
- **Addressed:** A pipeline (encoder → VAE → BO) for a *restricted, behavioral-level* op-amp space of 2–3 stages, 26 subgraph types and ≤8 subgraph nodes, with simulation-labelled data.
- **Still open:**
  - transistor-level realization of the behavioral gm stages;
  - stability and other specifications as constraints rather than a weighted sum;
  - power and area costs (absent from FoM);
  - process variation and parasitics;
  - other circuit classes;
  - topologies outside the predefined basis and slot structure;
  - realistic magnitudes (the dataset median is 134 dB gain and UGFs of up to 32 GHz, §6.2).

---

## 2. Methodology and architecture (Phase 2)

### 2.1 Circuit representation

**Behavioral abstraction [P §4, App. A; C `graph_to_fom.py:165–249`].** Each single-stage op-amp is replaced by a voltage-controlled current source (VCCS) `gm` with a fixed output parasitic, `R0 = 1 MΩ ‖ C0 = 50 fF` to ground. Compensation and feed-forward elements are R, C, series R–C or parallel R–C.

**Three graph levels** exist in the code:

| Level | Nodes | Edges | Used by |
|---|---|---|---|
| Circuit (netlist) | Nets: VIN, n1, n2, OUT, GND, internal | Devices | Simulator only |
| **Device DAG** (`g_all`, "original graph") | Devices: R, C, ±gm±, plus In/Out | Signal flow; parallel devices become parallel nodes | D-VAE, DAGNN and other baselines |
| **Subgraph DAG** (`g`, "transformed graph" G′) | Subgraphs from the 26-type basis, plus In/Out | Signal flow between subgraphs | CktGNN |

**Role swap [P App. A].** In Fig. 6 the sub-circuits are *edges* between nets (input, node 1, node 2, output). The paper then "convert[s] the role of nodes and edges", so sub-circuits become *nodes* of the DAG.

**Feedback handling [P §4].** A feedback path is drawn in the forward direction and marked with a direction flag (`+gm+` forward, `+gm-` feedback). In the netlist the IN and OUT terminals are swapped back (`graph_to_fom.py:519–545`). This makes every circuit graph acyclic while preserving the electrical loop.

**Polarity [C].** `-gm` multiplies gm by −1 (`polarity_node`). Under Spectre's VCCS convention, current gm·V(IN) leaves OUT, so a `+gm` stage is **inverting**. Our simulator matches the labels best with this convention (95.2% vs 94.6% BW agreement on the first 1000 circuits [E]).

**Device values [C `graph_to_fom.py:70–126`; P §5.1].** Stored as integers v ∈ {1,…,101} and scaled as R = 0.1·v MΩ, C = 10·v fF, gm = 0.1·v mS. That matches the paper's ranges of 10⁵–10⁷ Ω, 10⁻¹⁴–10⁻¹² F and 10⁻⁴–10⁻² S, but the values are sampled **uniformly on a linear grid** (`utils_src.py:141,174`), so about 90% of resistors lie above 1 MΩ.

**Worked example: training circuit #0** [E; output of `scripts/ocb_sim.py`]

```
Subgraph DAG (CktGNN view)                  Device DAG (baseline view)
v0 In        pos0                            In
v1 +gm+      pos2 gm=58      <- v0           gm(58)  <- In
v2 -gm+      pos3 gm=84      <- v1           gm(84)  <- gm(58)
v3 C||-gm+   pos4 c=39 gm=9  <- v2           C(39), gm(9)     <- gm(84)     (two parallel nodes)
v4 C         pos5 c=33       <- v2           C(33)            <- gm(84)
v5 R||-gm+   pos6 r=35 gm=11 <- v2           R(35), gm(11)    <- gm(84)
v6 Out       pos1            <- v3,v4,v5     Out <- all five dangling devices

Netlist (Spectre semantics, values scaled)
stage1: VCCS vin->n1, gm=+5.8 mS (inverting); 1M||50f at n1
stage2: VCCS n1->n2,  gm=-8.4 mS (non-inverting); 1M||50f at n2
stage3: VCCS n2->out, gm=-0.9 mS; C1=390 fF n2-out (Miller); 1M||50f at out
slot5 : C = 330 fF between n2 and out
slot6 : VCCS n2->out, gm=-1.1 mS; R1=3.5 MOhm n2-out; 1M||50f at out (feed-forward)

Simulated : gain 164.68 dB, UGF 12.463 GHz, phase@UGF -11.25 deg
Label     : gain 2.84211 (=164.68/100 + 1.1953 offset), bw 12.46541, pm 0.12498 (=-11.25/-90 mod 4)
```

This example shows two properties used later in the critique:
- **Electrical non-uniqueness.** The slot-5 capacitor (330 fF) is electrically in parallel with stage 3's Miller capacitor (390 fF). The circuit is identical to one with a single 720 fF Miller capacitor and no slot-5 node, but the graph is different.
- **Lossy device DAG.** In the device DAG, C(39) and C(33) are structurally indistinguishable (both n2 → Out), so the baselines cannot tell a Miller capacitor inside stage 3 from a separate bridging capacitor. The two happen to be equivalent here, but in general they need not be.

**Graph → netlist conversion** [C `graph_to_fom.py`]:
- main-path nodes (indices 2, 3, 4) are chained VIN → n1 → n2 → OUT;
- any other subgraph connects the output net of its predecessor to the output net of its successor;
- feedback types swap their terminals.

The script reads the authors' `.txt` format and writes Spectre netlists. **No code converts a *decoded* igraph back to this format** [C]. `ocb_sim.py` now does it directly from igraph.

**What the representation preserves or loses**
- **Preserved:** small-signal behavior of the behavioral model.
- **Lost or abstracted:** transistor-level realization, biasing, swing, noise, power, nonlinearity, offset, and parasitics beyond the fixed 1 MΩ ‖ 50 fF.
- **Graph-level ambiguity:** parallel or series combinations of like elements across slots produce different graphs for the same circuit.
- **Assumptions required:** a linear behavioral model is adequate; topology is restricted to the "main path plus ≤3 extra slots" template.

### 2.2 Ordered subgraph basis

**Definition [P Def. 3.1].** A basis 𝔹 = {g₁,…,g_K} with a total order o.

**The op-amp basis [P App. A, Fig. 7; C `models_ig.py:25–109`]:** 24 device subgraphs plus In and Out, giving 26 node types.

| Type id | Content | Connection | Count |
|---|---|---|---|
| 0, 1 | In, Out | – | 2 |
| 2, 3 | R, C | – | 2 |
| 4, 5 | R–C | series, parallel | 2 |
| 6–9 | gm: (+/−polarity) × (forward/feedback) | – | 4 |
| 10–13 | C ‖ gm (4 polarity × direction variants) | parallel | 4 |
| 14–17 | R ‖ gm | parallel | 4 |
| 18–21 | C ‖ R ‖ gm | parallel | 4 |
| 22–25 | (R–C series) ‖ gm | series R–C across the stage | 4 |

**Order o [P App. A].**
- Among size-1 subgraphs: o(gm) > o(R) > o(C), with forward and positive polarity ranked ahead of feedback and negative.
- Larger subgraphs rank above smaller ones.
- At equal size, parallel ranks above series.
- Remaining ties are broken lexicographically.

**Graphlizer rules [P §3.1].**
1. A node belonging to several basis subgraphs takes the one with the highest order.
2. Connected nodes that select the same subgraph are merged into one G′ node, carrying that subgraph's representation.

App. C gives the uniqueness construction: choose the decomposition with the fewest subgraphs, and break ties by the lexicographically largest sorted tuple of orders.

**Pseudocode, written from the paper. The released code does not contain it [C].**
```
f(G, B, o):
  candidates = all partitions of V(G) into connected node sets, each isomorphic (typed) to some g in B
  keep candidates with the minimum number of parts               # rule (1) in App. C
  among those, pick the one whose sorted tuple (o(g_1..g_K)) is lexicographically largest   # rule (2)
  build G': one node per part; edge (P,Q) iff G has an edge from a node in P to a node in Q
  return G'
```

**Search-space reduction.** A subgraph node carries its internal topology implicitly, so the decoder chooses among 26 types for ≤8 nodes instead of 10 types for ≤16 nodes with explicit internal edges. In OCB the space is further restricted by the generator and by the slot structure [E, `results/audit/position_type_counts.csv`]:
- the main-path slots 2–4 only ever contain types {6, 7, 10, 11};
- slots 5 and 6 contain any of the 24 device types;
- slot 7 contains only passives {2, 3, 4, 5}.

**Theorem 3.2 in plain words [P].** If 𝔹 contains every single-node subgraph (so a decomposition always exists), and every basis subgraph has exactly one "entry" node and one "exit" node for edges crossing its boundary, then an injective *f* exists.

Why the boundary condition matters: if a subgraph had two possible entry nodes, knowing only that "an edge goes from subgraph P to subgraph Q" would not tell you *which* internal node receives it. Two different circuits would then map to the same G′.

**Assessment.**
- The App. C proof mainly shows that *f* is **well defined** (a deterministic tie-break gives a unique output). That is necessary but is not injectivity. Injectivity rests on the single-sentence boundary argument ("connections between subgraphs are uniquely dependent on the input DAG"). It is plausible for this basis but informal.
- The theorem concerns **graph identity, not electrical function**. The worked example shows distinct G′ for the same circuit, so the latent space must represent electrically equivalent circuits at different points.
- As noted in finding 3, the code never runs *f*.

**What is op-amp specific and what generalizes.**
- *Specific:* the gm-centred basis, polarity and direction flags, the main-path slot template, and "no R/C in the main path".
- *General:* the two-level idea (a known motif vocabulary with message passing inside and between motifs), the boundary-node condition, and order-based canonicalization. These carry over to filters (biquad sections as motifs), LDOs, or NAS cells.

### 2.3 Two-level GNN architecture

**Inner GNN, as the paper describes it [P §3.2].** The subgraph g_v′ is treated as an undirected graph: several undirected message-passing layers (Eq. 1) followed by graph pooling produce h_v′. **The paper specifies no layer count, hidden size or pooling type.**

**Inner GNN in the code.**
- *Dead code* (`subc_GNN`, `models_ig.py:130–253`):
  - Embedding: `Linear(type one-hot) ⊕ Linear(value)` gives 2·out_feat dimensions.
  - Two layers of the form `x ← ReLU(D⁻¹(Wx + A·Wx))`.
  - Sum or mean readout.
  - It is never instantiated, and it would raise a TypeError (`inverse_adj` arity).
- *Actually used* (`models_ig.py:391–396, 489–500`): no per-subgraph network at all.
  - The node input is `x′_v′ = onehot(type, 26) ⊕ onehot(pos, 9)`, a 35-dimensional vector.
  - Device values enter only at graph level: `Hd = MLP_{27→24→8}([r,c,gm] for each of the 9 slots)`.

**Equation (1), generic message passing:**
`a_v^{t+1} = A({h_u^t : (u,v) ∈ E})`, `h_v^{t+1} = U(h_v^t, a_v^{t+1})`, where A aggregates the neighbors' states and U updates node v.

**Equation (2), outer aggregation:** `a_v′ = Σ_{u′∈N(v′)} g(z_u′) ⊗ m(z_u′)`.
- N(v′) is the set of **predecessors** of v′ in G′.
- z_u′ is the predecessor's hidden state.
- g is a gating network, `σ(Linear)` (`models_ig.py:321–324`).
- m is a mapping network, `Linear` with no bias so that zero padding stays zero (L329–331).
- ⊗ is elementwise multiplication.
- In the code, z_u′ is concatenated with the predecessor's position one-hot before g and m are applied (vs = hs + 9 = 310 dimensions, L401–411).

**Equation (3), outer update:** `z_v′ = GRU(input = concat(x′_v′, h_v′), hidden = a_v′)`.
- x′_v′ is the subgraph-type one-hot.
- h_v′ is the inner-GNN output (the paper's version); the code uses the position one-hot instead.
- The GRU is a `GRUCell(35 → 301)` (L291).

**Processing order and readout.** Nodes are processed in topological order, which the data loader guarantees by sorting (`utils.py:357–418`). The graph representation is the hidden state of the output node (`_get_graph_state`, L461–475). Bidirectional encoding exists but is off by default.

**Tensor shapes per batch of B circuits (code defaults)**

| Quantity | Shape |
|---|---|
| Node input X | B × 35 |
| Padded predecessor states | B × max_pred × 310 |
| Aggregated message a_v′ | B × 301 |
| Graph state H_g | B × 301 |
| Device embedding H_d | B × 8 |
| Concatenation [H_g ⊕ H_d] | B × 309 |
| μ, log σ² | B × 66 |

**Parameter counts [E]:** CktGNN 2,506,128; D-VAE 2,237,003 (hs = 301, nz = 66).

**Why the paper expects it to win [P §3.3, §5.2].** Inner GNNs add depth and context within motifs and parallelize well, while the directed outer pass encodes computation. In the released code the expected gains must instead come from three things: a smaller graph (≤8 nodes vs ≤16), the slot prior, and a dedicated device-value pathway. This is exactly what the ablations in §7 test.

### 2.4 VAE and optimization

**Encoder.** μ, log σ² = FC(H_g ⊕ H_d) [C L502]. The paper says "two FC layers" [P App. E], which matches (one per output).

**Reparameterization.** `z = μ + 0.01·ε·σ` during training (`eps_scale = 0.01`, L505–512). The noise is 100× smaller than a standard VAE's; this is inherited from D-VAE and not mentioned in the paper. In eval mode z = μ.

**Decoder** [C L519–596; P App. F]:
1. `H0 = tanh(FC(z))`, and all 27 device values are predicted at once from H0 (`df_fc`).
2. At each step: predict the subgraph type (`add_vertex`, a softmax over 26) and the slot (`add_pos`, a softmax over 9) from the current graph state. Write the device values for that slot from the step-1 prediction.
3. GRU-update the new node, then predict edges from every earlier node with `σ(MLP[h_vi ⊕ pos_i ⊕ h_new ⊕ pos_new])`, updating after each edge.
4. Stop at type = Out, connecting all dangling nodes to Out, or force Out at step 8.

The paper (App. E/F) instead says the decoder "uses the same undirected message passing as CktGNN" and predicts node features per subgraph from h_{v′_{i−1}}. **Neither is true of the code.**

**Loss** [C L598–693; P App. E] = −log p(types) − log p(positions) + BCE(edges) + 0.01·MSE(device values)/300 + β·KL, with β = 0.005 (matching the paper).
- The positional term and the tiny device-value weight are undocumented.
- The `recon` value that `main.py` logs excludes the device-value term.

**Optimizer.** Adam, lr 1e-4, batch 64, `ReduceLROnPlateau(factor 0.1)`. Patience is **10 in the code vs 20 in the paper**; epochs are **300 in the README vs 200 in the paper** [C `main.py:222`; P App. E].

**Latent-space evaluation (SGP)** [C `sgp_regression.py`]:
- Encode the train and test sets to μ.
- Negate and standardize each target with training statistics.
- Fit a sparse GP with M = 500 inducing points by Adam (lr 5e-4, 100 iterations, minibatch 1000).
- Report test RMSE and Pearson r.
- Repeat for SGP seeds 1–9, always with the **same** checkpoint.

**BO, as the paper describes it [P §5.4].** Batch BO with batch size 50, expected improvement, 10 iterations, reporting the FoM of the best circuit found.

**BO, as the repository implements it** [C `bo_search.py`, `search_methods/bo_cktbench301.py`]:
- The search is a *pool* search over the 50,000 pre-simulated Ckt-Bench-301 circuits; nothing is decoded or simulated.
- Surrogate: Bohamiann (SG-HMC Bayesian neural net) or DNGO.
- 10 random initial points; each round picks the top 5 by EI; budget 150 queries; 10 fresh random points are added after 20 rounds without improvement; 20 runs.
- Regret is measured against the hard-coded pool optimum 197.2296. We verified this equals `max(perform301.fom)` [E].

**Expected improvement.** The standard formula is `EI(x) = σ(x)·[u·Φ(u) + φ(u)]` with `u = (μ(x) − f*)/σ(x)`, where f* is the incumbent best value and Φ, φ are the normal CDF and PDF. The code (`bo_cktbench301.py:72–73`, same in DNGO) instead computes `ei = σ·(u·Φ(u) + 1 + φ(u))` with **f* fixed at 1.0** (FoMs are on a scale of about 10²).
- The same form, with 0.95, appears in NAS codebases built for accuracy targets; I did not verify which one was the source.
- Here it makes u ≫ 0 for nearly every candidate, so the ranking reduces to roughly μ + σ, a UCB-like rule rather than EI.
- [H, to confirm once pybnn is installed] pybnn's `predict` returns a predictive *variance*, which would make "σ" a variance.

**End-to-end pipeline trace**

| Stage | Paper | Released code |
|---|---|---|
| Circuit → graph | graphlizer *f* | generated directly as a subgraph DAG |
| Graph → latent | two-level GNN | GRU over (type, pos) plus device MLP |
| Latent → optimization | batch-EI BO | Bohamiann/DNGO pool search, or SGP regression |
| Latent → circuit | decoder | decoder (D-VAE style) |
| Circuit → simulation | Spectre via netlist script | `.txt` → Spectre script only, with no bridge from decoded graphs |
| Simulation → FoM | `1.2·|gain_dB|/100 + 1.6·PM/(−90) + 10·UGF/1e9` (L660) | stored labels |

---

## 3. Dataset and experimental setup (Phase 3)

**Measured dataset facts [E, `results/audit/audit_summary.json`, from `scripts/audit_dataset.py`]**
- **Split:** first 9000 rows train, last 1000 test (`utils.py:421`); no validation split exists. All 10,000 rows are marked `valid = 1`.
- **Topologies:** 4,552 distinct among 10,000 circuits; 9,999 distinct circuits once device values are included (one exact duplicate).
  - Topologies appear in consecutive **pairs** (4,996 runs of length 2), whereas the generator's default re-samples a topology every K = 20 circuits (`amp_generator.py:13,436`). The released data was therefore generated with an undocumented setting.
- **Train/test overlap:** 126 of 1000 test circuits (12.6%) share their topology with a training circuit (58 topologies), so the test set is mostly, but not entirely, topology-disjoint.
- **Size:** 7,994 three-stage and 2,006 two-stage circuits. Subgraph DAGs have 4–8 nodes; device DAGs have 4–16.
- **Label scaling:** gain, PM, BW and FoM are stored *after* scaling and shifting:
  - gain = gain_dB/100 − min + 1e-5 (the min over the paper-era data, offset 1.1953);
  - pm = PM/(−90) − min + 1e-5;
  - bw = UGF/1e9, unshifted.
  - The `pm` column contains values below the shift floor (min −2.35), consistent with rows being replaced after the shift was computed.
- **FoM formula:** FoM = 1.2·gain + 1.6·pm + 10·bw − 1.40 (std 0.22 across rows), consistent with `graph_to_fom.py:660` applied before the shift.
- **Value grid:** 101 integer levels for R, C and gm, sampled uniformly.

**Phase-3 component table**

| Component | Original implementation | Evidence | Unresolved details |
|---|---|---|---|
| Dataset generation | Random template sampling. Main slots 2,3 always present with types {6,7,10,11}; slot 4 present with p = 0.8; extra slots 5–7 with p = 0.6; device values uniform on 1..101. 10k circuits; Ckt-Bench-301 has 50k (5.5% marked invalid). | [C] `utils_src.py:85–111`, `amp_generator.py`; [E] audit | K (topology reuse) differs between code and data; exact generator version and seed for the release; how invalid rows were "replaced" |
| Graph conversion | No graphlizer. Generator writes subgraph-level rows; loader builds subgraph and device igraphs, topologically sorted. | [C] `utils.py:290–421` | Ordered-basis tie-breaking cannot be exercised |
| Node / edge features | CktGNN: type (26) + slot (9) one-hots; device values r, c, gm ∈ 1..101 per slot. Baselines: device type (10) + scalar value. Edges untyped. | [C] `models_ig.py:391–396, 489–500, 828–832` | The paper never mentions slot features |
| Model architecture | D-VAE-style GRU encoder (hs 301), gated sum, slot-augmented messages, device MLP 27→24→8, nz = 66; inner GNN absent | [C] `models_ig.py:256–335` | Inner-GNN depth and width (paper) unspecified |
| Training procedure | Adam 1e-4, batch 64, ReduceLROnPlateau(0.1, patience 10), 300 epochs (README), seed 1 | [C] `main.py:72–99, 221–222`; [P] App. E says 200 epochs, patience 20 | Which setting produced the paper's numbers |
| VAE objective | Type NLL + slot NLL + edge BCE + 0.01·MSE/300 + 0.005·KL; eps_scale 0.01 | [C] `models_ig.py:598–693, 505` | Slot-loss weight (`pos_scale = 0.5` is defined but unused) |
| Circuit simulation | Spectre netlist of behavioral subcircuits; Ocean AC sweep 1 Hz–100 GHz at 100 points/decade; gain = dB20 at the first point; `phaseMargin`; `unityGainFreq` | [C] `graph_to_fom.py`, `ocean.txt`; [E] open re-implementation matches 95.7% | ~2.7% unexplained mismatches; Ocean's crossing selection and PM branch conventions |
| Bayesian optimization | Paper: batch EI, batch 50, 10 iterations. Code: Bohamiann/DNGO pool search on Ckt-Bench-301, init 10, top-5, budget 150, 20 runs, EI with f* = 1.0 | [P] §5.4; [C] `bo_cktbench301.py` | How Table 2's BO was run (decode + simulate or pool?) and with what FoM scale (paper ≈ 33 vs dataset median 41, max 327) |
| Evaluation metrics | SGP RMSE / r on standardized negated targets; recon = exact match of types + in-neighbors (no values or slots); validity = DAG + 1 In/Out + no dangling nodes + no feedback-gm in slots 2–4; novelty = 1 − (fraction of valid DAG decodes found in train) | [C] `sgp_regression.py`, `utils.py:98–285` | Paper's validity criterion (3) differs from the code; trial count (paper 5, code 9 SGP seeds, OCB report 10) |

---

## 4. Reproduction checklist (Phase 4)

### 4.1 Environment
| Item | Original | This machine | Plan |
|---|---|---|---|
| Python | 3.7 [C README] | 3.13.14 (Windows 11) | **Track B (modern port)** as the main line; Track A (a py3.7 conda env) only if a Theano cross-check proves necessary |
| PyTorch | 1.8.0 | 2.13.0+cpu (system), reused through the `.venv` | Works for CktGNN and D-VAE [E] |
| PyG | 1.7.2 | not installed | Needed only for DAGNN; port `torch_geometric.nn.glob` → `torch_geometric.nn.pool`, stub or replace `torch_sparse` in `layers/batch.py` |
| igraph | python-igraph (unpinned) | 1.0.0 in `.venv` | Works [E] |
| torchvision | imported by `models_ig.py`, unused | not installed | Stubbed in `scripts/run_vae.py` |
| Theano | required by `search_methods/sparse_gp*.py` | not installable on 3.13 | Port the SparseGP (same FITC/EP equations) to PyTorch; validate on synthetic data and on saved latents |
| pybnn | required by BO | not installed | `pip install git+https://github.com/automl/pybnn`, or re-implement DNGO and use a GP-EI baseline |
| Simulator | Cadence Spectre + Ocean (commercial) | unavailable | **Done:** `scripts/ocb_sim.py`, validated against the labels |
| Hardware | P100 12 GB / GTX 1080 Ti [P App. D] | RTX 3050 6 GB laptop GPU (CUDA 13 driver), 12 CPU threads | CPU is viable (below); installing a CUDA build of torch (~2.5 GB) is optional |

**Measured training cost on CPU [E, `results/vae/smoke_*`]**
- **CktGNN:** 0.45 s per batch, ≈ 63 s per epoch, so ≈ 3.5 h (200 epochs) to 5.3 h (300 epochs) per seed.
- **D-VAE:** 1.3 s per batch, ≈ 3 min per epoch, so ≈ 10–15 h per seed.
- **Evaluation cost (not yet measured):** 100k test decodes plus 10k prior decodes plus an O(10⁴ × 9·10³) Python novelty scan.
- **Parallelism:** with 12 threads, 3–4 seeds can run in parallel.

### 4.2 Milestones
| # | Milestone | Inputs → expected outputs | Files | Tests / done criteria | Expected failure points | Status |
|---|---|---|---|---|---|---|
| 1 | Environment | → working `.venv` | `.venv`, `requirements.txt` | Import all modules; one training batch | Theano, pybnn, PyG | **Done for CktGNN/D-VAE** [E]; SGP/BO/DAGNN pending |
| 2 | Dataset | repo pkl/csv → loaded graphs + audit | `scripts/audit_dataset.py` | Counts match the README (10k, 9k/1k) | — | **Done** [E] |
| 3 | Graph parsing | pkl → igraph pairs | authors' loader | Topological order; In first, Out last | — | **Done** (verified on all 10k) |
| 4 | Ordered basis | basis table + order o | (none in repo) | Implement *f* and check that circuits round-trip | Not in repo | Pending; needed only to test Thm 3.2 |
| 5 | Graph transformation | device DAG → subgraph DAG | (none in repo) | `f(g_all) == g` for all 10k | Ambiguous parallel nodes in the device DAG | Pending |
| 6 | Inner GNN | subgraph → h_v′ | `subc_GNN` (dead code) | Fix arity; unit-test shapes | Unspecified hyperparameters | Pending (it is a *proposal* now, §7) |
| 7 | Outer directed GNN | G′ → H_g | `models_ig.py` | Permutation tests: same G′ → same μ | — | Runs [E] |
| 8 | VAE training | train set → checkpoints, loss CSV | `scripts/run_vae.py` | Loss decreases; loss logs match `main.py` | Runtime | Smoke-tested [E]; full runs pending |
| 9 | Decoding / reconstruction | z → G′ | `models_ig.decode` | Recon accuracy vs the paper and the OCB report | Stochastic decoding variance | Pending |
| 10 | Simulation | G′ → gain/UGF/PM/FoM | `scripts/ocb_sim.py` | ≥95% label agreement | Ocean conventions | **Done** (95.7% / 99.9%) [E] |
| 11 | BO | latents → best FoM vs budget | `search_methods/*` | Regret curves; 20 seeds | pybnn; EI bug | Pending |
| 12 | Baselines | DAGNN, then PACE/GCN/GIN/NGNN/Graphormer | `layers/dagnn_pyg.py`; others missing | Same training budget and latent size | Missing code; PyG port | Pending |
| 13 | Figures and tables | results → Tables 1–2, Fig. 3 | to be written | CSV + plots | — | Pending |

**Commands available now** (run from the project root):
```bash
.venv/Scripts/python.exe scripts/audit_dataset.py
```
```bash
.venv/Scripts/python.exe scripts/validate_simulator.py --n 10000
```
```bash
.venv/Scripts/python.exe scripts/run_vae.py --model CktGNN --epochs 300 --patience 10 --seed 1 --device cpu
```
```bash
.venv/Scripts/python.exe scripts/run_vae.py --model CktGNN --epochs 200 --patience 20 --seed 1 --device cpu --out results/vae/CktGNN_paperhp_seed1
```

### 4.3 Comparison tables to fill (paper values entered; reproduction pending)

> **Update 2026-10-14:** filled in for CktGNN and D-VAE (3 seeds each) in [`phase2_reproduction.md`](phase2_reproduction.md) §3.
| Metric | Paper (Table 1/2) | Authors' results on released Ckt-Bench-101/301 (arXiv v2 App. G, Tables 3–5) | Reproduced | Δ abs | Δ rel | Explanation |
|---|---:|---:|---:|---:|---:|---|
| Gain RMSE / r | 0.607 / 0.791 | 0.682 / 0.723 | – | | | Dataset changed (README) |
| BW RMSE / r | 0.873 / 0.479 | 0.596 / 0.806 | – | | | |
| PM RMSE / r | 0.973 / 0.217 | 0.993 / 0.288 | – | | | |
| FoM RMSE / r | 0.854 / 0.491 | 0.597 / 0.806 | – | | | FoM ≈ BW, so these track BW |
| Topology reconstruction | 0.397 | 0.453 | – | | | Not comparable across models (§6.3) |
| Valid DAGs / circuits | 98.92 / 98.92 % | 93.10 / 91.92 % | – | | | Code criterion ≠ paper criterion |
| Novel circuits | 92.29 % | 96.34 % | – | | | Computed over valid DAGs, with duplicates |
| Final BO FoM | 33.436 (unknown scale) | Bohamiann 175.85 ± 11.26; DNGO 163.29 ± 16.57 (Ckt-Bench-301, budget 150) | – | | | Different procedure and scale |

*The ± values in the paper are SGP-reseeding deviations; we will report VAE-retraining deviations over ≥3 seeds, plus 95% CIs.*

---

## 5. Missing information and reproduction risks

### 5.1 Paper ↔ code discrepancies
| # | Item | Paper | Code / release | Impact |
|---|---|---|---|---|
| D1 | Inner GNN | Present, Eq. (3) | Absent from the encoder; dead code is broken | Core claim untestable as released |
| D2 | Decoder | "Same undirected message passing as CktGNN"; per-subgraph feature MLPs | D-VAE GRU; all values from z at once | Generation quality not attributable to CktGNN |
| D3 | Slot (pos) prior | Not mentioned | Encoder, messages, decoder head, edge MLP | Confounds every comparison with baselines |
| D4 | Epochs / LR patience | 200 / 20 | 300 (README) / 10 | Unknown which produced the tables |
| D5 | Prior sampling for validity | N(0, I) | N(μ_train, σ_train) | Validity and novelty optimistic relative to the stated protocol |
| D6 | Validity criterion | No R/C in the main path | No feedback-gm in slots 2–4; no simulatability check | Validity not comparable |
| D7 | Trials | 5 [App. D]; 10 seeds for the released data [arXiv v2 App. G] | 9 SGP seeds on one checkpoint (`sgp_regression.py:261`) | ± underestimates true variance; "10 runs" cannot be produced by the released script without retraining |
| D8 | Dataset | 10k "distinct" op-amps; App. G (v2): "10000 successfully simulated circuits" | Released Ckt-Bench-101 replaced invalid rows; 7.9% placeholders remain; topologies come in pairs | Exact Table 1 numbers unreachable; the App. G claim is contradicted by the data |
| D9 | BO | Batch EI, batch 50, 10 iterations | Pool search, Bohamiann/DNGO, budget 150, EI with f* = 1 | Table 2 BO not reproducible as described |
| D10 | FoM scale | Best ≈ 33.4 | Ckt-Bench-101 median 41, max 327 | Different FoM normalization or data |
| D11 | Hyperparameters | Unspecified | hs = 301, nz = 66, emb 24/8, eps_scale = 0.01, value-loss weight 0.01/300 | Now known from code |
| D12 | Baselines | 8 models | Only CktGNN, D-VAE, DAGNN (`DVAE_GCN` is broken: argument misalignment in `super().__init__`, missing `import math`) | 5 baselines need external code |
| D13 | Baseline lists | Table 1 has Pathformer, not Graphormer; Table 2 and Fig. 3 have Graphormer, not Pathformer | — | Internal inconsistency |

### 5.2 Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SGP port differs numerically from Theano | Medium | Table 1 numbers shift | Port the exact equations; also report a standard GP or GPyTorch SGPR; compare ranking stability |
| DAGNN port changes behavior | Medium | DAGNN comparison | Minimal API shims; unit tests on small DAGs |
| Missing baselines (PACE etc.) | High | Partial Table 1 | Use the official PACE/NGNN repos; at minimum keep D-VAE, DAGNN, CktGNN ± ablations |
| Seed variance larger than reported gaps | High [H] | Conclusions change | ≥3 VAE seeds (5 if compute allows); paired tests |
| CPU runtime | Medium | Delays | Run seeds in parallel; optional CUDA torch |
| Simulator mismatch on ~2.7% | Low | Small label noise for closed-loop BO | Report it; use our simulator consistently for every method |
| Evaluation runtime (novelty scan) | Medium | Hours | Hash-based topology lookup (exact equivalent of `is_same_DAG`) |

---

## 6. Critical evaluation of the original paper (Phase 5, first pass)

Each entry gives: **Evidence** → *Why it matters* → *Likely impact* → *How to test* → *Feasible improvement*. Entries are marked **Demonstrated** (evidence in hand) or **Hypothesis** (needs an experiment).

### 6.1 Model
1. **No inner GNN in the released model. Demonstrated.** [C] Evidence: §0.1. *It matters* because the paper attributes its gains to inner message passing. *Impact:* the headline mechanism is unverified. *Test:* implement the inner GNN as described and compare it with the released model over ≥3 seeds. *Improvement:* a working two-level encoder (proposal A1).
2. **Undocumented slot prior. Demonstrated** (presence) / **Hypothesis** (that it drives the gains). [C] *It matters* because baselines see no slots, so comparisons are confounded. *Test:* CktGNN with `--no-pos` (already exposed in `run_vae.py`), and D-VAE given the *subgraph* DAG with and without slots. *Improvement:* matched-information baselines.
3. **The basis restricts discoverable circuits. Demonstrated** for the data. [E] Main slots only ever contain 4 types; there are ≤3 extra slots and ≤8 nodes. *Impact:* "novel" circuits are recombinations inside a small template. *Test:* count the template's reachable space versus generated novel circuits. *Improvement:* learnable or extensible bases (proposal A4).
4. **Representation is not canonical with respect to electrical function. Demonstrated** by the §2.1 example. *Impact:* redundant latent regions, so recon and novelty count electrically identical circuits as different. *Test:* canonicalize (merge parallel/series like elements across slots) and recount duplicates and novelty. *Improvement:* electrical canonicalization or equivalence-aware augmentation (A5).
5. **Theorem 3.2 is informal and unused. Demonstrated.** [P][C] *Impact:* low for practice, medium for the theory claims. *Test/improvement:* implement *f* and verify round-trips on all 10k.
6. **The decoder generates structurally valid but not necessarily functional circuits. Hypothesis** (strongly suggested). The validity check never builds a netlist; 7.9% of even the *training* circuits never reach 0 dB. *Test:* run decoded circuits through `ocb_sim.py` and report the fraction with a UGF, and with PM > 45°.
7. **Directed message passing ignores bidirectional electrical coupling.** Loading and Miller effects travel backwards; `bidirectional` is off. **Hypothesis.** *Test:* turn on `--bidirectional` or add a reverse pass.

### 6.2 Dataset
1. **FoM ≈ BW. Demonstrated** [E]: r = 0.9996, 98.8% of variance. *Impact:* every FoM result is a BW result; BO rewards high UGF regardless of stability or gain. *Improvement:* a constrained formulation (C1).
2. **Placeholder labels. Demonstrated** [E]: 7.9%. *Impact:* label noise; "valid" is overstated. *Improvement:* relabel with `ocb_sim.py` and a `valid` flag based on whether a UGF exists.
3. **The PM label is a wrapped phase, not a margin. Demonstrated** [E] (≡ phase at UGF mod 360° in 99.9% of cases; corr −0.12 with the lag-based margin). *Impact:* PM prediction (Table 1) and the PM term in FoM have unclear physical meaning, which may explain why every model has r ≈ 0.2–0.3 on PM. *Test:* recompute PM with a branch-consistent definition and re-run the regression.
4. **Unrealistic magnitudes. Demonstrated** [E]: median gain 134 dB (172 dB among the top-10% FoM circuits); UGF up to 32 GHz, from gm ≤ 10 mS driving 50 fF loads. *Impact:* results do not transfer to real CMOS op-amps (typically 60–120 dB and MHz to hundreds of MHz GBW). *Improvement:* realistic loads (pF), power-aware FoM, transistor-level mapping (D3).
5. **Linear-grid value sampling** over two decades. **Demonstrated** [C]. *Impact:* low values are under-sampled. *Improvement:* log-uniform sampling.
6. **Topology pairs and 12.6% test-topology overlap. Demonstrated** [E]. *Impact:* mild leakage and not a cross-topology generalization test. *Test:* re-split by topology (group split).
7. **Simulator dependence.** Behavioral linear models only. **Demonstrated.** Our open simulator removes the vendor lock-in but keeps the abstraction.

### 6.3 Evaluation
1. **± reflects SGP noise only. Demonstrated** [C]. *Test:* retrain the VAE over seeds.
2. **Cross-model reconstruction accuracy is not comparable. Demonstrated** [C]. CktGNN reconstructs a ≤8-node, 26-type DAG; the baselines reconstruct ≤16-node, 10-type DAGs; values and slots are ignored. *Improvement:* evaluate reconstruction at the circuit level, with the netlist and values within tolerance.
3. **Validity and novelty definitions. Demonstrated** [C]. Validity: DAG + one In/Out + no dangling node + no feedback-gm in the main path. It does *not* check duplicate or missing slots, simulatability or functionality. Novelty is computed over all valid DAG decodes including repeats, at topology level only.
4. **Lower prediction error does not imply better circuits. Hypothesis.** *Test:* correlate SGP RMSE with BO regret across encoders and seeds.
5. **Statistical significance. Missing** [P]. Many gaps in Table 1 are 0.01–0.04 RMSE, smaller than plausible seed variance [H].
6. **Fairness.** Parameter counts, input information (the slot prior) and training budgets are not matched or reported [P][C].
7. **Bit-identical BO FoMs across encoders. Demonstrated** [P Table 2]. This suggests BO converged to the same few circuits, so the "best FoM" metric has very low resolution.
8. **The paper's own Table 1 contradicts "consistently state-of-the-art". Demonstrated** [P]. On PM, D-VAE (RMSE 0.956, r 0.301) and Pathformer (r 0.297) beat CktGNN (0.973, 0.217).
9. **The BO code is not EI. Demonstrated** [C] (f* = 1.0 constant, "+1" term) / **Hypothesis** (σ is a variance).

### 6.4 Practical
- **Commercial-simulator dependence** (now solved for the behavioral level [E]).
- **No process variation, parasitics or layout**; no transistor-level validation [P]. These are the main barriers to industrial use.
- **No human-design comparison is shown**, despite the abstract's claim [P]. Fig. 8 shows the BO circuits only.
- **Transfer to other circuit classes** requires a new basis, a new slot template and a new generator [P App. A].

---

## 7. Ranked shortlist of improvements (Phase 6, first pass)

**Scoring (1–5):**
- *Novelty:* 5 = no close prior work found after searching; 1 = known technique applied as-is.
- *Gain:* expected improvement on the stated metrics.
- *Feasibility:* doable by one student on this laptop in ≤6–8 weeks.
- *Validation:* ease of a clean, measurable experiment.
- *Publication:* realistic venue fit.

Novelty scores are provisional. The quick search (§8) found close work only for B2 (TTARO) and partially for C1 (Lu et al. DATE 2022, without constraints); a full Phase-10 search is still required.

| ID | Proposed improvement | Limitation addressed | Technical change | Expected benefit | Difficulty | Compute | Main risk | Nov | Gain | Feas | Val | Pub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C1** | **Stability-aware, simulator-in-the-loop constrained latent BO** | FoM ≈ BW; PM label meaningless; pool-only BO | Relabel with `ocb_sim` (gain, UGF, branch-consistent PM, closed-loop pole check, power proxy Σgm); constrained EI (EI·P(feasible)) or qNEHVI on (UGF, gain, PM, power); decode → simulate each query | Finds high-GBW circuits that are *stable*; honest objective | Medium | Low (sim ≈ 0.6 ms/circuit) | Decoder validity limits the gains | 4 | 4 | 5 | 5 | 4 |
| **R1** | **Controlled re-evaluation and ablation of CktGNN** | §6.1(1–2), §6.3 | Multi-seed; ±slot prior; D-VAE on the subgraph DAG; matched params; circuit-level recon/validity; strong supervised baselines | Establishes what actually drives the gains | Low–medium | Medium (CPU hours) | Missing baselines | 3 | – | 5 | 5 | 4 (reproducibility track) |
| **A1** | Working two-level encoder (the inner GNN as described) | §6.1(1) | Fixed `subc_GNN` (values and internal topology) → h_v′ fed to the GRU, as in Eq. 3 | Tests the paper's core mechanism | Medium | Medium | Gains may be nil (still a result) | 2 | 2 | 4 | 5 | 3 |
| **B1** | Property-aware latent space | Unsupervised latents give poor SGP and BO | Multi-task heads (gain, UGF, PM) jointly with the VAE (the `--predictor` stub exists but is broken) | Better RMSE and BO sample efficiency | Low | Low | Known technique (e.g. property-supervised chemical VAEs) | 2 | 4 | 5 | 5 | 3 |
| **B2** | Online representation adaptation in BO (deep-kernel GP) | Fixed embeddings | Deep-kernel GP updated per query | Lower regret | Medium | Low | **Close prior work:** TTARO (Amin et al., arXiv 2608.12687, Aug 2026) does this on OCB | 1 | 3 | 4 | 4 | 2 |
| **A5** | Electrical canonicalization / equivalence-aware augmentation | Non-canonical graphs | Merge parallel/series like elements; contrastive or augmented training | Smoother latent space; honest novelty | Low–medium | Low | Modest gains | 3 | 2 | 5 | 4 | 3 |
| **C2** | Robust BO under device mismatch | No variation | Monte-Carlo ±σ on R/C/gm in `ocb_sim`; optimize a quantile | Variation-robust designs | Medium | Low–medium | Behavioral model limits realism | 3 | 3 | 4 | 4 | 3 |
| **D1** | Topology-grouped split and cross-benchmark generalization | 12.6% topology leakage | GroupKFold by topology; train 101 → test 301 | Honest generalization numbers | Low | Low | — | 2 | – | 5 | 5 | 3 (part of R1) |
| **D2** | Realistic-magnitude benchmark ("OCB-Phys") | 134 dB, 32 GHz | pF loads, power term, log-uniform sampling, re-generated with `ocb_sim` | Transferable conclusions | Medium | Low | Must re-run all baselines | 3 | – | 4 | 4 | 4 (datasets track) |
| **D3** | Transistor-level validation (ngspice + open PDK such as SKY130) | Behavioral abstraction | Map gm stages to sized OTA cells; simulate the best circuits | Real-world credibility | High | Medium | Mapping effort; PDK setup on Windows | 4 | – | 2 | 3 | 5 |
| **A4** | Learnable / extensible subgraph basis | Fixed basis | Motif mining + basis selection inside training | Discover out-of-basis circuits | High | High | Hard to make work | 5 | ? | 1 | 2 | 5 |
| **A2** | Attention or bidirectional outer pass | Directed-only coupling | Transformer over subgraph nodes, or bidirectional GRU | Small RMSE gains | Low | Low | Incremental | 1 | 2 | 5 | 5 | 2 |

### Top three directions

1. **R1, a controlled re-evaluation (do first; it is the baseline every later claim needs).** It is cheap, almost certain to yield a publishable reproducibility result (e.g. the ML Reproducibility Challenge or TMLR), and it decides whether A1 is worth doing. Core ablations:
   - CktGNN as released, with and without the slot prior;
   - D-VAE on the subgraph DAG, with and without slots;
   - D-VAE and DAGNN on the device DAG;
   - a random forest / MLP supervised ceiling;
   - all of the above over ≥3 seeds, with circuit-level validity measured by `ocb_sim`.
2. **C1, stability-aware constrained BO with an open simulator in the loop (main research contribution).** It directly addresses the most consequential demonstrated flaw (FoM ≈ BW, unstable optima). It is very feasible now that the simulator exists and has a crisp hypothesis:
   - *Hypothesis:* at equal simulator budget, constrained latent BO finds circuits with higher UGF subject to PM ≥ 60° and gain ≥ G_min than FoM-BO or random search does.
   - Close work found: TTARO (2026) uses OCB's FoM unconstrained; Lu et al. DATE 2022 uses VGAE + BO without stability constraints. No paper found combining latent BO over op-amp topologies with explicit stability constraints, which supports a novelty score of 4 pending the full search.
3. **B1 + A5, a property-aware, equivalence-aware latent space (supporting contribution).** Low risk, likely to improve both SGP metrics and C1's sample efficiency. Novelty is modest on its own, so it is best framed as a component of C1.

**Recommended project arc:** baseline reproduction (Milestones 8–9, 11) → R1 ablations → C1 with B1/A5 as ablated components → optional D3 transistor-level validation of the top designs for the paper's "real circuit" section.

---

## 8. Literature found so far (quick novelty check; the full Phase-10 review remains)

| Work | Relation to CktGNN / this project | Source |
|---|---|---|
| Lu, Lei, Yang, Zeng, Shang. *Topology Optimization of Operational Amplifier in Continuous Space via Graph Embedding.* DATE 2022 | VGAE embedding of behavioral op-amp DAGs plus BO, before CktGNN; not cited by it | [X] [DATE proceedings](https://past.date-conference.com/proceedings-archive/2022/html/0422.html) |
| Lu et al. *Automated Compensation Scheme Design for Operational Amplifier via Bayesian Optimization.* DAC 2021 | Behavioral-model graph mapping that OCB builds on (cited) | [X] [IEEE Xplore](https://ieeexplore.ieee.org/document/9586306) |
| Amin, Dutta, Franzon. *Test-Time Analog Circuit Representation Adaptation for BO (TTARO).* arXiv 2608.12687, Aug 2026 | Uses Ckt-Bench-101/301 and CktGNN/D-VAE/DAGNN encoders with the default FoM; no constraints; does not discuss label issues. Overlaps B2, and motivates C1 | [X] [arXiv](https://arxiv.org/abs/2608.12687) |
| Gao, Cao, Yang, Zhang. *AnalogGenie.* ICLR 2025; *AnalogGenie-Lite*, ICML 2025 | Same group; GPT-style transistor-level topology generation, which moves beyond the fixed basis | [X] [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html), [ICML 2025](https://icml.cc/virtual/2025/poster/45643) |
| LaMAGIC2; AUTOCIRCUIT-RL (ICML 2025); AnalogCoder-Pro (2025) | LLM-based topology generation (mostly power converters / transistor level) | [X] [LaMAGIC2](https://arxiv.org/abs/2506.10235v1), [AUTOCIRCUIT-RL](https://proceedings.mlr.press/v267/vijayaraghavan25a.html), [AnalogCoder-Pro](https://arxiv.org/abs/2508.02518v1) |
| ATOM (Shen et al.); INTO-OA (WL-kernel GP) | Behavior-level op-amp topology synthesis with freeze-thaw BO or graph-kernel GPs. **Only secondary descriptions found; verify** | [X] via the TTARO related-work section |
| *Compensation-Aware Topology Synthesis of Multistage Op-Amp* (AnalogSyn), ACM TODAES | Pole-zero-guided pruning of infeasible topologies, relevant to stability constraints (C1) | [X] [citedrive](https://www.citedrive.com/en/discovery/compensation-aware-topology-synthesis-of-multistage-op-amp/) |
| *Exploiting Function-Family Structure in Analog Circuit Optimization*, arXiv 2512.00712 | Constrained sizing BO; notes gain/BW/PM coupling through pole-zero spacing | [X] [arXiv](https://arxiv.org/abs/2512.00712) |

---

## 9. Next steps and decisions

**Ready to run now (no model changes):**
1. Baseline reproduction: CktGNN seeds 1–3, repo hyperparameters (300 epochs, patience 10), on CPU in parallel (~5–6 h wall-clock). Then the same with paper hyperparameters (200 / 20).
2. D-VAE seeds 1–3 (~15 h each on CPU; consider CUDA torch).
3. Port the SparseGP to PyTorch and validate it; implement fast hashed novelty; add circuit-level validity via `ocb_sim`.
4. Install pybnn (or implement a GP-EI baseline) and reproduce the Ckt-Bench-301 BO with both the original and a corrected EI.

**Decisions for you:**
- Compute: CPU-only (works) or install CUDA PyTorch in `.venv` (~2.5 GB) for the RTX 3050?
- Is a multi-hour background training run on this laptop acceptable?
- Target output: reproducibility report (R1) only, or R1 + C1 research paper?

## Appendix: files produced in this phase
| Path | Content |
|---|---|
| `scripts/audit_dataset.py` → `results/audit/` | Dataset audit, simple baselines, position/type table |
| `scripts/ocb_sim.py` | Open-source AC simulator for OCB circuits (Spectre-semantics re-implementation) |
| `scripts/validate_simulator.py` → `results/simulator/` | Label validation for 3 netlist conventions; per-circuit CSVs; `label_diagnostics.json` (placeholders, crossing ambiguity, PM convention, stability statistics) |
| `scripts/run_vae.py` → `results/vae/` | Faithful VAE training/evaluation runner (CktGNN, D-VAE; `--no-pos` ablation) |
| `scripts/paths.py`, `scripts/setup_data.py` | Data-root configuration; copies/extracts the OCB datasets to `D:\CktGNN_data\datasets` |
| `external/CktGNN` | Unmodified official repo @ `416cd03` |
