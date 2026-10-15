# Handover: CktGNN replication and extension

*Last updated: 2026-10-15. State: Phases 1–8 complete and pushed; nothing running.*

This file lets someone continue the project without the original working session. Read it first, then the eight phase reports; Phase 8 §3 summarises what the optimisation study can claim. Phases 5–8 were pre-registered: each protocol file was committed before its results existed.

---

## 1. What this project is

A reproduction and critical evaluation of **CktGNN** (Dong et al., *Circuit Graph Neural Network for Electronic Design Automation*, ICLR 2023; [arXiv 2308.16406](https://arxiv.org/abs/2308.16406); [authors' code](https://github.com/zehao-dong/CktGNN)), plus a first research extension: stability-aware circuit optimisation.

| Phase | Report | One-line result |
|---|---|---|
| 1. Analysis | [`reports/phase1_report.md`](reports/phase1_report.md) | The released code differs from the paper (no inner GNN, a hidden slot prior); FoM ≈ 99% bandwidth; ~8% of labels come from failed simulations; an open simulator reproduces the Cadence labels (~96%) |
| 2. Reproduction | [`reports/phase2_reproduction.md`](reports/phase2_reproduction.md) | CktGNN and D-VAE, 3 seeds each: the authors' released-data numbers reproduce. "CktGNN is state of the art" does not hold (it wins BW/FoM, loses gain); the paper's error bars are 5–20× too small |
| 3. Ablations | [`reports/phase3_ablation.md`](reports/phase3_ablation.md) | CktGNN's latent advantage comes from the subgraph representation alone; the slot prior helps generation; CktGNN's value pathway halves physical device values. Corrected after Phase 4: CktGNN generates the fewest stable circuits |
| 4. Optimisation | [`reports/phase4_optimization.md`](reports/phase4_optimization.md) | 63% of OCB circuits are open-loop unstable; optimising the paper's FoM picks unstable circuits; constrained BO finds feasible circuits in 16/18 runs (FoM-BO 9/18), but its best-design gain over random search is not significant in a good latent space |
| 5. A3 + SCBO | [`reports/phase5_report.md`](reports/phase5_report.md) | A3 shows CktGNN's value pathway alone explains its reconstruction gain and its unphysical device values. SCBO does not beat cBO or random search (pre-registered, negative) because the constraint score is flat for unstable circuits; an exploratory graded score repairs it. With 20 runs, cBO clearly beats random search and FoM-BO in CktGNN's space |
| 6. Property-aware latent | [`reports/phase6_report.md`](reports/phase6_report.md) | A2 + a jointly trained property predictor: better reconstruction, far better SGP regression (FoM RMSE 0.475, the best of all models), predictor AUC 0.94 for "meets spec"; cBO's best design improves (median 17.6 vs 6.6 GHz) but not significantly (Holm p = 0.28) |
| 7. Confirmatory + replication | [`reports/phase7_report.md`](reports/phase7_report.md) | Graded-stability SCBO does not beat plain SCBO on fresh seeds (Holm p = 1.0). Several best-design conclusions from Phases 5–6 flip between seed sets; feasibility-rate conclusions replicate. Best-design comparisons need ≥ 40 runs or a lower-variance endpoint |
| 8. Endpoint reliability | [`reports/phase8_report.md`](reports/phase8_report.md) | AUC is no more reliable than final best (cross-seed-set sign agreement 6/15 vs 7/15); feasibility share is (ρ = 0.92, 13/15, post hoc). With 40 paired runs only cBO > SCBO in P's space survives correction |

---

## 2. Where everything lives

| What | Where |
|---|---|
| Code, reports, small results | this repository: https://github.com/DevaanshGupta8/cktgnn-replication (branch `main`) |
| Authors' code (not in this repo) | `external/CktGNN`, unmodified, commit `416cd035f79dd8cfcb60ac0a4792b43255adf0b7` |
| Datasets, checkpoints, full results | data folder, default `D:\CktGNN_data` (override with the `CKTGNN_DATA` environment variable; see `scripts/paths.py`) |
| Small results under version control | `results/`, refreshed by `bash scripts/export_results.sh` |

Data-folder layout:
```
<DATA>/datasets/CktBench101|CktBench301/   copied/extracted by scripts/setup_data.py
<DATA>/results/vae/<MODEL>_seed<k>/        config, train_log.csv, last.pt, model_checkpoint*.pth, latents.npz,
                                           results.json, sgp_results.csv, circuit_eval.json
<DATA>/results/bo/<MODEL>_<METHOD>_seed<k>.csv   Phase 4 optimisation traces (not bit-reproducible, see Phase 5 §B.7)
<DATA>/results/bo5/<MODEL>_<METHOD>_seed<k>.csv  Phases 5-6 traces (seeds 1-20, deterministic)
<DATA>/results/bo7/<MODEL>_<METHOD>_seed<k>.csv  Phase 7 traces (seeds 21-40)
<DATA>/results/simulator/analysis_all.csv  simulator labels of all 10,000 circuits (scripts/circuit_labels.py)
<DATA>/results/summary/                    aggregated tables, tests, reference evaluations
<DATA>/results/audit|simulator/            Phase 1 outputs
```
Trained runs that exist: `CktGNN_seed1..3`, `DVAE_seed1..3`, `SubgDVAE_seed1..3` (ablation A1), `SubgDVAEpos_seed1..3` (A2), `SubgDVAEslot_seed1..3` (A3), `SubgDVAEprop_seed1..3` (P), all 300 epochs.

---

## 3. Setting up from scratch

```bash
git clone https://github.com/DevaanshGupta8/cktgnn-replication
```
```bash
git clone https://github.com/zehao-dong/CktGNN cktgnn-replication/external/CktGNN
```
```bash
git -C cktgnn-replication/external/CktGNN checkout 416cd035f79dd8cfcb60ac0a4792b43255adf0b7
```
Then, inside `cktgnn-replication`:
```bash
python -m venv --system-site-packages .venv
```
```bash
.venv/Scripts/python.exe -m pip install "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
```
```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
```bash
.venv/Scripts/python.exe scripts/setup_data.py
```
The original environment: Windows 11, Python 3.13, PyTorch 2.13.0+cu130, RTX 3050 6 GB laptop GPU, i5-12450HX, 16 GB RAM. numpy, scipy, pandas, matplotlib, networkx and tqdm come from the system interpreter (`--system-site-packages`); versions are listed in `requirements.txt`.

Sanity checks after setup:
```bash
.venv/Scripts/python.exe tests/test_fast_models.py --device cpu
```
(must print `ALL EQUIVALENT`)
```bash
.venv/Scripts/python.exe tests/test_ablation_models.py
```
(must print `ALL CONSISTENT`)
```bash
.venv/Scripts/python.exe scripts/validate_simulator.py --n 10000
```
(about 95.7% BW+gain match; see `results/simulator/label_diagnostics.json`)

---

## 4. Code map

| Script | Purpose |
|---|---|
| `paths.py` | All locations (repo, data folder) |
| `setup_data.py` | Populate the data folder from the authors' repo |
| `audit_dataset.py` | Dataset audit, simple baselines (Phase 1) |
| `ocb_sim.py` | **Open simulator**: netlist rules of `graph_to_fom.py`, MNA AC solver, `measure`, `stability` (exact poles), `analyze` |
| `validate_simulator.py` | Simulator vs released labels |
| `fast_models.py` | Batched, numerically equivalent `encode`/`loss` for the authors' CktGNN and DVAE (5–9× faster) |
| `ablation_models.py` | A1/A2/A3/P models (`SubgDVAE` with `use_pos`, `slot_struct`, `n_props`); `decompositions`/`infer_slots`, which read a slot-free graph as an OCB op-amp |
| `run_vae.py` | Training and evaluation runner (`--model CktGNN|DVAE|SubgDVAE|SubgDVAEpos|SubgDVAEslot|SubgDVAEprop`, `--resume`, `--eval-only`, `--eval-device`) |
| `circuit_labels.py` | Simulator labels for every real circuit (used by P) |
| `launch_job.sh` | Launch one run with logs in the data folder; auto-resumes (`DEVICE=cuda bash scripts/launch_job.sh MODEL SEED --epochs 300 --patience 10`) |
| `sparse_gp_torch.py`, `sgp_regression.py` | PyTorch port of the authors' Theano sparse GP; the paper's latent-regression protocol |
| `eval_circuits.py` | Circuit-level evaluation of generated circuits (`--reference` for real data) |
| `latent_bo.py`, `run_bo_jobs.sh` | Optimisation (`--model ... --method rs|fom_bo|cbo|scbo|scbo_g --seed k --out DIR`); the runner takes `PAR=n OUT=dir`, skips finished runs and pins BLAS to one thread per job |
| `bo_auc.py` | Phase 8 endpoints (AUC, time to first feasible) and their reliability across seed sets |
| `aggregate_results.py`, `plot_results.py`, `plot_ablation.py`, `plot_bo.py` | Tables, figures, significance tests (`plot_bo.py --dir bo5` for Phases 5–6) |
| `export_results.sh` | Copy small results into `results/` before committing |
| `finish_run_on_cpu.sh` | Legacy helper: move the evaluation of an already-running old-style job to the CPU |

---

## 5. Gotchas (learned the hard way)

1. **Train on the GPU, evaluate on the CPU.** With the fast path the GPU is fastest (CktGNN 0.13 s, D-VAE 0.25 s per batch). The authors' per-graph *decoder* (used for CktGNN/D-VAE evaluation) is ~15× slower on the GPU, hence `--eval-device cpu` (the default).
2. **RAM is the parallelism limit.** Each training run uses ~1–1.4 GB and each SGP/BO process ~0.5–0.8 GB. Three GPU training runs side by side is the safe maximum on 16 GB.
3. **On Windows, stopping a background shell does not always kill its Python child.** Kill leftovers by command line, e.g. matching `run_vae.py` with `Get-CimInstance Win32_Process`.
4. **`models_ig.py` caches the device on first use.** After `model.to(...)`, set `model.device = None` (`run_vae.py` does this).
5. **Theano does not install on Python 3.13.** Use `sparse_gp_torch.py`. With the authors' settings (lr 5e-4, 100 epochs) the SGP underfits; keep those settings for comparability, but be aware of it.
6. **Phase margin is not a stability test on OCB.** Most circuits are open-loop unstable. Use `ocb_sim.stability` (poles).
7. **Slot-free subgraph graphs can be ambiguous.** OCB uses different edge meanings for main-path and extra subgraphs. 21% of real circuits have more than one reading; only 1.7% differ electrically. D-VAE's device graphs cannot be turned into netlists.
8. **Windows paths.** Use `D:/...` (not `/d/...`) inside Python strings. Set `PYTHONIOENCODING=utf-8` when printing tables with ± or Δ.
9. **Learning rate.** `ReduceLROnPlateau` (patience 10) never fired in 300 epochs, and no run converged. Results could change with longer training.
10. **BLAS threads.** numpy/sklearn use every core by default; one BO job slowed GPU training by ~65%. `run_bo_jobs.sh` sets `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
11. **SGP regression is the slow step**: about 1–2 h per run on a busy CPU (9 folds × 100 epochs). BO runs only need `last.pt` and `latents.npz`, so they can start before the SGP finishes.
12. **Never seed with `hash()`.** Python salts string hashes per process; Phase 4's BO runs cannot be re-run bit-for-bit for this reason. `latent_bo.py` now uses `zlib.crc32`.

---

## 6. Known deviations from the authors' setup (all documented in the reports)

- Released Ckt-Bench-101 instead of the paper's (unreleased) dataset.
- 300 epochs / patience 10 (README) rather than 200 / 20 (paper).
- Batched training path (verified equivalent), float64 PyTorch SGP instead of Theano, hashed novelty (exact).
- Training on GPU vs evaluation on CPU (different random streams, same distributions).
- A1/A2 use a batched decoder with the same sampling steps as D-VAE's.
- Open simulator instead of Cadence Spectre (validated; ~2.7% of circuits unexplained).

---

## 7. Open work, in priority order

*Done in Phases 5–6: the A3 ablation, SCBO with 20 runs per cell, and the property-aware latent space.*

*Phase 7 tested graded SCBO (not confirmed) and showed that 20-run best-design comparisons are unreliable here.*

1. **Optimiser comparisons:** use the feasible-proposal share (or a similar per-proposal measure) as the primary endpoint; it is reliable across seed sets (Phase 8). Best-design claims need far more runs or a larger budget per run (AUC did not help).
2. **Use P's predictor inside the optimiser** (feasibility filter or GP prior mean).
3. **Why the value pathway helps reconstruction:** test feeding normalised (not raw 1–101) device values to A2/A3, or dropping them from the decoder input.
4. **Missing paper baselines:** DAGNN (needs a PyG 2.x port of `layers/dagnn_pyg.py`), PACE/GCN/GIN/NGNN/Graphormer (not in the authors' repo).
5. **The paper's BO on Ckt-Bench-301:** needs `pybnn`; the authors' EI code is wrong (incumbent fixed at 1.0, stray `+1`).
6. **Paper hyperparameters** (200 epochs, patience 20), and longer training to check convergence.
7. **Transistor-level validation** of the best feasible designs (ngspice + an open PDK such as SKY130).

Scope decision still open with the project owner: a reproducibility report only, or a reproducibility report plus a research paper built on Phase 4.

---

## 8. Conventions

- **Evidence tags in the reports:** [P] paper, [C] code-verified, [E] measured here, [X] external, [H] hypothesis.
- **One change at a time:** each ablation or method change gets ≥ 3 seeds and paired tests.
- **Before committing:** run `bash scripts/export_results.sh`; never commit checkpoints, datasets, `.venv`, `external/` or the paper PDF (`.gitignore` covers these).
- **Commit messages** describe the phase or change; results stay reproducible from the scripts listed in each report's "Reproduce" section.
