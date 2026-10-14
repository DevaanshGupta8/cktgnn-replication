# Handover: CktGNN replication and extension

*Last updated: 2026-10-14. State: Phases 1–4 complete and pushed; nothing running.*

This file lets someone continue the project without the original working session. Read it first, then the four phase reports.

---

## 1. What this project is

A reproduction and critical evaluation of **CktGNN** (Dong et al., *Circuit Graph Neural Network for Electronic Design Automation*, ICLR 2023; [arXiv 2308.16406](https://arxiv.org/abs/2308.16406); [authors' code](https://github.com/zehao-dong/CktGNN)), plus a first research extension: stability-aware circuit optimisation.

| Phase | Report | One-line result |
|---|---|---|
| 1. Analysis | [`reports/phase1_report.md`](reports/phase1_report.md) | The released code differs from the paper (no inner GNN, a hidden slot prior); FoM ≈ 99% bandwidth; ~8% of labels come from failed simulations; an open simulator reproduces the Cadence labels (~96%) |
| 2. Reproduction | [`reports/phase2_reproduction.md`](reports/phase2_reproduction.md) | CktGNN and D-VAE, 3 seeds each: the authors' released-data numbers reproduce. "CktGNN is state of the art" does not hold (it wins BW/FoM, loses gain); the paper's error bars are 5–20× too small |
| 3. Ablations | [`reports/phase3_ablation.md`](reports/phase3_ablation.md) | CktGNN's latent advantage comes from the subgraph representation alone; the slot prior helps generation; CktGNN's value pathway halves physical device values. Corrected after Phase 4: CktGNN generates the fewest stable circuits |
| 4. Optimisation | [`reports/phase4_optimization.md`](reports/phase4_optimization.md) | 63% of OCB circuits are open-loop unstable; optimising the paper's FoM picks unstable circuits; constrained BO finds feasible circuits in 16/18 runs (FoM-BO 9/18), but its best-design gain over random search is not significant in a good latent space |

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
<DATA>/results/bo/<MODEL>_<METHOD>_seed<k>.csv   Phase 4 optimisation traces
<DATA>/results/summary/                    aggregated tables, tests, reference evaluations
<DATA>/results/audit|simulator/            Phase 1 outputs
```
Trained runs that exist: `CktGNN_seed1..3`, `DVAE_seed1..3`, `SubgDVAE_seed1..3` (ablation A1), `SubgDVAEpos_seed1..3` (A2), all 300 epochs.

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
| `ablation_models.py` | A1/A2 models (`SubgDVAE`); `decompositions`/`infer_slots`, which read a slot-free graph as an OCB op-amp |
| `run_vae.py` | Training and evaluation runner (`--model CktGNN|DVAE|SubgDVAE|SubgDVAEpos`, `--resume`, `--eval-only`, `--eval-device`) |
| `launch_job.sh` | Launch one run with logs in the data folder; auto-resumes (`DEVICE=cuda bash scripts/launch_job.sh MODEL SEED --epochs 300 --patience 10`) |
| `sparse_gp_torch.py`, `sgp_regression.py` | PyTorch port of the authors' Theano sparse GP; the paper's latent-regression protocol |
| `eval_circuits.py` | Circuit-level evaluation of generated circuits (`--reference` for real data) |
| `latent_bo.py`, `run_bo_jobs.sh` | Phase 4 optimisation (`--model CktGNN|SubgDVAEpos --method rs|fom_bo|cbo --seed k`) |
| `aggregate_results.py`, `plot_results.py`, `plot_ablation.py`, `plot_bo.py` | Tables, figures, significance tests |
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

1. **A3 ablation:** A2 plus slots in the edge scorer and messages, to separate the bundled last ablation step (~3 GPU-hours).
2. **Trust-region constrained BO** (SCBO-style) and ≥ 20 runs per cell, to fix cBO's bimodality and settle the best-design comparison.
3. **Property-aware latent space:** joint VAE + predictor of gain/UGF/stability (the authors' `--predictor` stub in `main.py` is broken).
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
