# Handover: CktGNN replication and extension

*Final handover, 2026-10-15. State: Phases 1–9 complete and pushed; nothing running; no training is needed to continue.*

This file is for the person taking over. It is meant to be enough on its own: what was done, where every file is, how to get a working setup **without training anything**, how to reproduce every number, what went wrong along the way, and what is left to do.

**Reading order:**
1. This file.
2. [`reports/project_summary.md`](reports/project_summary.md): the paper's claims vs our findings, and the status of the proposed improvements.
3. The phase reports you need (§1).

Phases 5–9 were pre-registered: each protocol file was committed (and from Phase 7 on, pushed) before its results existed.

---

## 1. What this project is and what it found

A reproduction and critical evaluation of **CktGNN** (Dong et al., *CktGNN: Circuit Graph Neural Network for Electronic Design Automation*, ICLR 2023; [arXiv 2308.16406](https://arxiv.org/abs/2308.16406); [authors' code](https://github.com/zehao-dong/CktGNN)). CktGNN learns a latent space of op-amp circuit graphs with a VAE and optimises circuits in it with Bayesian optimisation (BO). The project went on to stability-aware optimisation and better generative models.

| Phase | Report | Result |
|---|---|---|
| 1. Analysis | [`phase1_report.md`](reports/phase1_report.md) | The released code differs from the paper: no inner GNN, an undocumented slot prior. FoM ≈ 99% bandwidth. ~8% of labels come from failed simulations. Our open simulator matches the Cadence labels on ~96% of circuits |
| 2. Reproduction | [`phase2_reproduction.md`](reports/phase2_reproduction.md) | CktGNN and D-VAE, 3 seeds each: the authors' released-data numbers reproduce. "CktGNN is state of the art" does not hold (it wins BW/FoM, loses gain). The paper's error bars are 5–20× too small |
| 3. Ablations | [`phase3_ablation.md`](reports/phase3_ablation.md) | CktGNN's latent advantage comes from the subgraph representation; the slot prior helps generation. CktGNN generates the fewest stable circuits |
| 4. Optimisation | [`phase4_optimization.md`](reports/phase4_optimization.md) | 63% of OCB circuits are open-loop unstable; optimising the paper's FoM picks unstable circuits. Constrained BO (cBO) finds spec-meeting circuits far more often |
| 5. A3 + SCBO | [`phase5_report.md`](reports/phase5_report.md) | CktGNN's value pathway alone explains its reconstruction gain and its unphysical device values. Trust-region constrained BO (SCBO) does not beat cBO (pre-registered, negative) |
| 6. Property-aware latent space | [`phase6_report.md`](reports/phase6_report.md) | P = A2 + a jointly trained property predictor: the best latent regression of all models (FoM RMSE 0.475) and AUC 0.94 for "meets spec". Its optimisation gain did not replicate (Phase 7) |
| 7. Confirmation + replication | [`phase7_report.md`](reports/phase7_report.md) | Graded-stability SCBO not confirmed on fresh seeds. Best-design conclusions from 20 runs flip between seed sets; feasibility-rate conclusions replicate |
| 8. Endpoint reliability | [`phase8_report.md`](reports/phase8_report.md) | AUC is no more reliable than the final best (sign agreement 6/15 vs 7/15); feasibility share is (ρ = 0.92, post hoc). §3 lists what the optimisation study can and cannot claim |
| 9. Better generation | [`phase9_report.md`](reports/phase9_report.md) | **Predictor-guided generation works** (confirmed): selecting latents with P's predictor raises the spec-meeting share of generated circuits 1.6× (2.62% vs 1.62%) and increases diversity. A linear classifier fitted afterwards does nothing for A2 or CktGNN. Normalised value inputs (Pn): reconstruction 0.24 → 0.32 and physical values 82% → 95% (primary endpoints p = 0.11 and 0.050, not significant), at the cost of worse latent regression (p = 0.007) |

**The models to build on:**
- **P (`SubgDVAEprop`) with predictor-guided generation** (`scripts/guided_generation.py`) when the goal is circuits that meet a stability spec. It has the best latent space and the best spec-meeting rate.
- **Pn (`SubgDVAEpropn`)** when reconstruction accuracy and physical device values matter more.

All trained runs are in the release; nothing needs retraining.

---

## 2. Where everything is

| What | Where |
|---|---|
| Code, reports, figures, all small results and logs | This repository, branch `main`: https://github.com/DevaanshGupta8/cktgnn-replication |
| **Trained models: all 21 runs**, final checkpoint + latents + config | GitHub release [`models-v1`](https://github.com/DevaanshGupta8/cktgnn-replication/releases/tag/models-v1), one zip per model (~80 MB each) plus `SHA256SUMS.txt` |
| Datasets (Ckt-Bench-101/301) | Not redistributed. `scripts/setup_data.py` copies them from the authors' repo |
| Authors' code | `external/CktGNN` (not in this repo; clone it, §3), unmodified, commit `416cd035f79dd8cfcb60ac0a4792b43255adf0b7` |
| The working data folder | `D:\CktGNN_data` on the original laptop. Change it with the `CKTGNN_DATA` environment variable (`scripts/paths.py`) |

**Trained runs** (each `<MODEL>_seed1..3`, 300 epochs, in the release):

| Run name | Model | Phase |
|---|---|---|
| `CktGNN` | the authors' model (batched, verified-equivalent training path) | 2 |
| `DVAE` | D-VAE baseline on the device graph | 2 |
| `SubgDVAE` | A1: D-VAE equations on the subgraph graph | 3 |
| `SubgDVAEpos` | A2: A1 + slot prior | 3 |
| `SubgDVAEslot` | A3: A2 + CktGNN's slot-aware messages/edge scorer | 5 |
| `SubgDVAEprop` | P: A2 + jointly trained property predictor | 6 |
| `SubgDVAEpropn` | Pn: P with normalised device-value inputs | 9 |

**`results/` in this repository** (copied from the data folder by `scripts/export_results.sh`):

| Path | Contents |
|---|---|
| `results/vae/<run>/` | Per run: `config.json`, `train_log.csv`, `stdout.log`, `results.json` (decoder metrics, predictor AUCs), `sgp_results.csv` + `sgp_log.txt` (latent regression), `circuit_eval.json` + log (10,000 simulated samples), `guided_generation.json` + `guided_latents.csv` (Phase 9) |
| `results/bo/`, `results/bo5/`, `results/bo7/` | One CSV per optimisation run (every simulation of every run), plus `log.txt`. `bo`: Phase 4; `bo5`: Phases 5–6 (seeds 1–20); `bo7`: Phase 7 (seeds 21–40) |
| `results/jobs/` | The job lists that defined every BO batch |
| `results/summary/` | Every aggregated table and test behind the reports (`per_run.csv`, `ablation_*`, `bo*_table/tests`, `bo_auc_*`, `guided_*`, …) |
| `results/simulator/` | Simulator validation, label diagnostics, and `analysis_all.csv`: simulator labels (gain, UGF, PM, pole stability, meets-spec) for all 10,000 real circuits |
| `results/audit/` | Phase 1 dataset audit |

**Not kept anywhere public:** intermediate checkpoints (epochs 50–250) and the smoke-test folders. Nothing in the reports depends on them.

---

## 3. Getting a working setup without training

### 3.1 Install (≈ 10 minutes)

```bash
git clone https://github.com/DevaanshGupta8/cktgnn-replication
```
```bash
cd cktgnn-replication
```
```bash
git clone https://github.com/zehao-dong/CktGNN external/CktGNN
```
```bash
git -C external/CktGNN checkout 416cd035f79dd8cfcb60ac0a4792b43255adf0b7
```
```bash
python -m venv .venv
```
Install PyTorch. The CPU build is enough for everything except training:
```bash
.venv/Scripts/python.exe -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu
```
or, to train on an NVIDIA GPU, the CUDA build used here:
```bash
.venv/Scripts/python.exe -m pip install "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
```
```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
On Linux/macOS, use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

### 3.2 Re-run every analysis immediately (no data folder needed)

Point the data path at the repository; all summary tables and figures are regenerated from `results/`:
```bash
export CKTGNN_DATA="$PWD"
```
```bash
.venv/Scripts/python.exe scripts/plot_ablation.py
```
```bash
.venv/Scripts/python.exe scripts/plot_bo.py --dir bo5
```
```bash
.venv/Scripts/python.exe scripts/plot_bo.py --dir bo7
```
```bash
.venv/Scripts/python.exe scripts/bo_auc.py
```
```bash
.venv/Scripts/python.exe scripts/guided_analysis.py --models SubgDVAEprop SubgDVAEpropn SubgDVAEpos CktGNN
```
These reproduce the committed tables and figures byte for byte; this was checked on a fresh clone.

### 3.3 Use the trained models (no training)

1. Choose a data folder, e.g. `~/cktgnn_data`, and set `CKTGNN_DATA` to it.
2. Copy the datasets into it:
   ```bash
   .venv/Scripts/python.exe scripts/setup_data.py
   ```
3. Download the models from the release, all at once with the GitHub CLI:
   ```bash
   gh release download models-v1 --repo DevaanshGupta8/cktgnn-replication --dir release_zips
   ```
   Or download them from the release page in a browser. Check them against `SHA256SUMS.txt`.
4. Unzip every model zip **into the data folder**; each zip contains `results/vae/<run>/`.
5. Copy the simulator labels in (P and Pn need them for evaluation):
   ```bash
   mkdir -p "$CKTGNN_DATA/results/simulator" && cp results/simulator/analysis_all.csv "$CKTGNN_DATA/results/simulator/"
   ```
   Alternatively, regenerate them with `scripts/circuit_labels.py`.

Then everything downstream works. Examples:

| Task | Command |
|---|---|
| Decoder metrics of a run | `.venv/Scripts/python.exe scripts/run_vae.py --model SubgDVAEprop --seed 1 --eval-only` |
| Latent SGP regression | `.venv/Scripts/python.exe scripts/sgp_regression.py --run $CKTGNN_DATA/results/vae/SubgDVAEprop_seed1` |
| Simulate 10,000 generated circuits | `.venv/Scripts/python.exe scripts/eval_circuits.py --run $CKTGNN_DATA/results/vae/SubgDVAEprop_seed1 --model SubgDVAEprop` |
| Guided generation | `.venv/Scripts/python.exe scripts/guided_generation.py --model SubgDVAEprop --seed 1` |
| One optimisation run | `.venv/Scripts/python.exe scripts/latent_bo.py --model SubgDVAEprop --method cbo --seed 1 --out <dir>` |
| A batch of optimisation runs | `PAR=4 OUT=<dir> bash scripts/run_bo_jobs.sh results/jobs/bo5_jobs.txt` |

**Exactness**, verified on a fresh clone with a data folder built only from the authors' repo and this release, without any training:
- optimisation runs (`latent_bo.py`), circuit evaluation (`eval_circuits.py`) and guided generation (`guided_generation.py`) reproduce the stored results **bit for bit**;
- `run_vae.py --eval-only` reproduces every rate and count exactly (reconstruction, validity, novelty, predictor AUCs). Continuous values (NLL, predictor RMSE) differ only at the 7th significant digit, from float32 summation order with a different CPU thread count;
- all analysis scripts in §3.2 regenerate the committed tables and figures with **zero changed files**.

The SGP regression is float64 and seeded, but was not re-run in that check (1–2 h per run).

### 3.4 Sanity checks
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
(≈ 95.7% bandwidth + gain match)

### 3.5 If you do want to train

```bash
DEVICE=cuda bash scripts/launch_job.sh SubgDVAEprop 1 --epochs 300 --patience 10
```
- Logs go to the data folder.
- Re-running the same command resumes from `last.pt`.
- Training takes 1.5–2 GPU-hours per model, three seeds in parallel on an RTX 3050 6 GB (D-VAE ≈ 4 h); much slower on a CPU.

---

## 4. Reproducing each phase

Each report ends with a **Reproduce** section listing its commands. In short:

| Phase | Generates | Analyses |
|---|---|---|
| 1 | `audit_dataset.py`, `validate_simulator.py` | in the scripts |
| 2, 3, 5, 6, 9B | `launch_job.sh` (training), `sgp_regression.py`, `eval_circuits.py` | `aggregate_results.py`, then `plot_results.py` (Phase 2), `plot_ablation.py` (Phases 3, 5, 6, 9B) |
| 4 | `run_bo_jobs.sh` with `latent_bo.py` → `results/bo` | `plot_bo.py` |
| 5, 6 | `run_bo_jobs.sh` → `results/bo5` (job lists in `results/jobs/`) | `plot_bo.py --dir bo5` |
| 7 | `run_bo_jobs.sh` → `results/bo7` | `plot_bo.py --dir bo7` |
| 8 | (uses bo5 + bo7) | `bo_auc.py` |
| 9A | `guided_generation.py` | `guided_analysis.py` |

---

## 5. Code map (`scripts/`)

| Script | Purpose |
|---|---|
| `paths.py` | All locations: repository, data folder (`CKTGNN_DATA`) |
| `setup_data.py` | Copy the datasets from the authors' repo into the data folder |
| `audit_dataset.py` | Dataset audit and simple baselines (Phase 1) |
| `ocb_sim.py` | **Open simulator**: the netlist rules of the authors' `graph_to_fom.py`, an MNA AC solver, `measure`, `stability` (exact poles), `analyze` |
| `validate_simulator.py`, `circuit_labels.py` | Simulator vs released labels; simulator labels for all real circuits |
| `fast_models.py` | Batched, numerically equivalent `encode`/`loss` for the authors' CktGNN and D-VAE (5–9× faster) |
| `ablation_models.py` | `SubgDVAE` with flags `use_pos` (A2), `slot_struct` (A3), `n_props` (P), `norm_values` (Pn). `decompositions`/`infer_slots` read a slot-free graph as an OCB op-amp |
| `run_vae.py` | Train / evaluate one run (`--model`, `--seed`, `--resume`, `--eval-only`, `--eval-device`) |
| `launch_job.sh` | Launch one run with logs in the data folder; auto-resumes |
| `sparse_gp_torch.py`, `sgp_regression.py` | PyTorch port of the authors' Theano sparse GP; the paper's latent-regression protocol |
| `eval_circuits.py` | Decode → read as an op-amp → simulate → score (functional, stable, meets spec, novel, values in range); `--reference` scores real data |
| `latent_bo.py` | Optimisation in a latent space: `rs`, `fom_bo` (the paper's objective), `cbo`, `scbo`, `scbo_g` |
| `run_bo_jobs.sh` | Parallel, resumable batch runner (`PAR=n OUT=dir`); pins BLAS to one thread per job |
| `guided_generation.py`, `guided_analysis.py` | Phase 9A: predictor-guided generation and its analysis |
| `demo.py` | Live demo, 4 steps (see [`DEMO.md`](DEMO.md)); `--auto` regenerates `demo_output/`; backup figures in `reports/demo/` |
| `aggregate_results.py`, `plot_results.py`, `plot_ablation.py`, `plot_bo.py`, `bo_auc.py` | Tables, figures and statistical tests |
| `export_results.sh` | Copy all small results and logs from the data folder into `results/` (run before committing) |
| `finish_run_on_cpu.sh` | Legacy helper from Phase 2 |

Tests: `tests/test_fast_models.py` (fast path ≡ authors' code), `tests/test_ablation_models.py` (A1–A3, P, Pn consistency).

---

## 6. Gotchas (learned the hard way)

1. **Train on the GPU, evaluate on the CPU.** The authors' per-graph decoder (used to evaluate CktGNN and D-VAE) is ~15× slower on the GPU; `--eval-device cpu` is the default.
2. **RAM limits parallelism.** A training run takes ~1–1.4 GB, an SGP or BO process ~0.5–0.8 GB. Three GPU training runs side by side is the maximum on 16 GB.
3. **BLAS threads.** numpy/sklearn use every core by default; one BO job slowed GPU training by ~65%. Set `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` (`run_bo_jobs.sh` sets them to 1).
4. **The SGP regression is the slowest step**: 1–2 h per run on a busy CPU. Optimisation runs need only `last.pt` and `latents.npz`, so they can run in parallel with it.
5. **Never seed with `hash()`.** Python salts string hashes per process, which is why Phase 4's BO runs cannot be re-run bit-for-bit. `latent_bo.py` uses `zlib.crc32`.
6. **Resuming on the GPU** used to crash (`RNG state must be a torch.ByteTensor`), because the saved CPU RNG state was loaded onto the GPU. Fixed in Phase 9 (`run_vae.py`, `load_state`).
7. **Check for real completion before chaining jobs.** A failed run still appends an `exit` line to `stdout.log`; wait for `results.json` instead.
8. **On Windows, stopping a background shell may leave its Python child running.** Kill leftovers by command line (`Get-CimInstance Win32_Process`).
9. **`models_ig.py` caches the device on first use.** After `model.to(...)`, set `model.device = None` (`run_vae.py` does this).
10. **Theano does not install on Python 3.13**, so `sparse_gp_torch.py` replaces it. With the authors' settings the SGP underfits; we kept them for comparability.
11. **Phase margin is not a stability test on OCB.** Use `ocb_sim.stability` (poles).
12. **Slot-free graphs can be read in more than one way** (21% of real circuits; 1.7% electrically different). D-VAE's device graphs cannot be turned into netlists at all.
13. **No run has converged** in 300 epochs; `ReduceLROnPlateau` never fired.
14. **Windows paths:** use `D:/...` in Python strings; set `PYTHONIOENCODING=utf-8` when printing ± or Δ.
15. **Best-design optimisation results are noisy.** With 20 runs, comparisons flip between seed sets (Phases 7–8). Use the feasible-proposal share, or many more runs.

---

## 7. Known deviations from the authors' setup

- The released Ckt-Bench-101 instead of the paper's (unreleased) dataset.
- 300 epochs / patience 10 (the authors' README) instead of 200 / 20 (the paper).
- Batched training path (verified equivalent); float64 PyTorch SGP instead of Theano; exact hashed novelty check.
- Training on the GPU, evaluation on the CPU: different random streams, same distributions.
- The ablation models use a batched decoder with the same sampling steps as D-VAE's.
- An open simulator instead of Cadence Spectre (validated; ~2.7% of circuits unexplained).

---

## 8. Open work, in priority order

1. **Close the stability gap.** Even guided P generates stable circuits at 6.8%, against 11.6% for real circuits. Options: a larger predictor weight λ (only λ = 1 was tried), a stability-weighted loss, or decoding with predictor feedback.
2. **Use P's predictor inside the optimiser:** as a feasibility filter for candidate latents, or as a GP prior mean. Guided generation shows the predictor carries real signal (Phase 9).
3. **Settle Pn:** more seeds (≥ 5), and the same normalisation applied to A2/A3 to separate it from the predictor. Phase 9's primary endpoints were p = 0.11 / 0.050 with 3 seeds.
4. **Optimiser comparisons:** use the feasible-proposal share as the primary endpoint (reliable across seed sets, Phase 8). Best-design claims need many more runs or a larger budget per run.
5. **Missing paper baselines:** DAGNN (needs a PyG 2.x port of `layers/dagnn_pyg.py`); PACE/GCN/GIN/NGNN/Graphormer (not in the authors' repo).
6. **The paper's BO on Ckt-Bench-301:** needs `pybnn`. The authors' EI code is wrong (incumbent fixed at 1.0, stray `+1`).
7. **Paper hyperparameters** (200 epochs, patience 20) and longer training, to check convergence.
8. **Transistor-level validation** of the best designs (ngspice + an open PDK such as SKY130). This is the first step towards the proposed *real-chip-conditions* improvement: PVT corners, Monte Carlo mismatch, parasitics.
9. **Power/area objectives** (the proposed *multi-objective* improvement). OCB circuits have no power or area, so this needs a proxy (e.g. total gm for bias current, total C for area) or the transistor-level flow of item 8. Constrained BO is the starting point; Pareto methods (qNEHVI) or RL are alternatives.
10. **Other circuit classes** (the proposed *dataset expansion*: filters, ADC sub-blocks, RF). This needs a generator and simulator per class; filters built from biquad blocks fit the subgraph idea best.

The status of every proposed improvement is in [`reports/project_summary.md`](reports/project_summary.md) §6.

**Open decision:** the project scope, i.e. a reproducibility report only, or a reproducibility report plus a research paper building on Phases 4–9.

---

## 9. Conventions

- **Evidence tags in the reports:** [P] paper, [C] verified in the code, [E] measured here, [X] external source, [H] hypothesis.
- **Pre-registration:** each new phase has a `phaseN_protocol.md`, committed and pushed before running; exploratory analyses are labelled as such.
- **One change at a time:** each model or method change gets ≥ 3 seeds (models) or ≥ 20 paired runs (optimisers), with stated tests and multiple-comparison correction.
- **Before committing:** `bash scripts/export_results.sh`. Never commit checkpoints, datasets, `.venv`, `external/` or the paper PDF (`.gitignore` covers them). New trained runs go into a GitHub release.
