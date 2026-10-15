# CktGNN replication and extension

A research project to reproduce, critically evaluate and extend
*CktGNN: Circuit Graph Neural Network for Electronic Design Automation* (ICLR 2023).

- Paper: ICLR 2023 · [OpenReview](https://openreview.net/pdf?id=NE2911Kq1sp) · [arXiv 2308.16406](https://arxiv.org/abs/2308.16406) (v2 adds Appendix G on the updated OCB)
- Official code + dataset (unmodified): [github.com/zehao-dong/CktGNN](https://github.com/zehao-dong/CktGNN) → [`external/CktGNN`](external/CktGNN), commit `416cd03`
- **Start here: [`HANDOVER.md`](HANDOVER.md)** (state, setup, code map, gotchas, open work)
- **Phase 1 report** (paper analysis, code verification, dataset audit, reproduction checklist, critique, ranked improvements): [`reports/phase1_report.md`](reports/phase1_report.md)
- **Phase 2 report** (baseline reproduction: CktGNN and D-VAE, 3 seeds each, vs paper and authors' released-data results): [`reports/phase2_reproduction.md`](reports/phase2_reproduction.md)
- **Phase 3 report** (ablation study: which of CktGNN's design choices explains its results, with simulator-based evaluation of generated circuits): [`reports/phase3_ablation.md`](reports/phase3_ablation.md)
- **Phase 4 report** (stability-aware optimisation: exact stability check, the paper's FoM objective vs constrained Bayesian optimisation): [`reports/phase4_optimization.md`](reports/phase4_optimization.md)
- **Phase 5 report** (ablation A3 separating CktGNN's value pathway from its slot-aware decoder; trust-region constrained BO with 20 runs per cell): [`reports/phase5_report.md`](reports/phase5_report.md), protocol [`reports/phase5_protocol.md`](reports/phase5_protocol.md)
- **Phase 6 report** (a property-aware latent space: VAE trained jointly with a stability/performance predictor): [`reports/phase6_report.md`](reports/phase6_report.md), protocol [`reports/phase6_protocol.md`](reports/phase6_protocol.md)
- **Phase 7 report** (confirmatory test of graded-stability SCBO on fresh seeds, and a replication check of the optimisation results): [`reports/phase7_report.md`](reports/phase7_report.md), protocol [`reports/phase7_protocol.md`](reports/phase7_protocol.md)

## Layout
| Path | Purpose |
|---|---|
| `scripts/audit_dataset.py` | Audit of OCB Ckt-Bench-101: splits, topologies, leakage, FoM decomposition, simple baselines |
| `scripts/ocb_sim.py` | Open-source AC (MNA) simulator that re-implements the Spectre/Ocean flow of `OCB/src/simulator/graph_to_fom.py` |
| `scripts/validate_simulator.py` | Checks `ocb_sim.py` against the released labels |
| `scripts/run_vae.py` | Faithful VAE training/evaluation runner reusing the authors' `layers/models_ig.py` |
| `scripts/paths.py` | Shared locations (code here, data on D:) |
| `scripts/setup_data.py` | Copies Ckt-Bench-101 and extracts Ckt-Bench-301 into the data folder |
| `scripts/launch_job.sh` | Launches one training run (`DEVICE=cuda` or `cpu`; logs to the data folder; auto-resumes from `last.pt`) |
| `scripts/fast_models.py` | Batched, numerically equivalent training path for the authors' CktGNN and D-VAE |
| `tests/test_fast_models.py` | Equivalence and speed test: fast path vs the authors' code |
| `tests/test_ablation_models.py` | Consistency tests for A1/A2/A3/P (batch independence, A3 slot wiring) |
| `scripts/circuit_labels.py` | Simulator labels (gain, UGF, PM, pole stability, spec) for all 10,000 real circuits |
| `scripts/finish_run_on_cpu.sh` | Moves the evaluation of an already-running job to the CPU |
| `scripts/sparse_gp_torch.py` | PyTorch port of the authors' Theano sparse GP |
| `scripts/sgp_regression.py` | SGP regression on saved latents (port of the authors' `sgp_regression.py`) |
| `scripts/ablation_models.py` | Ablation variants A1/A2/A3 and the property-aware model P, plus the slot-free op-amp reader |
| `scripts/eval_circuits.py` | Circuit-level evaluation: decode, read as op-amp, simulate, score functionality/phase margin/novelty |
| `scripts/aggregate_results.py`, `scripts/plot_results.py`, `scripts/plot_ablation.py` | Summary tables, figures and significance tests |
| `scripts/export_results.sh` | Copies the small result files from the data folder into `results/` |
| `scripts/latent_bo.py`, `scripts/run_bo_jobs.sh`, `scripts/plot_bo.py` | Phases 4–6: latent-space optimisation (random search, FoM-BO, constrained BO, SCBO) and its analysis |

## Results in this repository
`results/` holds the small result files (audit, simulator validation, per-run configs, training logs, evaluation and regression results, summary tables), copied from the data folder by `scripts/export_results.sh`. Checkpoints and datasets are not included.

## Data folder
Datasets, checkpoints and results live in **`D:\CktGNN_data`**, not in this folder:

```
D:\CktGNN_data\
  datasets\CktBench101\   ckt_bench_101*.pkl, perform101.csv   (copied from the repo)
  datasets\CktBench301\   ckt_bench_301*.pkl, perform301.csv   (extracted from the repo zips)
  results\audit\          dataset audit
  results\simulator\      simulator validation and label diagnostics
  results\vae\<run>\      config, training log, checkpoints, evaluation results
```
To use another location, set the `CKTGNN_DATA` environment variable before running the scripts.

## Environment
Windows 11, Python 3.13, PyTorch 2.13.0+cu130 in `.venv` (other packages come from the system interpreter).

**Training speed.** The authors' code builds each graph node by node in Python, so it is dominated by tiny-tensor overhead. `scripts/fast_models.py` batches the same computation over the 64 graphs of a batch. It uses the same parameters and equations; `tests/test_fast_models.py` checks that losses, latents and gradients match the original to float32 precision. `run_vae.py` uses it by default (`--impl original` keeps the authors' code).

| Seconds per training step (batch 64) | Original, CPU | Original, GPU | Fast, CPU | Fast, GPU |
|---|---|---|---|---|
| CktGNN | 0.71 | 1.36 | 0.32 | **0.13** |
| D-VAE | 2.28 | 3.16 | 0.93 | **0.25** |

Train on the GPU and evaluate on the CPU (the default `--eval-device cpu`): evaluation decoding still uses the authors' per-graph decoder, which is about 15x slower on the GPU. Three GPU runs fit side by side on the RTX 3050 (each uses about 1.4 GB of RAM and 0.7 GB of VRAM); 300 epochs then take about 1.6 h for CktGNN and 3.8 h for D-VAE.

Get the authors' code (not included here) at the commit used in this study:

```bash
git clone https://github.com/zehao-dong/CktGNN external/CktGNN
```
```bash
git -C external/CktGNN checkout 416cd035f79dd8cfcb60ac0a4792b43255adf0b7
```

Create the environment:

```bash
python -m venv --system-site-packages .venv
```
```bash
.venv/Scripts/python.exe -m pip install "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130
```
```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Reproduce the Phase 1 results
Populate the data folder (run once):
```bash
.venv/Scripts/python.exe scripts/setup_data.py
```
```bash
.venv/Scripts/python.exe scripts/audit_dataset.py
```
```bash
.venv/Scripts/python.exe scripts/validate_simulator.py --n 10000
```
Smoke-test training (about 10 s):
```bash
.venv/Scripts/python.exe scripts/run_vae.py --model CktGNN --epochs 1 --max-batches 15 --skip-eval --device cpu --out D:/CktGNN_data/results/vae/smoke_cktgnn
```
Full baseline runs on the GPU. Re-running the same command resumes an interrupted run:
```bash
DEVICE=cuda bash scripts/launch_job.sh CktGNN 1 --epochs 300 --patience 10
```
```bash
DEVICE=cuda bash scripts/launch_job.sh DVAE 1 --epochs 300 --patience 10
```
SGP regression on a finished run:
```bash
.venv/Scripts/python.exe scripts/sgp_regression.py --run D:/CktGNN_data/results/vae/CktGNN_seed1
```
