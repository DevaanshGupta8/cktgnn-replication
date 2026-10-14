"""Stability-aware circuit optimisation in a learned latent space (Phase 4).

Design problem (fixed before any optimisation was run):
  maximise   UGF (unity-gain frequency)
  subject to functional (0 dB crossing), open-loop stable, closed-loop stable as a unity-gain buffer
             (pole check, scripts/ocb_sim.stability), phase margin >= 60 deg, DC gain >= 60 dB.
Only 3.0% of the 10,000 OCB circuits meet this specification.

Each evaluation decodes a latent point deterministically (argmax decoding) with a trained VAE (CktGNN or
A2 = SubgDVAEpos), reads it as an OCB op-amp (own slots if consistent, else the first structural reading),
clips device values to the dataset range and simulates it. Methods (same budget, same 20 initial points):
  rs      random search: z ~ N(mu_train, sigma_train)
  fom_bo  the paper's objective: GP + expected improvement on the OCB FoM, no constraints
  cbo     constrained BO (Gardner et al., ICML 2014): GP on log10(UGF) for functional points, GP on a
          constraint score; acquisition = EI x P(feasible); maximise P(feasible) until a feasible point exists.
  scbo    trust-region constrained BO (SCBO, Eriksson & Poloczek, AISTATS 2021; Phase 5): same two GPs but with
          ARD kernels, constrained Thompson sampling inside a trust region centred on the best point (best
          feasible, else least infeasible). Settings fixed before running, see SCBO_* below.

Usage:  .venv/Scripts/python.exe scripts/latent_bo.py --model CktGNN --method cbo --seed 1
Writes: <DATA_ROOT>/results/bo/<model>_<method>_seed<k>.csv (one row per evaluation)
"""
import argparse
import math
import os
import pickle
import sys
import time
import types
import warnings
import zlib

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO, BENCH101, RESULTS  # noqa: E402
sys.path.insert(0, REPO)
_tv = types.ModuleType('torchvision'); _tv.datasets = _tv.transforms = None
sys.modules.setdefault('torchvision', _tv)
from ablation_models import decompositions  # noqa: E402
from eval_circuits import build_model, own_main, dag_key  # noqa: E402
from ocb_sim import analyze  # noqa: E402

PM_MIN, GAIN_MIN = 60.0, 60.0
# SCBO settings (fixed before running). Domain: standardised latent units clipped to [-BOX, BOX], mapped to [0, 1]^d.
# Length limits, perturbation probability and candidate rule follow the SCBO/TuRBO papers. The failure tolerance is
# budget-adapted: the default max(4, d) = 66 would never shrink the region within 180 BO steps.
SCBO_BOX = 3.0
SCBO_L_INIT, SCBO_L_MIN, SCBO_L_MAX = 0.8, 0.5 ** 7, 1.6
SCBO_SUCC_TOL, SCBO_FAIL_TOL = 3, 10
SCBO_N_CAND = 2000          # exact joint Thompson samples need an n_cand x n_cand posterior covariance
warnings.filterwarnings('ignore')


def evaluate(g):
    """Simulate a decoded graph. Returns a dict with feasibility, objective and diagnostics."""
    r = {'decomposable': False, 'functional': False, 'feasible': False, 'ugf_ghz': np.nan, 'gain_db': np.nan,
         'pm_lag': np.nan, 'ol_stable': False, 'cl_stable': False, 'fom': 0.0}
    dec = decompositions(g)
    if not dec:
        return r
    om = own_main(g)
    main = om if om is not None and om in dec else dec[0]
    h = g.copy()
    n = h.vcount()
    pos = {0: 0, n - 1: 1}
    for k, v in enumerate(main):
        pos[v] = 2 + k
    extra = 5
    for v in range(n):
        if v not in pos:
            pos[v] = extra
            extra = min(extra + 1, 7)
    h.vs['pos'] = [pos[v] for v in range(n)]
    for a in ('r', 'c', 'gm'):
        h.vs[a] = [float(np.clip(round(float(x)), 1, 101)) for x in h.vs[a]]
    r['decomposable'] = True
    try:
        m = analyze(h)
    except (np.linalg.LinAlgError, AssertionError, KeyError, ValueError):
        return r
    r.update(gain_db=float(m['gain_db']), ol_stable=bool(m['ol_stable']), cl_stable=bool(m['cl_stable']))
    if np.isfinite(m['ugf_hz']) and np.isfinite(m['gain_db']):
        r.update(functional=True, ugf_ghz=float(m['ugf_hz']) / 1e9, pm_lag=float(m['pm_lag']))
        # OCB FoM as in graph_to_fom.py, with Ocean's phase output = wrapped phase at the UGF (Phase 1 finding)
        r['fom'] = 1.2 * abs(m['gain_db']) / 100 + 1.6 * m['phase_ugf_wrapped'] / (-90) + 10 * r['ugf_ghz']
        r['feasible'] = bool(r['ol_stable'] and r['cl_stable'] and r['pm_lag'] >= PM_MIN and r['gain_db'] >= GAIN_MIN)
    r['_graph'] = h
    return r


def constraint_score(r):
    """Continuous score, >= 0 iff feasible (used by the constraint GP)."""
    if not (r['functional'] and r['ol_stable'] and r['cl_stable']):
        return -60.0
    return float(min(r['pm_lag'] - PM_MIN, r['gain_db'] - GAIN_MIN, 60.0))


def fit_gp(X, y):
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5) \
        + WhiteKernel(1e-3, (1e-6, 1e0))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(X, y)


def fit_gp_ard(X, y, d, warm=None):
    """ARD Matern-5/2 GP; `warm` (the previous step's fitted kernel) is used as the optimiser's starting point."""
    k = warm if warm is not None else ConstantKernel(1.0, (1e-3, 1e3)) *         Matern(length_scale=np.full(d, 0.5), length_scale_bounds=(5e-3, 4.0), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0, random_state=0).fit(X, y)


def thompson(gp, C, rng):
    """One joint posterior sample at the rows of C (Cholesky; sklearn's sample_y uses an SVD, ~40x slower)."""
    m, cov = gp.predict(C, return_cov=True)
    jitter = 1e-8 * max(np.mean(np.diag(cov)), 1e-12)
    for _ in range(5):
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(len(C)))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    return m + L @ rng.standard_normal(len(C))


class SCBOState:
    """Trust-region bookkeeping of SCBO (success/failure counters, length doubling/halving, restarts)."""

    def __init__(self):
        self.length, self.n_succ, self.n_fail, self.restarts = SCBO_L_INIT, 0, 0, 0
        self.best_key = None                       # (feasible, objective or -violation), compared lexicographically
        self.kc = self.ko = None                   # fitted kernels, warm starts for the next step

    def update(self, key):
        improved = self.best_key is None or key[0] > self.best_key[0] or             (key[0] == self.best_key[0] and key[1] > self.best_key[1] + 1e-3 * abs(self.best_key[1]))
        if improved:
            self.best_key, self.n_succ, self.n_fail = key, self.n_succ + 1, 0
        else:
            self.n_succ, self.n_fail = 0, self.n_fail + 1
        if self.n_succ == SCBO_SUCC_TOL:
            self.length, self.n_succ = min(2 * self.length, SCBO_L_MAX), 0
        elif self.n_fail == SCBO_FAIL_TOL:
            self.length, self.n_fail = self.length / 2, 0
        if self.length < SCBO_L_MIN:               # collapse: restart the region (centre stays the global best)
            self.length, self.n_succ, self.n_fail, self.restarts = SCBO_L_INIT, 0, 0, self.restarts + 1


def scbo_key(r):
    """Ranking used for the trust-region centre and success test: feasible beats infeasible; feasible points by
    log10(UGF), infeasible ones by the constraint score (higher = less violated)."""
    return (1, float(np.log10(r['ugf_ghz']))) if r['feasible'] else (0, constraint_score(r))


def scbo_propose(Xc, rows, state, rng, d):
    """One SCBO step in the unit cube. Returns the next point (unit cube) and the ARD length scales used."""
    keys = [scbo_key(r) for r in rows]
    centre = Xc[max(range(len(keys)), key=lambda i: keys[i])]
    c = np.array([constraint_score(r) for r in rows])
    gpc = fit_gp_ard(Xc, c, d, state.kc)
    state.kc = gpc.kernel_
    ls = np.asarray(gpc.kernel_.k1.k2.length_scale, dtype=float)
    func = np.array([r['functional'] for r in rows])
    gpo = None
    if func.sum() >= 3:
        lu = np.log10(np.array([r['ugf_ghz'] for r in rows])[func])
        gpo = fit_gp_ard(Xc[func], lu, d, state.ko)
        state.ko = gpo.kernel_
        ls = np.sqrt(ls * np.asarray(gpo.kernel_.k1.k2.length_scale, dtype=float))   # both models shape the region
    w = ls / ls.mean()
    w = w / np.prod(w) ** (1.0 / d)
    lb = np.clip(centre - w * state.length / 2, 0.0, 1.0)
    ub = np.clip(centre + w * state.length / 2, 0.0, 1.0)
    pert = lb + (ub - lb) * rng.random((SCBO_N_CAND, d))
    mask = rng.random((SCBO_N_CAND, d)) <= min(20.0 / d, 1.0)
    mask[np.arange(SCBO_N_CAND), rng.integers(d, size=SCBO_N_CAND)] = True
    cand = np.where(mask, pert, centre)
    cs = thompson(gpc, cand, rng)
    feas = cs >= 0
    if feas.any() and gpo is not None:
        fs = thompson(gpo, cand[feas], rng)
        return cand[feas][int(np.argmax(fs))]
    if feas.any():
        return cand[feas][rng.integers(feas.sum())]
    return cand[int(np.argmax(cs))]


def ei(mu, sd, best):
    sd = np.maximum(sd, 1e-9)
    u = (mu - best) / sd
    return sd * (u * norm.cdf(u) + norm.pdf(u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['CktGNN', 'SubgDVAEpos', 'SubgDVAEslot', 'SubgDVAEprop'], required=True)
    ap.add_argument('--method', choices=['rs', 'fom_bo', 'cbo', 'scbo'], required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--budget', type=int, default=200)
    ap.add_argument('--n-init', type=int, default=20)
    ap.add_argument('--n-cand', type=int, default=3000)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(RESULTS, 'bo'), help='folder for the trace CSV')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    vae_seed = (args.seed - 1) % 3 + 1                      # BO runs cycle over the 3 trained VAEs
    run = os.path.join(RESULTS, 'vae', f'{args.model}_seed{vae_seed}')
    model = build_model(args.model)
    model.load_state_dict(torch.load(os.path.join(run, 'last.pt'), map_location='cpu', weights_only=False)['model'])
    model.eval()
    Z = np.load(os.path.join(run, 'latents.npz'))['Z_train']
    mu_z, sd_z = Z.mean(0), Z.std(0)
    with open(os.path.join(BENCH101, 'ckt_bench_101.pkl'), 'rb') as f:
        train_keys = {dag_key(x[0]) for x in pickle.load(f)[0]}

    rng = np.random.default_rng(1000 + args.seed)            # same initial design for every method in a run
    init = mu_z + sd_z * rng.standard_normal((args.n_init, len(mu_z)))
    # method stream: a stable hash (Phase 4 used Python's hash(), which is salted per process, so its runs
    # could not be re-run bit-for-bit)
    rng_m = np.random.default_rng(zlib.crc32(f'{args.method}-{args.seed}'.encode()))
    to_u = lambda z: (z - mu_z) / sd_z                          # GP works in standardised latent units

    X, rows, tr_len = [], [], []
    t0 = time.time()
    to_c = lambda u: u / (2 * SCBO_BOX) + 0.5                  # SCBO's unit cube (clipped box +-SCBO_BOX)
    state = SCBOState() if args.method == 'scbo' else None

    def run_eval(z):
        with torch.no_grad():
            g = model.decode(torch.tensor(z[None], dtype=torch.float32), stochastic=False)[0]
        r = evaluate(g)
        h = r.pop('_graph', None)
        r['novel'] = bool(r['functional'] and dag_key(g) not in train_keys)
        r['circuit'] = repr((tuple(h.vs['type']), tuple(h.vs['pos']), tuple(h.vs['r']), tuple(h.vs['c']),
                             tuple(h.vs['gm']), tuple(h.get_edgelist()))) if h is not None else ''
        X.append(to_u(z)); rows.append(r)

    for z in init:
        run_eval(z)
        if state is not None:
            state.update(scbo_key(rows[-1]))
        tr_len.append(np.nan)
    if state is not None:                          # the initial design only sets the centre, not the region size
        state.length, state.n_succ, state.n_fail = SCBO_L_INIT, 0, 0
    while len(rows) < args.budget:
        if args.method == 'scbo':
            tr_len.append(state.length)
            xc = scbo_propose(np.clip(to_c(np.array(X)), 0, 1), rows, state, rng_m, len(mu_z))
            run_eval(mu_z + sd_z * (xc - 0.5) * 2 * SCBO_BOX)
            state.update(scbo_key(rows[-1]))
            continue
        tr_len.append(np.nan)
        if args.method == 'rs':
            z = mu_z + sd_z * rng_m.standard_normal(len(mu_z))
        else:
            Xa = np.array(X)
            cand = rng_m.standard_normal((args.n_cand, len(mu_z)))           # prior samples (standardised)
            if args.method == 'fom_bo':
                y = np.array([r['fom'] for r in rows])
                top = Xa[np.argsort(-y)[:5]]
            else:
                func = np.array([r['functional'] for r in rows])
                feas = np.array([r['feasible'] for r in rows])
                lu = np.log10(np.array([r['ugf_ghz'] for r in rows]))
                top = Xa[feas][np.argsort(-lu[feas])[:5]] if feas.any() else Xa[np.argsort(-np.array([constraint_score(r) for r in rows]))[:5]]
            local = top[rng_m.integers(len(top), size=args.n_cand // 2)] + 0.3 * rng_m.standard_normal((args.n_cand // 2, len(mu_z)))
            cand = np.vstack([cand, local])
            if args.method == 'fom_bo':
                gp = fit_gp(Xa, y)
                m, s = gp.predict(cand, return_std=True)
                acq = ei(m, s, y.max())
            else:
                c = np.array([constraint_score(r) for r in rows])
                gpc = fit_gp(Xa, c)
                mc, sc = gpc.predict(cand, return_std=True)
                pf = norm.cdf(mc / np.maximum(sc, 1e-9))
                if feas.any() and func.sum() >= 3:
                    gpo = fit_gp(Xa[func], lu[func])
                    mo, so = gpo.predict(cand, return_std=True)
                    acq = ei(mo, so, lu[feas].max()) * pf
                else:
                    acq = pf
            z = mu_z + sd_z * cand[int(np.argmax(acq))]
        run_eval(z)

    df = pd.DataFrame(rows)
    df.insert(0, 'eval', np.arange(1, len(df) + 1))
    df['tr_length'] = tr_len
    best = np.where(df.feasible, df.ugf_ghz, -np.inf)
    df['best_feasible_ugf'] = np.maximum.accumulate(best)
    df['n_feasible_unique'] = [len(set(df.circuit[:i + 1][df.feasible[:i + 1]])) for i in range(len(df))]
    out = args.out
    os.makedirs(out, exist_ok=True)
    df.assign(model=args.model, method=args.method, seed=args.seed, vae_seed=vae_seed) \
      .to_csv(os.path.join(out, f'{args.model}_{args.method}_seed{args.seed}.csv'), index=False)
    fb = df.best_feasible_ugf.iloc[-1]
    extra = f', {state.restarts} TR restarts, final length {state.length:.3f}' if state is not None else ''
    print(f'{args.model} {args.method} seed {args.seed}: best feasible UGF {fb:.3f} GHz, '
          f'{df.feasible.mean():.3f} of evaluations feasible, {time.time() - t0:.0f}s{extra}')


if __name__ == '__main__':
    main()
