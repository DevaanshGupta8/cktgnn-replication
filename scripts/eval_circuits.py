"""Circuit-level evaluation of generated circuits (subgraph-DAG models: CktGNN, SubgDVAE, SubgDVAEpos, SubgDVAEslot).

The authors' "valid circuit" metric is a structural check. Here every decoded graph is read as an OCB
op-amp (scripts/ablation_models.decompositions), turned into a netlist and simulated with
scripts/ocb_sim.py:
  valid_dag      authors' is_valid_DAG
  decomposable   at least one reading as an OCB op-amp (2-3 forward gm stages + single-input/output extras)
  ambiguous      more than one reading (the slot-free graph does not determine the circuit)
  slot_consistent  (slot models) the model's own slots form one of the valid readings
  values_ok      every device value present in the subgraph lies within the dataset range (1..101)
  functional     simulates and has a unity-gain crossing (Ocean would report a UGF)
  pm45           functional and lag-based phase margin >= 45 deg (NOT a stability test; kept for reference)
  stable         functional, open-loop stable and closed-loop stable as a unity-gain buffer (pole check)
  meets_spec     stable, phase margin >= 60 deg and DC gain >= 60 dB (the Phase 4 specification)
  novel          functional and topology (type + edges, as in is_same_DAG) not in the training set
Sampling follows the authors' prior-validity protocol: 1000 latent points from N(mu_train, sigma_train),
each decoded 10 times. `--reference` scores the real test/train circuits with the same pipeline.

Usage:
  .venv/Scripts/python.exe scripts/eval_circuits.py --run D:/CktGNN_data/results/vae/CktGNN_seed1 --model CktGNN
  .venv/Scripts/python.exe scripts/eval_circuits.py --reference
"""
import argparse
import json
import os
import pickle
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO, BENCH101, RESULTS  # noqa: E402
sys.path.insert(0, REPO)
_tv = types.ModuleType('torchvision'); _tv.datasets = _tv.transforms = None
sys.modules.setdefault('torchvision', _tv)
from utils import is_valid_DAG, SUBG_INDI  # noqa: E402
from ablation_models import SubgDVAE, decompositions  # noqa: E402
from fast_models import FastCktGNN  # noqa: E402
from ocb_sim import analyze  # noqa: E402


def dag_key(g):
    return (g.vcount(),) + tuple((int(g.vs[i]['type']), frozenset(g.neighbors(i, 'in'))) for i in range(g.vcount()))


def build_model(name):
    if name == 'CktGNN':
        return FastCktGNN(max_n=8, max_pos=8, nvt=26, subn_nvt=40, START_TYPE=0, END_TYPE=1,
                          emb_dim=24, feat_emb_dim=8, hs=301, nz=66)
    return SubgDVAE(max_n=8, nvt=26, START_TYPE=0, END_TYPE=1, hs=301, nz=66, use_pos=name != 'SubgDVAE',
                    slot_struct=name == 'SubgDVAEslot', n_props=7 if name.startswith('SubgDVAEprop') else 0,
                    norm_values=name == 'SubgDVAEpropn')


def own_main(g):
    """Main chain according to the graph's own 'pos' labels (slots 2, 3, optional 4, each exactly once)."""
    if 'pos' not in g.vs.attributes():
        return None
    by = {}
    for v in g.vs:
        by.setdefault(int(v['pos']), []).append(v.index)
    if any(len(by.get(p, [])) != 1 for p in (2, 3)) or len(by.get(4, [])) > 1:
        return None
    return [by[p][0] for p in (2, 3, 4) if p in by]


def simulate(g, main):
    """Simulate g read with the given main chain; device values are clipped to the dataset range."""
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
    try:
        return analyze(h)
    except (np.linalg.LinAlgError, AssertionError, KeyError, ValueError):
        return None


def values_ok(g):
    names = ('r', 'c', 'gm')
    for v in g.vs:
        for i in SUBG_INDI.get(int(v['type']), []):
            x = float(v[names[i]])
            if not (0.5 <= x <= 101.5):
                return False
    return True


def score(G, train_keys, slot_model=False, true_slots=False, return_rows=False):
    rows = []
    for g in G:
        r = {'valid_dag': bool(is_valid_DAG(g, subg=True))}
        dec = decompositions(g) if r['valid_dag'] else []
        r['decomposable'] = len(dec) > 0
        r['ambiguous'] = len(dec) > 1
        main = None
        if true_slots or slot_model:
            om = own_main(g)
            r['slot_consistent'] = om is not None and om in dec
            main = om if r['slot_consistent'] else None
        if main is None and dec:
            main = dec[0]
        r['values_ok'] = values_ok(g)
        m = simulate(g, main) if main is not None else None
        r['simulates'] = m is not None
        r['functional'] = bool(m is not None and np.isfinite(m['gain_db']) and np.isfinite(m['ugf_hz']))
        if r['functional']:
            r.update(gain_db=float(m['gain_db']), ugf_ghz=float(m['ugf_hz']) / 1e9, pm_lag=float(m['pm_lag']))
        r['pm45'] = r['functional'] and r.get('pm_lag', -1) >= 45          # kept for comparison; not a stability test
        r['stable'] = bool(r['functional'] and m['ol_stable'] and m['cl_stable'])
        r['meets_spec'] = bool(r['stable'] and r['pm_lag'] >= 60 and r['gain_db'] >= 60)
        r['novel'] = r['functional'] and dag_key(g) not in train_keys
        rows.append(r)
    out = {k: float(np.mean([r[k] for r in rows])) for k in
           ('valid_dag', 'decomposable', 'ambiguous', 'values_ok', 'simulates', 'functional', 'pm45', 'stable', 'meets_spec', 'novel')}
    if 'slot_consistent' in rows[0]:
        out['slot_consistent'] = float(np.mean([r['slot_consistent'] for r in rows]))
    func = [r for r in rows if r['functional']]
    out['n'] = len(rows)
    out['unique_functional_topologies'] = len({dag_key(g) for g, r in zip(G, rows) if r['functional']})
    out['unique_spec_topologies'] = len({dag_key(g) for g, r in zip(G, rows) if r['meets_spec']})
    if func:
        out['median_gain_db'] = float(np.median([r['gain_db'] for r in func]))
        out['median_ugf_ghz'] = float(np.median([r['ugf_ghz'] for r in func]))
    return (out, rows) if return_rows else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run')
    ap.add_argument('--model', choices=['CktGNN', 'SubgDVAE', 'SubgDVAEpos', 'SubgDVAEslot', 'SubgDVAEprop', 'SubgDVAEpropn'])
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--n-latent', type=int, default=1000)
    ap.add_argument('--decode-times', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    with open(os.path.join(BENCH101, 'ckt_bench_101.pkl'), 'rb') as f:
        train, test = pickle.load(f)
    train_g = [x[0] for x in train]
    train_keys = {dag_key(g) for g in train_g}

    if args.reference:
        res = {'test': score([x[0] for x in test], train_keys, true_slots=True),
               'train': score(train_g, set(), true_slots=True)}
        out = os.path.join(RESULTS, 'summary', 'circuit_eval_reference.json')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, 'w'), indent=2)
        print(json.dumps(res, indent=2))
        return

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build_model(args.model)
    st = torch.load(os.path.join(args.run, 'last.pt'), map_location='cpu', weights_only=False)
    model.load_state_dict(st['model'])
    model.eval()
    Z = np.load(os.path.join(args.run, 'latents.npz'))['Z_train']
    mean, std = torch.tensor(Z.mean(0)), torch.tensor(Z.std(0))
    G = []
    with torch.no_grad():
        for start in range(0, args.n_latent, 128):
            z = torch.randn(min(128, args.n_latent - start), model.nz) * std + mean
            for _ in range(args.decode_times):
                G.extend(model.decode(z))
    res = score(G, train_keys, slot_model=args.model != 'SubgDVAE')
    res.update(model=args.model, run=os.path.basename(args.run), epoch=st['epoch'])
    json.dump(res, open(os.path.join(args.run, 'circuit_eval.json'), 'w'), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
