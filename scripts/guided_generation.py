"""Phase 9 Part A: predictor-guided generation (reports/phase9_protocol.md).

For each trained run: draw a pool of latents z ~ N(mu_train, sigma_train), score them with a "meets spec" predictor,
and compare decoding the top fraction of the pool (guided) with decoding a random subset of the same pool (control).
Predictor: P's jointly trained head (models SubgDVAEprop*), or for other models a logistic regression fitted on the
training latents with the simulator's "meets spec" labels.

Usage:  .venv/Scripts/python.exe scripts/guided_generation.py --model SubgDVAEprop --seed 1
Writes: <DATA_ROOT>/results/vae/<model>_seed<k>/guided_generation.json and guided_latents.csv (per-latent outcomes)
"""
import argparse
import json
import os
import pickle
import sys
import types

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO, BENCH101, RESULTS  # noqa: E402
sys.path.insert(0, REPO)
_tv = types.ModuleType('torchvision'); _tv.datasets = _tv.transforms = None
sys.modules.setdefault('torchvision', _tv)
from eval_circuits import build_model, score, dag_key  # noqa: E402

POOL, KEEP, DECODES = 20000, 1000, 10
DOSE = [1.0, 0.5, 0.2, 0.05, 0.01]           # top fractions of the pool for the dose-response curve


def decode_score(model, Z, train_keys, slot_model, rng_seed):
    """Decode every latent DECODES times; return the aggregate and the per-latent meets-spec rate."""
    torch.manual_seed(rng_seed)
    G = []
    with torch.no_grad():
        for s in range(0, len(Z), 128):
            z = torch.tensor(Z[s:s + 128], dtype=torch.float32)
            for _ in range(DECODES):
                G.extend((i, g) for i, g in zip(range(s, s + len(z)), model.decode(z)))
    out, rows = score([g for _, g in G], train_keys, slot_model=slot_model, return_rows=True)
    lat = pd.DataFrame({'latent': [i for i, _ in G], 'meets_spec': [r['meets_spec'] for r in rows],
                        'stable': [r['stable'] for r in rows]}).groupby('latent').mean()
    return out, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    run = os.path.join(RESULTS, 'vae', f'{args.model}_seed{args.seed}')
    model = build_model(args.model)
    model.load_state_dict(torch.load(os.path.join(run, 'last.pt'), map_location='cpu', weights_only=False)['model'])
    model.eval()
    lat = np.load(os.path.join(run, 'latents.npz'))
    Ztr = lat['Z_train']
    with open(os.path.join(BENCH101, 'ckt_bench_101.pkl'), 'rb') as f:
        train_keys = {dag_key(x[0]) for x in pickle.load(f)[0]}

    rng = np.random.default_rng(9000 + args.seed)
    pool = Ztr.mean(0) + Ztr.std(0) * rng.standard_normal((POOL, Ztr.shape[1]))
    if args.model.startswith('SubgDVAEprop'):
        predictor = 'jointly trained head'
        with torch.no_grad():
            s = model.prop(torch.tensor(pool, dtype=torch.float32))[:, 3].numpy()
    else:
        predictor = 'logistic regression on training latents'
        y = pd.read_csv(os.path.join(RESULTS, 'simulator', 'analysis_all.csv'))['feasible'].values[:len(Ztr)]
        sc = StandardScaler().fit(Ztr)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Ztr), y)
        s = clf.decision_function(sc.transform(pool))
    order = np.argsort(-s)
    slot_model = args.model != 'SubgDVAE'

    res = {'model': args.model, 'seed': args.seed, 'predictor': predictor, 'pool': POOL, 'keep': KEEP,
           'decodes': DECODES}
    guided_idx = order[:KEEP]
    control_idx = rng.choice(POOL, KEEP, replace=False)
    per_latent = []
    for arm, idx in (('guided', guided_idx), ('control', control_idx)):
        out, lat_df = decode_score(model, pool[idx], train_keys, slot_model, rng_seed=args.seed)
        res[arm] = out
        lat_df['arm'] = arm
        per_latent.append(lat_df)
        print(f"{args.model} seed {args.seed} {arm:8s} meets_spec {out['meets_spec']:.4f} stable {out['stable']:.4f} "
              f"functional {out['functional']:.3f} novel {out['novel']:.3f} unique_spec {out['unique_spec_topologies']}",
              flush=True)
    res['dose_response'] = {}
    for q in DOSE:
        top = order[:int(q * POOL)]
        idx = top if len(top) <= KEEP else rng.choice(top, KEEP, replace=False)
        out, _ = decode_score(model, pool[idx], train_keys, slot_model, rng_seed=args.seed)
        res['dose_response'][str(q)] = out
        print(f"  top {q:>5.0%}: meets_spec {out['meets_spec']:.4f} stable {out['stable']:.4f} "
              f"unique_spec {out['unique_spec_topologies']}", flush=True)
    json.dump(res, open(os.path.join(run, 'guided_generation.json'), 'w'), indent=2)
    pd.concat(per_latent).to_csv(os.path.join(run, 'guided_latents.csv'))


if __name__ == '__main__':
    main()
