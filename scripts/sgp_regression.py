"""SGP regression on saved VAE latents (port of external/CktGNN/sgp_regression.py).

Protocol kept from the original: targets are negated, standardised with training statistics,
M = 500 inducing points, Adam lr 5e-4, 100 epochs, minibatch 1000, one SGP per target,
folds = SGP seeds 1..9 on the same latents. Reads latents.npz written by run_vae.py.

Usage:  .venv/Scripts/python.exe scripts/sgp_regression.py --run D:/CktGNN_data/results/vae/CktGNN_seed1
Writes: <run>/sgp_results.csv and prints the mean +- std over folds.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as sps
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_gp_torch import SparseGP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, help='run folder containing latents.npz')
ap.add_argument('--folds', type=int, default=9, help='original script uses seeds 1..9')
ap.add_argument('--M', type=int, default=500)
ap.add_argument('--max-iter', type=int, default=100)
ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--threads', type=int, default=None)
args = ap.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

d = np.load(os.path.join(args.run, 'latents.npz'))
X_train, X_test = d['Z_train'].astype(np.float64), d['Z_test'].astype(np.float64)
names = [str(s) for s in d['label_names']]
rows = []
for fold in range(1, args.folds + 1):
    np.random.seed(fold); torch.manual_seed(fold)
    for j, name in enumerate(names):
        ytr, yte = -d['Y_train'][:, j:j + 1], -d['Y_test'][:, j:j + 1]
        mu, sd = ytr.mean(), ytr.std()
        ytr, yte = (ytr - mu) / sd, (yte - mu) / sd
        gp = SparseGP(args.M, X_train.shape[0]).train_via_adam(
            X_train, ytr, max_iterations=args.max_iter, minibatch_size=2 * args.M, learning_rate=args.lr)
        pred, var = gp.predict(X_test)
        rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
        r = float(sps.pearsonr(pred.ravel(), yte.ravel())[0])
        ll = float(np.mean(sps.norm.logpdf(pred - yte, scale=np.sqrt(var))))
        rows.append({'fold': fold, 'target': name, 'rmse': rmse, 'pearson': r, 'test_ll': ll})
        print(f'fold {fold} {name:4s} rmse {rmse:.4f} pearson {r:.4f}', flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.run, 'sgp_results.csv'), index=False)
print(df.groupby('target')[['rmse', 'pearson']].agg(['mean', 'std']).round(4))
