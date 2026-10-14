"""Collect finished VAE runs into the reproduction comparison table.

Reads <DATA_ROOT>/results/vae/<MODEL>_seed<k>/{results.json, sgp_results.csv} and writes
<DATA_ROOT>/results/summary/reproduction_table.{csv,md}: mean +- std over training seeds next to
the paper (Table 1/2, older dataset) and the authors' released-data results (arXiv v2 App. G).

SGP metrics are first averaged over the SGP folds of each run, then mean +- std is taken over
training seeds, so the reported spread reflects retraining (the paper's +- reflects SGP folds only).

Usage:  .venv/Scripts/python.exe scripts/aggregate_results.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS  # noqa: E402

PAPER = {   # Table 1 and Table 2 of the ICLR paper (older dataset)
    'CktGNN': {'gain_rmse': .607, 'gain_r': .791, 'bw_rmse': .873, 'bw_r': .479, 'pm_rmse': .973, 'pm_r': .217,
               'fom_rmse': .854, 'fom_r': .491, 'recon_acc': .397, 'valid_dag': .9892, 'valid_ckt': .9892, 'novel': .9229},
    'DVAE': {'gain_rmse': .681, 'gain_r': .739, 'bw_rmse': .914, 'bw_r': .394, 'pm_rmse': .956, 'pm_r': .301,
             'fom_rmse': .897, 'fom_r': .374, 'recon_acc': .271, 'valid_dag': .8212, 'valid_ckt': .7393, 'novel': .9715},
}
APP_G = {   # arXiv v2 Appendix G, Tables 3-4 (released Ckt-Bench-101)
    'CktGNN': {'gain_rmse': .682, 'gain_r': .723, 'bw_rmse': .596, 'bw_r': .806, 'pm_rmse': .993, 'pm_r': .288,
               'fom_rmse': .597, 'fom_r': .806, 'recon_acc': .453, 'valid_dag': .9310, 'valid_ckt': .9192, 'novel': .9634},
    'DVAE': {'gain_rmse': .467, 'gain_r': .881, 'bw_rmse': .756, 'bw_r': .667, 'pm_rmse': .999, 'pm_r': .265,
             'fom_rmse': .755, 'fom_r': .668, 'recon_acc': .149, 'valid_dag': .8975, 'valid_ckt': .3953, 'novel': .9678},
}
METRICS = ['gain_rmse', 'gain_r', 'bw_rmse', 'bw_r', 'pm_rmse', 'pm_r', 'fom_rmse', 'fom_r',
           'recon_acc', 'valid_dag', 'valid_ckt', 'novel']


def main():
    rows = []
    for run in sorted(glob.glob(os.path.join(RESULTS, 'vae', '*_seed*'))):
        m = re.match(r'(.+)_seed(\d+)$', os.path.basename(run))
        if not m:
            continue
        row = {'model': m.group(1), 'seed': int(m.group(2))}
        rj = os.path.join(run, 'results.json')
        if os.path.exists(rj):
            row.update({k: v for k, v in json.load(open(rj)).items() if k in METRICS})
        sc = os.path.join(run, 'sgp_results.csv')
        if os.path.exists(sc):
            s = pd.read_csv(sc).groupby('target')[['rmse', 'pearson']].mean()
            for t in s.index:
                row[f'{t}_rmse'], row[f'{t}_r'] = s.loc[t, 'rmse'], s.loc[t, 'pearson']
            row['sgp_folds'] = int(pd.read_csv(sc)['fold'].nunique())
        rows.append(row)
    if not rows:
        print('no finished runs yet'); return
    runs = pd.DataFrame(rows)
    out = os.path.join(RESULTS, 'summary')
    os.makedirs(out, exist_ok=True)
    runs.to_csv(os.path.join(out, 'per_run.csv'), index=False)

    table = []
    for model, g in runs.groupby('model'):
        for k in METRICS:
            vals = g[k].dropna().values if k in g else np.array([])
            mean = vals.mean() if len(vals) else np.nan
            table.append({'model': model, 'metric': k, 'paper': PAPER.get(model, {}).get(k, np.nan),
                          'authors_released_data': APP_G.get(model, {}).get(k, np.nan),
                          'reproduced_mean': mean, 'reproduced_std': vals.std(ddof=1) if len(vals) > 1 else np.nan,
                          'n_seeds': len(vals),
                          'diff_vs_released': mean - APP_G.get(model, {}).get(k, np.nan)})
    t = pd.DataFrame(table)
    t.to_csv(os.path.join(out, 'reproduction_table.csv'), index=False)
    with open(os.path.join(out, 'reproduction_table.md'), 'w', encoding='utf-8') as f:
        f.write('| Model | Metric | Paper | Authors (released data) | Reproduced mean ± std | Seeds | Δ vs released |\n|---|---|---:|---:|---:|---:|---:|\n')
        for _, r in t.iterrows():
            rep = '–' if np.isnan(r.reproduced_mean) else f'{r.reproduced_mean:.3f}' + ('' if np.isnan(r.reproduced_std) else f' ± {r.reproduced_std:.3f}')
            d = '–' if np.isnan(r.diff_vs_released) else f'{r.diff_vs_released:+.3f}'
            f.write(f'| {r.model} | {r.metric} | {r.paper:.3f} | {r.authors_released_data:.3f} | {rep} | {r.n_seeds} | {d} |\n')
    print(open(os.path.join(out, 'reproduction_table.md'), encoding='utf-8').read())


if __name__ == '__main__':
    main()
