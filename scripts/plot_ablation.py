"""Figures and tests for the CktGNN decomposition (ablation) study.

Variants, in decomposition order:
  DVAE        D-VAE on the device DAG                         (baseline)
  SubgDVAE    A1: D-VAE on the subgraph DAG, no slots          -> effect of the representation
  SubgDVAEpos A2: A1 + slot prior                              -> effect of the slot prior
  SubgDVAEslot A3: A2 + slots in messages and edge scorer      -> effect of slot-aware structure decoding
  CktGNN      A3 with the slot-indexed device-value pathway    -> effect of the value pathway
Reads per_run.csv (aggregate_results.py) and circuit_eval.json files; writes
  reports/figures/ablation_sgp.png, ablation_decoder.png, ablation_circuits.png
  <DATA_ROOT>/results/summary/ablation_table.{csv,md}, ablation_tests.csv

Usage:  .venv/Scripts/python.exe scripts/plot_ablation.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RESULTS  # noqa: E402
from plot_results import style, SURFACE, INK, INK2, MUTED  # noqa: E402

FIG = os.path.join(ROOT, 'reports', 'figures')
ORDER = ['DVAE', 'SubgDVAE', 'SubgDVAEpos', 'SubgDVAEslot', 'CktGNN']
LABEL = {'DVAE': 'D-VAE (device DAG)', 'SubgDVAE': 'A1: D-VAE on subgraph DAG', 'SubgDVAEpos': 'A2: A1 + slots',
         'SubgDVAEslot': 'A3: A2 + slots in edges/messages',
         'CktGNN': 'CktGNN (slots + value vector)'}
# reference palette categorical slots, fixed per entity (CktGNN/D-VAE keep the colours of the Phase 2 figures)
COLOR = {'CktGNN': '#2a78d6', 'DVAE': '#eb6834', 'SubgDVAE': '#1baf7a', 'SubgDVAEpos': '#eda100', 'SubgDVAEslot': '#e87ba4'}
CIRC_RAW = ['valid_dag', 'decomposable', 'slot_consistent', 'values_ok', 'functional', 'stable', 'meets_spec', 'novel']
CIRC = ['c_' + k for k in CIRC_RAW]   # prefixed: 'valid_dag'/'novel' also exist as the authors' decoder metrics
CIRC_NAMES = ['Valid DAG', 'Readable as\nop-amp', 'Own slots\nconsistent', 'Device values\nin range',
              'Functional\n(has UGF)', 'Stable\n(pole check)', 'Meets spec\n(stable, PM≥60°,\ngain≥60 dB)', 'Functional\n& novel']
PAIRS = [('DVAE', 'SubgDVAE', 'representation'), ('SubgDVAE', 'SubgDVAEpos', 'slot prior'),
         ('SubgDVAEpos', 'SubgDVAEslot', 'slot structure'), ('SubgDVAEslot', 'CktGNN', 'value pathway'),
         ('SubgDVAEpos', 'CktGNN', 'slot structure + value pathway'), ('DVAE', 'CktGNN', 'total')]


def bars(ax, runs, metrics, names, models, ylabel):
    w = 0.8 / len(models)
    x = np.arange(len(metrics))
    for i, m in enumerate(models):
        g = runs[runs.model == m]
        pos = x - 0.4 + w * (i + 0.5)
        means = [g[k].mean() if k in g and g[k].notna().any() else np.nan for k in metrics]
        ax.bar(pos, means, w * 0.92, color=COLOR[m], label=LABEL[m], zorder=2)
        for j, k in enumerate(metrics):
            if k in g:
                vals = g[k].dropna().values
                ax.scatter(np.full(len(vals), pos[j]), vals, s=12, color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=4)
    ax.set_xticks(x, names)
    ax.set_ylabel(ylabel)
    style(ax)


def main():
    runs = pd.read_csv(os.path.join(RESULTS, 'summary', 'per_run.csv'))
    circ = []
    for f in glob.glob(os.path.join(RESULTS, 'vae', '*_seed*', 'circuit_eval.json')):
        d = json.load(open(f))
        circ.append({'model': d['model'], 'seed': int(d['run'].split('seed')[1]), **{'c_' + k: d.get(k, np.nan) for k in CIRC_RAW}})
    if circ:
        runs = runs.merge(pd.DataFrame(circ), on=['model', 'seed'], how='left')
    models = [m for m in ORDER if m in set(runs.model)]

    targets, tnames = ['gain', 'bw', 'pm', 'fom'], ['Gain', 'BW', 'PM', 'FoM']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    bars(axes[0], runs, [f'{t}_rmse' for t in targets], tnames, models, 'Test RMSE (lower is better)')
    axes[0].set_title('SGP prediction error from latents'); axes[0].set_ylim(0, 1.1)
    bars(axes[1], runs, [f'{t}_r' for t in targets], tnames, models, "Pearson's r (higher is better)")
    axes[1].set_title('SGP prediction correlation'); axes[1].set_ylim(0, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, 'ablation_sgp.png'), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars(ax, runs, ['recon_acc', 'valid_dag', 'valid_ckt', 'novel'],
         ['Topology\nreconstruction', 'Valid DAGs', 'Valid circuits\n(authors\' check)', 'Novel'], models, 'Fraction')
    ax.set_ylim(0, 1.05); ax.set_title("Decoder metrics as defined in the authors' code")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(FIG, 'ablation_decoder.png'), dpi=160); plt.close(fig)

    ref_path = os.path.join(RESULTS, 'summary', 'circuit_eval_reference.json')
    sub = [m for m in models if m != 'DVAE']
    if circ and os.path.exists(ref_path):
        ref = json.load(open(ref_path))['test']
        fig, ax = plt.subplots(figsize=(12, 4.4))
        bars(ax, runs, CIRC, CIRC_NAMES, sub, 'Fraction of 10,000 generated circuits')
        x = np.arange(len(CIRC))
        ax.scatter(x, [ref.get(k, np.nan) for k in CIRC_RAW], marker='_', s=900, color=INK, linewidth=2, zorder=5,
                   label='real test circuits (reference)')
        ax.set_ylim(0, 1.05)
        ax.set_title('Circuit-level evaluation with the open simulator (D-VAE graphs cannot be read as netlists)')
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(os.path.join(FIG, 'ablation_circuits.png'), dpi=160); plt.close(fig)

    metrics = [f'{t}_{s}' for t in targets for s in ('rmse', 'r')] + ['recon_acc', 'valid_dag', 'valid_ckt', 'novel'] + \
        [c for c in CIRC if c in runs]
    table = []
    for m in models:
        g = runs[runs.model == m]
        row = {'model': m, 'n_seeds': len(g)}
        for k in metrics:
            vals = g[k].dropna().values if k in g else []
            row[k] = f'{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}' if len(vals) > 1 else (f'{vals[0]:.3f}' if len(vals) else '–')
        table.append(row)
    t = pd.DataFrame(table)
    t.to_csv(os.path.join(RESULTS, 'summary', 'ablation_table.csv'), index=False)
    with open(os.path.join(RESULTS, 'summary', 'ablation_table.md'), 'w', encoding='utf-8') as f:
        tt = t.set_index('model').T
        f.write('| Metric | ' + ' | '.join(LABEL[m] for m in tt.columns) + ' |\n|---|' + '---:|' * len(tt.columns) + '\n')
        for k, r in tt.iterrows():
            f.write(f'| {k} | ' + ' | '.join(str(v) for v in r.values) + ' |\n')

    tests = []
    for a, b, what in PAIRS:
        for k in metrics:
            if k not in runs:
                continue
            x1, x2 = runs[runs.model == a][k].dropna().values, runs[runs.model == b][k].dropna().values
            if len(x1) > 1 and len(x2) > 1:
                tt_, p = stats.ttest_ind(x2, x1, equal_var=False)
                tests.append({'step': what, 'from': a, 'to': b, 'metric': k, 'from_mean': x1.mean(), 'to_mean': x2.mean(),
                              'change': x2.mean() - x1.mean(), 'p_value': p})
    pd.DataFrame(tests).to_csv(os.path.join(RESULTS, 'summary', 'ablation_tests.csv'), index=False)
    print(open(os.path.join(RESULTS, 'summary', 'ablation_table.md'), encoding='utf-8').read())
    tdf = pd.DataFrame(tests)
    if len(tdf):
        print(tdf.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
