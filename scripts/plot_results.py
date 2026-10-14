"""Figures and significance tests for the baseline reproduction.

Reads <DATA_ROOT>/results/summary/per_run.csv (from aggregate_results.py) and the training logs, and writes
  reports/figures/repro_sgp.png      SGP RMSE and Pearson r per target, per model (seed dots + references)
  reports/figures/repro_decoder.png  reconstruction / validity / novelty per model
  reports/figures/repro_training.png training loss curves (one panel per model)
  <DATA_ROOT>/results/summary/significance.csv  Welch t-tests CktGNN vs D-VAE across training seeds

Usage:  .venv/Scripts/python.exe scripts/plot_results.py
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RESULTS  # noqa: E402
from aggregate_results import PAPER, APP_G  # noqa: E402

FIG = os.path.join(ROOT, 'reports', 'figures')
os.makedirs(FIG, exist_ok=True)
# reference palette (dataviz skill): categorical slots 1-2, light-mode chrome
SERIES = {'CktGNN': '#2a78d6', 'DVAE': '#eb6834'}
LABEL = {'CktGNN': 'CktGNN', 'DVAE': 'D-VAE'}
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.edgecolor': AXIS, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False})


def style(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def grouped(ax, runs, metrics, names, ylabel, refs=True):
    models = [m for m in SERIES if m in set(runs.model)]
    width = 0.36
    x = np.arange(len(metrics))
    for i, m in enumerate(models):
        g = runs[runs.model == m]
        means = [g[k].mean() for k in metrics]
        pos = x + (i - (len(models) - 1) / 2) * (width + 0.02)
        ax.bar(pos, means, width, color=SERIES[m], label=f'{LABEL[m]} (mean of {len(g)} seeds)', zorder=2)
        for j, k in enumerate(metrics):
            vals = g[k].dropna().values
            ax.scatter(np.full(len(vals), pos[j]), vals, s=22, color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=4)
            if refs:
                ax.scatter(pos[j], APP_G[m][k], marker='D', s=46, facecolor='none', edgecolor=INK, linewidth=1.3, zorder=5)
                ax.scatter(pos[j], PAPER[m][k], marker='x', s=40, color=MUTED, linewidth=1.6, zorder=5)
    ax.set_xticks(x, names)
    ax.set_ylabel(ylabel)
    style(ax)


def legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(color=SERIES[m], label=LABEL[m] + ' (mean)') for m in SERIES]
    h += [Line2D([], [], marker='o', ls='', color=INK, markeredgecolor=SURFACE, label='individual training seed'),
          Line2D([], [], marker='D', ls='', markerfacecolor='none', markeredgecolor=INK, label='authors, released data (arXiv v2 App. G)'),
          Line2D([], [], marker='x', ls='', color=MUTED, label='paper Table 1/2 (older dataset)')]
    return h


def main():
    runs = pd.read_csv(os.path.join(RESULTS, 'summary', 'per_run.csv'))
    targets = ['gain', 'bw', 'pm', 'fom']
    tnames = ['Gain', 'BW', 'PM', 'FoM']

    # 1. SGP regression on the latent space
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    grouped(axes[0], runs, [f'{t}_rmse' for t in targets], tnames, 'Test RMSE (standardized; lower is better)')
    axes[0].set_title('SGP prediction error from VAE latents')
    axes[0].set_ylim(0, 1.1)
    grouped(axes[1], runs, [f'{t}_r' for t in targets], tnames, "Pearson's r (higher is better)")
    axes[1].set_title('SGP prediction correlation')
    axes[1].set_ylim(0, 1.0)
    fig.legend(handles=legend_handles(), loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(FIG, 'repro_sgp.png'), dpi=160)
    plt.close(fig)

    # 2. decoder metrics
    dm = ['recon_acc', 'valid_dag', 'valid_ckt', 'novel']
    fig, ax = plt.subplots(figsize=(8, 4.2))
    grouped(ax, runs, dm, ['Topology\nreconstruction', 'Valid DAGs', 'Valid circuits', 'Novel circuits'], 'Fraction')
    ax.set_ylim(0, 1.05)
    ax.set_title('Decoder metrics (code-defined; not comparable across graph representations)')
    fig.legend(handles=legend_handles(), loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(FIG, 'repro_decoder.png'), dpi=160)
    plt.close(fig)

    # 3. training curves (separate panels: the two losses are on different scales)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, m in zip(axes, SERIES):
        ends = []
        for path in sorted(glob.glob(os.path.join(RESULTS, 'vae', f'{m}_seed*', 'train_log.csv'))):
            seed = re.search(r'seed(\d+)', path).group(1)
            log = pd.read_csv(path)
            ax.plot(log.epoch, log.recon, color=SERIES[m], linewidth=1.4, alpha=0.35 + 0.2 * int(seed))
            ends.append((seed, log.epoch.iloc[-1], log.recon.iloc[-20:].median()))
        ys = [e[2] for e in ends]
        if ends and max(ys) / min(ys) < 1.25:     # curves end together: one label instead of overlapping ones
            ax.annotate(f'seeds {ends[0][0]}–{ends[-1][0]}', (ends[0][1], np.median(ys)), color=INK2, fontsize=8,
                        xytext=(4, 0), textcoords='offset points', va='center')
        else:
            for seed, xe, ye in ends:
                ax.annotate(f'seed {seed}', (xe, ye), color=INK2, fontsize=8, xytext=(4, 0),
                            textcoords='offset points', va='center')
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training reconstruction loss per graph')
        ax.set_title(f'{LABEL[m]}: training loss (lr stayed at 1e-4)')
        style(ax)
        ax.set_xlim(0, 330)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'repro_training.png'), dpi=160)
    plt.close(fig)

    # 4. Welch t-tests, CktGNN vs D-VAE, across training seeds
    rows = []
    for k in [f'{t}_{s}' for t in targets for s in ('rmse', 'r')] + dm:
        a = runs[runs.model == 'CktGNN'][k].dropna().values
        b = runs[runs.model == 'DVAE'][k].dropna().values
        if len(a) > 1 and len(b) > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append({'metric': k, 'CktGNN_mean': a.mean(), 'CktGNN_std': a.std(ddof=1), 'DVAE_mean': b.mean(),
                         'DVAE_std': b.std(ddof=1), 'diff': a.mean() - b.mean(), 'welch_t': t, 'p_value': p,
                         'n': f'{len(a)} vs {len(b)}'})
    sig = pd.DataFrame(rows)
    sig.to_csv(os.path.join(RESULTS, 'summary', 'significance.csv'), index=False)
    print(sig.round(4).to_string(index=False))
    print('figures written to', FIG)


if __name__ == '__main__':
    main()
