"""Analysis of the Phase 4 optimisation study (results of scripts/latent_bo.py).

Writes reports/figures/bo_convergence.png, bo_feasibility.png and
<DATA_ROOT>/results/summary/bo_table.{csv,md}, bo_tests.csv.

Usage:  .venv/Scripts/python.exe scripts/plot_bo.py
"""
import glob
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
METHODS = ['rs', 'fom_bo', 'cbo']
MLABEL = {'rs': 'Random search', 'fom_bo': "FoM-BO (paper's objective)", 'cbo': 'Constrained BO (ours)'}
MCOLOR = {'rs': '#898781', 'fom_bo': '#eb6834', 'cbo': '#2a78d6'}
SPACE = {'CktGNN': 'CktGNN latent space', 'SubgDVAEpos': 'A2 latent space (per-node device values)'}
ORACLE_UGF = 30.577      # best feasible UGF among the 9,000 training circuits (same specification)


def main():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RESULTS, 'bo', '*_seed*.csv'))], ignore_index=True)
    df['best'] = df.best_feasible_ugf.replace(-np.inf, 0.0)
    spaces = [s for s in SPACE if s in set(df.model)]

    fig, axes = plt.subplots(1, len(spaces), figsize=(6 * len(spaces), 4.2), squeeze=False)
    for ax, sp in zip(axes[0], spaces):
        for m in METHODS:
            d = df[(df.model == sp) & (df.method == m)]
            if d.empty:
                continue
            piv = d.pivot_table(index='eval', columns='seed', values='best').astype(float)
            med, lo, hi = piv.median(axis=1), piv.quantile(0.25, axis=1), piv.quantile(0.75, axis=1)
            ax.plot(piv.index, med, color=MCOLOR[m], linewidth=2, label=f'{MLABEL[m]} (median of {piv.shape[1]})')
            ax.fill_between(piv.index, lo, hi, color=MCOLOR[m], alpha=0.15, linewidth=0)
        ax.axhline(ORACLE_UGF, color=INK2, linestyle='--', linewidth=1)
        ax.annotate('best feasible circuit in the 9,000 training circuits', (5, ORACLE_UGF), xytext=(0, 4),
                    textcoords='offset points', color=INK2, fontsize=8)
        ax.axvline(20, color=MUTED, linewidth=0.8, linestyle=':')
        ax.set_xlabel('Simulations'); ax.set_ylabel('Best feasible UGF so far (GHz; 0 = none yet)')
        ax.set_title(SPACE[sp]); ax.set_ylim(0, 35); style(ax)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, 'bo_convergence.png'), dpi=160); plt.close(fig)

    rows = []
    for (sp, m), d in df.groupby(['model', 'method']):
        fin = d.sort_values('eval').groupby('seed').tail(1)
        first = d[d.feasible].groupby('seed')['eval'].min()
        top = d.loc[d.groupby('seed')['fom'].idxmax()]           # the circuit each run ranks best by FoM
        rows.append({'space': sp, 'method': m, 'runs': fin.seed.nunique(),
                     'runs_with_feasible': int((fin.best > 0).sum()),
                     'best_feasible_ugf_median': fin.best.median(), 'best_feasible_ugf_mean': fin.best.mean(),
                     'best_feasible_ugf_std': fin.best.std(ddof=1),
                     'frac_evals_feasible': d.feasible.mean(), 'frac_evals_stable': (d.ol_stable & d.cl_stable & d.functional).mean(),
                     'unique_feasible_per_run': fin.n_feasible_unique.mean(),
                     'evals_to_first_feasible_median': first.median() if len(first) else np.nan,
                     'max_fom_circuit_feasible': top.feasible.mean(),
                     'max_fom_circuit_stable': (top.ol_stable & top.cl_stable).mean(),
                     'feasible_novel_frac': (d.feasible & d.novel).sum() / max(d.feasible.sum(), 1)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RESULTS, 'summary', 'bo_table.csv'), index=False)

    tests = []
    for sp in spaces:
        fin = df[df.model == sp].sort_values('eval').groupby(['method', 'seed']).tail(1).pivot(index='seed', columns='method', values='best')
        for a, b in (('cbo', 'fom_bo'), ('cbo', 'rs'), ('fom_bo', 'rs')):
            if a in fin and b in fin:
                x, y = fin[a].values, fin[b].values
                p = stats.wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
                tests.append({'space': sp, 'a': a, 'b': b, 'median_a': np.median(x), 'median_b': np.median(y),
                              'a_better_in_runs': int((x > y).sum()), 'ties': int((x == y).sum()), 'wilcoxon_p': p})
        ff = df[df.model == sp].groupby(['method', 'seed']).feasible.mean().unstack(0)
        if 'cbo' in ff and 'fom_bo' in ff:
            tests.append({'space': sp, 'a': 'cbo', 'b': 'fom_bo', 'metric': 'frac_evals_feasible',
                          'median_a': ff.cbo.median(), 'median_b': ff.fom_bo.median(),
                          'a_better_in_runs': int((ff.cbo > ff.fom_bo).sum()), 'wilcoxon_p': stats.wilcoxon(ff.cbo, ff.fom_bo).pvalue})
    pd.DataFrame(tests).to_csv(os.path.join(RESULTS, 'summary', 'bo_tests.csv'), index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.26
    for i, m in enumerate(METHODS):
        vals = [t[(t.space == sp) & (t.method == m)].frac_evals_feasible.mean() for sp in spaces]
        ax.bar(np.arange(len(spaces)) + (i - 1) * (w + 0.02), vals, w, color=MCOLOR[m], label=MLABEL[m], zorder=2)
    ax.axhline(0.0305, color=INK2, linestyle='--', linewidth=1)
    ax.annotate('share of real circuits meeting the spec (3.0%)', (-0.45, 0.0305), xytext=(0, 4), textcoords='offset points',
                color=INK2, fontsize=8)
    ax.set_xticks(np.arange(len(spaces)), [SPACE[s] for s in spaces])
    ax.set_ylabel('Fraction of simulations that meet the spec'); style(ax)
    ax.set_title('How often each method proposes a feasible (stable, PM ≥ 60°, gain ≥ 60 dB) circuit')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'bo_feasibility.png'), dpi=160); plt.close(fig)

    with open(os.path.join(RESULTS, 'summary', 'bo_table.md'), 'w', encoding='utf-8') as f:
        r3 = t.round(3)
        f.write('| ' + ' | '.join(r3.columns) + ' |\n|' + '---|' * len(r3.columns) + '\n')
        for _, row in r3.iterrows():
            f.write('| ' + ' | '.join(str(v) for v in row.values) + ' |\n')
    print(t.round(3).to_string(index=False))
    print(pd.DataFrame(tests).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
