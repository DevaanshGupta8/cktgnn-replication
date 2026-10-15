"""Analysis of the optimisation studies (results of scripts/latent_bo.py).

Phase 4 (--dir bo, default): reports/figures/bo_convergence.png, bo_feasibility.png and
<DATA_ROOT>/results/summary/bo_table.{csv,md}, bo_tests.csv.
Phase 5 (--dir bo5): the same files with prefix bo5_. Primary tests (SCBO vs cBO and vs random search in each
latent space) are Holm-corrected across those four comparisons, as fixed in reports/phase5_protocol.md.

Usage:  .venv/Scripts/python.exe scripts/plot_bo.py [--dir bo5]
"""
import argparse
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
METHODS = ['rs', 'fom_bo', 'cbo', 'scbo', 'scbo_g']
MLABEL = {'rs': 'Random search', 'fom_bo': "FoM-BO (paper's objective)", 'cbo': 'Constrained BO',
          'scbo': 'Trust-region constrained BO (SCBO)', 'scbo_g': 'SCBO, graded stability (exploratory)'}
MCOLOR = {'rs': '#898781', 'fom_bo': '#eb6834', 'cbo': '#2a78d6', 'scbo': '#1baf7a', 'scbo_g': '#eda100'}
PRIMARY = [('scbo', 'cbo'), ('scbo', 'rs')]                      # Phase 5 primary comparisons, in each of
PHASE5_SPACES = ('CktGNN', 'SubgDVAEpos')                        # these two latent spaces (Holm across 4)
SECONDARY = [('cbo', 'fom_bo'), ('cbo', 'rs'), ('fom_bo', 'rs'), ('scbo', 'fom_bo')]
EXPLORATORY = [('scbo_g', 'scbo'), ('scbo_g', 'cbo'), ('scbo_g', 'rs')]    # added after the Phase 5 results
# Phase 7 (reports/phase7_protocol.md): confirmatory test of scbo_g on fresh seeds 21-40, Holm across 3 spaces
TESTS = {'bo7': dict(primary=[('scbo_g', 'scbo')], primary_spaces=('CktGNN', 'SubgDVAEpos', 'SubgDVAEprop'),
                     secondary=[('scbo_g', 'cbo'), ('scbo_g', 'rs'), ('scbo', 'cbo'), ('cbo', 'rs'), ('scbo', 'rs')],
                     exploratory=[])}
BIMODAL = 0.20                                                   # a run "exploits a basin" above this feasible share
SPACE = {'CktGNN': 'CktGNN latent space', 'SubgDVAEpos': 'A2 latent space (per-node device values)',
         'SubgDVAEprop': 'P latent space (A2 + property predictor)'}
ORACLE_UGF = 30.577      # best feasible UGF among the 9,000 training circuits (same specification)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='bo', help='bo (Phase 4) or bo5 (Phase 5)')
    args = ap.parse_args()
    pre = args.dir
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RESULTS, args.dir, '*_seed*.csv'))], ignore_index=True)
    df['best'] = df.best_feasible_ugf.replace(-np.inf, 0.0)
    spaces = [s for s in SPACE if s in set(df.model)]
    methods = [m for m in METHODS if m in set(df.method)]
    cfg = TESTS.get(args.dir, dict(primary=PRIMARY, primary_spaces=PHASE5_SPACES, secondary=SECONDARY,
                                   exploratory=EXPLORATORY))

    fig, axes = plt.subplots(1, len(spaces), figsize=(6 * len(spaces), 4.2), squeeze=False)
    for ax, sp in zip(axes[0], spaces):
        for m in methods:
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
    fig.legend(h, l, loc='lower center', ncol=min(len(methods), 4), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, f'{pre}_convergence.png'), dpi=160); plt.close(fig)

    rows = []
    for (sp, m), d in df.groupby(['model', 'method']):
        fin = d.sort_values('eval').groupby('seed').tail(1)
        first = d[d.feasible].groupby('seed')['eval'].min()
        top = d.loc[d.groupby('seed')['fom'].idxmax()]           # the circuit each run ranks best by FoM
        per_run_feas = d.groupby('seed').feasible.mean()
        rows.append({'space': sp, 'method': m, 'runs': fin.seed.nunique(),
                     'runs_with_feasible': int((fin.best > 0).sum()),
                     'best_feasible_ugf_median': fin.best.median(), 'best_feasible_ugf_mean': fin.best.mean(),
                     'best_feasible_ugf_std': fin.best.std(ddof=1),
                     'frac_evals_feasible': d.feasible.mean(), 'frac_evals_stable': (d.ol_stable & d.cl_stable & d.functional).mean(),
                     'unique_feasible_per_run': fin.n_feasible_unique.mean(),
                     'runs_over_20pct_feasible': int((per_run_feas > BIMODAL).sum()),
                     'evals_to_first_feasible_median': first.median() if len(first) else np.nan,
                     'max_fom_circuit_feasible': top.feasible.mean(),
                     'max_fom_circuit_stable': (top.ol_stable & top.cl_stable).mean(),
                     'feasible_novel_frac': (d.feasible & d.novel).sum() / max(d.feasible.sum(), 1)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RESULTS, 'summary', f'{pre}_table.csv'), index=False)

    tests = []
    for sp in spaces:
        fin = df[df.model == sp].sort_values('eval').groupby(['method', 'seed']).tail(1).pivot(index='seed', columns='method', values='best')
        for a, b in cfg['primary'] + cfg['secondary'] + cfg['exploratory']:
            if a in fin and b in fin:
                both = fin[[a, b]].dropna()                       # paired: same seeds, same initial design
                x, y = both[a].values, both[b].values
                p = stats.wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
                tests.append({'space': sp, 'a': a, 'b': b, 'metric': 'best_feasible_ugf', 'n': len(x),
                              'primary': (a, b) in cfg['primary'] and sp in cfg['primary_spaces'], 'exploratory': (a, b) in cfg['exploratory'], 'median_a': np.median(x), 'median_b': np.median(y),
                              'a_better_in_runs': int((x > y).sum()), 'ties': int((x == y).sum()), 'wilcoxon_p': p})
        ff = df[df.model == sp].groupby(['method', 'seed']).feasible.mean().unstack(0)
        for a, b in (('cbo', 'fom_bo'), ('scbo', 'cbo'), ('scbo', 'rs'), ('scbo_g', 'scbo')):
            if a in ff and b in ff:
                both = ff[[a, b]].dropna()
                tests.append({'space': sp, 'a': a, 'b': b, 'metric': 'frac_evals_feasible', 'n': len(both), 'primary': False,
                              'exploratory': a == 'scbo_g' and args.dir != 'bo7',
                              'median_a': both[a].median(), 'median_b': both[b].median(),
                              'a_better_in_runs': int((both[a] > both[b]).sum()), 'ties': int((both[a] == both[b]).sum()),
                              'wilcoxon_p': stats.wilcoxon(both[a], both[b]).pvalue if np.any(both[a] != both[b]) else 1.0})
    tt = pd.DataFrame(tests)
    if len(tt) and tt.primary.any():
        tt.loc[tt.primary, 'holm_p'] = holm(tt.loc[tt.primary, 'wilcoxon_p'])
    tt.to_csv(os.path.join(RESULTS, 'summary', f'{pre}_tests.csv'), index=False)

    # Phase 6 (reports/phase6_protocol.md): P's latent space vs A2's, per method; unpaired Mann-Whitney U.
    # Primary: cBO and SCBO on best feasible UGF, Holm across the two.
    if {'SubgDVAEprop', 'SubgDVAEpos'} <= set(spaces):
        fin = df.sort_values('eval').groupby(['model', 'method', 'seed']).tail(1)
        feas = df.groupby(['model', 'method', 'seed']).feasible.mean().rename('frac').reset_index()
        cross = []
        for m in methods:
            for metric, src, col in (('best_feasible_ugf', fin, 'best'), ('frac_evals_feasible', feas, 'frac')):
                x = src[(src.model == 'SubgDVAEprop') & (src.method == m)][col].values
                y = src[(src.model == 'SubgDVAEpos') & (src.method == m)][col].values
                if len(x) > 1 and len(y) > 1:
                    p = stats.mannwhitneyu(x, y, alternative='two-sided').pvalue if len(set(x) | set(y)) > 1 else 1.0
                    cross.append({'method': m, 'metric': metric, 'n_P': len(x), 'n_A2': len(y),
                                  'primary': metric == 'best_feasible_ugf' and m in ('cbo', 'scbo'),
                                  'median_P': np.median(x), 'median_A2': np.median(y), 'mean_P': x.mean(), 'mean_A2': y.mean(),
                                  'P_runs_feasible': int((x > 0).sum()) if col == 'best' else np.nan,
                                  'A2_runs_feasible': int((y > 0).sum()) if col == 'best' else np.nan, 'mannwhitney_p': p})
        ct = pd.DataFrame(cross)
        if ct.primary.any():
            ct.loc[ct.primary, 'holm_p'] = holm(ct.loc[ct.primary, 'mannwhitney_p'])
        ct.to_csv(os.path.join(RESULTS, 'summary', f'{pre}_P_vs_A2_tests.csv'), index=False)
        print(ct.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8 + 1.5 * (len(methods) - 3), 4))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        x0 = np.arange(len(spaces)) - 0.4 + w * (i + 0.5)
        vals = [t[(t.space == sp) & (t.method == m)].frac_evals_feasible.mean() for sp in spaces]
        ax.bar(x0, vals, w * 0.9, color=MCOLOR[m], label=MLABEL[m], zorder=2)
        for j, sp in enumerate(spaces):                          # per-run shares: shows the bimodality
            pr = df[(df.model == sp) & (df.method == m)].groupby('seed').feasible.mean().values
            ax.scatter(np.full(len(pr), x0[j]), pr, s=10, color=INK, edgecolor=SURFACE, linewidth=0.6, zorder=4)
    ax.axhline(0.0305, color=INK2, linestyle='--', linewidth=1)
    ax.annotate('share of real circuits meeting the spec (3.0%)', (-0.45, 0.0305), xytext=(0, 4), textcoords='offset points',
                color=INK2, fontsize=8)
    ax.set_xticks(np.arange(len(spaces)), [SPACE[s] for s in spaces])
    ax.set_ylabel('Fraction of simulations that meet the spec'); style(ax)
    ax.set_title('How often each method proposes a feasible (stable, PM ≥ 60°, gain ≥ 60 dB) circuit\n'
                 '(bars: mean; dots: individual runs)', fontsize=10)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=min(len(methods), 3), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06 * ((len(methods) + 2) // 3), 1, 1))
    fig.savefig(os.path.join(FIG, f'{pre}_feasibility.png'), dpi=160); plt.close(fig)

    with open(os.path.join(RESULTS, 'summary', f'{pre}_table.md'), 'w', encoding='utf-8') as f:
        r3 = t.round(3)
        f.write('| ' + ' | '.join(r3.columns) + ' |\n|' + '---|' * len(r3.columns) + '\n')
        for _, row in r3.iterrows():
            f.write('| ' + ' | '.join(str(v) for v in row.values) + ' |\n')
    print(t.round(3).to_string(index=False))
    print(tt.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
