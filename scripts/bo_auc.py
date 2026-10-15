"""Phase 8: area-under-the-curve endpoint for the optimisation runs, and its reliability across seed sets.

Pre-specified in reports/phase8_protocol.md. Reads the traces of Phases 5-7 (<DATA>/results/bo5: seeds 1-20,
<DATA>/results/bo7: seeds 21-40) and computes per run
  auc        mean best-feasible-UGF-so-far over simulations 21-200 (GHz; 0 before the first feasible circuit)
  t_first    simulation index of the first feasible circuit (201 = none found)
  final      best feasible UGF after 200 simulations (the earlier endpoint)
then (1) the reliability of auc vs final across the two seed sets and (2) the method comparisons.
Writes <DATA>/results/summary/bo_auc_{runs,reliability,tests}.csv and reports/figures/bo_auc{,_reliability}.png.
Post hoc (added after the pre-specified results were seen, labelled as such): the same reliability check for the
share of feasible proposals in simulations 21-200 ('frac'), as a contrast.

Usage:  .venv/Scripts/python.exe scripts/bo_auc.py
"""
import glob
import itertools
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
from plot_results import style, SURFACE, INK, INK2  # noqa: E402
from plot_bo import MLABEL, MCOLOR, SPACE, holm  # noqa: E402

FIG = os.path.join(ROOT, 'reports', 'figures')
N_INIT = 20
SPACES = ['CktGNN', 'SubgDVAEpos', 'SubgDVAEprop']
FULL = {'CktGNN': ['rs', 'cbo', 'scbo', 'scbo_g'], 'SubgDVAEpos': ['rs', 'cbo', 'scbo', 'scbo_g'],
        'SubgDVAEprop': ['rs', 'cbo', 'scbo']}                    # the 11 cells with 40 seeds
PRIMARY = [('cbo', 'rs'), ('scbo', 'cbo')]                         # per space; Holm across the 6
ENDPOINTS = {'auc': 'AUC (mean best-so-far, GHz)', 'final': 'Final best (GHz)',
             'frac': 'Share of feasible proposals (post hoc)'}
SHORT = {'rs': 'Random', 'fom_bo': 'FoM-BO', 'cbo': 'cBO', 'scbo': 'SCBO', 'scbo_g': 'SCBO\ngraded'}


def per_run():
    rows = []
    for d in ('bo5', 'bo7'):
        for f in glob.glob(os.path.join(RESULTS, d, '*_seed*.csv')):
            t = pd.read_csv(f).sort_values('eval')
            best = t.best_feasible_ugf.replace(-np.inf, 0.0).values
            feas = t['eval'][t.feasible]
            rows.append({'model': t.model.iloc[0], 'method': t.method.iloc[0], 'seed': int(t.seed.iloc[0]),
                         'set': 'A' if t.seed.iloc[0] <= 20 else 'B', 'auc': best[N_INIT:].mean(),
                         't_first': int(feas.min()) if len(feas) else 201, 'final': best[-1],
                         'frac': t.feasible.values[N_INIT:].mean()})
    return pd.DataFrame(rows)


def paired(df, sp, a, b, col, seeds=None):
    d = df[df.model == sp]
    if seeds is not None:
        d = d[d.seed.isin(seeds)]
    piv = d.pivot(index='seed', columns='method', values=col)
    if a not in piv or b not in piv:
        return None
    both = piv[[a, b]].dropna()
    x, y = both[a].values, both[b].values
    p = stats.wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    return {'n': len(x), 'median_a': np.median(x), 'median_b': np.median(y), 'median_diff': np.median(x - y),
            'a_better': int((x > y).sum()), 'b_better': int((x < y).sum()), 'p': p}


def reliability(df):
    out, sign_rows = [], []
    cells = [(sp, m) for sp in SPACES for m in FULL[sp]]
    for col in ENDPOINTS:
        mA = [df[(df.model == sp) & (df.method == m) & (df.set == 'A')][col].median() for sp, m in cells]
        mB = [df[(df.model == sp) & (df.method == m) & (df.set == 'B')][col].median() for sp, m in cells]
        rho = stats.spearmanr(mA, mB).correlation
        shifts = sum(stats.mannwhitneyu(df[(df.model == sp) & (df.method == m) & (df.set == 'A')][col],
                                        df[(df.model == sp) & (df.method == m) & (df.set == 'B')][col]).pvalue < 0.05
                     for sp, m in cells)
        agree = 0
        n_pairs = 0
        for sp in SPACES:
            for a, b in itertools.combinations(FULL[sp], 2):
                rA = paired(df, sp, a, b, col, range(1, 21))
                rB = paired(df, sp, a, b, col, range(21, 41))
                sa, sb = np.sign(rA['median_diff']), np.sign(rB['median_diff'])
                ok = bool(sa == sb and sa != 0)
                agree += ok
                n_pairs += 1
                sign_rows.append({'endpoint': col, 'space': sp, 'a': a, 'b': b, 'median_diff_A': rA['median_diff'],
                                  'median_diff_B': rB['median_diff'], 'same_sign': ok})
        out.append({'endpoint': col, 'post_hoc': col == 'frac', 'spearman_cell_medians': rho, 'comparisons_same_sign': agree,
                    'comparisons': n_pairs, 'cells_shifted_p05': shifts, 'cells': len(cells),
                    'median_A': mA, 'median_B': mB})
    return pd.DataFrame(out), pd.DataFrame(sign_rows), cells


def main():
    df = per_run()
    summ = os.path.join(RESULTS, 'summary')
    df.to_csv(os.path.join(summ, 'bo_auc_runs.csv'), index=False)

    rel, signs, cells = reliability(df)
    rel.drop(columns=['median_A', 'median_B']).to_csv(os.path.join(summ, 'bo_auc_reliability.csv'), index=False)
    signs.to_csv(os.path.join(summ, 'bo_auc_sign_agreement.csv'), index=False)

    tests = []
    for sp in SPACES:
        for a, b in PRIMARY:
            r = paired(df, sp, a, b, 'auc')
            tests.append({'space': sp, 'a': a, 'b': b, 'endpoint': 'auc', 'primary': True, **r})
        for a, b in [('scbo_g', 'scbo'), ('scbo_g', 'cbo'), ('scbo_g', 'rs'), ('scbo', 'rs')]:
            r = paired(df, sp, a, b, 'auc')
            if r:
                tests.append({'space': sp, 'a': a, 'b': b, 'endpoint': 'auc', 'primary': False, **r})
        for col in ('t_first', 'final'):
            for a, b in PRIMARY:
                tests.append({'space': sp, 'a': a, 'b': b, 'endpoint': col, 'primary': False,
                              **paired(df, sp, a, b, col)})
    for m in ('rs', 'cbo', 'scbo'):                      # P vs A2, unpaired
        x = df[(df.model == 'SubgDVAEprop') & (df.method == m)].auc.values
        y = df[(df.model == 'SubgDVAEpos') & (df.method == m)].auc.values
        tests.append({'space': 'P vs A2', 'a': m, 'b': m, 'endpoint': 'auc', 'primary': False, 'n': len(x),
                      'median_a': np.median(x), 'median_b': np.median(y), 'median_diff': np.median(x) - np.median(y),
                      'p': stats.mannwhitneyu(x, y).pvalue})
    tt = pd.DataFrame(tests)
    tt.loc[tt.primary, 'holm_p'] = holm(tt.loc[tt.primary, 'p'])
    tt.to_csv(os.path.join(summ, 'bo_auc_tests.csv'), index=False)

    # cell table
    tab = df.groupby(['model', 'method']).agg(n=('auc', 'size'), auc_median=('auc', 'median'), auc_mean=('auc', 'mean'),
                                              auc_std=('auc', 'std'), final_median=('final', 'median'),
                                              final_std=('final', 'std'), t_first_median=('t_first', 'median'),
                                              found=('t_first', lambda s: int((s <= 200).sum()))).reset_index()
    tab.to_csv(os.path.join(summ, 'bo_auc_table.csv'), index=False)

    # figure 1: AUC per run, per space
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, sp in zip(axes, SPACES):
        ms = [m for m in ['rs', 'fom_bo', 'cbo', 'scbo', 'scbo_g'] if ((df.model == sp) & (df.method == m)).any()]
        for i, m in enumerate(ms):
            v = df[(df.model == sp) & (df.method == m)].auc.values
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=MCOLOR[m], edgecolor=SURFACE,
                       linewidth=0.6, zorder=3)
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, linewidth=2, zorder=4)
            ax.annotate(f'n={len(v)}', (i, -1.2), ha='center', fontsize=7.5, color=INK2, annotation_clip=False)
        ax.set_xticks(range(len(ms)), [SHORT[m] for m in ms], fontsize=8.5)
        ax.set_title(SPACE[sp], fontsize=10)
        style(ax)
    axes[0].set_ylabel('AUC: mean best feasible UGF so far,\nsimulations 21–200 (GHz)')
    fig.suptitle('Area under the best-so-far curve, one dot per run (black bar: median)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'bo_auc.png'), dpi=160)
    plt.close(fig)

    # figure 2: reliability, set A vs set B cell medians for both endpoints
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (_, r) in zip(axes, rel.iterrows()):
        lim = max(max(r.median_A), max(r.median_B)) * 1.1 + (0.5 if r.endpoint != 'frac' else 0.01)
        ax.plot([0, lim], [0, lim], color=INK2, linewidth=1, linestyle='--', zorder=1)
        for (sp, m), a, b in zip(cells, r.median_A, r.median_B):
            ax.scatter(a, b, s=46, color=MCOLOR[m], edgecolor=SURFACE, linewidth=1, zorder=3)
            ax.annotate({'CktGNN': 'C', 'SubgDVAEpos': 'A2', 'SubgDVAEprop': 'P'}[sp], (a, b), xytext=(4, 3),
                        textcoords='offset points', fontsize=7.5, color=INK2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel('Median, seeds 1–20'); ax.set_ylabel('Median, seeds 21–40')
        ax.set_title(f"{ENDPOINTS[r.endpoint]}\nSpearman ρ = {r.spearman_cell_medians:.2f}; "
                     f"{r.comparisons_same_sign}/{r.comparisons} comparisons agree in sign", fontsize=9.5)
        style(ax)
    h = [plt.Line2D([], [], marker='o', linestyle='', color=MCOLOR[m], label=SHORT[m].replace('\n', ' '))
         for m in ['rs', 'cbo', 'scbo', 'scbo_g']]
    fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIG, 'bo_auc_reliability.png'), dpi=160)
    plt.close(fig)

    pd.set_option('display.width', 200)
    print(rel.drop(columns=['median_A', 'median_B']).round(3).to_string(index=False))
    print(tab.round(2).to_string(index=False))
    print(tt.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
