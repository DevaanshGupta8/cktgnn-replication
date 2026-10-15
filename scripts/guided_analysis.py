"""Phase 9 Part A analysis: guided vs control generation (reports/phase9_protocol.md).

Per model: per-seed meets-spec rates, and a bootstrap 95% CI of the guided - control difference, resampling latents
within each seed and arm (2,000 resamples, pooled over seeds). Also the dose-response curve and diversity counts.
Writes <DATA_ROOT>/results/summary/guided_generation.csv, guided_dose_response.csv and reports/figures/guided_generation.png.

Usage:  .venv/Scripts/python.exe scripts/guided_analysis.py [--models SubgDVAEprop SubgDVAEpos CktGNN]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, RESULTS  # noqa: E402
from plot_results import style, INK2  # noqa: E402

LABEL = {'SubgDVAEprop': 'P (jointly trained predictor)', 'SubgDVAEpropn': 'Pn (jointly trained predictor)',
         'SubgDVAEpos': 'A2 (fitted classifier)', 'CktGNN': 'CktGNN (fitted classifier)'}
COLOR = {'SubgDVAEprop': '#2a78d6', 'SubgDVAEpropn': '#1baf7a', 'SubgDVAEpos': '#eda100', 'CktGNN': '#eb6834'}
B = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['SubgDVAEprop', 'SubgDVAEpos', 'CktGNN'])
    args = ap.parse_args()
    rows, dose = [], []
    rng = np.random.default_rng(0)
    for m in args.models:
        per_seed, lat = [], []
        for s in (1, 2, 3):
            run = os.path.join(RESULTS, 'vae', f'{m}_seed{s}')
            if not os.path.exists(os.path.join(run, 'guided_generation.json')):
                continue
            r = json.load(open(os.path.join(run, 'guided_generation.json')))
            per_seed.append(r)
            d = pd.read_csv(os.path.join(run, 'guided_latents.csv'))
            d['seed'] = s
            lat.append(d)
            for q, o in r['dose_response'].items():
                dose.append({'model': m, 'seed': s, 'top_fraction': float(q), 'meets_spec': o['meets_spec'],
                             'stable': o['stable'], 'unique_spec_topologies': o['unique_spec_topologies']})
        if not per_seed:
            continue
        lat = pd.concat(lat)
        groups = {(s, a): g.meets_spec.values for (s, a), g in lat.groupby(['seed', 'arm'])}
        seeds = sorted({s for s, _ in groups})
        diffs = []
        for _ in range(B):
            gm = np.mean(np.concatenate([rng.choice(groups[(s, 'guided')], len(groups[(s, 'guided')])) for s in seeds]))
            cm = np.mean(np.concatenate([rng.choice(groups[(s, 'control')], len(groups[(s, 'control')])) for s in seeds]))
            diffs.append(gm - cm)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        g = [r['guided'] for r in per_seed]
        c = [r['control'] for r in per_seed]
        rows.append({'model': m, 'predictor': per_seed[0]['predictor'], 'seeds': len(per_seed),
                     'guided_meets_spec': np.mean([x['meets_spec'] for x in g]),
                     'control_meets_spec': np.mean([x['meets_spec'] for x in c]),
                     'guided_better_in_seeds': int(sum(a['meets_spec'] > b['meets_spec'] for a, b in zip(g, c))),
                     'diff': np.mean([x['meets_spec'] for x in g]) - np.mean([x['meets_spec'] for x in c]),
                     'diff_ci_lo': lo, 'diff_ci_hi': hi,
                     'ratio': np.mean([x['meets_spec'] for x in g]) / np.mean([x['meets_spec'] for x in c]),
                     'guided_stable': np.mean([x['stable'] for x in g]), 'control_stable': np.mean([x['stable'] for x in c]),
                     'guided_functional': np.mean([x['functional'] for x in g]),
                     'control_functional': np.mean([x['functional'] for x in c]),
                     'guided_values_ok': np.mean([x['values_ok'] for x in g]),
                     'control_values_ok': np.mean([x['values_ok'] for x in c]),
                     'guided_novel': np.mean([x['novel'] for x in g]), 'control_novel': np.mean([x['novel'] for x in c]),
                     'guided_unique_spec': np.mean([x['unique_spec_topologies'] for x in g]),
                     'control_unique_spec': np.mean([x['unique_spec_topologies'] for x in c]),
                     'guided_unique_functional': np.mean([x['unique_functional_topologies'] for x in g]),
                     'control_unique_functional': np.mean([x['unique_functional_topologies'] for x in c]),
                     'confirmed': bool(all(a['meets_spec'] > b['meets_spec'] for a, b in zip(g, c)) and lo > 0)})
    t = pd.DataFrame(rows)
    dd = pd.DataFrame(dose)
    summ = os.path.join(RESULTS, 'summary')
    t.to_csv(os.path.join(summ, 'guided_generation.csv'), index=False)
    dd.to_csv(os.path.join(summ, 'guided_dose_response.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in t.model:
        d = dd[dd.model == m].groupby('top_fraction')[['meets_spec', 'unique_spec_topologies']].mean().sort_index()
        for ax, col in zip(axes, ['meets_spec', 'unique_spec_topologies']):
            dc = d if col == 'meets_spec' else d[d.index >= 0.05]   # top 1% = 200 latents only: counts not comparable
            ax.plot(dc.index, dc[col] * (100 if col == 'meets_spec' else 1), marker='o', markersize=5, linewidth=2,
                    color=COLOR[m], label=LABEL[m])
    for ax in axes:
        ax.set_xscale('log'); ax.invert_xaxis()
        ax.set_xticks([1, 0.5, 0.2, 0.05, 0.01], ['all', 'top 50%', 'top 20%', 'top 5%', 'top 1%'])
        ax.set_xlabel('Latents kept (ranked by predicted "meets spec")')
        style(ax)
    axes[0].axhline(2.8, color=INK2, linestyle='--', linewidth=1)
    axes[0].annotate('real test circuits (2.8%)', (1, 2.8), xytext=(4, 4), textcoords='offset points', fontsize=8, color=INK2)
    axes[0].set_ylabel('Generated circuits meeting the spec (%)')
    axes[1].set_ylabel('Distinct spec-meeting topologies\n(per 10,000 generated circuits)')
    axes[0].set_title('Guided generation: share meeting the spec', fontsize=10)
    axes[1].set_title('Guided generation: diversity\n(top 1% omitted: only 2,000 circuits, not comparable)', fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(t), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(ROOT, 'reports', 'figures', 'guided_generation.png'), dpi=160)
    plt.close(fig)
    pd.set_option('display.width', 220)
    print(t.round(4).T.to_string())
    print(dd.groupby(['model', 'top_fraction'])[['meets_spec', 'stable', 'unique_spec_topologies']].mean().round(4))


if __name__ == '__main__':
    main()
