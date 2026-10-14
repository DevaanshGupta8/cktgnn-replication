"""Audit of the released OCB Ckt-Bench-101 dataset.

Answers, with numbers, the dataset questions from Phase 3 of the research plan:
split sizes, number of distinct topologies, train/test topology overlap
(leakage), duplicate circuits, the FoM formula, how much of FoM is explained
by BW, type/position statistics, and how strong simple non-GNN baselines are
on the exact train/test split used by the paper's code.

Usage:  .venv/Scripts/python.exe scripts/audit_dataset.py
Writes: <DATA_ROOT>/results/audit/*.csv, <DATA_ROOT>/results/audit/audit_summary.json
"""
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BENCH101, RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, 'audit')
os.makedirs(OUT, exist_ok=True)

MAIN_PATH_POS = (2, 3, 4)
PASSIVE_TYPES = (2, 3, 4, 5)          # R, C, R-C series, R-C parallel
FEEDBACK_GM_TYPES = (8, 9)            # pure gm with feedback direction


def topo_key(g):
    """Topology identity used by the paper's is_same_DAG: (type, sorted preds) per vertex."""
    return tuple((v['type'], tuple(sorted(g.predecessors(v.index)))) for v in g.vs)


def topo_pos_key(g):
    """Topology + position slot (what CktGNN's encoder actually sees, minus device values)."""
    return tuple((v['type'], v['pos'], tuple(sorted(g.predecessors(v.index)))) for v in g.vs)


def full_key(g):
    return tuple((v['type'], v['pos'], v['r'], v['c'], v['gm'],
                  tuple(sorted(g.predecessors(v.index)))) for v in g.vs)


def flat_features(g, max_pos=9):
    """Hand-crafted fixed-length vector: one-hot type per position slot + r/c/gm per slot."""
    type_oh = np.zeros((max_pos, 26))
    vals = np.zeros((max_pos, 3))
    for v in g.vs:
        type_oh[v['pos'], v['type']] = 1
        vals[v['pos']] = (v['r'], v['c'], v['gm'])
    edges = np.zeros((max_pos, max_pos))
    for e in g.es:
        edges[g.vs[e.source]['pos'], g.vs[e.target]['pos']] = 1
    return np.concatenate([type_oh.ravel(), vals.ravel() / 100.0, edges.ravel()])


def main():
    with open(os.path.join(BENCH101, 'ckt_bench_101.pkl'), 'rb') as f:
        train, test = pickle.load(f)
    perf = pd.read_csv(os.path.join(BENCH101, 'perform101.csv'), index_col=0)
    summary = {'n_train': len(train), 'n_test': len(test), 'n_perf_rows': len(perf)}

    G = [p[0] for p in train] + [p[0] for p in test]       # CktGNN (subgraph-level) graphs
    G_raw = [p[1] for p in train] + [p[1] for p in test]  # device-level graphs used by baselines
    is_test = np.array([0] * len(train) + [1] * len(test), bool)

    # ---------- labels ----------
    summary['valid_fraction'] = float(perf['valid'].mean())
    # Recover the constant in fom = 1.2*gain + 1.6*pm + 10*bw + c (gain/pm are min-shifted in the csv)
    resid = perf['fom'] - (1.2 * perf['gain'] + 1.6 * perf['pm'] + 10 * perf['bw'])
    summary['fom_formula_const_mean'] = float(resid.mean())
    summary['fom_formula_const_std'] = float(resid.std())
    var_terms = {'1.2*gain': np.var(1.2 * perf['gain']), '1.6*pm': np.var(1.6 * perf['pm']),
                 '10*bw': np.var(10 * perf['bw'])}
    summary['fom_term_variances'] = {k: float(v) for k, v in var_terms.items()}
    summary['fom_variance'] = float(np.var(perf['fom']))
    summary['corr_fom_bw'] = float(np.corrcoef(perf['fom'], perf['bw'])[0, 1])
    summary['corr_fom_gain'] = float(np.corrcoef(perf['fom'], perf['gain'])[0, 1])
    summary['corr_fom_pm'] = float(np.corrcoef(perf['fom'], perf['pm'])[0, 1])
    summary['label_describe'] = perf[['gain', 'pm', 'bw', 'fom']].describe().round(4).to_dict()

    # ---------- topology statistics ----------
    tk = [topo_key(g) for g in G]
    tpk = [topo_pos_key(g) for g in G]
    fk = [full_key(g) for g in G]
    summary['distinct_topologies_all'] = len(set(tk))
    summary['distinct_topologies_with_pos_all'] = len(set(tpk))
    summary['distinct_full_circuits_all'] = len(set(fk))
    train_topos = set(k for k, t in zip(tk, is_test) if not t)
    test_topos = [k for k, t in zip(tk, is_test) if t]
    summary['distinct_topologies_train'] = len(train_topos)
    summary['distinct_topologies_test'] = len(set(test_topos))
    summary['test_circuits_whose_topology_is_in_train'] = int(sum(k in train_topos for k in test_topos))
    summary['test_topologies_seen_in_train'] = int(len(set(test_topos) & train_topos))
    full_train = set(k for k, t in zip(fk, is_test) if not t)
    summary['test_circuits_exactly_duplicated_in_train'] = int(sum(k in full_train for k, t in zip(fk, is_test) if t))

    # Block structure: the generator re-samples a topology only every K=20 circuits.
    same_as_prev = [tk[i] == tk[i - 1] for i in range(1, len(tk))]
    runs, cur = [], 1
    for s in same_as_prev:
        if s:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    summary['topology_run_length_counts'] = dict(Counter(runs).most_common(8))
    topo_counts = Counter(tk)
    summary['circuits_per_topology_quantiles'] = {
        q: float(np.quantile(list(topo_counts.values()), q)) for q in (0, 0.25, 0.5, 0.75, 1.0)}

    # ---------- graph size / type / position statistics ----------
    summary['subgraph_nodes_per_circuit'] = dict(Counter(g.vcount() for g in G))
    summary['device_nodes_per_circuit'] = dict(Counter(g.vcount() for g in G_raw))
    summary['n_stages'] = dict(Counter(3 if any(v['pos'] == 4 for v in g.vs) else 2 for g in G))
    pos_type = Counter((v['pos'], v['type']) for g in G for v in g.vs)
    pd.DataFrame([{'pos': p, 'type': t, 'count': c} for (p, t), c in sorted(pos_type.items())]) \
        .to_csv(os.path.join(OUT, 'position_type_counts.csv'), index=False)
    # Paper App. E criterion (3): "In the main path, there is no R node nor C node."
    main_passive = sum(any(v['pos'] in MAIN_PATH_POS and v['type'] in PASSIVE_TYPES for v in g.vs) for g in G)
    main_fbgm = sum(any(v['pos'] in MAIN_PATH_POS and v['type'] in FEEDBACK_GM_TYPES for v in g.vs) for g in G)
    summary['dataset_circuits_with_passive_in_main_path'] = int(main_passive)
    summary['dataset_circuits_with_feedback_gm_in_main_path'] = int(main_fbgm)
    vals = {k: [v[k] for g in G for v in g.vs if v[k] != 0] for k in ('r', 'c', 'gm')}
    summary['device_value_ranges'] = {k: [int(min(x)), int(max(x)), len(set(x))] for k, x in vals.items()}

    # ---------- how hard is the prediction task? simple baselines on the same split ----------
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from scipy.stats import pearsonr

    X = np.stack([flat_features(g) for g in G])
    rows = []
    for target in ('gain', 'bw', 'pm', 'fom'):
        y = perf[target].values.astype(float)
        ytr, yte = y[~is_test], y[is_test]
        mu, sd = ytr.mean(), ytr.std()
        ztr, zte = (ytr - mu) / sd, (yte - mu) / sd   # same standardisation as sgp_regression.py
        # (a) predict the training mean
        rows.append({'target': target, 'model': 'train-mean', 'rmse_std': float(np.sqrt(np.mean(zte ** 2))), 'pearson': np.nan})
        # (b) topology lookup: mean label of training circuits with identical topology
        by_topo = pd.Series(ztr).groupby([str(k) for k, t in zip(tk, is_test) if not t]).mean()
        pred = np.array([by_topo.get(str(k), 0.0) for k, t in zip(tk, is_test) if t])
        rows.append({'target': target, 'model': 'topology-lookup', 'rmse_std': float(np.sqrt(np.mean((pred - zte) ** 2))),
                     'pearson': float(pearsonr(pred, zte)[0])})
        # (c) ridge regression and (d) random forest on hand-crafted features
        for name, mdl in (('ridge(hand-feats)', RidgeCV(alphas=np.logspace(-3, 3, 13))),
                          ('random-forest(hand-feats)', RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0))):
            mdl.fit(X[~is_test], ztr)
            pred = mdl.predict(X[is_test])
            rows.append({'target': target, 'model': name, 'rmse_std': float(np.sqrt(np.mean((pred - zte) ** 2))),
                         'pearson': float(pearsonr(pred, zte)[0])})
    base = pd.DataFrame(rows)
    base.to_csv(os.path.join(OUT, 'simple_baselines.csv'), index=False)
    summary['simple_baselines'] = base.round(4).to_dict(orient='records')

    with open(os.path.join(OUT, 'audit_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    sys.exit(main())
