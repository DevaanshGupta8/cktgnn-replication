"""Check whether the open-source simulator (ocb_sim.py) reproduces the released OCB labels.

perform101.csv stores gain = gain_dB/100 - min(.) + 1e-5, pm = PM/(-90) - min(.) + 1e-5,
bw = UGF/1e9 (unshifted), so BW can be compared directly and gain/PM up to a constant offset.
Several netlist conventions are tried; the one that matches best is reported.

Usage:  .venv/Scripts/python.exe scripts/validate_simulator.py [--n 10000]
Writes: <DATA_ROOT>/results/simulator/validation_<variant>.csv, validation_summary.json, label_diagnostics.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocb_sim import transfer_function, measure  # noqa: E402
from paths import BENCH101, RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, 'simulator')
os.makedirs(OUT, exist_ok=True)

VARIANTS = {
    'spectre_sign': dict(gm_sign=-1.0, pos7_to_ground=False),
    'flipped_sign': dict(gm_sign=+1.0, pos7_to_ground=False),
    'spectre_sign_pos7gnd': dict(gm_sign=-1.0, pos7_to_ground=True),
}


def affine_fit(x, y):
    """Best y ~ x + c (slope fixed at 1); returns offset and residual stats."""
    m = np.isfinite(x) & np.isfinite(y)
    c = np.median(y[m] - x[m])
    r = y[m] - x[m] - c
    return float(c), float(np.median(np.abs(r))), float(np.mean(np.abs(r) < 1e-2 * max(1.0, np.std(y[m]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10000)
    args = ap.parse_args()
    with open(os.path.join(BENCH101, 'ckt_bench_101.pkl'), 'rb') as f:
        train, test = pickle.load(f)
    G = ([p[0] for p in train] + [p[0] for p in test])[:args.n]
    perf = pd.read_csv(os.path.join(BENCH101, 'perform101.csv'), index_col=0).iloc[:args.n]

    summary = {}
    for name, kw in VARIANTS.items():
        rows = [measure(transfer_function(g, **kw)) for g in G]
        df = pd.DataFrame(rows)
        df['bw_label'] = perf['bw'].values
        df['gain_label'] = perf['gain'].values
        df['pm_label'] = perf['pm'].values
        df['bw_sim'] = df['ugf_hz'] / 1e9
        df.to_csv(os.path.join(OUT, f'validation_{name}.csv'), index=False)

        bw_ok = np.isclose(df['bw_sim'].fillna(0), df['bw_label'], rtol=1e-2, atol=1e-4)
        rel = np.abs(df['bw_sim'] - df['bw_label']) / np.maximum(df['bw_label'], 1e-9)
        s = {'bw_match_rate_1pct': float(bw_ok.mean()),
             'bw_median_rel_err': float(np.nanmedian(rel)),
             'bw_pearson': float(df[['bw_sim', 'bw_label']].corr().iloc[0, 1]),
             'n_no_unity_crossing_sim': int(df['ugf_hz'].isna().sum()),
             'n_bw_label_zero': int((df['bw_label'] == 0).sum())}
        c, mad, frac = affine_fit(df['gain_db'].values / 100, df['gain_label'].values)
        s['gain_offset'] = c; s['gain_median_abs_resid'] = mad; s['gain_match_rate'] = frac
        s['gain_pearson'] = float(np.corrcoef(df['gain_db'], df['gain_label'])[0, 1])
        for ph in ('phase_ugf_wrapped', 'phase_ugf_unwrapped'):
            for pm_def, pm in (('180+phase', 180 + df[ph]), ('phase', df[ph]), ('-phase', -df[ph])):
                x = (pm / -90).values
                c, mad, frac = affine_fit(x, df['pm_label'].values)
                s[f'pm[{pm_def},{ph}]'] = {'offset': c, 'median_abs_resid': mad, 'match_rate': frac}
        summary[name] = s
        print(name, json.dumps(s, indent=1))
    with open(os.path.join(OUT, 'validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    if args.n == len(perf) == 10000:
        label_diagnostics(G, perf)


def label_diagnostics(G, perf, offset_gain=1.1953102):
    """Placeholder rows, multi-crossing ambiguity, PM convention and stability statistics
    for the best-matching (Spectre-sign) netlist convention."""
    from ocb_sim import FREQS
    df = pd.read_csv(os.path.join(OUT, 'validation_spectre_sign.csv'))
    placeholder = np.isclose(df.gain_label, offset_gain, atol=2e-6) & (df.bw_label < 1e-3)
    real = ~placeholder
    bw_ok = np.isclose(df.bw_sim.fillna(0), df.bw_label, rtol=1e-2, atol=1e-4)
    gain_ok = np.abs(df.gain_label - (df.gain_db / 100 + offset_gain)) < 1e-3
    ok = real & bw_ok & gain_ok
    # BW mismatches explained by Ocean picking a different 0-dB crossing
    lf = np.log10(FREQS)
    other_crossing = 0
    for i in np.where(real & ~bw_ok)[0]:
        m = 20 * np.log10(np.abs(transfer_function(G[i])))
        ks = np.where(np.sign(m[:-1]) != np.sign(m[1:]))[0]
        xs = [10 ** (lf[k] + m[k] / (m[k] - m[k + 1]) * (lf[k + 1] - lf[k])) / 1e9 for k in ks]
        other_crossing += any(abs(x - df.bw_label[i]) / df.bw_label[i] < 1e-2 for x in xs)
    d = df[ok].copy()
    d['fom'] = perf['fom'].values[ok]
    # label pm = phase(UGF)/(-90) + shift, i.e. -90*pm_label == phase (mod 360)
    r = (-90 * d.pm_label - d.phase_ugf_unwrapped).values
    pm_mod360 = float(np.mean(np.abs(((r + 720) + 180) % 360 - 180) < 0.5))
    # indicative margin: 180 deg minus the phase lag accumulated from DC to the UGF
    d['pm_lag'] = 180 - (d.phase_dc - d.phase_ugf_unwrapped)
    q90, q99 = d.fom.quantile(0.9), d.fom.quantile(0.99)
    out = {
        'placeholder_rows': int(placeholder.sum()),
        'placeholder_rows_without_sim_down_crossing': int((placeholder & df.ugf_hz.isna()).sum()),
        'non_placeholder_rows': int(real.sum()),
        'bw_and_gain_match': int(ok.sum()),
        'gain_match': int((real & gain_ok).sum()),
        'bw_mismatch': int((real & ~bw_ok).sum()),
        'bw_mismatch_matching_another_crossing': int(other_crossing),
        'pm_label_equals_phase_at_ugf_mod360': pm_mod360,
        'corr_pm_label_vs_pm_lag': float(np.corrcoef(d.pm_label, d.pm_lag)[0, 1]),
        'median_gain_db_matched': float(d.gain_db.median()),
        'frac_pm_lag_below_45': {'all_matched': float(np.mean(d.pm_lag < 45)),
                                 'top10pct_fom': float(np.mean(d[d.fom >= q90].pm_lag < 45)),
                                 'top1pct_fom': float(np.mean(d[d.fom >= q99].pm_lag < 45))},
        'frac_pm_lag_below_0': {'all_matched': float(np.mean(d.pm_lag < 0)),
                                'top10pct_fom': float(np.mean(d[d.fom >= q90].pm_lag < 0))},
    }
    with open(os.path.join(OUT, 'label_diagnostics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
