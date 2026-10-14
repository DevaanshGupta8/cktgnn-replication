"""Simulator labels for every Ckt-Bench-101 circuit (train then test, the order of perform101.csv).

Runs ocb_sim.analyze on each real circuit (true slots and device values) and adds the Phase 4 specification
flags. Used by the property-aware VAE (Phase 6) and as the reference for circuit-level statistics.

Writes <DATA_ROOT>/results/simulator/analysis_all.csv
Usage:  .venv/Scripts/python.exe scripts/circuit_labels.py
"""
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BENCH101, RESULTS  # noqa: E402
from ocb_sim import analyze  # noqa: E402

PM_MIN, GAIN_MIN = 60.0, 60.0                    # the Phase 4 specification (latent_bo.py)
COLS = ['gain_db', 'ugf_hz', 'phase_ugf_wrapped', 'phase_ugf_unwrapped', 'phase_dc', 'n_crossings', 'pm_lag',
        'dc_gain', 'ol_stable', 'cl_stable', 'ol_worst_rel_re', 'cl_worst_rel_re']


def main():
    warnings.filterwarnings('ignore')
    with open(os.path.join(BENCH101, 'ckt_bench_101.pkl'), 'rb') as f:
        train, test = pickle.load(f)
    rows = [{k: analyze(x[0])[k] for k in COLS} for x in train + test]
    df = pd.DataFrame(rows)
    df['functional'] = np.isfinite(df.ugf_hz) & np.isfinite(df.gain_db)
    df['feasible'] = df.functional & df.ol_stable & df.cl_stable & (df.pm_lag >= PM_MIN) & (df.gain_db >= GAIN_MIN)
    df['split'] = ['train'] * len(train) + ['test'] * len(test)
    out = os.path.join(RESULTS, 'simulator', 'analysis_all.csv')
    df.to_csv(out, index=False)
    print(f'{len(df)} circuits -> {out}')
    print(df.groupby('split')[['functional', 'ol_stable', 'cl_stable', 'feasible']].mean().round(4))


if __name__ == '__main__':
    main()
