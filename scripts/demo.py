"""Live demo of the project (laptop CPU; no training, no GPU).

Steps (press Enter between them):
  1. A real op-amp from the benchmark: its graph, its simulated frequency response, and the stability check.
  2. The trained model (P) designs new circuits live; each is simulated and judged.
  3. Our improvement: predictor-guided generation vs random generation, measured live.
  4. The paper's objective vs ours: same start, same budget; the paper's "best" circuit is unstable.
Figures are shown in windows and saved to demo_output/ (a backup set is in reports/demo/).

Needs the trained P checkpoints (release models-v1) in the data folder; see HANDOVER.md section 3.3.
Takes about 3 minutes of computing in total; step 3 is the longest (about 2 minutes).
Usage:  .venv/Scripts/python.exe scripts/demo.py            (interactive)
        .venv/Scripts/python.exe scripts/demo.py --auto     (no pauses, e.g. to regenerate the backup figures)
"""
import argparse
import ast
import os
import pickle
import subprocess
import sys
import types
import warnings

import igraph
import matplotlib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, REPO, BENCH101, RESULTS  # noqa: E402
sys.path.insert(0, REPO)
_tv = types.ModuleType('torchvision'); _tv.datasets = _tv.transforms = None
sys.modules.setdefault('torchvision', _tv)
from eval_circuits import build_model, score, dag_key  # noqa: E402
from latent_bo import evaluate  # noqa: E402
from ocb_sim import transfer_function, FREQS, NEG_GM_TYPES, FEEDBACK_TYPES  # noqa: E402

MODEL, MODEL_SEED = 'SubgDVAEprop', 1
BO_SEED, BO_BUDGET = 8, 60                 # an illustrative run; statistics over 40 runs are in Phases 7-8
OUT = os.path.join(ROOT, 'demo_output')
BLUE, ORANGE, GREEN, RED, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#898781', '#0b0b0b'
ARGS = None


# ---------------------------------------------------------------- helpers
def say(text=''):
    print(text, flush=True)


def pause(msg='Press Enter for the next step...'):
    if not ARGS.auto:
        input(f'\n{msg}')


def show(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    say(f'   [figure saved: demo_output/{name}]')
    if not ARGS.auto:
        import matplotlib.pyplot as plt
        plt.show(block=False)
        plt.pause(0.5)


def block_label(v):
    t = int(v['type'])
    base = {0: 'IN', 1: 'OUT', 2: 'R', 3: 'C', 4: 'R–C', 5: 'R‖C'}
    if t in base:
        s = base[t]
    else:
        s = ('−gm' if t in NEG_GM_TYPES else '+gm') + (' fb' if t in FEEDBACK_TYPES else '')
        s += {10: ' ‖C', 14: ' ‖R', 18: ' ‖R‖C', 22: ' +R–C'}.get(t - (t - 10) % 4 if t >= 10 else 0, '')
    return s


def verdict(r):
    if not r['decomposable']:
        return 'not a valid op-amp structure', RED
    if not r['functional']:
        return 'no unity-gain crossing (not functional)', RED
    if not (r['ol_stable'] and r['cl_stable']):
        return f"UNSTABLE (pole in right half-plane), UGF {r['ugf_ghz']:.1f} GHz", RED
    if not r['feasible']:
        return f"stable, but misses spec (gain {r['gain_db']:.0f} dB, PM {r['pm_lag']:.0f}°)", ORANGE
    return f"✓ MEETS SPEC: gain {r['gain_db']:.0f} dB, UGF {r['ugf_ghz']:.1f} GHz, PM {r['pm_lag']:.0f}°", GREEN


def draw_graph(ax, g, title, colour=INK):
    """Circuit graph, left to right in signal-flow order: main amplifier stages in blue, other blocks in grey."""
    n = g.vcount()
    level = [0] * n
    for v in g.topological_sorting():
        for u in g.predecessors(v):
            level[v] = max(level[v], level[u] + 1)
    cols = {}
    for v in range(n):
        cols.setdefault(level[v], []).append(v)
    xy = {}
    for lv, vs in cols.items():
        for i, v in enumerate(vs):
            xy[v] = (lv, (i - (len(vs) - 1) / 2) * 1.2)
    for a, b in g.get_edgelist():
        ax.annotate('', xy=xy[b], xytext=xy[a],
                    arrowprops=dict(arrowstyle='-|>', color=GREY, lw=1.2, shrinkA=16, shrinkB=16))
    pos = g.vs['pos'] if 'pos' in g.vs.attributes() else [9] * n
    for v in range(n):
        main = pos[v] in (2, 3, 4)
        fc = '#d7e7fa' if main else ('#eeeeea' if int(g.vs[v]['type']) > 1 else '#ffffff')
        ax.text(*xy[v], block_label(g.vs[v]), ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.35', fc=fc, ec=BLUE if main else GREY, lw=1.2))
    xs = [p[0] for p in xy.values()]
    ys = [p[1] for p in xy.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    ax.axis('off')
    ax.set_title(title, fontsize=9.5, color=colour)


def draw_bode(ax_m, ax_p, g, r, label='', colour=BLUE):
    H = transfer_function(g)
    ax_m.semilogx(FREQS, 20 * np.log10(np.abs(H)), color=colour, lw=1.8, label=label)
    ax_p.semilogx(FREQS, np.degrees(np.unwrap(np.angle(H))), color=colour, lw=1.8)
    ax_m.axhline(0, color=GREY, lw=0.8, ls='--')
    if r.get('functional'):
        ax_m.axvline(r['ugf_ghz'] * 1e9, color=colour, lw=0.8, ls=':')
    ax_m.set_ylabel('Gain (dB)')
    ax_p.set_ylabel('Phase (°)')
    ax_p.set_xlabel('Frequency (Hz)')
    for a in (ax_m, ax_p):
        a.grid(alpha=0.3)


def graph_from_trace(s):
    t, pos, r, c, gm, edges = ast.literal_eval(s)
    g = igraph.Graph(directed=True)
    g.add_vertices(len(t))
    g.vs['type'], g.vs['pos'], g.vs['r'], g.vs['c'], g.vs['gm'] = list(t), list(pos), list(r), list(c), list(gm)
    g.add_edges(list(edges))
    return g


# ---------------------------------------------------------------- steps
def step1(train, test, labels):
    import matplotlib.pyplot as plt
    say('\n' + '=' * 78)
    say('STEP 1. A real op-amp from the benchmark (Open Circuit Benchmark, 10,000 circuits)')
    say('=' * 78)
    say('Each circuit is a graph: blocks (amplifier stages "gm", resistors R, capacitors C)')
    say('connected in signal-flow order. Our open simulator computes its frequency response and')
    say('checks stability exactly (poles of the circuit).')
    lab = labels[labels.split == 'test'].reset_index(drop=True)
    good = int(lab.index[lab.feasible & (lab.ugf_hz > 5e9)][0])
    bad = int(lab.index[lab.functional & ~lab.ol_stable & (lab.ugf_hz > 5e9)][0])
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), gridspec_kw=dict(height_ratios=[1.1, 1, 0.8]))
    for col, (i, name) in enumerate([(good, 'a working op-amp'), (bad, 'an UNSTABLE one (63% of the benchmark)')]):
        g = test[i][0].copy()
        r = evaluate(g)
        text, colour = verdict(r)
        say(f'   Test circuit {i}: {text}')
        draw_graph(axes[0, col], r['_graph'], f'Real test circuit #{i}: {name}\n{text}', colour)
        draw_bode(axes[1, col], axes[2, col], r['_graph'], r, colour=GREEN if r['feasible'] else RED)
    fig.suptitle('Step 1 · Real circuits from the benchmark, simulated', fontsize=12)
    fig.tight_layout()
    show(fig, 'step1_real_circuits.png')
    say('\nKey point: the paper scores circuits by a "figure of merit" that is ~99% bandwidth and')
    say('ignores stability, although most benchmark circuits are unstable.')


def step2(model, Ztr, train_keys):
    import matplotlib.pyplot as plt
    say('\n' + '=' * 78)
    say('STEP 2. The trained AI model designs new circuits, live')
    say('=' * 78)
    say('The model (a graph variational autoencoder) maps circuits to points in a 66-dimensional')
    say('"latent space". We pick random points and let the decoder build a circuit for each;')
    say('then the simulator judges every circuit.')
    rng = np.random.default_rng(2026)
    z = Ztr.mean(0) + Ztr.std(0) * rng.standard_normal((6, Ztr.shape[1]))
    torch.manual_seed(2026)
    with torch.no_grad():
        G = model.decode(torch.tensor(z, dtype=torch.float32))
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
    for k, (g, ax) in enumerate(zip(G, axes.ravel())):
        r = evaluate(g)
        text, colour = verdict(r)
        novel = 'new design' if dag_key(g) not in train_keys else 'seen in training data'
        say(f'   Design {k + 1}: {text}  [{novel}]')
        draw_graph(ax, r.get('_graph', g), f'AI design {k + 1} ({novel})\n{text}', colour)
    fig.suptitle('Step 2 · Six circuits generated by the AI, each simulated', fontsize=12)
    fig.tight_layout()
    show(fig, 'step2_generated.png')
    say('\nKey point: the AI produces valid-looking, often new circuits, but only a few percent')
    say('of random designs are stable and meet a standard op-amp specification.')


def step3(model, Ztr, train_keys):
    import matplotlib.pyplot as plt
    from guided_generation import POOL, KEEP, decode_score
    say('\n' + '=' * 78)
    say('STEP 3. Our improvement: predictor-guided generation (re-run live)')
    say('=' * 78)
    say('Our model P was trained together with a predictor of "will this circuit meet the spec?".')
    say('Guided generation: score 20,000 latent points with the predictor, keep the best 5%, decode')
    say('them. Control: decode random points from the same pool. This is the exact pre-registered')
    say('Phase 9 experiment for model seed 1: 10,000 circuits per arm. Simulating (about 2 minutes)...')
    rng = np.random.default_rng(9000 + MODEL_SEED)          # identical to scripts/guided_generation.py
    pool = Ztr.mean(0) + Ztr.std(0) * rng.standard_normal((POOL, Ztr.shape[1]))
    with torch.no_grad():
        s = model.prop(torch.tensor(pool, dtype=torch.float32))[:, 3].numpy()
    guided_idx = np.argsort(-s)[:KEEP]
    control_idx = rng.choice(POOL, KEEP, replace=False)
    res = {}
    for name, idx in (('Random generation', control_idx), ('Guided generation (ours)', guided_idx)):
        res[name], _ = decode_score(model, pool[idx], train_keys, True, rng_seed=MODEL_SEED)
        say(f"   {name:26s}: meets spec {100 * res[name]['meets_spec']:.2f}%   stable {100 * res[name]['stable']:.1f}%   "
            f"distinct working designs {res[name]['unique_spec_topologies']}")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    names = list(res)
    for ax, key, title, mult in [(axes[0], 'meets_spec', 'Circuits meeting the spec (%)', 100),
                                 (axes[1], 'stable', 'Stable circuits (%)', 100),
                                 (axes[2], 'unique_spec_topologies', 'Distinct designs meeting the spec', 1)]:
        vals = [res[n][key] * mult for n in names]
        bars = ax.bar(names, vals, color=[GREY, BLUE], width=0.55)
        for b, v in zip(bars, vals):
            ax.annotate(f'{v:.2f}' if mult == 100 else f'{v:.0f}', (b.get_x() + b.get_width() / 2, v),
                        ha='center', va='bottom', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', labelsize=8.5)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Step 3 · Guided vs random generation, 10,000 circuits each (live re-run)', fontsize=12)
    fig.tight_layout()
    show(fig, 'step3_guided.png')
    say('\nKey point: same model, no retraining, but choosing WHERE to sample with the learned')
    say('predictor gives more circuits that meet the spec and more distinct working designs.')
    say('Over all 3 trained seeds: 2.62% vs 1.62% (1.6x), better in every seed, 95% CI of the')
    say('difference +0.45 to +1.56 points (Phase 9, pre-registered).')


def step4():
    import matplotlib.pyplot as plt
    say('\n' + '=' * 78)
    say("STEP 4. Optimisation: the paper's objective vs ours (same start, same budget)")
    say('=' * 78)
    say(f"Both optimisers search the latent space for the fastest op-amp, {BO_BUDGET} simulations each.")
    say("  Paper's method (FoM-BO): maximise the figure of merit, no stability check.")
    say('  Ours (constrained BO): maximise speed SUBJECT TO stable, PM >= 60°, gain >= 60 dB.')
    out = os.path.join(OUT, 'bo')
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    traces = {}
    for method in ('fom_bo', 'cbo'):
        say(f'   running {method} ...', )
        subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'latent_bo.py'), '--model', MODEL,
                        '--method', method, '--seed', str(BO_SEED), '--budget', str(BO_BUDGET), '--out', out],
                       check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        traces[method] = pd.read_csv(os.path.join(out, f'{MODEL}_{method}_seed{BO_SEED}.csv'))
    fom = traces['fom_bo'].loc[traces['fom_bo'].fom.idxmax()]
    cb = traces['cbo'][traces['cbo'].feasible]
    best = cb.loc[cb.ugf_ghz.idxmax()] if len(cb) else None
    picks = [("Paper's method: its best-FoM circuit", fom)] + ([('Our method: best circuit meeting the spec', best)]
                                                               if best is not None else [])
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), gridspec_kw=dict(height_ratios=[1.1, 1, 0.8]))
    for col, (title, row) in enumerate(picks):
        g = graph_from_trace(row.circuit)
        r = evaluate(g)
        text, colour = verdict(r)
        say(f'   {title}: {text}')
        draw_graph(axes[0, col], r['_graph'], f'{title}\n{text}', colour)
        draw_bode(axes[1, col], axes[2, col], r['_graph'], r, colour=GREEN if r['feasible'] else RED)
    fig.suptitle(f"Step 4 · Paper's objective vs ours, {BO_BUDGET} simulations each (live run)", fontsize=12)
    fig.tight_layout()
    show(fig, 'step4_optimisation.png')
    say(f"\n   Feasible circuits proposed in this run: paper's method {int(traces['fom_bo'].feasible.sum())}, "
        f"ours {int(traces['cbo'].feasible.sum())} (of {BO_BUDGET})")
    t = pd.read_csv(os.path.join(RESULTS, 'summary', 'bo5_table.csv'))
    t = t[t.space == MODEL].set_index('method')
    say('\nAcross 20 full runs per method (Phases 5-8), in this latent space:')
    for m, name in [('rs', 'random search'), ('fom_bo', "paper's FoM-BO"), ('cbo', 'constrained BO (ours)')]:
        say(f"   {name:24s} {100 * t.loc[m, 'frac_evals_feasible']:5.1f}% of simulations meet the spec; "
            f"found a spec-meeting circuit in {int(t.loc[m, 'runs_with_feasible'])}/20 runs")
    say(f"   The circuit the paper's method ranks best is stable in only "
        f"{int(round(20 * t.loc['fom_bo', 'max_fom_circuit_stable']))}/20 runs.")
    say("\nKey point: the paper's objective rewards fast but UNSTABLE circuits (they would oscillate);")
    say('adding stability as a constraint finds working circuits instead.')


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--auto', action='store_true', help='no pauses and no windows (regenerates the figures)')
    ap.add_argument('--steps', default='1234')
    ARGS = ap.parse_args()
    warnings.filterwarnings('ignore')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    matplotlib.use('Agg' if ARGS.auto else 'TkAgg')
    torch.set_num_threads(4)
    say('\nCktGNN replication project: live demo')
    say('Paper: Dong et al., "CktGNN", ICLR 2023. Everything below runs live on this laptop (CPU).')
    with open(os.path.join(BENCH101, 'ckt_bench_101.pkl'), 'rb') as f:
        train, test = pickle.load(f)
    labels = pd.read_csv(os.path.join(RESULTS, 'simulator', 'analysis_all.csv'))
    train_keys = {dag_key(x[0]) for x in train}
    run = os.path.join(RESULTS, 'vae', f'{MODEL}_seed{MODEL_SEED}')
    model = build_model(MODEL)
    model.load_state_dict(torch.load(os.path.join(run, 'last.pt'), map_location='cpu', weights_only=False)['model'])
    model.eval()
    Ztr = np.load(os.path.join(run, 'latents.npz'))['Z_train']
    steps = {'1': lambda: step1(train, test, labels), '2': lambda: step2(model, Ztr, train_keys),
             '3': lambda: step3(model, Ztr, train_keys), '4': step4}
    for i, k in enumerate(ARGS.steps):
        if i:
            pause()
        steps[k]()
    say('\nDemo finished. Figures: demo_output/. Full results: reports/ and HANDOVER.md.')
    if not ARGS.auto:
        input('Press Enter to close.')


if __name__ == '__main__':
    main()
