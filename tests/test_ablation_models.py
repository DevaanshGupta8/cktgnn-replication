"""Consistency test for the ablation models A1/A2/A3 (scripts/ablation_models.py).

There is no reference implementation to compare against, so this checks invariants instead:
  1. batch independence: the loss of a mixed-size batch equals the sum of the per-graph losses
     (exercises the masked teacher-forcing loop and the closed-form sum over padded positions);
     the same holds for mu/logvar;
  2. A3 really uses slots in the messages and the edge scorer, with CktGNN's layer shapes;
  3. decode runs and returns graphs with slots for A2/A3/P;
  4. P (property head) keeps batch independence, with labels that include non-functional (NaN) targets.

Usage:  .venv/Scripts/python.exe tests/test_ablation_models.py
"""
import os
import pickle
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from paths import REPO, BENCH101  # noqa: E402
sys.path.insert(0, REPO)
_tv = types.ModuleType('torchvision'); _tv.datasets = _tv.transforms = None
sys.modules.setdefault('torchvision', _tv)
from ablation_models import SubgDVAE  # noqa: E402

torch.set_num_threads(2)
with open(os.path.join(BENCH101, 'ckt_bench_101.pkl'), 'rb') as f:
    train, _ = pickle.load(f)
G = [x[0] for x in train[:200]]
batch = sorted(G, key=lambda g: g.vcount())[::25][:8]   # mixed vertex counts
assert len({g.vcount() for g in batch}) > 1
for i, g in enumerate(batch):                    # synthetic property labels; every other circuit "non-functional"
    g['props'] = [i % 2, 1, 0, (i // 2) % 2] + ([0.5 * i, -0.3, 1.2] if i % 2 else [float('nan')] * 3)

VARIANTS = {'A1': dict(use_pos=False), 'A2': dict(use_pos=True), 'A3': dict(use_pos=True, slot_struct=True),
            'P': dict(use_pos=True, n_props=7)}
ok = True
for name, kw in VARIANTS.items():
    torch.manual_seed(0)
    m = SubgDVAE(**kw).eval()          # eval: reparameterize returns mu
    with torch.no_grad():
        mu, lv = m.encode(batch)
        out = m.loss(mu, lv, batch)
        single = [m.loss(*m.encode([g]), [g]) for g in batch]
        mus = torch.cat([m.encode([g])[0] for g in batch])
    errs = [abs(float(out[i]) - sum(float(s[i]) for s in single)) / max(1.0, abs(float(out[i])))
            for i in range(len(out))]
    err_mu = float((mu - mus).abs().max())
    good = max(errs) < 1e-5 and err_mu < 1e-5   # float32
    ok &= good
    n_par = sum(p.numel() for p in m.parameters())
    print(f'{name}: {n_par} params, batch-vs-single rel. err {max(errs):.1e}, mu err {err_mu:.1e} '
          f'{"OK" if good else "FAIL"}')

    if name == 'A3':                             # slots must reach messages and edge scorer, as in CktGNN
        torch.set_grad_enabled(False)
        S = torch.randn(1, 3, m.hs)
        p1, p2 = torch.tensor([[0, 2, 5]]), torch.tensor([[0, 3, 5]])
        d_msg = float((m._msg_s(S, p1) - m._msg_s(S, p2)).abs().max())
        e = lambda p: m.add_edge(torch.cat([m._e(S[:, 1], p[:, 1]), m._e(S[:, 2], p[:, 2])], -1))  # noqa: E731
        d_edge = float((e(p1) - e(p2)).abs().max())
        shapes = (m.add_edge[0].in_features == 2 * (m.hs + m.n_pos) and m.gate_forward[0].in_features == m.hs + m.n_pos)
        good = d_msg > 1e-6 and d_edge > 1e-6 and shapes
        ok &= good
        print(f'A3 slot wiring: message change {d_msg:.2e}, edge-score change {d_edge:.2e}, '
              f'CktGNN layer shapes {shapes} {"OK" if good else "FAIL"}')
        torch.set_grad_enabled(True)

    with torch.no_grad():
        D = m.decode(torch.randn(16, m.nz))
    has_pos = all('pos' in g.vs.attributes() for g in D)
    print(f'{name}: decoded {len(D)} graphs, sizes {sorted({g.vcount() for g in D})}, slots {has_pos}')

print('ALL CONSISTENT' if ok else 'FAILURES')
sys.exit(0 if ok else 1)
