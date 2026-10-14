"""Equivalence test: batched FastCktGNN/FastDVAE vs the authors' CktGNN/DVAE.

Same weights, same batches, same RNG seed for the reparameterisation noise; compares
mu/logvar, every returned loss term, and the gradient of every parameter.
Also reports the speed-up per training step.

Usage:  .venv/Scripts/python.exe tests/test_fast_models.py [--device cpu|cuda]
"""
import argparse
import os
import pickle
import sys
import time
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from paths import REPO, BENCH101  # noqa: E402
sys.path.insert(0, REPO)
_tv = types.ModuleType('torchvision'); _tv.datasets = _tv.transforms = None
sys.modules.setdefault('torchvision', _tv)
from layers.models_ig import CktGNN, DVAE  # noqa: E402
from fast_models import FastCktGNN, FastDVAE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cpu')
ap.add_argument('--threads', type=int, default=2)
args = ap.parse_args()
torch.set_num_threads(args.threads)
dev = torch.device(args.device)

with open(os.path.join(BENCH101, 'ckt_bench_101.pkl'), 'rb') as f:
    train, test = pickle.load(f)

CFG = {
    'CktGNN': (0, CktGNN, FastCktGNN, dict(max_n=8, max_pos=8, nvt=26, subn_nvt=40, START_TYPE=0, END_TYPE=1,
                                            emb_dim=24, feat_emb_dim=8, hs=301, nz=66)),
    'DVAE': (1, DVAE, FastDVAE, dict(max_n=24, nvt=10, feat_nvt=103, START_TYPE=8, END_TYPE=9, hs=301, nz=66)),
}


def run(model, batch, seed):
    model.zero_grad()
    gb = model._collate_fn(batch)
    mu, logvar = model.encode(gb)
    torch.manual_seed(seed)
    out = model.loss(mu, logvar, gb)
    out[0].backward()
    grads = {k: (p.grad.detach().clone() if p.grad is not None else None) for k, p in model.named_parameters()}
    return mu.detach(), logvar.detach(), [float(o) for o in out], grads


def max_rel(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-12))


ok_all = True
for name, (k, Orig, Fast, kw) in CFG.items():
    torch.manual_seed(0)
    orig = Orig(**kw).to(dev)
    fast = Fast(**kw).to(dev)
    fast.load_state_dict(orig.state_dict())
    for m in (orig, fast):
        m.train()
    batches = {'train[0:64]': [x[k] for x in train[:64]],
               'test[100:164]': [x[k] for x in test[100:164]],
               'train[5000:5007]': [x[k] for x in train[5000:5007]]}
    for bname, batch in batches.items():
        mu_o, lv_o, out_o, g_o = run(orig, batch, seed=123)
        mu_f, lv_f, out_f, g_f = run(fast, batch, seed=123)
        loss_err = max(abs(a - b) / max(1.0, abs(a)) for a, b in zip(out_o, out_f))
        mu_err = max(max_rel(mu_f, mu_o), max_rel(lv_f, lv_o))
        grad_err = max(max_rel(g_f[p], g_o[p]) for p in g_o if g_o[p] is not None)
        missing = [p for p in g_o if (g_o[p] is None) != (g_f[p] is None)]
        ok = loss_err < 1e-5 and mu_err < 1e-5 and grad_err < 1e-3 and not missing   # grads: float32 rounding on saturated sigmoids
        ok_all &= ok
        print(f'{name:6s} {bname:16s} loss terms rel.err {loss_err:.1e} | mu/logvar {mu_err:.1e} | '
              f'grads {grad_err:.1e} | {"OK" if ok else "MISMATCH"}  (loss {out_o[0]:.4f} vs {out_f[0]:.4f})')
        if missing:
            print('   gradient presence differs for', missing)
    # timing of a full training step (encode + loss + backward)
    batch = [x[k] for x in train[64:128]]
    for label, m in (('original', orig), ('fast', fast)):
        run(m, batch, 1)
        t = time.time()
        for s in range(3):
            run(m, batch, s)
        print(f'{name:6s} {label:8s} {(time.time() - t) / 3:.3f} s per training step on {args.device}')
print('ALL EQUIVALENT' if ok_all else 'SOME MISMATCHES')
sys.exit(0 if ok_all else 1)
