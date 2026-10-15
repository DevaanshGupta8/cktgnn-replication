"""Reproduction runner for the CktGNN / D-VAE VAE experiments on Ckt-Bench-101.

Mirrors external/CktGNN/main.py, but
  * imports only layers/models_ig.py (so torch_geometric/torch_sparse are not needed for
    CktGNN and DVAE); torchvision, imported but unused by models_ig.py, is stubbed;
  * removes the trailing pdb.set_trace() and the deletion of old result files;
  * exposes the paper-vs-repo discrepancies as flags (--epochs, --patience);
  * logs per-epoch losses to CSV and evaluation results to JSON;
  * saves a resumable state every epoch (--resume) and the test/train latent means for the
    later SGP regression;
  * by default trains with the batched, numerically equivalent encode/loss of scripts/fast_models.py
    (--impl original uses the authors' per-graph code; see tests/test_fast_models.py);
  * computes the novelty ratio with an exact hash of what utils.is_same_DAG compares
    (vertex count, per-index type, per-index in-neighbour set) instead of an O(N*M) scan.
The authors' files are not modified.

Examples
  smoke test : python scripts/run_vae.py --model CktGNN --epochs 1 --max-batches 20 --skip-eval
  repo setup : python scripts/run_vae.py --model CktGNN --epochs 300 --patience 10 --seed 1
  paper setup: python scripts/run_vae.py --model CktGNN --epochs 200 --patience 20 --seed 1
  evaluate   : python scripts/run_vae.py --model CktGNN --seed 1 --eval-only
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO, BENCH101, RESULTS  # noqa: E402
sys.path.insert(0, REPO)
_tv = types.ModuleType('torchvision')
_tv.datasets, _tv.transforms = types.ModuleType('datasets'), types.ModuleType('transforms')
sys.modules.setdefault('torchvision', _tv)

import utils as ckt_utils  # noqa: E402  (authors' utils.py)
from layers.models_ig import CktGNN, DVAE  # noqa: E402
from fast_models import FastCktGNN, FastDVAE  # noqa: E402
from ablation_models import SubgDVAE  # noqa: E402
from utils import is_same_DAG, prior_validity  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--model', default='CktGNN', choices=['CktGNN', 'DVAE', 'SubgDVAE', 'SubgDVAEpos', 'SubgDVAEslot', 'SubgDVAEprop', 'SubgDVAEpropn'],
               help='SubgDVAE / SubgDVAEpos: ablations A1 / A2 (D-VAE on the subgraph DAG without / with slots); '
                    'SubgDVAEslot: A3 (A2 plus slots in messages and edge scorer, as in CktGNN); '
                    'SubgDVAEprop: A2 plus a jointly trained property predictor (Phase 6); '
                    'SubgDVAEpropn: SubgDVAEprop with normalised device-value inputs (Phase 9)')
p.add_argument('--prop-weight', type=float, default=1.0, help='weight of the property loss (SubgDVAEprop)')
p.add_argument('--epochs', type=int, default=300)
p.add_argument('--patience', type=int, default=10, help='ReduceLROnPlateau patience (repo: 10, paper: 20)')
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--batch-size', type=int, default=64)
p.add_argument('--infer-batch-size', type=int, default=128)
p.add_argument('--hs', type=int, default=301)
p.add_argument('--nz', type=int, default=66)
p.add_argument('--emb-dim', type=int, default=24)
p.add_argument('--feat-emb-dim', type=int, default=8)
p.add_argument('--no-pos', action='store_true',
               help="authors' pos=False: removes the slot one-hot from messages only; the GRU input, "
                    "decoder slot head and edge MLP still use slots (a partial ablation)")
p.add_argument('--seed', type=int, default=1)
p.add_argument('--impl', default='fast', choices=['fast', 'original'],
               help="fast: batched encode/loss (same parameters and equations); original: authors' per-graph code")
p.add_argument('--device', default='cpu',
               help='with --impl fast the GPU is fastest (RTX 3050: CktGNN 0.13 s, D-VAE 0.25 s per step); '
                    'with --impl original the CPU is faster')
p.add_argument('--eval-device', default='cpu',
               help="device for the evaluation decodes; the authors' per-graph decoder is ~15x faster on CPU than GPU")
p.add_argument('--threads', type=int, default=None, help='torch CPU threads (set when running jobs in parallel)')
p.add_argument('--max-batches', type=int, default=None, help='limit batches per epoch (smoke tests)')
p.add_argument('--skip-eval', action='store_true')
p.add_argument('--eval-only', action='store_true', help='load last.pt from the run folder and only evaluate')
p.add_argument('--resume', action='store_true', help='continue training from last.pt in the run folder')
p.add_argument('--save-interval', type=int, default=50)
p.add_argument('--out', default=None)
args = p.parse_args()

if args.threads:
    torch.set_num_threads(args.threads)
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
device = torch.device(args.device)
tag = f"{args.model}{'_nopos' if args.no_pos else ''}_seed{args.seed}"
out_dir = args.out or os.path.join(RESULTS, 'vae', tag)
os.makedirs(out_dir, exist_ok=True)
last_path = os.path.join(out_dir, 'last.pt')
log_path = os.path.join(out_dir, 'train_log.csv')

with open(os.path.join(BENCH101, 'ckt_bench_101.pkl'), 'rb') as f:
    train_set, test_set = pickle.load(f)
perf = pd.read_csv(os.path.join(BENCH101, 'perform101.csv'), index_col=0)
k = 1 if args.model == 'DVAE' else 0            # D-VAE uses the device DAGs, the others the subgraph DAGs
train_data = [x[k] for x in train_set]
test_data = [x[k] for x in test_set]
del train_set, test_set                          # keep only the graph level this model uses
train_eval = list(train_data)                    # fixed order (train_data is shuffled in place)
if args.model.startswith('SubgDVAEprop'):                 # simulator labels, same order as perform101.csv (train, then test)
    lab = pd.read_csv(os.path.join(RESULTS, 'simulator', 'analysis_all.csv'))
    flags = lab[['functional', 'ol_stable', 'cl_stable', 'feasible']].astype(float).values
    reg = np.column_stack([np.log10(lab.ugf_hz), lab.gain_db, lab.pm_lag]).astype(float)
    reg[~lab.functional.values] = np.nan
    tr = reg[:len(train_eval)]
    prop_mean, prop_std = np.nanmean(tr, 0), np.nanstd(tr, 0)   # train statistics only
    reg = np.clip((reg - prop_mean) / prop_std, -3, 3)
    props = np.column_stack([flags, reg])
    for g, y in zip(train_eval + test_data, props):
        g['props'] = y.tolist()

torch.backends.cuda.matmul.allow_tf32 = False   # plain fp32, as on the paper's P100/1080 Ti
torch.backends.cudnn.allow_tf32 = False
Ckt, Dvae = (FastCktGNN, FastDVAE) if args.impl == 'fast' else (CktGNN, DVAE)
if args.model == 'CktGNN':                      # constants hard-coded in main.py
    model = Ckt(max_n=8, max_pos=8, nvt=26, subn_nvt=40, START_TYPE=0, END_TYPE=1,
                   emb_dim=args.emb_dim, feat_emb_dim=args.feat_emb_dim, hs=args.hs, nz=args.nz,
                   pos=not args.no_pos)
    subg = True
elif args.model.startswith('SubgDVAE'):
    model = SubgDVAE(max_n=8, nvt=26, START_TYPE=0, END_TYPE=1, hs=args.hs, nz=args.nz,
                     use_pos=args.model != 'SubgDVAE', slot_struct=args.model == 'SubgDVAEslot',
                     n_props=7 if args.model.startswith('SubgDVAEprop') else 0, prop_weight=args.prop_weight,
                     norm_values=args.model == 'SubgDVAEpropn')
    subg = True
else:
    model = Dvae(max_n=24, nvt=10, feat_nvt=103, START_TYPE=8, END_TYPE=9, hs=args.hs, nz=args.nz)
    subg = False
model.to(device)
n_params = sum(p_.numel() for p_ in model.parameters())
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=args.patience)


def save_state(epoch):
    torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng_torch': torch.get_rng_state(),
                'rng_numpy': np.random.get_state(), 'rng_python': random.getstate()}, last_path + '.tmp')
    os.replace(last_path + '.tmp', last_path)


def load_state():
    st = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(st['model']); optimizer.load_state_dict(st['optimizer'])
    scheduler.load_state_dict(st['scheduler'])
    torch.set_rng_state(st['rng_torch'].cpu())  # map_location may have moved it to the GPU
    np.random.set_state(st['rng_numpy']); random.setstate(st['rng_python'])
    return st['epoch']


def train_epoch():
    model.train()
    random.shuffle(train_data)
    tot = {'loss': 0., 'recon': 0., 'kld': 0., 'type': 0., 'pos': 0., 'feat': 0., 'prop': 0.}
    batch, nb, t0 = [], 0, time.time()
    for i, g in enumerate(train_data):
        batch.append(g)
        if len(batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            gb = model._collate_fn(batch)
            mu, logvar = model.encode(gb)
            loss, recon, kld, type_l, pos_l, feat_l = model.loss(mu, logvar, gb)
            loss.backward()
            optimizer.step()
            for key, val in zip(tot, (loss, recon, kld, -type_l, -pos_l, feat_l, getattr(model, 'last_prop_loss', 0.))):
                tot[key] += float(val.detach()) if torch.is_tensor(val) else float(val)
            batch, nb = [], nb + 1
            if args.max_batches and nb >= args.max_batches:
                break
    n = min(len(train_data), nb * args.batch_size)
    return {k_: v / n for k_, v in tot.items()}, time.time() - t0, nb


@torch.no_grad()
def recon_accuracy(encode_times=10, decode_times=10):
    """Same protocol as main.py:test() (note: in eval mode reparameterize returns mu,
    so the encode_times repetitions are identical; only decoding is stochastic)."""
    model.eval()
    n_perfect, nll, batch = 0, 0., []
    for i, g in enumerate(test_data):
        batch.append(g)
        if len(batch) == args.infer_batch_size or i == len(test_data) - 1:
            gb = model._collate_fn(batch)
            mu, logvar = model.encode(gb)
            _, r, *_ = model.loss(mu, logvar, gb)
            nll += float(r)
            for _ in range(encode_times):
                z = model.reparameterize(mu, logvar)
                for _ in range(decode_times):
                    n_perfect += sum(is_same_DAG(a, b) for a, b in zip(gb, model.decode(z)))
            batch = []
    return nll / len(test_data), n_perfect / (len(test_data) * encode_times * decode_times)


def dag_key(g):
    """Exactly the information utils.is_same_DAG compares."""
    return (g.vcount(),) + tuple((int(g.vs[i]['type']), frozenset(g.neighbors(i, 'in'))) for i in range(g.vcount()))


def fast_ratio_same_DAG(G0, G1):
    """Drop-in replacement for utils.ratio_same_DAG (fraction of G1 found in G0)."""
    keys = {dag_key(g) for g in G0}
    return sum(dag_key(g) in keys for g in G1) / len(G1)


@torch.no_grad()
def save_latents():
    """Latent means for train/test plus labels, as consumed by sgp_regression.py."""
    model.eval()
    Z = [ckt_utils.extract_latent_z(d, model, 'igraph', 0, args.infer_batch_size) for d in (train_eval, test_data)]
    lab = perf[['gain', 'bw', 'pm', 'fom']].values
    np.savez(os.path.join(out_dir, 'latents.npz'), Z_train=Z[0], Z_test=Z[1],
             Y_train=lab[:len(train_eval)], Y_test=lab[len(train_eval):], label_names=np.array(['gain', 'bw', 'pm', 'fom']))


start_epoch = 0
if args.resume or args.eval_only:
    start_epoch = load_state()
    print(f'loaded {last_path} (epoch {start_epoch})')
else:
    json.dump(vars(args), open(os.path.join(out_dir, 'config.json'), 'w'), indent=2)
    with open(log_path, 'w') as f:
        f.write('epoch,lr,seconds,loss,recon,kld,type,pos,feat' + (',prop' if args.model.startswith('SubgDVAEprop') else '') + '\n')
print(f'{tag}: {n_params} parameters, device={device}, threads={torch.get_num_threads()}')

if not args.eval_only:
    for epoch in range(start_epoch + 1, args.epochs + 1):
        stats, secs, nb = train_epoch()
        scheduler.step(stats['loss'])
        lr_now = optimizer.param_groups[0]['lr']
        cols = [k_ for k_ in stats if k_ != 'prop' or args.model.startswith('SubgDVAEprop')]   # old log format otherwise
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{lr_now},{secs:.2f}," + ','.join(f'{stats[k_]:.5f}' for k_ in cols) + '\n')
        print(f"epoch {epoch} ({nb} batches, {secs:.1f}s) " + ' '.join(f'{k_}={v:.4f}' for k_, v in stats.items()), flush=True)
        save_state(epoch)
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), os.path.join(out_dir, f'model_checkpoint{epoch}.pth'))

results = {'n_params': n_params, 'epochs': args.epochs, 'device': str(device), 'eval_device': args.eval_device}
if not args.skip_eval:
    if torch.device(args.eval_device) != device:
        device = torch.device(args.eval_device)
        model.to(device)
        model.device = None                          # models_ig caches the device on first use
    t0 = time.time()
    save_latents()
    results['test_recon_nll'], results['recon_acc'] = recon_accuracy()
    results['recon_seconds'] = time.time() - t0
    ckt_utils.ratio_same_DAG = fast_ratio_same_DAG   # prior_validity looks this name up at call time
    with torch.no_grad():
        v_dag, v_ckt, novel = prior_validity(train_eval, model, infer_batch_size=args.infer_batch_size,
                                             data_type='igraph', subg=subg, device=device, scale_to_train_range=True)
    results.update({'valid_dag': v_dag, 'valid_ckt': v_ckt, 'novel': novel, 'eval_seconds': time.time() - t0})
    if args.model.startswith('SubgDVAEprop'):             # predictor accuracy on the test circuits (from the latent mean)
        from sklearn.metrics import roc_auc_score
        model.eval()
        Zt = torch.tensor(np.load(os.path.join(out_dir, 'latents.npz'))['Z_test'], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model.prop(Zt).cpu().numpy()
        Y = np.array([g['props'] for g in test_data])
        for j, name in enumerate(['functional', 'ol_stable', 'cl_stable', 'feasible']):
            results[f'prop_auc_{name}'] = float(roc_auc_score(Y[:, j], out[:, j]))
        for j, name in enumerate(['log10_ugf', 'gain', 'pm'], start=4):
            ok = np.isfinite(Y[:, j])
            results[f'prop_rmse_{name}'] = float(np.sqrt(np.mean((out[ok, j] - Y[ok, j]) ** 2)))   # standardised units
json.dump(results, open(os.path.join(out_dir, 'results.json'), 'w'), indent=2)
print(json.dumps(results, indent=2))
