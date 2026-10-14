"""Ablation models for the CktGNN decomposition study (Phase 3, R1).

SubgDVAE: the authors' D-VAE equations (gated-sum GRU encoder, autoregressive decoder, vertex-id
messages, same losses and weights) applied to CktGNN's *subgraph* DAG instead of the device DAG.
Each node's input is onehot(subgraph type, 26) concatenated with its raw device values [r, c, gm]
(D-VAE feeds its raw scalar value the same way), and the decoder predicts [r, c, gm] per node from
the graph state, as D-VAE does for its scalar. With use_pos=True the CktGNN position slot is added
as a node feature and as a decoder head with CktGNN's padding convention (variant A2).

  A1 = SubgDVAE(use_pos=False): CktGNN's graph representation without the slot prior
  A2 = SubgDVAE(use_pos=True):  A1 plus the slot prior, still with per-node device values
  A3 = SubgDVAE(use_pos=True, slot_struct=True): A2 plus slots in the messages and the edge scorer, exactly
       as CktGNN uses them (messages from [state, slot one-hot] instead of [state, vertex id]; edge scorer on
       [state_i, slot_i, state_new, slot_new]). A3 differs from CktGNN only in the device-value pathway.
  P  = SubgDVAE(use_pos=True, n_props=7): A2 plus a property predictor on mu, trained jointly (Phase 6). Targets
       per circuit (simulator labels, scripts/circuit_labels.py, attached as g['props']): 4 binary flags
       [functional, open-loop stable, closed-loop stable, meets spec] and 3 standardised, clipped regressions
       [log10 UGF, gain dB, lag PM] that count only for functional circuits.

`decompositions`/`infer_slots` recover main-path slots from structure under the generator's semantics so that
slot-free graphs can be checked with the authors' validity function and simulated with ocb_sim. Without slots
the reading can be ambiguous (several valid decompositions); this is measured, not hidden.
"""
import igraph
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_models import FastDVAE

N_FEAT = 3  # r, c, gm
N_FLAGS = 4  # props layout: 4 binary flags, then 3 regression targets (NaN when not functional)


FORWARD_GM = {6, 7, 10, 11, 14, 15, 18, 19, 22, 23}   # gm subgraphs with forward direction


def decompositions(g, max_stages=3):
    """All ways to read a slot-free subgraph DAG as an OCB op-amp under the generator's semantics:
    a main chain In -> m1 -> ... -> mk -> Out (k = 2..max_stages, forward gm subgraphs, consecutive
    edges present, no other edges among In/main/Out), and every other node an "extra" with exactly one
    predecessor P and one successor Q, P in {In}+chain and Q in chain+{Out}, P strictly before Q.
    Every edge must be explained by this reading. Returns a list of main chains (vertex lists)."""
    import itertools
    n = g.vcount()
    if n < 4:
        return []
    start, end = 0, n - 1
    inner = [v for v in range(1, n - 1)]
    edges = set(g.get_edgelist())
    found = []
    for k in range(2, max_stages + 1):
        for chain in itertools.permutations([v for v in inner if g.vs[v]['type'] in FORWARD_GM], k):
            seq = [start, *chain, end]
            if any((seq[i], seq[i + 1]) not in edges for i in range(len(seq) - 1)):
                continue
            order = {v: i for i, v in enumerate(seq)}
            main_edges = {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}
            ok, explained = True, set(main_edges)
            for x in inner:
                if x in order:
                    continue
                P, Q = g.predecessors(x), g.successors(x)
                if len(P) != 1 or len(Q) != 1 or P[0] not in order or Q[0] not in order or order[P[0]] >= order[Q[0]]                         or P[0] == end or Q[0] == start:
                    ok = False
                    break
                explained |= {(P[0], x), (x, Q[0])}
            if ok and explained == edges:
                found.append(list(chain))
    return found


def infer_slots(g):
    """Assign 'pos' from the first valid decomposition (main chain -> slots 2, 3, 4; extras -> 5, 6, 7, 7...).
    Sets g['n_decomp'] (0 = not an OCB op-amp, >1 = ambiguous) and g['main_path']."""
    dec = decompositions(g)
    g['n_decomp'] = len(dec)
    main = dec[0] if dec else []
    g['main_path'] = main
    n = g.vcount()
    pos = {0: 0, n - 1: 1}
    for k, v in enumerate(main):
        pos[v] = 2 + k
    extra = 5
    for v in range(n):
        if v not in pos:
            pos[v] = extra
            extra = min(extra + 1, 7)
    g.vs['pos'] = [pos[v] for v in range(n)]
    return g


def _arrays(G, max_n, pad_type, pos_pad):
    B = len(G)
    n = np.array([g.vcount() for g in G])
    types = np.full((B, max_n), pad_type, dtype=np.int64)
    feats = np.zeros((B, max_n, N_FEAT), dtype=np.float32)
    pos = np.full((B, max_n), pos_pad, dtype=np.int64)
    A = np.zeros((B, max_n, max_n), dtype=np.float32)
    for b, g in enumerate(G):
        types[b, :n[b]] = g.vs['type']
        feats[b, :n[b], 0], feats[b, :n[b], 1], feats[b, :n[b], 2] = g.vs['r'], g.vs['c'], g.vs['gm']
        if 'pos' in g.vs.attributes():
            pos[b, :n[b]] = g.vs['pos']
        for e in g.get_edgelist():
            A[b, e[0], e[1]] = 1.0
    return n, types, feats, pos, A


class SubgDVAE(FastDVAE):
    def __init__(self, max_n=8, nvt=26, START_TYPE=0, END_TYPE=1, hs=301, nz=66, use_pos=False, max_pos=8,
                 slot_struct=False, n_props=0, prop_weight=1.0):
        super().__init__(max_n=max_n, nvt=nvt, feat_nvt=103, START_TYPE=START_TYPE, END_TYPE=END_TYPE, hs=hs, nz=nz)
        assert use_pos or not slot_struct
        self.use_pos = use_pos
        self.slot_struct = slot_struct
        self.n_pos = max_pos + 1                     # CktGNN: 1 sudo + 8 slots
        n_in = nvt + N_FEAT + (self.n_pos if use_pos else 0)
        self.grue_forward = nn.GRUCell(n_in, hs)
        self.grue_backward = nn.GRUCell(n_in, hs)
        self.grud = nn.GRUCell(n_in, hs)
        self.fc_feat = nn.Sequential(nn.Linear(hs, 64), nn.ReLU(), nn.Linear(64, N_FEAT))
        if use_pos:
            self.add_pos = nn.Sequential(nn.Linear(hs, hs * 2), nn.ReLU(), nn.Linear(hs * 2, self.n_pos))
        if slot_struct:                              # CktGNN's layer shapes (vs = hs + slots, edge input + 2 slots)
            vs = hs + self.n_pos
            self.vs = vs
            self.gate_forward = nn.Sequential(nn.Linear(vs, hs), nn.Sigmoid())
            self.gate_backward = nn.Sequential(nn.Linear(vs, hs), nn.Sigmoid())
            self.mapper_forward = nn.Sequential(nn.Linear(vs, hs, bias=False))
            self.mapper_backward = nn.Sequential(nn.Linear(vs, hs, bias=False))
            self.add_edge = nn.Sequential(nn.Linear(2 * vs, hs * 4), nn.ReLU(), nn.Linear(hs * 4, 1))
        self.e_dim = hs + (self.n_pos if slot_struct else 0)
        self.n_props, self.prop_weight = n_props, prop_weight
        if n_props:
            self.prop = nn.Sequential(nn.Linear(nz, 128), nn.ReLU(), nn.Linear(128, n_props))
        self.last_prop_loss = 0.0

    def prop_loss(self, mu, G):
        """Summed over the batch: BCE of the 4 flags + squared error of the regressions on functional circuits."""
        y = self._t(np.array([g['props'] for g in G], dtype=np.float32))
        out = self.prop(mu)
        bce = F.binary_cross_entropy_with_logits(out[:, :N_FLAGS], y[:, :N_FLAGS], reduction='sum')
        reg_y = y[:, N_FLAGS:]
        ok = torch.isfinite(reg_y)
        se = ((out[:, N_FLAGS:] - torch.nan_to_num(reg_y)) ** 2 * ok).sum()
        return bce + se

    def _poh(self, pos):
        return F.one_hot(pos, self.n_pos).float()

    def _msg_s(self, S_stack, pos_prefix):
        """Messages from the existing vertices: D-VAE's [state, vertex id], or CktGNN's [state, slot] (A3)."""
        if not self.slot_struct:
            return self._msg(S_stack)
        M = torch.cat([S_stack, self._poh(pos_prefix)], -1)
        return self.gate_forward(M) * self.mapper_forward(M)

    def _e(self, h, pos_v):
        """Vertex representation seen by the edge scorer (A3 appends the slot one-hot, as CktGNN does)."""
        return torch.cat([h, self._poh(pos_v)], -1) if self.slot_struct else h

    def _x(self, types_v, feats_v, pos_v=None):
        parts = [F.one_hot(types_v, self.nvt).float(), feats_v]
        if self.use_pos:
            parts.append(F.one_hot(pos_v, self.n_pos).float())
        return torch.cat(parts, 1)

    def encode(self, G):
        if not isinstance(G, list):
            G = [G]
        B = len(G)
        n, types, feats, pos, A = _arrays(G, self.max_n, self.START_TYPE, 0)
        types, feats, pos, A = self._t(types), self._t(feats), self._t(pos), self._t(A)
        S = [self.grue_forward(self._x(types[:, 0], feats[:, 0], pos[:, 0]), self._get_zero_hidden(B))]
        for v in range(1, int(n.max())):
            Hin = (A[:, :v, v].unsqueeze(-1) * self._msg_s(torch.stack(S, 1), pos[:, :v])).sum(1)
            S.append(self.grue_forward(self._x(types[:, v], feats[:, v], pos[:, v]), Hin))
        S = torch.stack(S, 1)
        Hg = S[torch.arange(B, device=S.device), self._t(n - 1)]
        return self.fc1(Hg), self.fc2(Hg)

    def _step_terms(self, Hlast, types_v, feats_v, pos_v, ar):
        vll = self.logsoftmax1(self.add_vertex(Hlast))[ar, types_v].sum()
        vl2 = self.topo_feat_scale * F.mse_loss(self.fc_feat(Hlast), feats_v, reduction='sum') / 100
        vl3 = self.logsoftmax1(self.add_pos(Hlast))[ar, pos_v].sum() if self.use_pos else torch.zeros((), device=Hlast.device)
        return vll, vl2, vl3

    def loss(self, mu, logvar, G_true, beta=0.005):
        B, N, dev = len(G_true), self.max_n, self.get_device()
        n, types, feats, pos, A = _arrays(G_true, N, self.START_TYPE, self.n_pos - 1)
        n_t, types, feats, pos, A = self._t(n), self._t(types), self._t(feats), self._t(pos), self._t(A)
        z = self.reparameterize(mu, logvar)
        H0 = self.tanh(self.fc3(z))
        start = self._x(torch.full((B,), self.START_TYPE, device=dev), torch.zeros(B, N_FEAT, device=dev),
                        torch.zeros(B, dtype=torch.long, device=dev))
        S = [self.grud(start, H0)]
        Hlast = S[0]
        zeros_e = torch.zeros(B, self.e_dim, device=dev)
        ar = torch.arange(B, device=dev)
        res, res_vll, res_vl2, res_vl3 = 0, 0, 0, 0
        n_max = int(n.max())
        for v in range(1, n_max):
            active = n_t > v
            vll, vl2, vl3 = self._step_terms(Hlast, types[:, v], feats[:, v], pos[:, v], ar)
            res = res + vll - vl2 + vl3
            res_vll, res_vl2, res_vl3 = res_vll + vll, res_vl2 + vl2, res_vl3 + vl3
            x_v = self._x(types[:, v], feats[:, v], pos[:, v])
            Sv = self.grud(x_v, self._get_zero_hidden(B))
            msg = self._msg_s(torch.stack(S, 1), pos[:, :v])
            mask = torch.zeros(B, v, device=dev)
            scores = []
            for vi in range(v - 1, -1, -1):
                h_vi = torch.where((n_t > vi).unsqueeze(1), self._e(S[vi], pos[:, vi]), zeros_e)
                h_cur = torch.where(active.unsqueeze(1), self._e(Sv, pos[:, v]), zeros_e)
                scores.append(self.sigmoid(self.add_edge(torch.cat([h_vi, h_cur], 1))))
                mask = mask.clone()
                mask[:, vi] = A[:, vi, v] * active
                Sv = self.grud(x_v, (mask.unsqueeze(-1) * msg).sum(1))
            gt = A[:, :v, v] * active.unsqueeze(1)
            res = res - F.binary_cross_entropy(torch.cat(scores[::-1], 1), gt, reduction='sum')
            S.append(Sv)
            Hlast = torch.where(active.unsqueeze(1), Sv, Hlast)
        K = N - n_max                                  # padded positions, summed in closed form
        if K > 0:
            vll, vl2, vl3 = self._step_terms(Hlast, types[:, n_max], feats[:, n_max], pos[:, n_max], ar)
            h_prev = self._e(torch.stack(S, 1), pos[:, :n_max])                 * (n_t.unsqueeze(1) > torch.arange(n_max, device=dev)).unsqueeze(-1)
            p_prev = self.sigmoid(self.add_edge(torch.cat([h_prev, torch.zeros_like(h_prev)], -1)))
            p0 = self.sigmoid(self.add_edge(torch.zeros(B, 2 * self.e_dim, device=dev)))
            res = res + K * (vll - vl2 + vl3) - K * F.binary_cross_entropy(p_prev, torch.zeros_like(p_prev), reduction='sum') \
                - (K * (K - 1) / 2) * F.binary_cross_entropy(p0, torch.zeros_like(p0), reduction='sum')
            res_vll, res_vl2, res_vl3 = res_vll + K * vll, res_vl2 + K * vl2, res_vl3 + K * vl3
        res = -res
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total = res + beta * kld
        if self.n_props:
            lp = self.prop_loss(mu, G_true)
            self.last_prop_loss = float(lp.detach())
            total = total + self.prop_weight * lp
        return total, res, kld, res_vll, res_vl3 if self.use_pos else torch.zeros(()), res_vl2

    @torch.no_grad()
    def decode(self, z, stochastic=True):
        """Batched version of DVAE.decode with the same sampling steps: sample the type (and slot), read the
        device values from fc_feat, then sample each candidate edge from the newest vertex backwards,
        updating its state after every decision; an END vertex is connected to all loose ends."""
        B, N, dev = z.shape[0], self.max_n, z.device
        H0 = self.tanh(self.fc3(z))
        types = torch.full((B, N), self.START_TYPE, dtype=torch.long, device=dev)
        feats = torch.zeros(B, N, N_FEAT, device=dev)
        pos = torch.zeros(B, N, dtype=torch.long, device=dev)
        A = torch.zeros(B, N, N, dtype=torch.bool, device=dev)
        n = torch.ones(B, dtype=torch.long, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        S = [self.grud(self._x(types[:, 0], feats[:, 0], pos[:, 0]), H0)]
        Hlast = S[0]
        for idx in range(1, N):
            active = ~finished
            if idx == N - 1:
                t = torch.full((B,), self.END_TYPE, dtype=torch.long, device=dev)
            else:
                logits = self.add_vertex(Hlast)
                t = torch.multinomial(F.softmax(logits, 1), 1).squeeze(1) if stochastic else logits.argmax(1)
            f = self.fc_feat(Hlast)
            if self.use_pos:
                pl = self.add_pos(Hlast)
                p = torch.multinomial(F.softmax(pl, 1), 1).squeeze(1) if stochastic else pl.argmax(1)
            else:
                p = torch.zeros(B, dtype=torch.long, device=dev)
            types[:, idx] = torch.where(active, t, types[:, idx])
            feats[:, idx] = torch.where(active.unsqueeze(1), f, feats[:, idx])
            pos[:, idx] = torch.where(active, p, pos[:, idx])
            is_end = active & (t == self.END_TYPE)
            x = self._x(t, f, p)
            Sv = self.grud(x, self._get_zero_hidden(B))
            msg = self._msg_s(torch.stack(S, 1), pos[:, :idx])
            mask = torch.zeros(B, idx, device=dev)
            for vi in range(idx - 1, -1, -1):
                score = self.sigmoid(self.add_edge(torch.cat([self._e(S[vi], pos[:, vi]), self._e(Sv, p)], 1))).squeeze(1)
                dec = (torch.rand_like(score) < score) if stochastic else (score > 0.5)
                dec = dec & active & ~is_end
                A[:, vi, idx] |= dec
                mask[:, vi] = dec.float()
                Sv = self.grud(x, (mask.unsqueeze(-1) * msg).sum(1))
            loose = (~A[:, :idx, :].any(-1)) & is_end.unsqueeze(1)
            A[:, :idx, idx] |= loose
            S.append(Sv)
            Hlast = torch.where(active.unsqueeze(1), Sv, Hlast)
            n = n + active.long()
            finished = finished | is_end
        G = []
        types, feats, pos, A, n = types.cpu().numpy(), feats.cpu().numpy(), pos.cpu().numpy(), A.cpu().numpy(), n.cpu().numpy()
        for b in range(B):
            g = igraph.Graph(directed=True)
            g.add_vertices(int(n[b]))
            g.vs['type'] = types[b, :n[b]].tolist()
            g.vs['r'], g.vs['c'], g.vs['gm'] = (feats[b, :n[b], k].tolist() for k in range(N_FEAT))
            g.add_edges([(u, v) for u, v in zip(*np.nonzero(A[b, :n[b], :n[b]]))])
            if self.use_pos:
                g.vs['pos'] = pos[b, :n[b]].tolist()
            else:
                infer_slots(g)
            G.append(g)
        return G
