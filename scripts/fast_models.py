"""Batched (vectorised) training path for the authors' CktGNN and DVAE.

FastCktGNN / FastDVAE subclass the classes in external/CktGNN/layers/models_ig.py and override
only `encode` and `loss`. They use the same parameters and the same equations, but process the
64 graphs of a batch as tensors instead of one igraph at a time. That removes ~95% of the tiny
tensor operations that dominate the original runtime. `decode` (used only for evaluation) is
inherited unchanged. State dicts are interchangeable with the original classes.

Faithfulness notes (all replicated from the original):
  * teacher forcing loops over every position up to max_n, and padded graphs still contribute
    their "START" type terms and edge BCE terms;
  * after each candidate edge, the new vertex's GRU state is recomputed from its current
    predecessors;
  * edge scores use the zero vector for vertices that do not exist in a graph.
Equivalence with the original (losses, mu/logvar, gradients) is checked by tests/test_fast_models.py.
"""
import numpy as np
import torch
import torch.nn.functional as F

from layers.models_ig import CktGNN, DVAE


def _batch_arrays(G, max_n, pad_type, pos_pad=None, attr=None):
    """Per-batch tensors: vertex counts, types (padded), optional position/feature, true adjacency."""
    B = len(G)
    n = np.array([g.vcount() for g in G])
    types = np.full((B, max_n), pad_type, dtype=np.int64)
    extra = np.full((B, max_n), pos_pad if pos_pad is not None else 0, dtype=np.int64 if attr == 'pos' else np.float32)
    A = np.zeros((B, max_n, max_n), dtype=np.float32)
    for b, g in enumerate(G):
        types[b, :n[b]] = g.vs['type']
        if attr is not None:
            extra[b, :n[b]] = g.vs[attr]
        for e in g.get_edgelist():
            A[b, e[0], e[1]] = 1.0
    return n, types, extra, A


class _FastMixin:
    def _collate_fn(self, G):
        return list(G)                       # the batched path never mutates the input graphs

    def _t(self, a, dtype=None):
        return torch.as_tensor(a, dtype=dtype, device=self.get_device())


class FastCktGNN(_FastMixin, CktGNN):

    def _msg(self, S_stack, pos_oh_prefix):
        M = torch.cat([S_stack, pos_oh_prefix], -1) if self.pos else S_stack
        return self.gate_forward(M) * self.mapper_forward(M)

    def _x(self, types_v, pos_v):
        return torch.cat([F.one_hot(types_v, self.nvt).float(), F.one_hot(pos_v, self.max_pos).float()], 1)

    def _dfs(self, G):
        dfs = np.zeros((len(G), 3 * self.max_pos), dtype=np.float32)
        for b, g in enumerate(G):
            for v in g.vs:
                dfs[b, v['pos'] * 3:(v['pos'] + 1) * 3] = (v['r'], v['c'], v['gm'])
        return self._t(dfs)

    def encode(self, G):
        assert not self.bidir
        if not isinstance(G, list):
            G = [G]
        B, N = len(G), self.max_n
        n, types, pos, A = _batch_arrays(G, N, self.START_TYPE, 0, 'pos')
        types, pos, A = self._t(types), self._t(pos), self._t(A)
        pos_oh = F.one_hot(pos, self.max_pos).float()
        S = [self.grue_forward(self._x(types[:, 0], pos[:, 0]), self._get_zero_hidden(B))]
        for v in range(1, int(n.max())):
            msg = self._msg(torch.stack(S, 1), pos_oh[:, :v])          # B x v x hs
            Hin = (A[:, :v, v].unsqueeze(-1) * msg).sum(1)
            S.append(self.grue_forward(self._x(types[:, v], pos[:, v]), Hin))
        S = torch.stack(S, 1)
        Hg = S[torch.arange(B, device=S.device), self._t(n - 1)]
        Hg = torch.cat([Hg, self.df_enc(self._dfs(G))], 1)
        return self.fc1(Hg), self.fc2(Hg)

    def loss(self, mu, logvar, G_true, beta=0.005, reg_scale=0.05, pos_scale=0.5):
        B, N, dev = len(G_true), self.max_n, self.get_device()
        n, types, pos, A = _batch_arrays(G_true, N, self.START_TYPE, self.max_pos - 1, 'pos')
        n_t, types, pos, A = self._t(n), self._t(types), self._t(pos), self._t(A)
        pos_oh = F.one_hot(pos, self.max_pos).float()
        true_dfs = self._dfs(G_true)

        z = self.reparameterize(mu, logvar)
        H0 = self.tanh(self.fc3(z))
        pred_dfs = self.df_fc(H0)
        start_x = self._x(torch.full((B,), self.START_TYPE, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
        S = [self.grud(start_x, H0)]                                     # final state of each vertex index
        Hlast = S[0]
        zeros_e = torch.zeros(B, self.hs + self.max_pos, device=dev)
        ar = torch.arange(B, device=dev)
        res, res_vl1, res_vl3 = 0, 0, 0
        for v in range(1, N):
            active = n_t > v                                             # graphs that receive vertex v
            type_scores, pos_scores = self.add_vertex(Hlast), self.add_pos(Hlast)
            vll = self.logsoftmax1(type_scores)[ar, types[:, v]].sum()
            vl3 = self.logsoftmax1(pos_scores)[ar, pos[:, v]].sum()
            res = res + vll + vl3
            res_vl1 = res_vl1 + vll
            res_vl3 = res_vl3 + vl3

            x_v = self._x(types[:, v], pos[:, v])
            Sv = self.grud(x_v, self._get_zero_hidden(B))
            msg = self._msg(torch.stack(S, 1), pos_oh[:, :v])            # predecessors' messages, B x v x hs
            cur_oh = pos_oh[:, v]
            mask = torch.zeros(B, v, device=dev)
            scores = []
            for vi in range(v - 1, -1, -1):
                h_vi = torch.where((n_t > vi).unsqueeze(1), torch.cat([S[vi], pos_oh[:, vi]], 1), zeros_e)
                h_cur = torch.where(active.unsqueeze(1), torch.cat([Sv, cur_oh], 1), zeros_e)
                scores.append(self.sigmoid(self.add_edge(torch.cat([h_vi, h_cur], 1))))
                mask = mask.clone()
                mask[:, vi] = A[:, vi, v] * active
                Sv = self.grud(x_v, (mask.unsqueeze(-1) * msg).sum(1))
            edge_scores = torch.cat(scores[::-1], 1)
            gt = A[:, :v, v] * active.unsqueeze(1)
            res = res - F.binary_cross_entropy(edge_scores, gt, reduction='sum')
            S.append(Sv)
            Hlast = torch.where(active.unsqueeze(1), Sv, Hlast)

        res_vl2 = self.topo_feat_scale * F.mse_loss(pred_dfs, true_dfs, reduction='sum') / 300
        res1 = -res
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res1 + res_vl2 + beta * kld, res1, kld, -res_vl1, -res_vl3, res_vl2


class FastDVAE(_FastMixin, DVAE):

    def _x(self, types_v, feats_v):
        return torch.cat([F.one_hot(types_v, self.nvt).float(), feats_v.unsqueeze(1)], 1)

    def _msg(self, S_stack):
        v = S_stack.shape[1]
        vid = torch.eye(self.max_n, device=S_stack.device)[:v].unsqueeze(0).expand(S_stack.shape[0], -1, -1)
        M = torch.cat([S_stack, vid], -1) if self.vid else S_stack
        return self.gate_forward(M) * self.mapper_forward(M)

    def encode(self, G):
        assert not self.bidir
        if not isinstance(G, list):
            G = [G]
        B = len(G)
        n, types, feats, A = _batch_arrays(G, self.max_n, self.START_TYPE, 0, 'feat')
        types, feats, A = self._t(types), self._t(feats), self._t(A)
        S = [self.grue_forward(self._x(types[:, 0], feats[:, 0]), self._get_zero_hidden(B))]
        for v in range(1, int(n.max())):
            Hin = (A[:, :v, v].unsqueeze(-1) * self._msg(torch.stack(S, 1))).sum(1)
            S.append(self.grue_forward(self._x(types[:, v], feats[:, v]), Hin))
        S = torch.stack(S, 1)
        Hg = S[torch.arange(B, device=S.device), self._t(n - 1)]
        return self.fc1(Hg), self.fc2(Hg)

    def loss(self, mu, logvar, G_true, beta=0.005):
        B, N, dev = len(G_true), self.max_n, self.get_device()
        n, types, feats, A = _batch_arrays(G_true, N, self.START_TYPE, 0, 'feat')
        n_t, types, feats, A = self._t(n), self._t(types), self._t(feats), self._t(A)

        z = self.reparameterize(mu, logvar)
        H0 = self.tanh(self.fc3(z))
        start_x = self._x(torch.full((B,), self.START_TYPE, device=dev), torch.zeros(B, device=dev))
        S = [self.grud(start_x, H0)]
        Hlast = S[0]
        zeros_h = torch.zeros(B, self.hs, device=dev)
        ar = torch.arange(B, device=dev)
        res, res_vll, res_vl2 = 0, 0, 0
        n_max = int(n.max())
        for v in range(1, n_max):
            active = n_t > v
            type_scores = self.add_vertex(Hlast)
            pred_feats = self.fc_feat(Hlast)
            vll = self.logsoftmax1(type_scores)[ar, types[:, v]].sum()
            vl2 = self.topo_feat_scale * F.mse_loss(pred_feats, feats[:, v:v + 1], reduction='sum') / 100
            res = res + vll - vl2
            res_vll = res_vll + vll
            res_vl2 = res_vl2 + vl2

            x_v = self._x(types[:, v], feats[:, v])
            Sv = self.grud(x_v, self._get_zero_hidden(B))
            msg = self._msg(torch.stack(S, 1))
            mask = torch.zeros(B, v, device=dev)
            scores = []
            for vi in range(v - 1, -1, -1):
                h_vi = torch.where((n_t > vi).unsqueeze(1), S[vi], zeros_h)
                h_cur = torch.where(active.unsqueeze(1), Sv, zeros_h)
                scores.append(self.sigmoid(self.add_edge(torch.cat([h_vi, h_cur], 1))))
                mask = mask.clone()
                mask[:, vi] = A[:, vi, v] * active
                Sv = self.grud(x_v, (mask.unsqueeze(-1) * msg).sum(1))
            edge_scores = torch.cat(scores[::-1], 1)
            gt = A[:, :v, v] * active.unsqueeze(1)
            res = res - F.binary_cross_entropy(edge_scores, gt, reduction='sum')
            S.append(Sv)
            Hlast = torch.where(active.unsqueeze(1), Sv, Hlast)

        # Positions n_max..N-1 exist in no graph of the batch. The original still loops over them;
        # every such iteration contributes identical terms (Hlast is final, the new vertex state is 0),
        # so they are summed in closed form: K copies of the type/feature terms, K copies of the edge
        # terms to vertices 0..n_max-1, and K(K-1)/2 copies of the all-zero edge term.
        K = N - n_max
        if K > 0:
            vll = self.logsoftmax1(self.add_vertex(Hlast))[ar, types[:, n_max]].sum()
            vl2 = self.topo_feat_scale * F.mse_loss(self.fc_feat(Hlast), feats[:, n_max:n_max + 1], reduction='sum') / 100
            h_prev = torch.stack(S, 1) * (n_t.unsqueeze(1) > torch.arange(n_max, device=dev)).unsqueeze(-1)
            p_prev = self.sigmoid(self.add_edge(torch.cat([h_prev, torch.zeros_like(h_prev)], -1)))
            bce_prev = F.binary_cross_entropy(p_prev, torch.zeros_like(p_prev), reduction='sum')
            p0 = self.sigmoid(self.add_edge(torch.zeros(B, 2 * self.hs, device=dev)))
            bce0 = F.binary_cross_entropy(p0, torch.zeros_like(p0), reduction='sum')
            res = res + K * (vll - vl2) - K * bce_prev - (K * (K - 1) / 2) * bce0
            res_vll = res_vll + K * vll
            res_vl2 = res_vl2 + K * vl2

        res = -res
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta * kld, res, kld, res_vll, torch.FloatTensor([0]), res_vl2
