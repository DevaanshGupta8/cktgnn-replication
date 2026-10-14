"""PyTorch port of the authors' Theano sparse GP (search_methods/sparse_gp.py + sparse_gp_theano_internal.py).

Only the path used by sgp_regression.py is ported: regression with deterministic inputs
(ignore_variances=True, input variances = 0), trained by minibatch Adam on the EP/PEP energy
of Bui et al. (arXiv:1602.04133), then predicting with the posterior (set_for_training = 0).
Equations, initialisation and the training loop follow the original line by line; the only
deliberate difference is float64 arithmetic (Theano ran in float32).
"""
import math

import numpy as np
import torch

LOG2PI = math.log(2 * math.pi)


def rbf(lls, lsf, x, z):
    """k(x, z) = sf * exp(-0.5 * sum_d (x_d - z_d)^2 / ls_d), with ls = exp(lls), sf = exp(lsf) (gauss.compute_kernel)."""
    ls = torch.exp(lls)
    r2 = (x * x / ls).sum(1, keepdim=True) - 2 * (x / ls) @ z.T + (1.0 / ls) @ (z.T ** 2)
    return torch.exp(lsf) * torch.exp(-0.5 * r2)


class SparseGP:
    def __init__(self, n_inducing, n_points, jitter=1e-3, dtype=torch.float64):
        self.M, self.n, self.jitter, self.dtype = n_inducing, n_points, jitter, dtype

    def _t(self, a):
        return torch.as_tensor(np.asarray(a), dtype=self.dtype)

    def initialize(self, X, y):
        """Inducing points = random subset; lls = log(0.5*(median squared distance + 1e-3));
        L ~ N(0, 1); m = targets at the inducing points (Sparse_GP.initialize)."""
        M = self.M
        assert X.shape[0] >= M
        sel = np.random.choice(X.shape[0], M, replace=False)
        sq = np.sum(X ** 2, 1)
        dist = sq[:, None] - 2 * X @ X.T + sq[None, :]
        med = np.median(dist[np.triu_indices(X.shape[0], 1)])
        self.lls = self._t(np.log(0.5 * (med + 1e-3)) * np.ones(X.shape[1])).requires_grad_()
        self.lsf = self._t(0.0).requires_grad_()
        self.z = self._t(X[sel]).requires_grad_()
        self.mParamPost = self._t(y[sel]).reshape(M, 1).requires_grad_()
        self.LParamPost = self._t(np.random.normal(size=(M, M))).requires_grad_()
        self.lvar_noise = self._t(0.0).requires_grad_()

    def params(self):
        return [self.lls, self.lsf, self.z, self.mParamPost, self.LParamPost, self.lvar_noise]

    def _posterior(self, s):
        """Cavity (s=1, training) or posterior (s=0, prediction) quantities (compute_output)."""
        n, eye = self.n, torch.eye(self.M, dtype=self.dtype)
        Kzz = rbf(self.lls, self.lsf, self.z, self.z) + eye * self.jitter * torch.exp(self.lsf)
        KzzInv = torch.linalg.inv(Kzz)
        LLt = self.LParamPost @ self.LParamPost.T
        covCavityInv = KzzInv + LLt * (n - s) / n
        covCavity = torch.linalg.inv(covCavityInv)
        meanCavity = covCavity @ ((n - s) / n * self.mParamPost)
        return dict(KzzInv=KzzInv, LLt=LLt, covCavityInv=covCavityInv, covCavity=covCavity,
                    meanCavity=meanCavity, B=KzzInv @ covCavity @ KzzInv - KzzInv,
                    KzzInvmeanCavity=KzzInv @ meanCavity)

    def _output(self, X, q):
        Kxz = rbf(self.lls, self.lsf, X, self.z)
        mean = Kxz @ q['KzzInvmeanCavity']
        v_out = torch.exp(self.lsf) + ((Kxz * (Kxz @ q['B'])).sum(1, keepdim=True))
        return mean, v_out.abs() + torch.exp(self.lvar_noise)

    def energy(self, Xb, yb):
        """getContributionToEnergy for a minibatch (training mode, s = 1)."""
        M, n = self.M, self.n
        q = self._posterior(1.0)
        covPosteriorInv = q['KzzInv'] + q['LLt']
        covPosterior = torch.linalg.inv(covPosteriorInv)
        meanPosterior = covPosterior @ self.mParamPost
        logZcav = 0.5 * M * LOG2PI + 0.5 * torch.logdet(q['covCavity']) + \
            0.5 * (q['meanCavity'].T @ q['covCavityInv'] @ q['meanCavity'])[0, 0]
        logZprior = 0.5 * M * LOG2PI - 0.5 * torch.logdet(q['KzzInv'])
        logZpost = 0.5 * M * LOG2PI + 0.5 * torch.logdet(covPosterior) + \
            0.5 * (meanPosterior.T @ covPosteriorInv @ meanPosterior)[0, 0]
        mean, var = self._output(Xb, q)
        logZ = -0.5 * torch.log(2 * math.pi * var) - 0.5 * (yb - mean) ** 2 / var
        return ((logZcav - logZpost) + logZpost / n - logZprior / n) * Xb.shape[0] + logZ.sum()

    @torch.no_grad()
    def predict(self, X):
        q = self._posterior(0.0)
        mean, var = self._output(self._t(X), q)
        return mean.numpy(), var.numpy()

    def train_via_adam(self, X, y, max_iterations=100, minibatch_size=1000, learning_rate=5e-4, log=None):
        """SparseGP.train_via_ADAM: initialise on a random subset of size minibatch_size, then
        max_iterations epochs of shuffled minibatch Adam on -energy."""
        n = X.shape[0]
        sel = np.random.choice(n, n, replace=False)[:min(n, minibatch_size)]
        self.initialize(X[sel], y[sel])
        opt = torch.optim.Adam(self.params(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        n_batches = int(np.ceil(n / minibatch_size))
        for it in range(max_iterations):
            perm = np.random.choice(n, n, replace=False)
            X, y = X[perm], y[perm]
            for i in range(n_batches):
                sl = slice(i * minibatch_size, min((i + 1) * minibatch_size, n))
                opt.zero_grad()
                loss = -self.energy(self._t(X[sl]), self._t(y[sl]))
                loss.backward()
                opt.step()
            if log is not None:
                log(it, float(loss))
        return self
