"""The five Layer 2 models behind one interface (eval/PROTOCOL.md section 5).

  m = Model(**params); m.fit(X)            X: train players x universe, values = enjoyment g (0 = not played)
  S = m.score(batch)                       batch: list of evaluated players (dicts from eval/data.py)
                                           returns a float32 array (len(batch) x |U|); -inf = cannot score
New players are always folded in from their visible history only, the same path the live app uses.
"""
import numpy as np
import scipy.sparse as sp

NEG = np.float32(-np.inf)


def _visible_matrix(batch, n_items, values="g"):
    rows, cols, vals = [], [], []
    for r, p in enumerate(batch):
        rows += [r] * len(p["vis_u"]); cols += list(p["vis_u"])
        vals += list(p["vis_g"] if values == "g" else np.ones(len(p["vis_u"]), np.float32))
    return sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(batch), n_items))


class MostPopular:
    name = "MostPopular"

    def __init__(self):
        pass

    def fit(self, X):
        self.pop = np.asarray((X >= 0.5).sum(0)).ravel().astype(np.float32)
        return self

    def score(self, batch):
        return np.tile(self.pop, (len(batch), 1))


class _GramModel:
    """Shared item-item co-occurrence (Gram) matrix over games with >= min_users train players."""

    _cache = {}                                          # same data + settings -> same Gram matrix; build once

    def _gram(self, X, min_users, binary):
        n = np.asarray((X > 0).sum(0)).ravel()
        self.vocab = np.where(n >= min_users)[0]
        key = (id(X), min_users, binary)
        if key not in _GramModel._cache:
            _GramModel._cache.clear()
            Xv = X[:, self.vocab]
            if binary:
                Xv = (Xv > 0).astype(np.float32)
            Xv = Xv.tocsc()
            _GramModel._cache[key] = (Xv.T @ Xv).toarray().astype(np.float32)
        return _GramModel._cache[key].copy()

    def _expand(self, part, n_items):
        S = np.full((part.shape[0], n_items), NEG, np.float32)
        S[:, self.vocab] = part
        return S


class ItemKNN(_GramModel):
    name = "item-kNN"

    def __init__(self, k=100, weight="g", min_users=50, shrink=10.0):
        self.k, self.weight, self.min_users, self.shrink = k, weight, min_users, shrink

    def fit(self, X):
        self.n_items = X.shape[1]
        G = self._gram(X, self.min_users, binary=self.weight == "binary")
        d = np.sqrt(np.diag(G)).astype(np.float32) + 1e-9
        S = G / d[:, None] / d[None, :]
        S *= G / (G + self.shrink)                       # shrink similarities that rest on few co-players
        np.fill_diagonal(S, 0)
        if self.k < S.shape[0]:                          # keep the k nearest neighbours of every game
            cut = np.partition(S, -self.k, axis=0)[-self.k]
            S[S < cut[None, :]] = 0
        self.S = sp.csr_matrix(S)
        self.pos = {u: i for i, u in enumerate(self.vocab)}
        return self

    def score(self, batch):
        V = _visible_matrix(batch, self.n_items, "g" if self.weight == "g" else "binary")[:, self.vocab]
        return self._expand(np.asarray((V @ self.S).todense(), np.float32), self.n_items)


class EASE(_GramModel):
    name = "EASE"

    def __init__(self, lam=500.0, min_users=100):
        self.lam, self.min_users = lam, min_users

    def fit(self, X):
        self.n_items = X.shape[1]
        G = self._gram(X, self.min_users, binary=True).astype(np.float64)
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0)
        self.B = B.astype(np.float32)
        return self

    def score(self, batch):
        V = _visible_matrix(batch, self.n_items, "binary")[:, self.vocab]
        return self._expand(np.asarray(V @ self.B, np.float32), self.n_items)


class ALS:
    name = "ALS"

    def __init__(self, factors=64, alpha=20.0, reg=0.05, iters=15, min_users=20):
        self.factors, self.alpha, self.reg, self.iters, self.min_users = factors, alpha, reg, iters, min_users

    def fit(self, X):
        from implicit.als import AlternatingLeastSquares
        self.n_items = X.shape[1]
        n = np.asarray((X > 0).sum(0)).ravel()
        self.vocab = np.where(n >= self.min_users)[0]
        Xv = X[:, self.vocab].tocsr()
        C = Xv.copy(); C.data = 1 + self.alpha * C.data
        m = AlternatingLeastSquares(factors=self.factors, regularization=self.reg, iterations=self.iters,
                                    random_state=0, calculate_training_loss=False)
        m.fit(C, show_progress=False)
        self.Y = np.asarray(m.item_factors, np.float64)
        self.YtY = self.Y.T @ self.Y + self.reg * np.eye(self.Y.shape[1])
        self.pos = np.full(self.n_items, -1); self.pos[self.vocab] = np.arange(len(self.vocab))
        return self

    def user_vectors(self, batch):
        out = np.zeros((len(batch), self.Y.shape[1]))
        for r, p in enumerate(batch):
            j = self.pos[p["vis_u"]]; ok = j >= 0
            if not ok.any():
                continue
            Yu = self.Y[j[ok]]; c = 1 + self.alpha * p["vis_g"][ok]
            out[r] = np.linalg.solve(self.YtY + (Yu * (c - 1)[:, None]).T @ Yu, (Yu * c[:, None]).sum(0))
        return out

    def score(self, batch):
        S = np.full((len(batch), self.n_items), NEG, np.float32)
        S[:, self.vocab] = (self.user_vectors(batch) @ self.Y.T).astype(np.float32)
        return S


class Hybrid:
    """w * z(ALS) + (1 - w) * z(content). Content = cosine between a game's description embedding and the player's
    enjoyment-weighted profile. Games ALS can't score get z(ALS) = 0, so new games can still surface via content."""
    name = "Hybrid"

    def __init__(self, als, emb, w=0.8):
        self.als, self.E, self.w = als, emb, w            # als: a fitted ALS; emb: |U| x d unit vectors (0 = none)

    def fit(self, X):
        return self

    def score(self, batch):
        A = self.als.score(batch).astype(np.float64)
        prof = np.zeros((len(batch), self.E.shape[1]))
        for r, p in enumerate(batch):
            prof[r] = (self.E[p["vis_u"]] * p["vis_g"][:, None]).sum(0)
        prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9
        Cn = prof @ self.E.T
        has = np.isfinite(A)
        mu = np.where(has, A, 0).sum(1, keepdims=True) / has.sum(1, keepdims=True)
        sd = np.sqrt(np.where(has, (A - mu) ** 2, 0).sum(1, keepdims=True) / has.sum(1, keepdims=True)) + 1e-9
        za = np.where(has, (A - mu) / sd, 0)
        hasc = np.abs(self.E).sum(1) > 0
        zc = (Cn - Cn[:, hasc].mean(1, keepdims=True)) / (Cn[:, hasc].std(1, keepdims=True) + 1e-9)
        zc[:, ~hasc] = 0
        S = self.w * za + (1 - self.w) * zc
        S[:, ~(has.any(0) | hasc)] = -np.inf
        return S.astype(np.float32)
