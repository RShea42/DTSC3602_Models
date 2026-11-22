"""EASE-safe as a scikit-learn Pipeline: LibraryEncoder -> EASEScorer -> SafeReranker.

Input to fit and predict is a long table, one row per (player, owned game):
    player_id, app_id, playtime_min, played (optional, default True)
- fit(train_rows):   learns the game vocabulary (games with >= min_users players) and EASE's item-item weights B.
- predict(rows):     each player's top-k picks as a table (player_id, rank, app_id). Played rows are the taste input;
                     every row given counts as owned and is never recommended.

Same math, masks and tie-breaking as Juan's eval (eval/models.py EASE, team_eval/rerank.py), so the lists match his
exactly; tests/test_pipeline.py and m1_check.py hold it to that.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .flags import NEG, profile, rowdot, tag_matrix


@lru_cache(maxsize=2)
def load_catalog(path):
    return pd.read_parquet(path)


@dataclass
class Libraries:
    """What flows between pipeline steps: players' libraries encoded over the model's game vocabulary."""
    player_ids: np.ndarray           # one per row of V / O
    app_ids: np.ndarray              # the vocabulary, one per column
    V: sp.csr_matrix                 # played games (binary): the taste input
    O: sp.csr_matrix                 # owned games (bool): never recommended
    played: pd.DataFrame             # played rows (player_id, app_id, playtime_min), all games, for the flags
    candidates: pd.DataFrame = None  # filled by EASEScorer.transform: player_id, rank, app_id, score


class LibraryEncoder(BaseEstimator, TransformerMixin):
    """Learns the vocabulary at fit (games with >= min_users distinct players); encodes libraries over it."""

    def __init__(self, min_users=100):
        self.min_users = min_users

    def fit(self, X, y=None):
        n = X.groupby("app_id").player_id.nunique()
        self.app_ids_ = np.sort(n[n >= self.min_users].index.to_numpy()).astype(np.int64)
        return self

    def transform(self, X):
        X = X if "played" in X else X.assign(played=True)
        pids, prow = np.unique(X.player_id.to_numpy(), return_inverse=True)
        col = pd.Series(np.arange(len(self.app_ids_)), index=self.app_ids_)
        c = col.reindex(X.app_id.to_numpy()).to_numpy()
        inv = ~np.isnan(c)
        played = X.played.to_numpy(bool)

        def mat(m, dtype):
            return sp.csr_matrix((np.ones(m.sum(), dtype), (prow[m], c[m].astype(np.int64))),
                                 shape=(len(pids), len(self.app_ids_)))
        V = mat(inv & played, np.float32)
        O = V.astype(bool) if played.all() else mat(inv, bool)
        return Libraries(pids, self.app_ids_, V, O, X.loc[played, ["player_id", "app_id", "playtime_min"]])


def app_candidates(app_ids, app_catalog_path, include_explicit=False):
    """Juan's live-app candidate rule (model/recommender.py, Recommender.__init__ and .recommend), same arithmetic:
    recommendable, not (rated < 65% with >= 50 reviews), >= 20 reviews, and not sexually explicit unless opted in."""
    g = load_catalog(app_catalog_path).set_index("app_id")
    pos, neg = g.steam_positive.fillna(0), g.steam_negative.fillna(0)
    n = pos + neg
    rating = ((pos / n.where(n > 0)).fillna(g.positive_ratio / 100)).reindex(app_ids).to_numpy()
    n_reviews = n.where(n > 0, g.user_reviews).reindex(app_ids).fillna(0).to_numpy()
    ok = g.recommendable.reindex(app_ids).fillna(False).to_numpy(bool) & ~((rating < 0.65) & (n_reviews >= 50)) & (n_reviews >= 20)
    if not include_explicit:
        ok &= ~g.explicit.reindex(app_ids).fillna(False).to_numpy(bool)
    return ok


class EASEScorer(BaseEstimator, TransformerMixin):
    """EASE (Steck 2019): B = I - P / diag(P) with P = (V'V + lam I)^-1, diagonal zeroed.
    transform keeps each player's top n_candidates allowed, unowned games with their scores.

    Allowed games (both are Juan's decisions):
      serve_filters=True  (default) his live app's rule, app_candidates(): what players are shown
      serve_filters=False his evaluation's rule: any recommendable game (how 0.2704 / 0.277 were scored)"""

    def __init__(self, lam=100.0, n_candidates=50, catalog_path=None, serve_filters=True, app_catalog_path=None,
                 include_explicit=False, dtype="float16", batch_size=1024):
        self.lam, self.n_candidates, self.catalog_path, self.dtype, self.batch_size = lam, n_candidates, catalog_path, dtype, batch_size
        self.serve_filters, self.app_catalog_path, self.include_explicit = serve_filters, app_catalog_path, include_explicit

    def allowed(self):
        key = (self.serve_filters, self.app_catalog_path, self.include_explicit, self.catalog_path)
        cache = self.__dict__.setdefault("_allowed", {})      # per fitted model; one entry per filter setting
        if key not in cache:
            cache[key] = self._allowed_mask()
        return cache[key]

    def _allowed_mask(self):
        if self.serve_filters:
            if self.app_catalog_path is None:
                raise ValueError("serve_filters=True needs app_catalog_path (Juan's data_clean/games.parquet)")
            return app_candidates(self.app_ids_, self.app_catalog_path, self.include_explicit)
        games = load_catalog(self.catalog_path)
        return np.isin(self.app_ids_, games.app_id[games.recommendable].to_numpy())

    def fit(self, L, y=None):
        G = (L.V.T @ L.V).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        del G
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0)
        self.B_ = B.astype(self.dtype)          # float16 halves the size; scores still computed in float32
        self.app_ids_ = L.app_ids
        return self

    def fit_transform(self, L, y=None):
        # Pipeline.fit calls this on training data: fit only. Scoring 11M training players would be wasted work,
        # and the next step (SafeReranker) learns nothing from candidates.
        self.fit(L)
        return L

    def __getstate__(self):                    # don't pickle the float32 working copy
        s = self.__dict__.copy()
        s.pop("_Bf", None)
        s.pop("_allowed", None)
        return s

    def transform(self, L):
        if not np.array_equal(L.app_ids, self.app_ids_):
            raise ValueError("libraries were encoded with a different vocabulary than EASE was fitted on")
        if getattr(self, "_Bf", None) is None:
            self._Bf = self.B_.astype(np.float32)
        rec_ok = self.allowed()
        has =np.asarray(L.V.sum(1)).ravel() > 0
        k = self.n_candidates
        rows = []
        for s in range(0, L.V.shape[0], self.batch_size):
            S = np.asarray(L.V[s:s + self.batch_size] @ self._Bf, np.float32)
            S[:, ~rec_ok] = -np.inf
            S[L.O[s:s + self.batch_size].toarray()] = -np.inf
            # Exact ties (common with one played game) are ordered as in the Juan code each mode reproduces:
            # app mode sorts all scores like his app (np.argsort(-s)); evaluation mode takes the top k first and
            # sorts them stably like his evaluation. Ties are the only place the two orders can differ.
            if self.serve_filters:
                top = np.argsort(-S, axis=1)[:, :k]
            elif k < S.shape[1]:
                top = np.argpartition(-S, k, axis=1)[:, :k]
            else:                                   # fewer games than candidates wanted: take them all
                top = np.tile(np.arange(S.shape[1]), (len(S), 1))
            for r in range(len(S)):
                if not has[s + r]:
                    continue
                t = top[r] if self.serve_filters else top[r][np.argsort(-S[r, top[r]], kind="stable")]
                t = t[np.isfinite(S[r, t])]
                pid = int(L.player_ids[s + r])
                rows += [(pid, i + 1, int(self.app_ids_[j]), float(S[r, j])) for i, j in enumerate(t)]
        L.candidates = pd.DataFrame(rows, columns=["player_id", "rank", "app_id", "score"])
        return L


class SafeReranker(BaseEstimator):
    """EASE-safe: within each player's candidates, new score = z(EASE score) - a*like_quit - b*short - c*negative;
    keep the top k. Flags as in team_eval/evaluate.py. a = b = c = 0 gives plain EASE.
    head: re-rank only each player's top `head` candidates and keep the rest in EASE order after them (Juan's app
    re-ranks the top 50 of 600). None = re-rank all candidates. k=None returns the whole order."""

    def __init__(self, catalog_path=None, a_quit=0.5, b_short=1.0, c_negative=10.0, k=10, head=None):
        self.catalog_path, self.a_quit, self.b_short, self.c_negative, self.k = catalog_path, a_quit, b_short, c_negative, k
        self.head = head

    def fit(self, L=None, y=None):
        games = load_catalog(self.catalog_path)
        self.T_ = tag_matrix(games).tocsr()
        self.gpos_ = pd.Series(np.arange(len(games)), index=games.app_id)
        self.short_ = (games.median_playtime_min <= 120).to_numpy()
        self.neg_ = games.rating.isin(NEG).to_numpy()
        return self

    def flags(self, L):
        """Candidates with their three flags (also used by the evaluation and the app's explanations)."""
        c = L.candidates.copy()
        if c.empty:
            return c.assign(quit=False, short=False, neg=False)
        pids = np.sort(c.player_id.unique())
        pidx = pd.Series(np.arange(len(pids)), index=pids)
        vis = L.played[L.played.player_id.isin(pids)]
        vg = vis.app_id.map(self.gpos_); vr = pidx.reindex(vis.player_id).to_numpy()
        ok = vg.notna().to_numpy() & ~np.isnan(vr)
        vgi, vpi, vpt = vg[ok].astype(int).to_numpy(), vr[ok].astype(int), vis.playtime_min.to_numpy()[ok]
        s = vpt < 120
        ps, pl = profile(self.T_, vgi[s], vpi[s], len(pids)), profile(self.T_, vgi[~s], vpi[~s], len(pids))
        has_s = np.bincount(vpi[s], minlength=len(pids)) > 0
        has_l = np.bincount(vpi[~s], minlength=len(pids)) > 0
        gi = c.app_id.map(self.gpos_).astype(int).to_numpy(); pi = pidx[c.player_id].to_numpy()
        has_tags = np.asarray(self.T_[gi].sum(1)).ravel() > 0
        c["quit"] = has_s[pi] & has_l[pi] & has_tags & (rowdot(self.T_, gi, ps, pi) > rowdot(self.T_, gi, pl, pi))
        c["short"] = self.short_[gi]
        c["neg"] = self.neg_[gi]
        return c

    def predict(self, L):
        return self.rerank(self.flags(L))

    def rerank(self, c):
        """Top k from flagged candidates (flags() output). Split out so tuning computes flags once per grid."""
        if c.empty:
            return pd.DataFrame(columns=["player_id", "rank", "app_id", "score"])
        top = c if self.head is None else c[c["rank"] <= self.head].copy()
        g = top.groupby("player_id").score
        z = (top.score - g.transform("mean")) / (g.transform("std") + 1e-9)
        top["s"] = z - self.a_quit * top.quit - self.b_short * top.short - self.c_negative * top.neg
        t = top.sort_values(["player_id", "s"], ascending=[True, False], kind="stable")
        if self.head is not None:                    # the rest follow in EASE order
            t = pd.concat([t.assign(part=0), c[c["rank"] > self.head].sort_values(["player_id", "rank"]).assign(part=1)])
            t = t.sort_values(["player_id", "part"], kind="stable")
        if self.k is not None:
            t = t.groupby("player_id").head(self.k)
        t = t.copy()
        t["rank"] = t.groupby("player_id").cumcount() + 1
        return t[["player_id", "rank", "app_id", "score"]].reset_index(drop=True)


def make_pipeline(catalog_path, app_catalog_path=None, serve_filters=True, include_explicit=False, lam=100.0, min_users=100,
                  a_quit=0.5, b_short=1.0, c_negative=10.0, k=10, n_candidates=50):
    """Every default is Juan's: EASE lam / min_users (eval/tuning.md), re-rank a/b/c (team_eval/rerank_choice.json),
    and his live app's candidate filters. For his evaluation setup pass serve_filters=False.
    catalog_path = team_eval/games.parquet (flags, as scored); app_catalog_path = data_clean/games.parquet (app filters)."""
    return Pipeline([
        ("encode", LibraryEncoder(min_users=min_users)),
        ("ease", EASEScorer(lam=lam, n_candidates=n_candidates, catalog_path=str(catalog_path), serve_filters=serve_filters,
                            app_catalog_path=None if app_catalog_path is None else str(app_catalog_path),
                            include_explicit=include_explicit)),
        ("safe", SafeReranker(catalog_path=str(catalog_path), a_quit=a_quit, b_short=b_short, c_negative=c_negative, k=k)),
    ])
