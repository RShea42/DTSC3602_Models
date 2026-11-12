"""Scoring harness (eval/PROTOCOL.md sections 2-4, 6): full-catalog ranking, no sampled negatives.

Candidates per player: recommendable games in the universe minus everything the player owns.
Targets: hidden games the player enjoyed (g >= 0.5) that are recommendable (protocol change 2); gain 2 if g >= 0.8.
Returns one row per player with every Layer 2 metric plus the top-10 list (Layer 3 reuses it).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

C = Path(__file__).resolve().parent / "cache"
DISC = 1 / np.log2(np.arange(2, 22))


def load(split):
    U = np.load(C / "universe.npy")
    rec = np.load(C / "recommendable.npy")
    players = pickle.load(open(C / f"{split}_players.pkl", "rb"))
    keep = []
    for p in players:                                   # targets: recommendable hidden hits only
        m = rec[p["hit_u"]] if len(p["hit_u"]) else np.zeros(0, bool)
        p = dict(p, hit_u=p["hit_u"][m], hit_gain=p["hit_gain"][m])
        if len(p["hit_u"]):
            keep.append(p)
    return U, rec, keep


def tail_items():
    n = np.load(C / "n_train_players.npy").astype(float)
    order = np.argsort(-n)
    head = order[: np.searchsorted(np.cumsum(n[order]) / n.sum(), 0.8) + 1]
    t = np.ones(len(n), bool); t[head] = False
    return t


def evaluate(model, players, rec, batch_size=256, keep_lists=True):
    tail = tail_items()
    pop = np.load(C / "n_train_players.npy")
    rows = []
    for k in range(0, len(players), batch_size):
        batch = players[k:k + batch_size]
        S = model.score(batch)
        S[:, ~rec] = -np.inf
        for r, p in enumerate(batch):
            s = S[r]
            s[np.setdiff1d(p["owned_u"], p["hid_u"])] = -np.inf     # owned games are excluded, except the hidden ones
            top = np.argpartition(-s, 20)[:20]
            top = top[np.argsort(-s[top], kind="stable")]
            top = top[np.isfinite(s[top])]
            gain = dict(zip(p["hit_u"].tolist(), p["hit_gain"].tolist()))
            hit = np.array([t in gain for t in top])
            n = len(gain)
            g_rel = np.array([gain.get(t, 0) for t in top[:10]])
            ideal = sorted(gain.values(), reverse=True)[:10]
            tail_hits = [u for u in gain if tail[u]]
            rows.append({
                "src": p["src"], "user": p["user"], "n_visible": p["n_visible"], "n_hits": n,
                "ndcg10": (hit[:10] * DISC[:len(hit[:10])]).sum() / DISC[:min(n, 10)].sum(),
                "ndcg10_graded": ((2.0 ** g_rel - 1) * DISC[:len(g_rel)]).sum() / ((2.0 ** np.array(ideal) - 1) * DISC[:len(ideal)]).sum(),
                "recall20": hit.sum() / min(n, 20),
                "tail_recall20": (np.isin(top, tail_hits).sum() / min(len(tail_hits), 20)) if tail_hits else np.nan,
                "arp10": float(pop[top[:10]].mean()) if len(top) else np.nan,
                "top10": top[:10].tolist() if keep_lists else None})
    return pd.DataFrame(rows)


def coverage(df, rec):
    items = {i for t in df.top10 for i in t}
    return len(items) / rec.sum()


def bucket(n):
    return pd.cut(n, [0, 4, 20, 100, 10**9], labels=["1-4", "5-20", "21-100", "100+"])


def train_matrix():
    return sp.load_npz(C / "train_g.npz").tocsr()
