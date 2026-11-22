"""Soft-fail flag helpers, copied unchanged from Juan's team_eval/evaluate.py so the pipeline is self-contained
(it must run on Modal without the DTSC3602_Models repo). Keep in sync with that file: the team scores with it."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

NEG = {"Mostly Negative", "Negative", "Very Negative", "Overwhelmingly Negative"}


def tag_matrix(games):
    """Row-normalized multi-hot of each game's first 10 user tags (genres if no tags), over the 400 most common tags."""
    lists = [list(t)[:10] if t is not None and len(t) else (list(g)[:10] if g is not None else [])
             for t, g in zip(games.tags, games.genres)]
    vocab = pd.Series([x for l in lists for x in l]).value_counts().index[:400]
    col = pd.Series(np.arange(len(vocab)), index=vocab)
    r, c = [], []
    for i, l in enumerate(lists):
        for x in l:
            if x in col.index:
                r.append(i); c.append(col[x])
    M = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(games), len(vocab)))
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()); n[n == 0] = 1
    return sp.diags(1 / n) @ M


def profile(T, gi, groups, n_groups, weights=None):
    """Sum of tag rows per group, row-normalized."""
    w = np.ones(len(gi), np.float32) if weights is None else weights.astype(np.float32)
    A = sp.csr_matrix((w, (groups, np.arange(len(gi)))), shape=(n_groups, len(gi)))
    P = A @ T[gi]
    n = np.sqrt(np.asarray(P.multiply(P).sum(1)).ravel()); n[n == 0] = 1
    return sp.diags(1 / n) @ P


def rowdot(A, ia, B, ib):
    return np.asarray(A[ia].multiply(B[ib]).sum(1)).ravel()
