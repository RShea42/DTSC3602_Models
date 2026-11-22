"""M0 baseline: reproduce Juan's team-eval numbers from the shared files only (no eval/cache).

Plain EASE: each player's visible played games (binary) summed over rows of B (model/artifacts/ease.npz),
masked to recommendable games not owned (hidden games stay allowed), top 50 kept.
EASE-safe: Juan's team_eval/rerank.py rule on that top 50 with rerank_choice.json (a=0.5, b=1, c=10).
Both lists are then scored by Juan's unmodified team_eval/evaluate.py.

  PYTHONUTF8=1 <ease-safe venv python> m0_baseline.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from paths import EASE_FILES, JUAN, TE
HERE = Path(__file__).resolve().parent
FILES = EASE_FILES
DATA = TE
sys.path.insert(0, str(JUAN / "team_eval"))
from evaluate import NEG, profile, rowdot, tag_matrix  # noqa: E402

HOLD = ["80_20", "75_25", "50_50"]
SAFE = json.loads((JUAN / "team_eval" / "rerank_choice.json").read_text())
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else FILES / "model" / "artifacts" / "ease.npz"   # optional: another ease.npz
OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else "m0_out")
OUT.mkdir(exist_ok=True)
z = np.load(MODEL)
B = z["B"].astype(np.float32)
ids = z["app_ids"].astype(np.int64)
col = pd.Series(np.arange(len(ids)), index=ids)
games = pd.read_parquet(DATA / "games.parquet")
val = pd.read_parquet(DATA / "validation.parquet")
rec_ok = np.isin(ids, games.app_id[games.recommendable].to_numpy())
log(f"B {B.shape}, recommendable in vocab {rec_ok.sum():,}")


def ease_top50(h):
    hcol = f"hidden_{h}"
    vis = val[val.played & ~val[hcol]]
    owned = val[~val[hcol]]                       # owned and not hidden: never recommend
    pids = np.sort(val.player_id.unique())
    prow = pd.Series(np.arange(len(pids)), index=pids)
    v = vis[vis.app_id.isin(col.index)]
    V = sp.csr_matrix((np.ones(len(v), np.float32), (prow[v.player_id].to_numpy(), col[v.app_id].to_numpy())),
                      shape=(len(pids), len(ids)))
    o = owned[owned.app_id.isin(col.index)]
    O = sp.csr_matrix((np.ones(len(o), bool), (prow[o.player_id].to_numpy(), col[o.app_id].to_numpy())),
                      shape=(len(pids), len(ids)))
    rows = []
    for k in range(0, len(pids), 1024):
        S = np.asarray(V[k:k + 1024] @ B, np.float32)
        S[:, ~rec_ok] = -np.inf
        S[O[k:k + 1024].toarray()] = -np.inf
        has = np.asarray(V[k:k + 1024].sum(1)).ravel() > 0
        top = np.argpartition(-S, 50, axis=1)[:, :50]
        for r in range(len(S)):
            if not has[r]:
                continue
            t = top[r][np.argsort(-S[r, top[r]], kind="stable")]
            t = t[np.isfinite(S[r, t])]
            rows += [(int(pids[k + r]), h, i + 1, int(ids[j]), float(S[r, j])) for i, j in enumerate(t)]
    return pd.DataFrame(rows, columns=["player_id", "holdout", "rank", "app_id", "score"])


def safe(c, h):
    """team_eval/rerank.py prepare() + top10(), same flag definitions, for one holdout."""
    T = tag_matrix(games).tocsr()
    gpos = pd.Series(np.arange(len(games)), index=games.app_id)
    vis = val[val.played & ~val[f"hidden_{h}"]]
    pids = np.sort(c.player_id.unique())
    pidx = pd.Series(np.arange(len(pids)), index=pids)
    vis = vis[vis.player_id.isin(pids)]
    vg = vis.app_id.map(gpos); vr = pidx.reindex(vis.player_id).to_numpy()
    ok = vg.notna().to_numpy() & ~np.isnan(vr)
    vgi, vpi, vpt = vg[ok].astype(int).to_numpy(), vr[ok].astype(int), vis.playtime_min.to_numpy()[ok]
    short = vpt < 120
    ps, pl = profile(T, vgi[short], vpi[short], len(pids)), profile(T, vgi[~short], vpi[~short], len(pids))
    has_s = np.bincount(vpi[short], minlength=len(pids)) > 0
    has_l = np.bincount(vpi[~short], minlength=len(pids)) > 0
    c = c.copy()
    gi = c.app_id.map(gpos).astype(int).to_numpy(); pi = pidx[c.player_id].to_numpy()
    has_tags = np.asarray(T[gi].sum(1)).ravel() > 0
    quit = has_s[pi] & has_l[pi] & has_tags & (rowdot(T, gi, ps, pi) > rowdot(T, gi, pl, pi))
    shortg = games.median_playtime_min.to_numpy()[gi] <= 120
    neg = games.rating.isin(NEG).to_numpy()[gi]
    g = c.groupby("player_id").score
    zs = (c.score - g.transform("mean")) / (g.transform("std") + 1e-9)
    c["s"] = zs - SAFE["a_quit"] * quit - SAFE["b_short"] * shortg - SAFE["c_negative"] * neg
    t = c.sort_values(["player_id", "s"], ascending=[True, False], kind="stable").groupby("player_id").head(10)
    t["rank"] = t.groupby("player_id").cumcount() + 1
    return t[["player_id", "holdout", "rank", "app_id"]]


plain, ease_safe = [], []
for h in HOLD:
    c = ease_top50(h)
    log(f"{h}: EASE top 50 for {c.player_id.nunique():,} players")
    plain.append(c[c["rank"] <= 10][["player_id", "holdout", "rank", "app_id"]])
    ease_safe.append(safe(c, h))
    log(f"{h}: EASE-safe re-ranked")
pd.concat(plain).to_parquet(OUT / "recs_ease.parquet", index=False)
pd.concat(ease_safe).to_parquet(OUT / "recs_ease_safe.parquet", index=False)
log(f"wrote {OUT.name}/recs_ease.parquet and recs_ease_safe.parquet")
