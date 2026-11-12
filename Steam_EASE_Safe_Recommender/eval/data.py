"""Shared evaluation data for Layers 2 and 3 (eval/PROTOCOL.md). Built once, cached in eval/cache/.

- Train rows: played play-history rows of TRAIN players only (test and validation players are excluded).
- Enjoyment g of a train row: percentile of its playtime among the same game's train players in the same source
  (games with < 10 train players in that source get 0.5).
- Enjoyment g of a validation/test row: its playtime's percentile within that same train distribution.
- Universe U: recommendable games + any game with >= 20 train players. Every model scores over U.
- Per evaluated player: visible (U index, g, playtime), hidden hits (U index, gain 1 or 2; hits outside U are
  unreachable but still count in the denominators), owned (everything owned, played or not), plus playtime of every
  hidden played game (Layer 3 regret).
Run: PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python eval/data.py
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

D = Path("C:/Users/juant/data_clean")
R = Path("C:/Users/juant/data_raw/steam_reviews")
C = Path(__file__).resolve().parent / "cache"
C.mkdir(exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

# ------------------------------------------------------------------ all rows (owned + played), per source
cat = pd.read_parquet(D / "games.parquet", columns=["app_id", "history_ok", "recommendable"])
hist_ok = set(cat.app_id[cat.history_ok.astype(bool)])
mc = pd.read_parquet(D / "interactions_mcauley.parquet", columns=["user_key", "app_id", "playtime_min"]).assign(src="mcauley")
kg = pd.read_parquet(D / "interactions_kaggle.parquet", columns=["user_key", "app_id", "playtime_min", "bulk_user"])
kg = kg[~kg.bulk_user.fillna(False)].drop(columns="bulk_user").assign(src="kaggle")
fr = pd.DataFrame([(r["uid"], a, pt) for r in map(json.loads, (R / "libraries.jsonl").open(encoding="utf-8")) if r["public"]
                   for a, pt, _, _ in r["games"]], columns=["user_key", "app_id", "playtime_min"]).assign(src="steam2026")
rows = pd.concat([mc, kg, fr], ignore_index=True)
del mc, kg, fr
rows["user_key"] = rows.user_key.astype(str)
rows["src"] = rows.src.astype("category")
rows = rows.drop_duplicates(["src", "user_key", "app_id"])
rows["played"] = (rows.playtime_min > 0) & rows.app_id.isin(hist_ok)
log(f"all rows {len(rows):,}")

players = pd.read_parquet(D / "splits" / "players.parquet")
players["user_key"] = players.user_key.astype(str)
rows = rows.merge(players, on=["src", "user_key"], how="left")
rows["split"] = rows.split.fillna("train")
hidden = pd.read_parquet(D / "splits" / "hidden_games.parquet", columns=["src", "user_key", "app_id", "hidden"])
hidden["user_key"] = hidden.user_key.astype(str)
rows = rows.merge(hidden, on=["src", "user_key", "app_id"], how="left")
rows["hidden"] = rows.hidden.fillna(False).astype(bool)
log("split joined " + str(rows.split.value_counts().to_dict()))

# ------------------------------------------------------------------ train enjoyment and reference distributions
tr = rows[(rows.split == "train") & rows.played][["src", "user_key", "app_id", "playtime_min"]].copy()
grp = tr.groupby(["src", "app_id"], observed=True).playtime_min
tr["g"] = grp.rank(pct=True).astype(np.float32)
small = grp.transform("size") < 10
tr.loc[small, "g"] = 0.5
log(f"train played rows {len(tr):,}")

# reference: sorted train playtimes per (src, app) for scoring validation/test rows
ref = tr[["src", "app_id", "playtime_min"]].copy()
ref["gid"] = ref.groupby(["src", "app_id"], observed=True).ngroup()
ref = ref.sort_values(["gid", "playtime_min"])
gid_of = ref.drop_duplicates(["src", "app_id"]).set_index(["src", "app_id"]).gid
key = ref.gid.to_numpy(np.int64) * 2**32 + ref.playtime_min.to_numpy(np.int64)
start = np.searchsorted(ref.gid.to_numpy(), np.arange(ref.gid.max() + 1))
size = np.diff(np.r_[start, len(ref)])


def g_eval(d):
    """Percentile of each row's playtime among the same (source, game) train players; 0.5 when < 10 of them."""
    gid = gid_of.reindex(pd.MultiIndex.from_arrays([d.src.astype(str), d.app_id])).to_numpy()
    out = np.full(len(d), 0.5, np.float32)
    ok = ~np.isnan(gid)
    gi = gid[ok].astype(np.int64)
    pos = np.searchsorted(key, gi * 2**32 + d.playtime_min.to_numpy(np.int64)[ok], side="right") - start[gi]
    v = pos / size[gi]
    v[size[gi] < 10] = 0.5
    out[ok] = v
    return out


# ------------------------------------------------------------------ universe and train matrix
n_tr = tr.groupby("app_id").user_key.size()
rec = set(cat.app_id[cat.recommendable])
U = np.array(sorted(rec | set(n_tr[n_tr >= 20].index)), dtype=np.int64)
uidx = pd.Series(np.arange(len(U)), index=U)
tr = tr[tr.app_id.isin(uidx.index)]
tr["user"] = tr.src.astype(str) + ":" + tr.user_key
users = pd.Index(tr.user.unique())
X = sp.csr_matrix((tr.g.to_numpy(np.float32), (users.get_indexer(tr.user), uidx[tr.app_id].to_numpy())), shape=(len(users), len(U)))
src_of_user = np.array([u.split(":", 1)[0] for u in users])
sp.save_npz(C / "train_g.npz", X)
np.save(C / "train_src.npy", src_of_user)
np.save(C / "universe.npy", U)
recommendable = np.isin(U, list(rec))
np.save(C / "recommendable.npy", recommendable)
n_train_players = np.asarray((X > 0).sum(0)).ravel()
n_train_enjoyed = np.asarray((X >= 0.5).sum(0)).ravel()
np.save(C / "n_train_players.npy", n_train_players)
np.save(C / "n_train_enjoyed.npy", n_train_enjoyed)
log(f"universe {len(U):,} games ({recommendable.sum():,} recommendable); train matrix {X.shape}, nnz {X.nnz:,}; "
    f"games with >=20 / >=50 / >=100 train players: {(n_train_players >= 20).sum():,} / {(n_train_players >= 50).sum():,} / "
    f"{(n_train_players >= 100).sum():,}")

# ------------------------------------------------------------------ evaluated players
ev = rows[rows.split != "train"].copy()
del rows, tr
# Kaggle has 2.7 M validation/test players; score a fixed hash subsample (protocol change 1): 9% of test, 3% of validation
import hashlib  # noqa: E402
kg_keep = {"test": 9, "validation": 3}
kk = ev[ev.src == "kaggle"][["user_key", "split"]].drop_duplicates()
kk = kk[[int.from_bytes(hashlib.sha256(f"steamrec-kaggle-sub-v1|{u}".encode()).digest()[:8], "big") % 100 < kg_keep[s]
         for u, s in zip(kk.user_key, kk.split)]]
ev = ev[(ev.src != "kaggle") | ev.user_key.isin(set(kk.user_key))]
log(f"evaluated rows after Kaggle subsample {len(ev):,}")
ev["g"] = np.float32(0)
pl = ev.played.to_numpy()
ev.loc[pl, "g"] = g_eval(ev[pl])
ev["u"] = ev.app_id.map(uidx).fillna(-1).astype(np.int64)
out = {"validation": [], "test": []}
for (s, u, split), d in ev.groupby(["src", "user_key", "split"], observed=True, sort=False):
    vis = d[d.played & ~d.hidden & (d.u >= 0)]
    hid = d[d.played & d.hidden]
    hits = hid[hid.g >= 0.5]
    if len(hits) == 0:
        continue
    out[split].append({
        "src": s, "user": u, "n_visible": int((d.played & ~d.hidden).sum()),
        "vis_u": vis.u.to_numpy(np.int64), "vis_g": vis.g.to_numpy(np.float32), "vis_pt": vis.playtime_min.to_numpy(np.int64),
        "hit_u": hits.u.to_numpy(np.int64), "hit_gain": np.where(hits.g >= 0.8, 2, 1).astype(np.int8),
        "hid_u": hid.u.to_numpy(np.int64), "hid_pt": hid.playtime_min.to_numpy(np.int64), "hid_g": hid.g.to_numpy(np.float32),
        "owned_u": d.u[d.u >= 0].to_numpy(np.int64)})
for split, L in out.items():
    pickle.dump(L, open(C / f"{split}_players.pkl", "wb"))
    c = pd.Series([p["src"] for p in L]).value_counts().to_dict()
    e = pd.Series([p["src"] for p in L if p["n_visible"] >= 5]).value_counts().to_dict()
    log(f"{split}: players with >= 1 hidden hit {c}; of those with >= 5 visible {e}")
json.dump({"universe": int(len(U)), "recommendable": int(recommendable.sum()), "train_players": int(X.shape[0]),
           "train_rows": int(X.nnz)}, open(C / "summary.json", "w"), indent=2)
log("done")
