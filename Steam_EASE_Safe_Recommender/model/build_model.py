"""Train the recommender: implicit ALS on "how much did this player enjoy each game".

Enjoyment = the player's playtime percentile among that game's players in the same source (the label chosen in
label/label_research.md, round 6). Confidence = 1 + ALPHA * g.

Sources (each kept as separate users):
  McAuley 2017 full libraries (played rows, play-history apps)
  Steam reviews collection 2026 full libraries (public reviewers)
  Kaggle reviews up to 2023 (users with >= 3 played games, bulk reviewers removed)

Held out for the offline check (never trained on): 2,000 McAuley users and 400 users from the 2026 libraries.
Run:  OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python model/build_model.py
Out:  model/artifacts/als.npz (item factors + ids), model/artifacts/items.parquet (popularity, playtime quantiles),
      model/artifacts/heldout.parquet
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

D = Path("C:/Users/juant/data_clean")
R = Path("C:/Users/juant/data_raw/steam_reviews")
A = Path(__file__).resolve().parent / "artifacts"
A.mkdir(exist_ok=True)
ALPHA, FACTORS, REG, ITERS, MIN_USERS = 20.0, 64, 0.05, 15, 20
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

# ------------------------------------------------------------------ load
mc = pd.read_parquet(D / "interactions_mcauley.parquet", columns=["user_key", "app_id", "playtime_min", "played", "history_ok"])
mc = mc[mc.played & mc.history_ok][["user_key", "app_id", "playtime_min"]].assign(src="mcauley")
fr = pd.DataFrame([(r["uid"], a, pt) for r in map(json.loads, (R / "libraries.jsonl").open(encoding="utf-8")) if r["public"]
                   for a, pt, _, _ in r["games"] if pt > 0], columns=["user_key", "app_id", "playtime_min"]).assign(src="steam2026")
kg = pd.read_parquet(D / "interactions_kaggle.parquet", columns=["user_key", "app_id", "playtime_min", "played", "history_ok", "bulk_user"])
kg = kg[kg.played & kg.history_ok & ~kg.bulk_user.fillna(False)][["user_key", "app_id", "playtime_min"]]
kg = kg[kg.groupby("user_key").app_id.transform("size") >= 3].assign(src="kaggle")
cat = pd.read_parquet(D / "games.parquet", columns=["app_id", "history_ok"])
fr = fr[fr.app_id.isin(set(cat.app_id[cat.history_ok.astype(bool)])) | ~fr.app_id.isin(set(cat.app_id))]
df = pd.concat([mc, fr, kg], ignore_index=True)
df["user"] = df.src + ":" + df.user_key.astype(str)
df = df.drop_duplicates(["user", "app_id"])
log(f"rows {len(df):,}  users {df.user.nunique():,}  " + str(df.groupby("src").user.nunique().to_dict()))

# ------------------------------------------------------------------ enjoyment g per (source, game)
grp = df.groupby(["src", "app_id"]).playtime_min
df["g"] = grp.rank(pct=True)
df.loc[grp.transform("size") < 10, "g"] = 0.5                    # too few players in this source to rank

# ------------------------------------------------------------------ held-out users
rng = np.random.default_rng(42)
pick = lambda s, n: set(rng.choice(sorted(df.user[(df.src == s) & (df.groupby("user").app_id.transform("size") >= 10)].unique()), n, replace=False))
held = pick("mcauley", 2000) | pick("steam2026", 400)
df[df.user.isin(held)][["user", "src", "app_id", "playtime_min", "g"]].to_parquet(A / "heldout.parquet")
tr = df[~df.user.isin(held)]

# ------------------------------------------------------------------ item table: vocab, popularity, playtime quantiles for live users
n_users = tr.groupby("app_id").user.nunique()
items = n_users[n_users >= MIN_USERS].index.to_numpy()
tr = tr[tr.app_id.isin(items)]
log(f"items {len(items):,}  train rows {len(tr):,}")
Q = np.linspace(0, 1, 101)
qs = {}
for src in ("steam2026", "mcauley", "kaggle"):                    # live users have lifetime playtime: prefer lifetime sources
    s = df[df.src == src].groupby("app_id").playtime_min
    ok = s.size()[lambda x: x >= 30].index
    for a, v in df[(df.src == src) & df.app_id.isin(ok) & ~df.app_id.isin(qs.keys())].groupby("app_id").playtime_min:
        qs[a] = (src, np.quantile(v.to_numpy(float), Q))
it = pd.DataFrame({"app_id": items, "n_users": n_users.loc[items].to_numpy(),
                   "n_enjoyed": tr[tr.g >= 0.5].groupby("app_id").size().reindex(items, fill_value=0).to_numpy()})
it["q_src"] = it.app_id.map(lambda a: qs.get(a, (None, None))[0])
it["q"] = it.app_id.map(lambda a: qs[a][1].tolist() if a in qs else None)
it.to_parquet(A / "items.parquet")

# ------------------------------------------------------------------ ALS
ui = {u: i for i, u in enumerate(tr.user.unique())}
ii = {a: i for i, a in enumerate(items)}
X = sp.csr_matrix(((1 + ALPHA * tr.g).astype(np.float32), (tr.user.map(ui).to_numpy(), tr.app_id.map(ii).to_numpy())),
                  shape=(len(ui), len(ii)))
del df, tr, kg, mc
log(f"matrix {X.shape}  nnz {X.nnz:,}; fitting ALS")
m = AlternatingLeastSquares(factors=FACTORS, regularization=REG, iterations=ITERS, random_state=0, calculate_training_loss=False)
m.fit(X, show_progress=False)
np.savez(A / "als.npz", item_factors=m.item_factors.astype(np.float32), app_ids=items,
         params=np.array([ALPHA, FACTORS, REG, ITERS, MIN_USERS]))
log("saved model/artifacts/als.npz")
