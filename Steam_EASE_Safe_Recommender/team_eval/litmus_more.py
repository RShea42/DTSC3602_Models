"""Rob's litmus test for EASE-safe and TasteForge, on the same 300 libraries as team_eval/litmus_compare.py.

Same players (same filter and seed), same split (each player's 10 most-played games hidden), same hit counting
(a hidden game found at rank <= 15 / 50 / 100). Rob's KNN, plain EASE and Rob's popularity baseline come from the
saved per-player results of that run (litmus_compare.csv), so his model is not re-run.
EASE-safe: EASE scores over recommendable, unowned games, then the app's own safe re-rank of the top 50
(model/recommender.py Recommender.safe_rerank). TasteForge: peyton-dev's recommend() (team_eval/tasteforge_on_team_eval.py).
  OPENBLAS_NUM_THREADS=2 PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python team_eval/litmus_more.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "model"))
import tasteforge_on_team_eval as tf  # noqa: E402
from recommender import Recommender  # noqa: E402

DATA = Path("C:/Users/juant/data_clean/team_eval")
HOLD, KS = 10, (15, 50, 100)

# same 300 players as litmus_compare.py
v = pd.read_parquet(DATA / "validation.parquet", columns=["player_id", "src", "app_id", "playtime_min", "played"])
v = v[v.src.isin(["mcauley", "steam2026"])]
n_played = v[v.played].groupby("player_id").size()
pool = n_played[n_played >= 20].index.to_numpy()
players = np.random.default_rng(42).choice(np.sort(pool), size=min(300, len(pool)), replace=False)
prev = pd.read_csv(HERE / "litmus_compare.csv").set_index("player_id")
assert set(prev.index) == set(players), "player sample differs from litmus_compare.py"

R = Recommender()
g = pd.read_parquet(DATA / "games.parquet", columns=["app_id", "recommendable"])
ok = np.isin(R.app_ids, g.app_id[g.recommendable].to_numpy())

rows = []
for n, pid in enumerate(players):
    lib = v[v.player_id == pid]
    owned = {int(a): int(m) for a, m in zip(lib.app_id, lib.playtime_min.fillna(0))}
    by_time = sorted(owned.items(), key=lambda kv: kv[1], reverse=True)
    hidden = [a for a, _ in by_time[:HOLD]]
    visible = dict(by_time[HOLD:])

    # EASE-safe
    vis_rows = [R.idx[a] for a, m in visible.items() if m > 0 and a in R.idx]
    s = R.B[vis_rows].astype(np.float32).sum(0) if vis_rows else np.zeros(len(R.app_ids), np.float32)
    s = np.where(ok, s, -np.inf)
    s[[R.idx[a] for a in visible if a in R.idx]] = -np.inf
    order = [j for j in np.argsort(-s)[:600] if np.isfinite(s[j])]
    order = R.safe_rerank(order, s, list(visible.items()))[:max(KS)]
    pos_e = {int(R.app_ids[j]): r + 1 for r, j in enumerate(order)}

    # TasteForge (a Steam import: owned games in its catalog with lifetime hours)
    t_owned = {}
    for a, m in visible.items():
        gid = tf.gid_of.get(a)
        if gid is not None:
            t_owned[gid] = min(9999.0, round(m / 60, 1))
    t_played = {gid: h for gid, h in t_owned.items() if h >= tf.MIN_HOURS}
    feats = {gid: {md: 1 / len(tf.G.at[gid, "modes"]) for md in tf.G.at[gid, "modes"]} for gid in t_owned}
    P = tf.profile(tf.G, t_played, feats)
    top = tf.recommend_fast(P, list(t_owned), n=max(KS))
    pos_t = {int(tf.G.at[gid, "app_id"]): r + 1 for r, (gid, _) in enumerate(top)}

    row = {"player_id": pid}
    for k in KS:
        row[f"easesafe_top{k}"] = sum(pos_e.get(a, 10 ** 9) <= k for a in hidden)
        row[f"tf_top{k}"] = sum(pos_t.get(a, 10 ** 9) <= k for a in hidden)
    rows.append(row)
    if (n + 1) % 50 == 0:
        print(f"{n + 1}/{len(players)}", flush=True)

df = pd.DataFrame(rows).set_index("player_id").join(prev)
df.to_csv(HERE / "litmus_all_models.csv")
cols = [("EASE-safe", "easesafe"), ("EASE", "ease"), ("Rob's KNN", "rob"), ("TasteForge", "tf"), ("Popularity (Rob's)", "pop")]
lines = [f"# Rob's blind test on every model ({len(df)} libraries, 10 most-played games hidden)", "",
         "| Hidden games found (of 10), average per player | " + " | ".join(c for c, _ in cols) + " |",
         "|---|" + "---:|" * len(cols)]
for k in KS:
    lines.append(f"| in the top {k} | " + " | ".join(f"{df[f'{p}_top{k}'].mean():.2f}" for _, p in cols) + " |")
    lines.append(f"| players with 1+ found in the top {k} | " +
                 " | ".join(f"{(df[f'{p}_top{k}'] > 0).mean():.0%}" for _, p in cols) + " |")
(HERE / "litmus_all_models.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
