"""Rob's litmus test (Steam_KNN_Recommender/litmus.py), run on Rob's KNN and our EASE over the same libraries.

Per player: hide the 10 most-played games (Rob's split), recommend from the rest, count hidden games found in the
top 15 / 50 / 100. Rob's model runs through his own code (recommend(..., full=True)) with recency off, because these
libraries have no last-played dates (same as his --no-recency). Hits count like his script: a hidden game counts when
it (or, for Rob's model, its game family) is at that rank or better.
Players: validation players from the shared team package (never in EASE's training data) with full libraries
(McAuley 2017 and Steam 2026; Kaggle rows are reviews, not libraries) and >= 20 played games. Fixed random sample.
  OPENBLAS_NUM_THREADS=4 PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python team_eval/litmus_compare.py [n_players]
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROB = Path("C:/Users/juant/team 3000/DTSC3602_Models/Steam_KNN_Recommender")
sys.path.insert(0, str(ROB))
from features import load_features  # noqa: E402
from recommend import RecConfig, exclusion_reasons, recommend  # noqa: E402
from litmus import popularity_ranking  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("C:/Users/juant/data_clean/team_eval")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
HOLD, KS = 10, (15, 50, 100)

v = pd.read_parquet(DATA / "validation.parquet", columns=["player_id", "src", "app_id", "playtime_min", "played"])
v = v[v.src.isin(["mcauley", "steam2026"])]
n_played = v[v.played].groupby("player_id").size()
pool = n_played[n_played >= 20].index.to_numpy()
players = np.random.default_rng(42).choice(np.sort(pool), size=min(N, len(pool)), replace=False)
src_of = v.drop_duplicates("player_id").set_index("player_id").src

f = load_features()
cfg = replace(RecConfig(), recency_half_life_days=0, recent_2weeks_boost=0)
z = np.load(ROOT / "model" / "artifacts" / "ease.npz")
B, ease_ids = z["B"], z["app_ids"].astype(np.int64)
eidx = {int(a): i for i, a in enumerate(ease_ids)}
g = pd.read_parquet(DATA / "games.parquet", columns=["app_id", "recommendable"])
ease_ok = np.isin(ease_ids, g.app_id[g.recommendable].to_numpy())

rows, t0 = [], time.time()
for n, pid in enumerate(players):
    lib = v[v.player_id == pid]
    owned = {int(a): int(m) for a, m in zip(lib.app_id, lib.playtime_min.fillna(0))}
    by_time = sorted(owned.items(), key=lambda kv: kv[1], reverse=True)
    hidden = [a for a, _ in by_time[:HOLD]]
    visible = {a: {"minutes": m, "minutes_2weeks": 0, "last_played": 0} for a, m in by_time[HOLD:]}

    # Rob's KNN, exactly as litmus.run ranks hidden games
    rank = recommend(f, visible, cfg, full=True)
    reasons = exclusion_reasons(f, set(visible), cfg)
    rob_ranks, rob_findable = [], 0
    for a in hidden:
        if a in f.row_of and not reasons.iat[f.row_of[a]]:
            rob_findable += 1
            fam = rank[rank.family_id == f.games.family_id.iat[f.row_of[a]]]
            if len(fam):
                rob_ranks.append(int(fam["rank"].iat[0]))
    rob_ranks = sorted(set(rob_ranks))                   # one family slot = one hit
    pop = popularity_ranking(f, set(visible), cfg).set_index("appid")["rank"]   # Rob's baseline
    pop_ranks = [int(pop[a]) for a in hidden if a in pop.index]

    # EASE: sum of item-to-item rows for visible played games; skip owned visible games
    vis_rows = [eidx[a] for a, x in visible.items() if x["minutes"] > 0 and a in eidx]
    s = B[vis_rows].astype(np.float32).sum(0) if vis_rows else np.zeros(len(ease_ids), np.float32)
    s[~ease_ok] = -np.inf
    s[[eidx[a] for a in visible if a in eidx]] = -np.inf
    order = np.argsort(-s)[:max(KS)]
    pos = {int(ease_ids[j]): r + 1 for r, j in enumerate(order)}
    ease_ranks = sorted(pos[a] for a in hidden if a in pos)
    ease_findable = sum(1 for a in hidden if a in eidx and ease_ok[eidx[a]])

    row = {"player_id": pid, "src": src_of[pid], "n_games": len(owned), "rob_findable": rob_findable,
           "ease_findable": ease_findable}
    for k in KS:
        row[f"rob_top{k}"] = sum(r <= k for r in rob_ranks)
        row[f"ease_top{k}"] = sum(r <= k for r in ease_ranks)
        row[f"pop_top{k}"] = sum(r <= k for r in pop_ranks)
    rows.append(row)
    if (n + 1) % 25 == 0:
        print(f"{n + 1}/{len(players)} players, {time.time() - t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(HERE / "litmus_compare.csv", index=False)
lines = [f"# Rob's litmus test on both models ({len(df)} players, 10 most-played games hidden, recency off)", "",
         "| | Rob KNN | EASE | Popularity (Rob's baseline) |", "|---|---:|---:|---:|",
         f"| hidden games the model can recommend at all (of 10) | {df.rob_findable.mean():.1f} | {df.ease_findable.mean():.1f} | |"]
for k in KS:
    lines.append(f"| avg hidden games found in top {k} | {df[f'rob_top{k}'].mean():.2f} | {df[f'ease_top{k}'].mean():.2f} | {df[f'pop_top{k}'].mean():.2f} |")
    lines.append(f"| players with 1+ found in top {k} | {(df[f'rob_top{k}'] > 0).mean():.0%} | {(df[f'ease_top{k}'] > 0).mean():.0%} | {(df[f'pop_top{k}'] > 0).mean():.0%} |")
for s_, d in df.groupby("src"):
    lines.append(f"| top 15, {s_} only ({len(d)} players) | {d.rob_top15.mean():.2f} | {d.ease_top15.mean():.2f} | {d.pop_top15.mean():.2f} |")
diff = df.ease_top15 - df.rob_top15
bs = [np.random.default_rng(i).choice(diff, len(diff)).mean() for i in range(2000)]
lines += ["", f"EASE minus Rob, hidden games found in top 15: {diff.mean():+.2f} "
          f"(95% CI {np.percentile(bs, 2.5):+.2f} to {np.percentile(bs, 97.5):+.2f})"]
(HERE / "litmus_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
