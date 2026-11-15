"""D3 error analysis: per-player NDCG@10 for every model on the shared head-to-head (1,000 validation players),
broken down by visible library size, data source, and popular vs niche hidden games.

NDCG@10 follows team_eval/evaluate.py exactly (binary gain on hidden enjoyed recommendable games; scored players have
>= 1 such game and >= 5 visible played games); the overall means are checked against results_all_models.md.
Popular = the 500 games with the most training players; niche = every other recommendable game.
  PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python team_eval/error_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = Path("C:/Users/juant/data_clean/team_eval_all")
MODELS = [("EASE-safe", "ease_safe"), ("EASE", "ease"), ("KNN (Rob)", "rob_knn"), ("TasteForge", "tasteforge"),
          ("MostPopular", "mostpopular")]
DISC = 1 / np.log2(np.arange(2, 12))
HOLD = ["80_20", "75_25", "50_50"]

games = pd.read_parquet(D / "games.parquet", columns=["app_id", "recommendable", "n_train_players"])
rec_ok = set(games.app_id[games.recommendable])
popular = set(games[games.recommendable].nlargest(500, "n_train_players").app_id)
val = pd.read_parquet(D / "validation.parquet")
recs = {k: pd.read_parquet(D / f"recs_{k}.parquet") for _, k in MODELS}

rows = []
for h in HOLD:
    hc = f"hidden_{h}"
    vis = val[val.played & ~val[hc]]
    tgt = val[val[hc] & val.enjoyed & val.app_id.isin(rec_ok)]
    n_vis = vis.groupby("player_id").size()
    n_tgt = tgt.groupby("player_id").size()
    pids = n_tgt.index[n_tgt.index.isin(n_vis.index[n_vis >= 5])]
    src = val.drop_duplicates("player_id").set_index("player_id").src
    tset = tgt.groupby("player_id").app_id.apply(set)
    for name, key in MODELS:
        r = recs[key]
        r = r[(r.holdout == h) & (r["rank"] <= 10)]
        lists = r.sort_values("rank").groupby("player_id").app_id.apply(list)
        for pid in pids:
            t = tset[pid]
            lst = lists.get(pid, [])
            dcg = sum(DISC[i] for i, a in enumerate(lst) if a in t)
            idcg = DISC[:min(len(t), 10)].sum()
            rows.append({"holdout": h, "model": name, "player_id": pid, "ndcg": dcg / idcg, "src": src[pid],
                         "n_visible": int(n_vis[pid]),
                         "pop_found": sum(a in t and a in popular for a in lst), "pop_total": sum(a in popular for a in t),
                         "niche_found": sum(a in t and a not in popular for a in lst),
                         "niche_total": sum(a not in popular for a in t)})
df = pd.DataFrame(rows)
df["size"] = pd.cut(df.n_visible, [4, 20, 100, 10 ** 6], labels=["5-20 games", "21-100 games", "100+ games"])
df.to_csv(HERE / "error_analysis_players.csv", index=False)

order = [m for m, _ in MODELS]
out = ["# D3 error analysis (shared head-to-head, 1,000 validation players)", ""]
overall = df.groupby(["holdout", "model"]).ndcg.mean().unstack()[order]
out += ["## Overall NDCG@10 (check: equals evaluate.py)", "", overall.round(4).to_markdown(), ""]
d = df[df.holdout == "80_20"]
for col, title in [("size", "visible library size"), ("src", "data source")]:
    t = d.groupby([col, "model"], observed=True).ndcg.mean().unstack()[order]
    n = d[d.model == order[0]].groupby(col, observed=True).size().rename("players")
    out += [f"## NDCG@10 by {title} (80/20)", "", t.join(n).round(4).to_markdown(), ""]
g = d.groupby("model")[["pop_found", "pop_total", "niche_found", "niche_total"]].sum().loc[order]
g["popular hidden games found %"] = (100 * g.pop_found / g.pop_total).round(1)
g["niche hidden games found %"] = (100 * g.niche_found / g.niche_total).round(1)
out += ["## Share of hidden enjoyed games found in the top 10, popular vs niche (80/20)", "",
        f"Popular = top 500 games by training players ({int(g.pop_total.iloc[0])} hidden games); "
        f"niche = all others ({int(g.niche_total.iloc[0])}).", "",
        g[["popular hidden games found %", "niche hidden games found %"]].to_markdown(), ""]
# where EASE-safe fails: players with zero hits
z = d[d.model == "EASE-safe"]
miss = z[z.ndcg == 0]
out += ["## Where EASE-safe finds nothing (80/20)", "",
        f"{len(miss)} of {len(z)} players ({len(miss) / len(z):.0%}) get no hidden enjoyed game in the top 10.", "",
        "| | players with no hit | all players |", "|---|---:|---:|",
        f"| median visible games | {miss.n_visible.median():.0f} | {z.n_visible.median():.0f} |",
        f"| share of their hidden games that are niche | {miss.niche_total.sum() / (miss.niche_total.sum() + miss.pop_total.sum()):.0%} | "
        f"{z.niche_total.sum() / (z.niche_total.sum() + z.pop_total.sum()):.0%} |",
        f"| share from Kaggle (reviews, not full libraries) | {(miss.src == 'kaggle').mean():.0%} | {(z.src == 'kaggle').mean():.0%} |", ""]
(HERE / "error_analysis.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
