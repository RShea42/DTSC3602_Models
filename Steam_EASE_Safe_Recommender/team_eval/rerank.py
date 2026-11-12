"""Flag-aware re-rank of EASE's top 50 ("EASE-safe"): push down picks that trip the team's soft-fail flags.

  new score = z(EASE score within the player's top 50) - a * like_quit - b * short_game - c * negative_reviews
Flags are the same as team_eval/evaluate.py. Rule fixed before looking at results: on the 75/25 holdout, choose the
(a, b, c) with the lowest share of flagged picks among settings that keep NDCG@10 >= 98% of plain EASE; then report
80/20 and 50/50 as checks. Caveat: the holdouts share the same validation players (different hidden games), so the
checks are not fully independent.
In:  eval/cache/team_top50_ease.parquet (team_eval/run_ours.py EASE)    Out: data_clean/team_eval/recs_ease_safe.parquet,
     team_eval/rerank_grid.md
  PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python team_eval/rerank.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import NEG, DISC, profile, rowdot, tag_matrix  # noqa: E402

DATA = Path("C:/Users/juant/data_clean/team_eval")
CAND = HERE.parent / "eval" / "cache" / "team_top50_ease.parquet"
games = pd.read_parquet(DATA / "games.parquet")
val = pd.read_parquet(DATA / "validation.parquet")
cand = pd.read_parquet(CAND)
T = tag_matrix(games).tocsr()
gpos = pd.Series(np.arange(len(games)), index=games.app_id)
rec_ok = set(games.app_id[games.recommendable])


def prepare(h):
    """Candidates of one holdout with flags, hit marks, and the scored-player set (same population as evaluate.py)."""
    hcol = f"hidden_{h}"
    v = val
    vis = v[v.played & ~v[hcol]]
    hid = v[v[hcol]]
    target = hid[hid.enjoyed & hid.app_id.isin(rec_ok)]
    n_vis = vis.groupby("player_id").size()
    n_tgt = target.groupby("player_id").size()
    pids = n_tgt.index[n_tgt.index.isin(n_vis.index[n_vis >= 5])]
    pidx = pd.Series(np.arange(len(pids)), index=pids)
    c = cand[(cand.holdout == h) & cand.player_id.isin(pids)].copy()
    c["hit"] = c.merge(target[["player_id", "app_id"]].assign(t=True), on=["player_id", "app_id"], how="left").t.fillna(False).astype(bool).to_numpy()
    c["hit_any"] = c.merge(hid[["player_id", "app_id"]].assign(t=True), on=["player_id", "app_id"], how="left").t.fillna(False).astype(bool).to_numpy()
    vg = vis.app_id.map(gpos); vr = pidx.reindex(vis.player_id).to_numpy()
    ok = vg.notna().to_numpy() & ~np.isnan(vr)
    vgi, vpi, vpt = vg[ok].astype(int).to_numpy(), vr[ok].astype(int), vis.playtime_min.to_numpy()[ok]
    short = vpt < 120
    ps, pl = profile(T, vgi[short], vpi[short], len(pids)), profile(T, vgi[~short], vpi[~short], len(pids))
    has_s = np.bincount(vpi[short], minlength=len(pids)) > 0
    has_l = np.bincount(vpi[~short], minlength=len(pids)) > 0
    gi = c.app_id.map(gpos).astype(int).to_numpy(); pi = pidx[c.player_id].to_numpy()
    has_tags = np.asarray(T[gi].sum(1)).ravel() > 0
    c["quit"] = has_s[pi] & has_l[pi] & has_tags & (rowdot(T, gi, ps, pi) > rowdot(T, gi, pl, pi))
    c["short"] = games.median_playtime_min.to_numpy()[gi] <= 120
    c["neg"] = games.rating.isin(NEG).to_numpy()[gi]
    g = c.groupby("player_id").score
    c["z"] = (c.score - g.transform("mean")) / (g.transform("std") + 1e-9)
    idcg = pd.Series([DISC[:min(n, 10)].sum() for n in n_tgt.reindex(pids)], index=pids)
    return c, idcg


def top10(c, a, b, cc):
    s = c.z - a * c.quit - b * c.short - cc * c.neg
    t = c.assign(s=s).sort_values(["player_id", "s"], ascending=[True, False])
    t = t.groupby("player_id").head(10)
    t["rank"] = t.groupby("player_id").cumcount() + 1
    return t


def metrics(t, idcg):
    dcg = (t.hit * DISC[t["rank"].to_numpy() - 1]).groupby(t.player_id).sum().reindex(idcg.index).fillna(0)
    flags = t.quit.astype(int) + t.short.astype(int) + t.neg.astype(int)
    return {"NDCG@10": float((dcg / idcg).mean()), "HitRate@10 (any)": float(t.groupby("player_id").hit_any.any().reindex(idcg.index).fillna(False).mean()),
            "Any flag %": 100 * float((flags >= 1).mean()), "like quit %": 100 * float(t.quit.mean()), "short %": 100 * float(t.short.mean())}


grid = list(itertools.product([0, 0.25, 0.5, 1, 2, 4], [0, 0.25, 0.5, 1, 2], [0, 10]))
prep = {h: prepare(h) for h in ("75_25", "80_20", "50_50")}
c, idcg = prep["75_25"]
rows = [{"a (quit)": a, "b (short)": b, "c (negative)": cc, **metrics(top10(c, a, b, cc), idcg)} for a, b, cc in grid]
G = pd.DataFrame(rows)
base = G[(G["a (quit)"] == 0) & (G["b (short)"] == 0) & (G["c (negative)"] == 0)].iloc[0]
ok = G[G["NDCG@10"] >= 0.98 * base["NDCG@10"]]
best = ok.sort_values(["Any flag %", "NDCG@10"], ascending=[True, False]).iloc[0]
a, b, cc = best["a (quit)"], best["b (short)"], best["c (negative)"]
print(f"chosen on 75/25: a={a} b={b} c={cc}: NDCG {best['NDCG@10']:.4f} (plain {base['NDCG@10']:.4f}), flags {best['Any flag %']:.1f}% (plain {base['Any flag %']:.1f}%)")

out = []
for h, (c, idcg) in prep.items():
    t = top10(c, a, b, cc)
    out.append(t[["player_id", "holdout", "rank", "app_id"]])
    print(h, "plain", metrics(top10(c, 0, 0, 0), idcg), "\n   safe", metrics(t, idcg))
pd.concat(out).to_parquet(DATA / "recs_ease_safe.parquet", index=False)
(HERE / "rerank_choice.json").write_text(json.dumps({"a_quit": a, "b_short": b, "c_negative": cc, "chosen_on": "75_25",
                                                    "rule": "lowest any-flag share with NDCG@10 >= 98% of plain EASE"}, indent=2))
(HERE / "rerank_grid.md").write_text("# EASE-safe re-rank grid (75/25 holdout)\n\n" + G.sort_values("Any flag %").to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")
