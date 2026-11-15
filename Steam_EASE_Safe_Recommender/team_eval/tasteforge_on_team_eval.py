"""Score TasteForge's recommender (zcrawfo6/Steamapp, branch peyton-dev, the newest version) on the shared team data.

Same 1,000 validation players as the KNN head-to-head (data_clean/team_eval_rob/validation.parquet), all three
holdouts, scored by the unchanged team_eval/evaluate.py next to EASE, EASE-safe, Rob's KNN and MostPopular.

Faithful to peyton-dev's app.py and games_db.py:
  - catalog: games_db.build_catalog over the `games` table (= data_clean/games.parquet), min 10 reviews,
    no explicit games, sparse tags filled from games_slim.csv
  - a Steam import: every owned game in the catalog with its lifetime hours; games with >= 2.5 h shape the profile
  - play-mode shares: even across the modes the game's tags suggest (the app's default)
  - recommend(G, P, owned, w_genre=70, budget=999 (no cap), n=10)
The genre-fit sum is computed with a sparse matrix instead of pandas explode/groupby (same numbers; checked
against the app's own function on sample players before the run).
  PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python team_eval/tasteforge_on_team_eval.py
"""
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

TF = Path("C:/Users/juant/team 3000/Steamapp")
sys.path.insert(0, str(TF))
import games_db as db  # noqa: E402  (peyton-dev version)

HERE = Path(__file__).resolve().parent
SRC = Path("C:/Users/juant/data_clean/team_eval")
ROBDIR = Path("C:/Users/juant/data_clean/team_eval_rob")       # the same 1,000 players
OUT = Path("C:/Users/juant/data_clean/team_eval_all")
HOLD = ["80_20", "75_25", "50_50"]
MIN_HOURS, W_GENRE, BUDGET, N = 2.5, 70, 999, 10

# ---- copied from peyton-dev app.py
MODES = ["story", "coop", "pvp"]
STORY_T = {"story rich", "singleplayer", "choices matter", "multiple endings", "visual novel", "interactive fiction",
           "rpg", "jrpg", "adventure", "action-adventure", "walking simulator"}
COOP_T = {"co-op", "online co-op", "local co-op", "co-op campaign"}
PVP_T = {"pvp", "competitive", "fighting", "2d fighter", "arena shooter", "esports", "moba", "battle royale",
         "hero shooter"}
MULTI_T = {"multiplayer", "massively multiplayer", "mmorpg", "local multiplayer"}


def modes_for(genres):
    s = {str(x).strip().lower() for x in genres}
    coop = bool(s & COOP_T)
    has = {"story": bool(s & STORY_T), "coop": coop, "pvp": bool(s & PVP_T or (s & MULTI_T and not coop))}
    return [m for m in MODES if has[m]] or ["story"]


def profile(G, played, feats):
    aff, tot, wp = {}, 0.0, 0.0
    for gid, h in played.items():
        g = G.loc[gid]
        tot += h
        wp += math.log1p(g.price) * h
        for x in g.genres:
            aff[x] = aff.get(x, 0) + h / len(g.genres)
    aff = {k: v / (tot or 1) for k, v in aff.items()}
    lp = wp / tot if tot else 0
    modes = {}
    for gid, h in played.items():
        for m, share in (feats.get(gid) or {}).items():
            modes[m] = modes.get(m, 0) + h * share
    return dict(tot=tot, aff=aff, lp=lp, avg_price=math.exp(lp) - 1, n=len(played), modes=modes)


def recommend_app(G, P, played, w_genre, budget, n=None):
    """The app's own recommend(), verbatim apart from returning (gid, score)."""
    pool = G[~G.index.isin(played) & (G.price <= budget)]
    aff = P["aff"]
    max_a = max(aff.values(), default=0.0001)
    w_g = w_genre / 100
    th = sum(P["modes"].values())
    share = {m: v / th for m, v in P["modes"].items()} if th else {}
    gm = pool.genres.explode().map(aff).fillna(0).groupby(level=0).sum() / pool.genres.str.len() / max_a
    d = np.log1p(pool.price) - P["lp"]
    pm = np.exp(-(d * d) / (2 * 1.1 * 1.1))
    q = (pool.rating - 70) / 30
    mf = pool.modes.apply(lambda ms: min(1, sum(share.get(m, 0) for m in ms))) if share else pd.Series(.5, pool.index)
    score = ((w_g * gm + (1 - w_g) * pm) * 0.8 + q * 0.1 + mf * 0.1).clip(0, 1)
    top = score.sort_values(ascending=False, kind="stable").index[:n]
    return [(int(g), float(score[g])) for g in top]


# ---- catalog, exactly as the app builds it
def csv_tags():
    d = pd.read_csv(TF / "games_slim.csv", usecols=["app_id", "tags"]).drop_duplicates("app_id")
    d = d[d.tags.notna() & (d.tags != "")]
    return {int(a): [t.strip() for t in tags.split(",") if t.strip()] for a, tags in zip(d.app_id, d.tags)}


raw = pd.read_parquet("C:/Users/juant/data_clean/games.parquet", columns=db.COLS.split(","))
raw["tags"] = raw.tags.apply(lambda t: list(t) if t is not None and not isinstance(t, float) else [])
G = db.build_catalog(raw, 10, extra_tags=csv_tags())
G["modes"] = G.genres.apply(modes_for)
print(f"TasteForge catalog: {len(G):,} games", flush=True)

# fast genre-fit: game x tag incidence matrix
vocab = {t: i for i, t in enumerate(sorted({t for gs in G.genres for t in gs}))}
rows = np.repeat(np.arange(len(G)), G.genres.str.len().to_numpy())
cols = np.array([vocab[t] for gs in G.genres for t in gs])
M = sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(G), len(vocab)))
glen = G.genres.str.len().to_numpy().astype(float)
lp_price = np.log1p(G.price.to_numpy())
q_all = ((G.rating - 70) / 30).to_numpy()
mode_mat = np.array([[m in ms for m in MODES] for ms in G.modes], dtype=float)
price_ok = (G.price <= BUDGET).to_numpy()


def recommend_fast(P, owned_idx, n=N):
    aff = P["aff"]
    max_a = max(aff.values(), default=0.0001)
    w_g = W_GENRE / 100
    a = np.zeros(len(vocab))
    for t, v in aff.items():
        a[vocab[t]] = v
    gm = (M @ a) / glen / max_a
    d = lp_price - P["lp"]
    pm = np.exp(-(d * d) / (2 * 1.1 * 1.1))
    th = sum(P["modes"].values())
    if th:
        sh = np.array([P["modes"].get(m, 0) / th for m in MODES])
        mf = np.minimum(1, mode_mat @ sh)
    else:
        mf = np.full(len(G), .5)
    score = np.clip((w_g * gm + (1 - w_g) * pm) * 0.8 + q_all * 0.1 + mf * 0.1, 0, 1)
    ok = price_ok.copy()
    ok[owned_idx] = False
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(-score[idx], kind="stable")][:n]
    return [(int(g), float(score[g])) for g in order]


gid_of = dict(zip(G.app_id.astype(int), G.index))
val = pd.read_parquet(ROBDIR / "validation.parquet")
players = np.sort(val.player_id.unique())


def player_input(lib, h):
    keep = lib[~lib[f"hidden_{h}"]]
    owned = {}
    for a, pl, pt in zip(keep.app_id, keep.played, keep.playtime_min.fillna(0)):
        g = gid_of.get(int(a))
        if g is not None:
            owned[g] = min(9999.0, round((pt if pl else 0) / 60, 1))
    played = {g: hr for g, hr in owned.items() if hr >= MIN_HOURS}
    feats = {g: {m: 1 / len(G.at[g, "modes"]) for m in G.at[g, "modes"]} for g in owned}
    return owned, played, feats


def main():
    # check: the fast version gives the app's exact lists
    checked = 0
    for pid in players[:15]:
        lib = val[val.player_id == pid]
        owned, played, feats = player_input(lib, "80_20")
        P = profile(G, played, feats)
        a = [g for g, _ in recommend_app(G, P, owned, W_GENRE, BUDGET, N)]
        b = [g for g, _ in recommend_fast(P, list(owned))]
        assert a == b, (pid, a, b)
        checked += 1
    print(f"fast scorer matches the app's recommend() on {checked} of {checked} players", flush=True)

    out, t0, empty = [], time.time(), 0
    for i, pid in enumerate(players):
        lib = val[val.player_id == pid]
        for h in HOLD:
            owned, played, feats = player_input(lib, h)
            if not played:
                empty += 1
            P = profile(G, played, feats)
            for r, (g, _) in enumerate(recommend_fast(P, list(owned)), 1):
                out.append((int(pid), h, r, int(G.at[g, "app_id"])))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(players)} players, {time.time() - t0:.0f}s", flush=True)
    print(f"runs with no game of 2.5+ hours (the app still lists games, scored on price, rating and modes): {empty}")

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(out, columns=["player_id", "holdout", "rank", "app_id"]).to_parquet(OUT / "recs_tasteforge.parquet")
    for f in ["games.parquet", "validation.parquet", "recs_rob_knn.parquet", "recs_ease.parquet", "recs_ease_safe.parquet",
              "recs_mostpopular.parquet"]:
        (OUT / f).write_bytes((ROBDIR / f).read_bytes())
    files = [str(OUT / f"recs_{m}.parquet") for m in ["ease_safe", "ease", "rob_knn", "tasteforge", "mostpopular"]]
    subprocess.run([sys.executable, str(HERE / "evaluate.py"), "--data", str(OUT), *files,
                    "--out", str(HERE / "results_all_models.md")], check=True)
    print("done -> team_eval/results_all_models.md")


if __name__ == "__main__":
    main()
