"""Score top-10 recommendation lists on the shared validation holdouts. Needs only pandas, numpy, pyarrow, scipy.

Usage:
  python evaluate.py --data <team_eval folder> recs_ours.parquet recs_yours.parquet [...] [--out results.md]

A recs file (parquet or csv) has one row per pick:
  player_id   from validation.parquet
  holdout     "80_20", "75_25" or "50_50"
  rank        1..10 (1 = best); rows with rank > 10 are ignored
  app_id      a Steam app id; only games with recommendable == True can count as hits
The model may only use the player's visible games (played games where hidden_<holdout> is False). Do not recommend
games the player already owns, except hidden ones; owned picks are counted as misses and reported.

Who is scored: players with >= 1 hidden enjoyed recommendable game and >= 5 visible played games (our main population).
A player with no picks still counts, with zeros.

Metrics (all on the top 10, averaged over players):
  NDCG@10           hidden ENJOYED games found, rewarded more near the top (our primary metric)
  HitRate@10        share of players with at least one hidden enjoyed game in the list
  HitRate@10 (any)  same, but any hidden PLAYED game counts (Rob's "did it find a hidden game")
  Coverage@10       share of recommendable games that appear in anyone's list
  Popularity pct    average popularity percentile of picks (100 = the most-played game)
  Tag distance      how unlike the player's visible games the picks are (0 = same tags, 1 = no shared tags)
  Playtime gap      |log10(game median playtime / player median playtime)|, over picks with a known median playtime;
                    0 = same, 0.3 = 2x apart. "Within 2x" = share of those picks at most 2x apart. Medians, because
                    playtime is heavily skewed; "(avg)" repeats it with averages (player mean vs game average), as Rob specified.
  Regret            among picks the player had actually played (hidden), share played under 2 hours
  Warning flags (share of picks), the composite soft-fail check:
    negative reviews   Steam rating Mostly Negative or worse (or under 40% positive with >= 10 reviews)
    short game         median playtime <= 120 minutes (known for about 22% of games)
    like quit games    the game's tags are closer to the player's visible games played < 2 h than to the ones played >= 2 h
                       (needs both kinds in the player's visible history)
    any flag / 2+ flags
With two or more files, the first is the reference: paired differences in NDCG@10 per holdout with a 95% bootstrap CI.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HOLDOUTS = ["80_20", "75_25", "50_50"]
NEG = {"Mostly Negative", "Negative", "Very Negative", "Overwhelmingly Negative"}
DISC = 1 / np.log2(np.arange(2, 12))


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


def score(recs, val, games, T, holdout):
    hcol = f"hidden_{holdout}"
    gpos = pd.Series(np.arange(len(games)), index=games.app_id)
    rec_ok = set(games.app_id[games.recommendable])
    v = val.copy()
    v["rec"] = v.app_id.isin(rec_ok)
    vis = v[v.played & ~v[hcol]]
    hid = v[v[hcol]]
    target = hid[hid.enjoyed & hid.rec]
    n_vis = vis.groupby("player_id").size()
    n_tgt = target.groupby("player_id").size()
    pids = n_tgt.index[n_tgt.index.isin(n_vis.index[n_vis >= 5])]
    P = pd.DataFrame(index=pids)
    P["n_target"] = n_tgt.reindex(pids)
    pidx = pd.Series(np.arange(len(pids)), index=pids)

    r = recs[(recs.holdout == holdout) & (recs["rank"] <= 10) & recs.player_id.isin(pids)].copy()
    r = r.sort_values(["player_id", "rank"]).drop_duplicates(["player_id", "app_id"])
    r["rank"] = r.groupby("player_id").cumcount() + 1
    owned = v[~v[hcol]][["player_id", "app_id"]].assign(owned=True)
    r = r.merge(owned, on=["player_id", "app_id"], how="left")
    r["owned"] = r.owned.fillna(False).astype(bool)
    r["hit"] = r.merge(target[["player_id", "app_id"]].assign(t=True), on=["player_id", "app_id"], how="left").t.fillna(False).to_numpy()
    r["hit"] = r.hit.astype(bool) & ~r.owned
    hp = hid[["player_id", "app_id", "playtime_min"]].rename(columns={"playtime_min": "hid_pt"})
    r = r.merge(hp, on=["player_id", "app_id"], how="left")
    r["gi"] = r.app_id.map(gpos)

    r["dcg"] = r.hit * DISC[(r["rank"] - 1).clip(0, 9).to_numpy()]
    idcg = np.array([DISC[:min(n, 10)].sum() for n in P.n_target])
    P["ndcg10"] = r.groupby("player_id").dcg.sum().reindex(pids).fillna(0).to_numpy() / idcg
    P["hit10"] = r.groupby("player_id").hit.any().reindex(pids).fillna(False).astype(float)
    P["hit10_any"] = r.assign(h=r.hid_pt.notna() & ~r.owned).groupby("player_id").h.any().reindex(pids).fillna(False).astype(float)

    # player-level references
    vg = vis.app_id.map(gpos)
    okv = vg.notna().to_numpy()
    vrow = pidx.reindex(vis.player_id).to_numpy()
    sel = okv & ~np.isnan(vrow)
    vgi, vpi, vpt = vg[sel].astype(int).to_numpy(), vrow[sel].astype(int), vis.playtime_min.to_numpy()[sel]
    prof = profile(T, vgi, vpi, len(pids), np.log1p(vpt))
    short = vpt < 120
    prof_s = profile(T, vgi[short], vpi[short], len(pids))
    prof_l = profile(T, vgi[~short], vpi[~short], len(pids))
    has_s = np.bincount(vpi[short], minlength=len(pids)) > 0
    has_l = np.bincount(vpi[~short], minlength=len(pids)) > 0
    med_pt = vis.groupby("player_id").playtime_min.median().reindex(pids).to_numpy()
    avg_pt = vis.groupby("player_id").playtime_min.mean().reindex(pids).to_numpy()

    k = r[r.gi.notna()].copy()
    k["pi"] = pidx[k.player_id].to_numpy()
    gi, pi = k.gi.astype(int).to_numpy(), k.pi.to_numpy()
    k["tagdist"] = 1 - rowdot(T, gi, prof, pi)
    has_tags = np.asarray(T[gi].sum(1)).ravel() > 0
    k.loc[~has_tags, "tagdist"] = np.nan
    gm = games.median_playtime_min.to_numpy()[gi]
    pm = med_pt[pi]
    with np.errstate(divide="ignore", invalid="ignore"):
        k["pt_gap"] = np.abs(np.log10(gm / pm))
    k.loc[~(gm > 0) | ~(pm > 0), "pt_gap"] = np.nan
    ga, pa = games.avg_playtime_min.to_numpy()[gi], avg_pt[pi]           # Rob's version: averages instead of medians
    with np.errstate(divide="ignore", invalid="ignore"):
        k["pt_gap_avg"] = np.abs(np.log10(ga / pa))
    k.loc[~(ga > 0) | ~(pa > 0), "pt_gap_avg"] = np.nan
    k["flag_neg"] = games.rating.isin(NEG).to_numpy()[gi]
    k["flag_short"] = gm <= 120
    both = has_s[pi] & has_l[pi] & has_tags
    k["flag_quit"] = both & (rowdot(T, gi, prof_s, pi) > rowdot(T, gi, prof_l, pi))
    k["n_flags"] = k.flag_neg.astype(int) + k.flag_short.astype(int) + k.flag_quit.astype(int)
    k["pop_pct"] = games.n_train_players.rank(pct=True).to_numpy()[gi] * 100

    played = k[k.hid_pt.notna()]
    out = {
        "players": len(pids), "picks per player": len(r) / len(pids), "owned picks %": 100 * r.owned.mean() if len(r) else np.nan,
        "not recommendable %": 100 * (~r.app_id.isin(rec_ok)).mean() if len(r) else np.nan,
        "NDCG@10": P.ndcg10.mean(), "HitRate@10": P.hit10.mean(), "HitRate@10 (any)": P.hit10_any.mean(),
        "Coverage@10": r.app_id[r.app_id.isin(rec_ok)].nunique() / len(rec_ok),
        "Popularity pct": k.pop_pct.mean(), "Tag distance": k.tagdist.mean(),
        "Playtime gap": k.pt_gap.mean(), "Within 2x %": 100 * (k.pt_gap.dropna() <= np.log10(2)).mean(),
        "Playtime known %": 100 * k.pt_gap.notna().mean(),
        "Playtime gap (avg)": k.pt_gap_avg.mean(), "Within 2x (avg) %": 100 * (k.pt_gap_avg.dropna() <= np.log10(2)).mean(),
        "Regret %": 100 * (played.hid_pt < 120).mean() if len(played) else np.nan,
        "Flag: negative %": 100 * k.flag_neg.mean(), "Flag: short %": 100 * k.flag_short.mean(),
        "Flag: like quit %": 100 * k.flag_quit.mean(), "Any flag %": 100 * (k.n_flags >= 1).mean(),
        "2+ flags %": 100 * (k.n_flags >= 2).mean()}
    return out, P.ndcg10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="+")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--out")
    ap.add_argument("--holdouts", default=",".join(HOLDOUTS))
    a = ap.parse_args()
    d = Path(a.data)
    games = pd.read_parquet(d / "games.parquet")
    val = pd.read_parquet(d / "validation.parquet")
    T = tag_matrix(games).tocsr()
    rd = lambda f: pd.read_parquet(f) if str(f).endswith(".parquet") else pd.read_csv(f)
    names = [Path(f).stem.removeprefix("recs_") for f in a.recs]
    lines = ["# Team comparison on validation players", ""]
    for h in a.holdouts.split(","):
        res, per = {}, {}
        for n, f in zip(names, a.recs):
            res[n], per[n] = score(rd(f), val, games, T, h)
            print(f"{h} {n}: NDCG@10 {res[n]['NDCG@10']:.4f}", flush=True)
        tab = pd.DataFrame(res)
        lines += [f"## Holdout {h.replace('_', '/')} (visible/hidden)", "", tab.to_markdown(floatfmt=".4f"), ""]
        if len(names) > 1:
            rng = np.random.default_rng(0)
            rows = []
            for n in names[1:]:
                diff = (per[n] - per[names[0]]).to_numpy()
                b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
                rows.append({"model": n, f"NDCG@10 minus {names[0]}": diff.mean(),
                             "95% CI": f"{np.percentile(b, 2.5):+.4f} to {np.percentile(b, 97.5):+.4f}"})
            lines += [pd.DataFrame(rows).to_markdown(index=False, floatfmt="+.4f"), ""]
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
