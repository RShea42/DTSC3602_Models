"""Litmus test: hide a player's N most-played games, recommend from the rest of the library,
and see whether the hidden favourites come back.

  python litmus.py <steamid or profile URL>          # fetches once, then reuses libraries/<id>.json
  python litmus.py --library libraries/<id>.json --holdout 10

Reports, for each hidden game, its rank among all recommendable games (or why it can't be
recommended), hits in the top 15/50/100, and a "most popular games" baseline for comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from features import load_features
from recommend import (LIBRARIES, RecConfig, exclusion_reasons, first_look, get_library, recommend,
                       refine)


def popularity_ranking(f, owned: set[int], cfg: RecConfig) -> pd.DataFrame:
    """Baseline: rank eligible games by review volume x review score, ignoring taste."""
    ok = exclusion_reasons(f, owned, cfg).eq("").to_numpy()
    g = f.games[ok]
    pop = g.assign(score=g.log_reviews * g.review_score).sort_values("score", ascending=False)
    return pop.assign(rank=range(1, len(pop) + 1))[["appid", "rank"]]


def split(library: dict[int, dict], holdout: int) -> tuple[dict, dict]:
    """Hide the holdout most-played games (by total playtime). Hidden games are neither owned
    nor used for taste."""
    by_time = sorted(library.items(), key=lambda kv: kv[1]["minutes"], reverse=True)
    return dict(by_time[:holdout]), dict(by_time[holdout:])


def run(f, library: dict[int, dict], holdout: int, cfg: RecConfig, label: str) -> pd.DataFrame:
    hidden, visible = split(library, holdout)

    ranking = recommend(f, visible, cfg, full=True)   # final list order, family cap applied
    reasons = exclusion_reasons(f, set(visible), cfg)
    pop = popularity_ranking(f, set(visible), cfg)

    rows = []
    for appid, info in hidden.items():
        row = {"game": None, "hours": round(info["minutes"] / 60), "rank": None,
               "popularity_rank": None, "note": ""}
        if appid not in f.row_of:
            row.update(game=f"appid {appid}", note="not in catalog")
            rows.append(row)
            continue
        r = f.row_of[appid]
        row["game"] = f.games.name.iat[r]
        if reasons.iat[r]:
            row["note"] = f"excluded: {reasons.iat[r]}"
        else:
            # Its own slot, or its family's slot if another edition/sequel took the family's place.
            fam = ranking[ranking.family_id == f.games.family_id.iat[r]]
            row["rank"] = int(fam["rank"].iat[0])
            if fam.appid.iat[0] != appid:
                row["note"] = f"as {fam['name'].iat[0]}"
            row["popularity_rank"] = int(pop.loc[pop.appid == appid, "rank"].iat[0])
        rows.append(row)
    res = pd.DataFrame(rows)

    testable = res["rank"].notna()
    print(f"\n=== {label} ===")
    with pd.option_context("display.width", 200, "display.max_colwidth", 50):
        print(res.to_string(index=False))
    print(f"\n{testable.sum()} of {holdout} hidden games can be recommended at all.")
    for k in (15, 50, 100):
        model = int(res.loc[res["rank"] <= k, "rank"].nunique())   # one family slot = one hit
        base = int((res.popularity_rank <= k).sum())
        print(f"  top {k:>3}: model finds {model}, popularity baseline finds {base}")
    print(f"  ({len(ranking):,} games are eligible in total)")
    top = recommend(f, visible, cfg)
    print("\nThe 15 recommendations from the visible library:")
    for i, x in top.iterrows():
        mark = "  <-- hidden favourite" if x.appid in hidden else ""
        print(f"  {i + 1:>2}. {x['name']}  (because you played {x.because}){mark}")
    return res


def run_first_look(f, library: dict[int, dict], holdout: int, cfg: RecConfig,
                   like_threshold: float = 0.5) -> None:
    """Same hidden favourites, through the 5-card first look. A simulated player swipes a card up
    if it closely resembles one of their hidden favourites (tag similarity >= like_threshold) and
    down otherwise: a stand-in for a real person whose taste includes those games."""
    hidden_lib, visible = split(library, holdout)
    hidden = [a for a in hidden_lib if a in f.row_of]
    hidden_families = set(f.games.family_id.iloc[f.rows(hidden)])

    pool, deck = first_look(f, visible, cfg)
    H = f.X[f.rows(hidden)]
    reactions, print_rows = {}, []
    for _, card in deck.iterrows():
        sims = (f.X[f.row_of[int(card.appid)]] @ H.T).toarray().ravel()
        closest = f.games.name.iat[f.rows(hidden)[int(sims.argmax())]]
        reactions[int(card.appid)] = bool(sims.max() >= like_threshold)
        hit = "  <-- hidden favourite's family" if card.family_id in hidden_families else ""
        print_rows.append(f"  {'UP  ' if reactions[int(card.appid)] else 'down'}  {card['name']:<42} "
                          f"closest hidden: {closest} ({sims.max():.2f}){hit}")
    final = refine(f, visible, pool, deck, reactions, cfg)

    def hits(df):
        found = df[df.family_id.isin(hidden_families)]
        return len(set(found.family_id)), ", ".join(found.name)

    print(f"\n=== First look (simulated swipes: up if similarity to a hidden favourite >= "
          f"{like_threshold}) ===")
    print("\n".join(print_rows))
    for label, df in [("Before swiping (top 15)", pool.head(cfg.n)),
                      (f"Pool of {len(pool)} (the ceiling)", pool),
                      ("After swiping (final 15)", final)]:
        n, names = hits(df)
        print(f"  {label:<28} {n} hidden-favourite families: {names or '-'}")
    print("\nFinal 15 after swiping:")
    for _, x in final.iterrows():
        mark = "  <-- hidden favourite's family" if x.family_id in hidden_families else ""
        print(f"  {x['rank']:>2}. {x['name']}{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user", nargs="?")
    ap.add_argument("--library", type=Path)
    ap.add_argument("--holdout", type=int, default=10)
    ap.add_argument("--first-look", action="store_true",
                    help="also run the 5-card first look with simulated swipes")
    ap.add_argument("--no-recency", action="store_true", help="ignore when games were last played")
    ap.add_argument("--refresh", action="store_true", help="re-fetch the library from Steam")
    args = ap.parse_args()
    if args.refresh:
        args.library = None
    if not args.library and not args.refresh and args.user and args.user.isdigit():
        cached = LIBRARIES / f"{args.user}.json"
        args.library = cached if cached.exists() else None
    if not (args.user or args.library):
        ap.error("give a Steam user or --library")
    f = load_features()
    library = get_library(args.user, args.library)
    print(f"Library: {len(library)} games, {sum(v['minutes'] > 0 for v in library.values())} played.")
    base = RecConfig()
    if args.no_recency:
        base = replace(base, recency_half_life_days=0, recent_2weeks_boost=0)
    for method in ("knn", "centroid"):
        run(f, library, args.holdout, replace(base, method=method), f"method = {method}")
    if args.first_look:
        run_first_look(f, library, args.holdout, base)


if __name__ == "__main__":
    main()
