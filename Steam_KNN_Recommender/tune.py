"""Blind-test sweep of RecConfig settings (no first look, no swipes).

For each test library, hide some games, recommend from the rest, and count how many hidden games
(or another game in their series) land in the top 10 / 15 / 50 of the plain ranked list. Each
setting is varied one at a time from the defaults, so you can see what each knob does.

  python tune.py --library libraries/<id>.json                     # your library, both blind tests
  python tune.py --library libraries/<id>.json --extra other.json   # add more libraries
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from features import load_features
from recommend import RecConfig, get_library, recommend

# Settings to sweep: name -> values to try (the default is marked in the output).
SWEEPS = {
    "collapse_franchises": [True, False],
    "recency_half_life_days": [0, 91, 182, 365, 730],
    "quality_weight": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "dislike_weight": [0.0, 0.25, 0.5, 1.0],
    "liked_count": [5, 10, 15, 25, 40],
    "knn_k": [1, 2, 3, 5],
    "max_per_anchor": [1, 2, 3, 15],
    "max_franchise_picks": [0, 1, 3, 15],
    "method": ["knn", "centroid"],
}
CUTOFFS = (10, 15, 50)


def holdout(library: dict[int, dict], by: str, n: int) -> tuple[list[int], dict]:
    """Hide the n most-played ('played') or most recently played ('recent') games."""
    if by == "played":
        order = sorted(library, key=lambda a: library[a]["minutes"], reverse=True)
    else:
        order = sorted((a for a in library if library[a]["minutes"] > 0),
                       key=lambda a: library[a]["last_played"], reverse=True)
    hidden = order[:n]
    return hidden, {a: v for a, v in library.items() if a not in set(hidden)}


def blind_hits(f, visible: dict, hidden: list[int], cfg: RecConfig) -> dict[int, int]:
    """{cutoff: number of hidden-game families in the top cutoff}."""
    families = set(f.games.family_id.iloc[f.rows(hidden)])
    ranked = recommend(f, visible, cfg, full=True).head(max(CUTOFFS))
    return {k: ranked.head(k).family_id[ranked.head(k).family_id.isin(families)].nunique()
            for k in CUTOFFS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path, required=True, help="main library (both blind tests)")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="more libraries (most-played test)")
    ap.add_argument("--hide", type=int, default=10)
    args = ap.parse_args()

    f = load_features()
    main_lib = get_library(None, args.library)
    tests = {"A: yours, most played": holdout(main_lib, "played", args.hide)}
    if any(v["last_played"] for v in main_lib.values()):
        tests["B: yours, recently played"] = holdout(main_lib, "recent", args.hide)
    for i, p in enumerate(args.extra):
        tests[f"{chr(67 + i)}: {p.stem}, most played"] = holdout(get_library(None, p), "played", args.hide)
    for name, (hidden, _) in tests.items():
        in_catalog = len(f.rows(hidden))
        print(f"{name}: {len(hidden)} hidden, {in_catalog} in the catalog "
              f"(max possible {len(set(f.games.family_id.iloc[f.rows(hidden)]))} series)")

    default = RecConfig()
    rows, start = [], time.time()
    for setting, values in SWEEPS.items():
        for value in values:
            cfg = replace(default, **{setting: value})
            row = {"setting": setting, "value": value,
                   "is_default": value == getattr(default, setting)}
            for name, (hidden, visible) in tests.items():
                hits = blind_hits(f, visible, hidden, cfg)
                for k in CUTOFFS:
                    row[f"{name.split(':')[0]}@{k}"] = hits[k]
            rows.append(row)
    res = pd.DataFrame(rows)
    top10 = [c for c in res.columns if c.endswith("@10")]
    res["total@10"] = res[top10].sum(axis=1)
    res["total@15"] = res[[c for c in res.columns if c.endswith("@15") and c != "total@15"]].sum(axis=1)
    res["value"] = res.apply(lambda r: f"{r.value} *" if r.is_default else str(r.value), axis=1)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(res.drop(columns="is_default").to_string(index=False))
    print(f"\n(* = current default; {len(rows)} settings tested in {time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
