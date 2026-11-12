"""Score Rob's KNN (DTSC3602_Models/Steam_KNN_Recommender) on the shared team package, next to our models.

Rob's model takes ~2 s per player, so we use a fixed random sample of N players (seed 42) drawn from the players
evaluate.py scores on the 80/20 holdout, and run all three holdouts for them. Every model's lists are cut to the same
players, and a reduced data folder (games.parquet + those players' validation rows) is scored by the unchanged
evaluate.py, so the comparison is like for like.
Rob's input per player: visible played games with minutes, plus owned unplayed games with 0 minutes (so he never
recommends them); hidden games are left out. Recency is off (the package has no last-played dates). Top 10 of his
final ranking (one game per family, as his app shows).
  PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python team_eval/rob_on_team_eval.py [N] [workers]
"""
import os
import subprocess
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROB = Path("C:/Users/juant/team 3000/DTSC3602_Models/Steam_KNN_Recommender")
HERE = Path(__file__).resolve().parent
DATA = Path("C:/Users/juant/data_clean/team_eval")
OUT = Path("C:/Users/juant/data_clean/team_eval_rob")
HOLD = ["80_20", "75_25", "50_50"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
W = int(sys.argv[2]) if len(sys.argv) > 2 else 4

_f = _cfg = None


def _init():
    global _f, _cfg
    sys.path.insert(0, str(ROB))
    from features import load_features
    from recommend import RecConfig
    _f = load_features()
    _cfg = replace(RecConfig(), recency_half_life_days=0, recent_2weeks_boost=0)


def _one(job):
    from recommend import recommend
    pid, h, lib = job
    try:
        rank = recommend(_f, lib, _cfg, full=True)
    except ValueError:                       # Rob's model can't recommend (no played game in his catalog): no picks
        return [(pid, h, 0, -1)]
    except Exception:                        # a crash inside Rob's code for this library: no picks, counted apart
        return [(pid, h, 0, -2)]
    top = rank.sort_values("rank").head(10)
    return [(pid, h, r + 1, int(a)) for r, a in enumerate(top.appid)]


def main():
    games = pd.read_parquet(DATA / "games.parquet")
    val = pd.read_parquet(DATA / "validation.parquet")
    rec_ok = set(games.app_id[games.recommendable])
    # players evaluate.py scores on 80/20: >= 1 hidden enjoyed recommendable game and >= 5 visible played games
    hid = val[val.hidden_80_20 & val.enjoyed & val.app_id.isin(rec_ok)].groupby("player_id").size()
    vis = val[val.played & ~val.hidden_80_20].groupby("player_id").size()
    pool = np.sort(hid.index[hid.index.isin(vis.index[vis >= 5])].to_numpy())
    players = np.random.default_rng(42).choice(pool, size=min(N, len(pool)), replace=False)
    v = val[val.player_id.isin(players)]

    jobs = []
    for pid, lib in v.groupby("player_id"):
        for h in HOLD:
            keep = lib[~lib[f"hidden_{h}"]]
            mins = np.where(keep.played, keep.playtime_min.fillna(0), 0).astype(int)
            jobs.append((int(pid), h, {int(a): {"minutes": int(m), "minutes_2weeks": 0, "last_played": 0}
                                       for a, m in zip(keep.app_id, mins)}))
    print(f"{len(players)} players x {len(HOLD)} holdouts = {len(jobs)} runs, {W} workers", flush=True)

    # Progress is saved after every finished run; a rerun skips what is already done.
    OUT.mkdir(exist_ok=True)
    # picks go to progress.csv; a run's "player,holdout" goes to done.csv only after all its picks are on disk,
    # so a run cut off mid-write is redone and its partial rows are ignored.
    prog, donef = OUT / "progress.csv", OUT / "done.csv"
    done = set()
    if donef.exists():
        d = pd.read_csv(donef, names=["player_id", "holdout"], dtype={"holdout": str}, on_bad_lines="skip")
        done = set(zip(d.player_id.astype(int), d.holdout))
        print(f"resuming: {len(done)} of {len(jobs)} runs already saved", flush=True)
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    t0 = time.time()
    with open(prog, "a", encoding="utf-8") as fp, open(donef, "a", encoding="utf-8") as fd, \
            Pool(W, initializer=_init) as p:
        for i, r in enumerate(p.imap_unordered(_one, todo, chunksize=2)):
            fp.write("".join(f"{a},{b},{c},{e}\n" for a, b, c, e in r))
            fp.flush()
            os.fsync(fp.fileno())
            fd.write(f"{r[0][0]},{r[0][1]}\n")
            fd.flush()
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"{len(done) + i + 1}/{len(jobs)}  {el / 60:.1f} min, ~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left", flush=True)
    d = pd.read_csv(donef, names=["player_id", "holdout"], dtype={"holdout": str}, on_bad_lines="skip")
    keep = set(zip(d.player_id.astype(int), d.holdout)) & {(j[0], j[1]) for j in jobs}
    pr = pd.read_csv(prog, names=["player_id", "holdout", "rank", "app_id"], dtype={"holdout": str},
                     on_bad_lines="skip").dropna()
    pr = pr.astype({"player_id": int, "rank": int, "app_id": int})
    pr = pr[[k in keep for k in zip(pr.player_id, pr.holdout)]].drop_duplicates(["player_id", "holdout", "rank"])
    if len(keep) < len(jobs):
        sys.exit(f"only {len(keep)} of {len(jobs)} runs finished; run again to resume")
    rows = list(pr.itertuples(index=False, name=None))

    recs = pd.DataFrame(rows, columns=["player_id", "holdout", "rank", "app_id"])
    failed = recs[recs["rank"] == 0]
    print(f"Rob's model returned no picks for {len(failed)} of {len(jobs)} runs, scored as zero: "
          f"{(failed.app_id == -1).sum()} with no played game in his catalog, "
          f"{(failed.app_id == -2).sum()} crashed inside his code", flush=True)
    recs[recs["rank"] > 0].to_parquet(OUT / "recs_rob_knn.parquet")
    games.to_parquet(OUT / "games.parquet")
    v.to_parquet(OUT / "validation.parquet")
    for m in ["ease", "ease_safe", "hybrid_ease", "mostpopular"]:
        r = pd.read_parquet(DATA / f"recs_{m}.parquet")
        r[r.player_id.isin(players)].to_parquet(OUT / f"recs_{m}.parquet")
    files = [str(OUT / f"recs_{m}.parquet") for m in ["ease", "rob_knn", "ease_safe", "mostpopular"]]
    subprocess.run([sys.executable, str(HERE / "evaluate.py"), "--data", str(OUT), *files,
                    "--out", str(HERE / "results_vs_rob.md")], check=True)
    print(f"done in {(time.time() - t0) / 60:.0f} min -> team_eval/results_vs_rob.md")


if __name__ == "__main__":
    main()
