"""M1 follow-up: the default (serve_filters=True) must match Juan's live app, and we report the app version's score.

1. Parity: for 500 validation players (80/20 visible library), Juan's Recommender (model/recommender.py) "For you"
   order before the live-price step vs the pipeline's default top 10. Live prices need the Steam store, so they stay
   out of both (they move to FastAPI in M4).
2. Score: the app version on all three holdouts with Juan's evaluate.score(), next to the evaluation version.
Uses m1_out/ease_safe_pipeline.joblib from m1_check.py.
"""
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from paths import EASE_FILES, JUAN, TE
HERE = Path(__file__).resolve().parent
FILES = EASE_FILES
os.environ["STEAMREC_ARTIFACTS"] = str(FILES / "model" / "artifacts")
os.environ["STEAMREC_DATA"] = str(FILES / "data_clean")
sys.path.insert(0, str(JUAN / "model"))
sys.path.insert(0, str(JUAN / "team_eval"))
from recommender import Recommender  # noqa: E402
from evaluate import score, tag_matrix  # noqa: E402

from steamrec.data import HOLDOUTS, holdout_input, predict_holdouts  # noqa: E402

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

pipe = joblib.load(HERE / "m1_out" / "ease_safe_pipeline.joblib")
pipe.set_params(ease__serve_filters=True, ease__app_catalog_path=str(FILES / "data_clean" / "games.parquet"))
juan = Recommender()
val = pd.read_parquet(TE / "validation.parquet")
log("loaded Juan's app model and the pipeline (app mode)")

# ---- 1. parity with Juan's app
inp = holdout_input(val, "80_20")
players = np.random.default_rng(7).choice(np.sort(inp[inp.played].player_id.unique()), 500, replace=False)
mine = pipe.predict(inp[inp.player_id.isin(players)])
same = differ = 0
for pid, lib in inp[inp.player_id.isin(players)].groupby("player_id"):
    played = [(int(a), int(p)) for a, p, pl in zip(lib.app_id, lib.playtime_min, lib.played) if pl and p > 0]
    s, rows = juan.taste(played)
    if s is None:
        continue
    mask = juan.candidate & ~juan.explicit
    for a in lib.app_id:
        i = juan.idx.get(int(a))
        if i is not None:
            mask[i] = False
    s = np.where(mask, s, -np.inf)
    order = [j for j in np.argsort(-s)[:600] if np.isfinite(s[j])]
    theirs = [int(juan.app_ids[j]) for j in juan.safe_rerank(order, s, played)[:10]]
    ours = mine[mine.player_id == pid].sort_values("rank").app_id.tolist()
    same += theirs == ours
    differ += theirs != ours
log(f"app parity: {same} of {same + differ} players get exactly Juan's app's top 10 ({differ} differ)")

# ---- 2. score the app version with Juan's evaluator
games = pd.read_parquet(TE / "games.parquet")
T = tag_matrix(games).tocsr()
recs_app = predict_holdouts(pipe, val)
recs_app.to_parquet(HERE / "m1_out" / "recs_ease_safe_app.parquet", index=False)
recs_eval = pd.read_parquet(HERE / "m1_out" / "recs_ease_safe.parquet")
rows = []
for h in HOLDOUTS:
    for name, r in [("evaluation version (Juan's scored setup)", recs_eval), ("app version (default)", recs_app)]:
        o, _ = score(r, val, games, T, h)
        rows.append({"holdout": h, "version": name, **{k: round(o[k], 4) for k in ["NDCG@10", "HitRate@10 (any)", "Any flag %", "Regret %", "Coverage@10"]}})
tab = pd.DataFrame(rows)
(HERE / "m1_out" / "results_app_vs_eval.md").write_text(tab.to_markdown(index=False), encoding="utf-8")
print(tab.to_markdown(index=False))
