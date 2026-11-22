"""Rubric check: matching local and deployed outputs.

The same 200 real libraries (validation players, 80/20 visible games) go to:
  1. the saved pipeline, loaded locally and called with plain scikit-learn  pipe.predict(X)
  2. the local FastAPI app        POST <local>/predict     (uvicorn app.api:app --port 8001)
  3. the deployed API on Modal    POST <deployed>/predict
Pass = every player gets the same games in the same order from all three, with the same scores.
Writes m5_out/requests.json (the exact inputs) and m5_out/verify_report.md.

  python m5_verify.py --local http://127.0.0.1:8001 --deployed https://<workspace>--steam-ease-safe-api.modal.run
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from paths import SERVE, TE
HERE = Path(__file__).resolve().parent
SF = SERVE
OUT = HERE / "m5_out"
OUT.mkdir(exist_ok=True)
K = 10

ap = argparse.ArgumentParser()
ap.add_argument("--local", default="http://127.0.0.1:8001")
ap.add_argument("--deployed")
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()

# ---- the inputs: 200 real libraries, saved so anyone can re-send them
val = pd.read_parquet(TE / "validation.parquet")
v = val[~val.hidden_80_20]
pids = np.random.default_rng(42).choice(np.sort(v[v.played].player_id.unique()), a.n, replace=False)
reqs = {int(p): {"library": [{"app_id": int(x), "playtime_min": float(t) if pl else 0.0}
                             for x, t, pl in zip(g.app_id, g.playtime_min, g.played)], "k": K}
        for p, g in v[v.player_id.isin(pids)].groupby("player_id")}
(OUT / "requests.json").write_text(json.dumps(reqs), encoding="utf-8")

# ---- 1. saved pipeline, plain scikit-learn, all libraries in one call
pipe = joblib.load(SF / "ease_safe_pipeline.joblib")
pipe.set_params(ease__app_catalog_path=str(SF / "games.parquet"), ease__catalog_path=str(SF / "team_eval_games.parquet"),
                safe__catalog_path=str(SF / "team_eval_games.parquet"), ease__n_candidates=600, safe__head=50, safe__k=K)
X = pd.DataFrame([(p, g["app_id"], g["playtime_min"], g["playtime_min"] > 0) for p, r in reqs.items() for g in r["library"]],
                 columns=["player_id", "app_id", "playtime_min", "played"])
pred = pipe.predict(X)
local_pipe = {p: [(int(x.app_id), float(x.score)) for x in g.itertuples()] for p, g in pred.groupby("player_id")}


def call(base):
    out, t = {}, []
    for p, body in reqs.items():
        s = time.time()
        r = requests.post(base.rstrip("/") + "/predict", json=body, timeout=300)
        r.raise_for_status()
        t.append(time.time() - s)
        out[p] = [(x["app_id"], x["score"]) for x in r.json()["picks"]]
    return out, t


def compare(name, other):
    same_games = sum([x for x, _ in local_pipe.get(p, [])] == [x for x, _ in other[p]] for p in reqs)
    diffs = [abs(s1 - s2) for p in reqs for (_, s1), (_, s2) in zip(local_pipe.get(p, []), other[p])]
    return {"compared with saved pipeline": name, "players": len(reqs), "same picks, same order": int(same_games),
            "max score difference": max(diffs) if diffs else 0.0, "identical scores": all(d == 0 for d in diffs)}


rows, timing = [], {}
for name, base in [("local FastAPI", a.local), ("deployed on Modal", a.deployed)]:
    if not base:
        continue
    s = time.time()
    res, t = call(base)
    timing[name] = {"first request s": round(t[0], 2), "median request s": round(float(np.median(t)), 3), "total s": round(time.time() - s, 1)}
    rows.append(compare(name, res))
    (OUT / f"responses_{name.split()[0]}.json").write_text(json.dumps({str(k): v for k, v in res.items()}), encoding="utf-8")

ok = all(r["same picks, same order"] == r["players"] and r["identical scores"] for r in rows)
report = ["# M5: local vs deployed outputs\n",
          f"{len(reqs)} real libraries (validation players, 80/20 visible games; inputs in m5_out/requests.json), top {K} picks each.",
          f"Players with no played game the model knows get no picks (counted as matching when every source returns none): "
          f"{sum(1 for p in reqs if not local_pipe.get(p))}.\n",
          pd.DataFrame(rows).to_markdown(index=False), "\n## Timing\n", pd.DataFrame(timing).T.to_markdown(),
          f"\n**Result: {'PASS' if ok else 'FAIL'}**"]
(OUT / "verify_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))
