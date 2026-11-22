"""Peyton's API with Juan's recipe vs with the scikit-learn pipeline: a side-by-side check to run before swapping.

Put your current modal_api.py in this folder (it is gitignored here; it is your code), then from this folder:
    PYTHONUTF8=1 python compare.py          (~8 min; needs the data in paths.py and serve_files/ from build_model.py)
Optional: TEST_PROFILE=<a public Steam profile name/link> adds the /player and /recommend/profile checks
(needs STEAM_API_KEY in the environment or in ../../Steam_EASE_Safe_Recommender/.env).
Three copies of your API, all built by your own build_api(), get the same requests on every route:
  original  your load_recommender(): Juan's model/recommender.py + Juan's ease.npz   (what taste-forge.com runs)
  swap A    the pipeline engine (../app/engine.py) with Juan's weights injected     -> proves the code path
  swap B    the pipeline engine with its own retrained weights                     -> what the swap serves
Responses must be identical (status + JSON). Store prices come from one fixed fake store on all three copies, and
/gem and /demo use the same seed. Local only: nothing is deployed. Report: report.md.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path[:0] = [str(HERE), str(PIPE)]
from paths import EASE_FILES as EASE, JUAN, SERVE  # noqa: E402

os.environ.update(STEAMREC_ARTIFACTS=str(EASE / "model" / "artifacts"), STEAMREC_DATA=str(EASE / "data_clean"),
                  STEAMREC_FILES=str(SERVE), API_TOKEN="dev", PYTHONUTF8="1")
env = JUAN / ".env"
if not os.environ.get("STEAM_API_KEY") and env.exists():
    os.environ["STEAM_API_KEY"] = next(l.split("=", 1)[1].strip() for l in env.read_text().splitlines() if l.startswith("STEAM_API_KEY="))
if not (HERE / "modal_api.py").exists():
    sys.exit("Put your modal_api.py in this folder first.")
PROFILE = os.environ.get("TEST_PROFILE", "").strip()

from fastapi.testclient import TestClient  # noqa: E402

import modal_api as P  # noqa: E402   your file, unchanged
from app import engine  # noqa: E402

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


def fake_prices(app_ids, cc="us"):
    out = {}
    for a in map(int, app_ids):
        if a % 7 == 0:
            out[a] = {"final": None, "initial": None, "discount": 0, "text": None}
        else:
            p = 0.0 if a % 11 == 0 else float(a % 60 + 1)
            out[a] = {"final": p, "initial": p, "discount": 0, "text": f"${p:.2f}"}
    return out


def load_ours():
    """Peyton's load_recommender(), with our engine in place of Juan's module (the one-line swap)."""
    R = engine

    def _key():
        k = os.environ.get("STEAM_API_KEY", "").strip()
        if not k:
            raise RuntimeError("STEAM_API_KEY is not set")
        return k
    R._key = _key
    R.get()
    return R


def build(R):
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: orig(12345)          # same seed for /gem and /demo on every copy
    try:
        return TestClient(P.build_api(R), raise_server_exceptions=False)
    finally:
        np.random.default_rng = orig


# ---- the three copies
RJ = P.load_recommender(str(JUAN / "model"))          # Juan's folder in this repo
RJ.live_prices = fake_prices
apis = {"original": build(RJ)}
log("original (Juan's recipe) loaded")
RO = load_ours()
engine.live_prices = fake_prices
recA = engine.get()
recA.B = recA.pipe["ease"].B_ = RJ.get().B                       # swap A: Juan's exact weights
recA.pipe["ease"]._Bf = None
apis["swap A"] = build(RO)
engine.get.cache_clear()
engine.get()                                                     # swap B: fresh engine, its own retrained weights
apis["swap B"] = build(RO)
log("swap A and swap B (pipeline) loaded")

# ---- the requests
H = {"Authorization": "Bearer dev"}
val = pd.read_parquet(EASE / "data_clean" / "team_eval" / "validation.parquet")
v = val[~val.hidden_80_20]
pids = np.random.default_rng(11).choice(np.sort(v[v.played].player_id.unique()), 200, replace=False)
libs = [[{"app_id": int(a), "hours": float(t) / 60} for a, t in zip(g.app_id, g.playtime_min)]
        for _, g in v[v.player_id.isin(pids)].groupby("player_id")]
ids = [int(a) for a in RJ.get().app_ids[::997]]
reqs = [("GET", "/health", None, None)]
reqs += [("GET", "/search", {"q": q}, None) for q in ["hades", "witcher", "portal", "a", "", "zzqqxx"]]
reqs += [("GET", "/games", {"ids": ",".join(map(str, ids + [1, 999999999]))}, None)]
reqs += [("GET", f"/game/{a}", None, None) for a in ids[:10] + [999999999]]
reqs += [("GET", "/gem", None, None)] * 5 + [("GET", "/demo", None, None)] * 5
for i, lib in enumerate(libs):
    owned = [g["app_id"] for g in lib]
    reqs.append(("POST", "/recommend", None, {"library": lib, "owned": owned}))
    if i < 60:
        reqs.append(("POST", "/recommend", None, {"library": lib, "owned": owned, "max_price": 20}))
        reqs.append(("POST", "/recommend", None, {"library": lib, "owned": owned, "include_explicit": True, "n": 5}))
    if i < 20:                                                       # no hours given: his "treat as loved" rule
        reqs.append(("POST", "/recommend", None, {"library": [{"app_id": g["app_id"]} for g in lib[:6]]}))
reqs += [("POST", "/recommend", None, {"library": []}), ("POST", "/recommend", None, {"library": [{"app_id": 999999999, "hours": 3}]})]
if PROFILE:                                                       # live Steam lookups (same answer for all copies)
    for q in [{"q": PROFILE}, {"q": PROFILE, "max_price": 15, "adult": 1}]:
        reqs.append(("GET", "/recommend/profile", q, None))
    reqs.append(("GET", "/player", {"q": PROFILE}, None))
log(f"{len(reqs)} requests per copy")


def send(c, method, path, params, body, headers=H):
    r = c.request(method, path, params=params, json=body, headers=headers)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


results = {name: [send(c, *rq) for rq in reqs] for name, c in apis.items()}
results_auth = {name: send(c, "GET", "/search", {"q": "hades"}, None, headers={}) for name, c in apis.items()}
log("all requests sent")


def first_diff(a, b, path=""):
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on one side"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} vs {len(b)} items"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
    elif a != b:
        return f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}"
    return None


def strip_scores(x):
    """The response without card scores (scores are rounded floats that move with tiny weight differences)."""
    if isinstance(x, dict):
        return {k: strip_scores(v) for k, v in x.items() if k not in ("score", "contrib")}
    if isinstance(x, list):
        return [strip_scores(v) for v in x]
    return x


lines = ["# Peyton's API, Juan's recipe vs the scikit-learn pipeline (local)\n",
         f"{len(reqs)} requests per copy + 1 without a token. Store prices: one fixed fake store for all copies.\n"]
summary = []
for name in ["swap A", "swap B"]:
    by_route, diffs = {}, []
    for rq, o, s in zip(reqs, results["original"], results[name]):
        route = rq[1] if not rq[1].startswith("/game/") else "/game/{id}"
        same = o == s
        t = by_route.setdefault(route, [0, 0])
        t[0] += 1
        t[1] += same
        if not same:
            rest = o[0] == s[0] and strip_scores(o[1]) == strip_scores(s[1])
            diffs.append((rq[1], (first_diff(o[1], s[1]) if o[0] == s[0] else f"status {o[0]} vs {s[0]}")
                          + ("; everything else identical (same games, order, prices, flags, explanations)" if rest else "; OTHER FIELDS DIFFER TOO")))
    auth_ok = results_auth[name] == results_auth["original"] and results_auth[name][0] == 401
    total = sum(t[0] for t in by_route.values())
    same = sum(t[1] for t in by_route.values())
    summary.append((name, same, total, auth_ok, diffs))
    lines.append(f"## {name}: {same} / {total} identical; no-token request refused (401) like the original: {auth_ok}\n")
    lines.append(pd.DataFrame([(r, t[0], t[1]) for r, t in by_route.items()], columns=["route", "requests", "identical"]).to_markdown(index=False))
    if diffs:
        lines.append("\nDifferences (first field that differs):\n")
        lines += [f"- `{p}` {d}" for p, d in diffs[:25]]
    lines.append("")
codes = pd.Series([s for s, _ in results["original"]]).value_counts().to_dict()
lines.append(f"Status codes from the original (to show error paths were exercised): {codes}")
(HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
