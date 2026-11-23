"""D4 check: the TasteForge API (modal_api.py, serving the scikit-learn pipeline) gives the same output locally and on Modal.

Same 200 real player libraries as the team's M5 check (Steam_EASE_Pipeline/m5_out/requests.json), sent to POST /recommend:
  local      modal_api.build_api(load_recommender(<pipeline>)) in-process, model from Steam_EASE_Pipeline/serve_files
  deployed   the Modal URL given (the pipeline build on the Volume)
  reference  (optional) another URL, e.g. the pre-swap live API, to show the swap keeps today's behaviour
"Picks" = app_id, order and score of every for_you / try_new card. "Full" = whole JSON response. Prices are live
Steam store prices fetched on each call, so a full-response mismatch caused only by a price change is reported as such.

Run (from Juan_EASE, pipeline venv):
  PYTHONUTF8=1 python verify_deployed.py --deployed <url> [--reference <url>]
Token: API_TOKEN env var or the .api_token file (never printed). Report: verify_out/verify_report.md
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent if (HERE.parent / "steamrec").is_dir() else HERE / "DTSC3602_Models" / "Steam_EASE_Pipeline"
OUT = HERE / "verify_out"
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--deployed", required=True)
ap.add_argument("--reference")
ap.add_argument("--n", type=int, default=10)
args = ap.parse_args()

TOKEN = os.environ.get("API_TOKEN") or (HERE / ".api_token").read_text().strip()
os.environ.update(STEAMREC_FILES=str(PIPE / "serve_files"), STEAMREC_DATA=str(HERE / "local_data"),
                  API_TOKEN=TOKEN, PYTHONUTF8="1")
os.environ.setdefault("STEAM_API_KEY", "unused-by-recommend")

sys.path.insert(0, str(HERE))
import modal_api as P  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# the same 200 libraries as the team's M5 check, in this API's /recommend format (hours; owned = whole library)
m5 = json.loads((PIPE / "m5_out" / "requests.json").read_text())
bodies = {}
for pid, r in m5.items():
    played = [x for x in r["library"] if x["playtime_min"] > 0]
    if not played:
        continue
    bodies[pid] = {"library": [{"app_id": x["app_id"], "hours": x["playtime_min"] / 60} for x in played],
                   "owned": [x["app_id"] for x in r["library"]], "n": args.n}
(OUT / "requests.json").write_text(json.dumps(bodies))

H = {"Authorization": f"Bearer {TOKEN}"}
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

log("loading the pipeline locally")
local = TestClient(P.build_api(P.load_recommender(str(PIPE))))
sources = {"local": lambda b: local.post("/recommend", json=b, headers=H)}
sess = requests.Session()
for name, url in (("deployed", args.deployed), ("reference", args.reference)):
    if url:
        u = url.rstrip("/") + "/recommend"
        sources[name] = lambda b, u=u: sess.post(u, json=b, headers=H, timeout=180)

resp, timing = {s: {} for s in sources}, {s: [] for s in sources}
for i, (pid, b) in enumerate(bodies.items()):
    for s, call in sources.items():          # same request to every source back to back (prices fetched together)
        t = time.time()
        r = call(b)
        timing[s].append(time.time() - t)
        resp[s][pid] = {"status": r.status_code, "json": r.json()}
    if i % 25 == 0:
        log(f"{i + 1}/{len(bodies)}")
for s in sources:
    (OUT / f"responses_{s}.json").write_text(json.dumps(resp[s]))


def picks(j):
    return [(c.get("app_id"), c.get("score")) for key in ("for_you", "try_new") for c in (j.get(key) or [])]


def strip_prices(o):
    if isinstance(o, dict):
        return {k: strip_prices(v) for k, v in o.items() if "price" not in k and k not in ("discount", "final", "initial")}
    if isinstance(o, list):
        return [strip_prices(v) for v in o]
    return o


rows = []
for s in [x for x in sources if x != "local"]:
    same_status = same_picks = same_full = same_noprice = 0
    for pid in bodies:
        a, b = resp["local"][pid], resp[s][pid]
        same_status += a["status"] == b["status"]
        same_picks += a["status"] == b["status"] and picks(a["json"]) == picks(b["json"])
        same_full += a == b
        same_noprice += a["status"] == b["status"] and strip_prices(a["json"]) == strip_prices(b["json"])
    rows.append(f"| local vs {s} | {len(bodies)} | {same_status} | {same_picks} | {same_noprice} | {same_full} |")

med = lambda v: sorted(v)[len(v) // 2]
n_cards = sum(len(picks(r["json"])) for r in resp["local"].values())
ok = all(picks(resp["local"][p]["json"]) == picks(resp["deployed"][p]["json"]) for p in bodies)
report = f"""# TasteForge API: local vs deployed (scikit-learn pipeline)

{len(bodies)} real player libraries (the team's M5 inputs, played games only, `n={args.n}`), `POST /recommend`;
{n_cards} recommendation cards per source. Inputs: `verify_out/requests.json`; raw responses: `verify_out/responses_*.json`.

- deployed: `{args.deployed}`
- reference: `{args.reference or '-'}`

| comparison | requests | same status | same picks (app, order, score) | same JSON except prices | identical JSON |
|:--|--:|--:|--:|--:|--:|
{chr(10).join(rows)}

| source | first request s | median s |
|:--|--:|--:|
{chr(10).join(f"| {s} | {timing[s][0]:.2f} | {med(timing[s]):.2f} |" for s in sources)}

**Result (local vs deployed picks): {'PASS' if ok else 'FAIL'}**
"""
(OUT / "verify_report.md").write_text(report)
print(report)
