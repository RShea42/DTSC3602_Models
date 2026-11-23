"""EASE-safe Steam recommender (Juan's model) served as an HTTPS API on Modal.

The model is the fitted scikit-learn pipeline (LibraryEncoder -> EASEScorer -> SafeReranker), unmodified:
  DTSC3602_Models/Steam_EASE_Pipeline  (RShea42/DTSC3602_Models, sparse clone): steamrec/ + app/engine.py
  app/engine.py has the same interface as Juan's model/recommender.py, so every route below is unchanged.
Modal Volume "steam-ease-artifacts":
  /pipeline  serve_files/ from Steam_EASE_Pipeline/build_model.py (ease_safe_pipeline.joblib + 3 catalog files)
             upload: modal volume put steam-ease-artifacts <Steam_EASE_Pipeline>/serve_files/ /pipeline
  /data      games.parquet, games_slim.csv (game pages, tags; upload_artifacts.ps1)
Previous version (Juan's recommender.py + ease.npz): modal_api.before_pipeline.py

Secrets (Modal secret "steam-ease-api"):
  STEAM_API_KEY  Steam Web API key (upstream reads it from .env; here it comes from the environment)
  API_TOKEN      shared bearer token; only the Vercel server knows it

Dev:     modal serve modal_api.py
Deploy:  modal deploy modal_api.py
Local (no Modal, same routes):
  $env:STEAMREC_FILES = "<Steam_EASE_Pipeline>/serve_files"; $env:STEAMREC_DATA = "<folder with games.parquet>"
  $env:API_TOKEN = "dev"
  python -m uvicorn modal_api:local_app --port 8000      (needs Steam_EASE_Pipeline/requirements.txt + uvicorn)

Routes (all need "Authorization: Bearer <API_TOKEN>" except /health):
  GET  /health
  GET  /search?q=hades
  GET  /games?ids=1,2,3
  GET  /game/{app_id}
  GET  /player?q=<steam id | profile link | custom name>
  POST /recommend          {"library": [{"app_id": 1, "hours": 12.5}], "owned": [..], "n": 10, "max_price": 20, "cc": "us", "include_explicit": false}
  GET  /recommend/profile?q=<steam id/link/name>&max_price=&adult=0|1&n=10
Steam IDs are never logged or stored (same rule as upstream app/server.py).
"""
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# the pipeline folder: parent when this file is in the repo (Steam_EASE_Pipeline/tasteforge_api/), else the local clone
PIPE = HERE.parent if (HERE.parent / "steamrec").is_dir() else HERE / "DTSC3602_Models" / "Steam_EASE_Pipeline"
REMOTE = "/root/ease"
VOL = "/vol"

app = modal.App(os.environ.get("MODAL_APP_NAME", "steam-ease-api"))   # set MODAL_APP_NAME=steam-ease-api-test to try it first
volume = modal.Volume.from_name("steam-ease-artifacts", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # pinned versions: a saved scikit-learn pipeline only loads reliably with the versions it was saved with
    .uv_pip_install("fastapi[standard]", requirements=[str(PIPE / "requirements.txt")])
    .env({"STEAMREC_FILES": f"{VOL}/pipeline", "STEAMREC_DATA": f"{VOL}/data", "PYTHONUTF8": "1",
          "MODAL_APP_NAME": app.name})
    .add_local_dir(PIPE / "steamrec", f"{REMOTE}/pipe/steamrec")
    .add_local_dir(PIPE / "app", f"{REMOTE}/pipe/app")
)

MAX_LIBRARY = 500
MAX_N = 30


# ---------------------------------------------------------------------------------------------- model loading
def load_recommender(pipeline_dir: str):
    """Import the pipeline engine (app/engine.py), take the Steam key from the environment, load the fitted
    pipeline (~335 MB of EASE weights) once."""
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    from app import engine as R  # noqa: E402  (was: import recommender as R)

    def _key():
        k = os.environ.get("STEAM_API_KEY", "").strip()
        if not k:
            raise RuntimeError("STEAM_API_KEY is not set")
        return k

    R._key = _key                          # upstream reads ROOT/.env; resolve_steamid/fetch_profile look it up at call time
    R.get()                                # loads B + catalogs once (lru_cache)
    return R


# ---------------------------------------------------------------------------------------------- API
def build_api(R):
    """FastAPI app around the upstream module R (already loaded)."""
    import hmac
    import math

    import numpy as np
    import pandas as pd
    from fastapi import Body, FastAPI, Header, HTTPException, Query
    from fastapi.responses import JSONResponse

    rec = R.get()
    G = rec.games                              # catalog indexed by app_id
    api = FastAPI(title="EASE-safe Steam recommender", docs_url=None, redoc_url=None, openapi_url=None)

    # ---- JSON-safe values (numpy scalars, NaN, arrays, timestamps)
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return None if math.isnan(v) or math.isinf(v) else float(v)
        if isinstance(v, pd.Timestamp):
            return None if pd.isna(v) else v.date().isoformat()
        if v is pd.NaT or v is pd.NA:
            return None
        return v

    def ok(body):
        return JSONResponse(clean(body))

    def auth(authorization: str | None):
        token = os.environ.get("API_TOKEN", "")
        if not token:
            raise HTTPException(503, "API_TOKEN not configured")
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "unauthorized")

    @api.exception_handler(R.SteamError)
    async def steam_error(_, e):
        return JSONResponse({"error": str(e)}, status_code=400)

    @api.exception_handler(Exception)
    async def any_error(_, e):                # no details: requests carry Steam IDs
        return JSONResponse({"error": f"Something went wrong: {type(e).__name__}"}, status_code=500)

    # ---- per-game summaries
    def _list(x):
        return [] if x is None or (isinstance(x, float) and math.isnan(x)) else [str(t) for t in x]

    # fuller user tags for games whose catalog tags are empty (Streamlit's games_slim.csv fill-in, gd.csv_tags)
    SLIM = {}
    slim_path = R.D / "games_slim.csv"
    if slim_path.exists():
        sl = pd.read_csv(slim_path, usecols=["app_id", "tags"]).dropna()
        SLIM = {int(a): [t.strip() for t in str(x).split(",") if t.strip()] for a, x in zip(sl.app_id, sl.tags)}

    def summary(a: int) -> dict | None:
        if a not in G.index:
            return None
        g = G.loc[a]
        j = rec.idx.get(a)
        pos, neg = g.steam_positive, g.steam_negative
        n = (0 if pd.isna(pos) else pos) + (0 if pd.isna(neg) else neg)
        rating = pos / n * 100 if n > 0 else g.positive_ratio
        raw = _list(g.tags) or SLIM.get(a) or _list(g.genres)
        return {"app_id": a, "title": g.title, "tags": rec.tags.get(a, [])[:5], "raw_tags": raw[:20],
                "genres": _list(g.genres), "image": g.header_image, "price": g.price_regular,
                "rating": None if pd.isna(rating) else round(float(rating)),
                "reviews": int(n) if n > 0 else (None if pd.isna(g.user_reviews) else int(g.user_reviews)),
                "in_catalog": True, "in_model": j is not None, "recommendable": bool(g.recommendable),
                "explicit": bool(g.explicit), "store_url": f"https://store.steampowered.com/app/{a}/"}

    # ---- game detail: stats vs genre peers and all games (port of Steamapp game_detail.py; games with >= 50 reviews)
    DET = pd.read_parquet(R.D / "games.parquet", columns=[
        "app_id", "developer", "publisher", "description", "rating", "owners_lo", "owners_hi",
        "steamspy_avg_playtime_min", "steamspy_median_playtime_min", "peak_ccu", "achievements"]).set_index("app_id")
    DET = DET[~DET.index.duplicated()].reindex(G.index)
    reviews = (G.steam_positive.fillna(0) + G.steam_negative.fillna(0)).where(lambda s: s > 0, G.user_reviews)
    stats = pd.DataFrame({
        "price": G.price_regular,
        "rating": (G.steam_positive / reviews.where(reviews > 0) * 100).fillna(G.positive_ratio),
        "reviews": reviews,
        "owners": (DET.owners_lo + DET.owners_hi) / 2,
        "hours": DET.steamspy_avg_playtime_min / 60,
        "median_hours": DET.steamspy_median_playtime_min / 60,
        "peak_ccu": DET.peak_ccu,
        "achievements": DET.achievements,
    })
    SKIP_ZERO = {"hours", "median_hours", "peak_ccu", "achievements"}
    NOT_PRIMARY = {"indie", "free to play", "early access", "casual", "massively multiplayer"}

    def primary_genre(genres):
        gs = _list(genres)
        return next((x for x in gs if x.lower() not in NOT_PRIMARY), gs[0] if gs else None)

    stats["pg"] = [primary_genre(x) for x in G.genres]
    ref = stats[reviews >= 50]

    def col_values(df, c):
        v = df[c].dropna().to_numpy(float)
        return v[v > 0] if c in SKIP_ZERO else v

    def metric(c, v, peers):
        has_v = not (v is None or pd.isna(v) or (c in SKIP_ZERO and v == 0))
        allv, pv = col_values(ref, c), col_values(peers, c)
        return {"value": float(v) if has_v else None,
                "genre_avg": float(pv.mean()) if len(pv) else None,
                "all_avg": float(allv.mean()) if len(allv) else None,
                "genre_pct_below": float((pv < v).mean() * 100) if has_v and len(pv) else None,
                "all_pct_below": float((allv < v).mean() * 100) if has_v and len(allv) else None}

    # hidden gems: well loved, lesser known, recommendable (bottom-right card in the Streamlit app)
    GEMS = ref.index[(ref.rating >= 92) & ref.reviews.between(300, 5000)
                     & G.recommendable.reindex(ref.index).fillna(False) & ~G.explicit.reindex(ref.index).fillna(False)].to_numpy()
    # demo profiles: popular games the model knows
    DEMO = np.array([a for a in ref.index[ref.reviews >= 20000] if a in rec.idx and not G.explicit.get(a, False)])
    rng = np.random.default_rng()

    # ------------------------------------------------------------------ routes
    @api.get("/health")
    def health():
        return {"ok": True, "games_in_model": len(rec.app_ids)}

    @api.get("/search")
    def search(q: str = "", authorization: str | None = Header(None)):
        auth(authorization)
        return ok([{**h, **(summary(h["app_id"]) or {})} for h in rec.search(q[:80])])

    @api.get("/games")
    def games(ids: str = "", authorization: str | None = Header(None)):
        auth(authorization)
        want = [int(x) for x in ids.split(",") if x.strip().isdigit()][:MAX_LIBRARY]
        return ok([s for s in map(summary, want) if s])

    @api.get("/gem")
    def gem(authorization: str | None = Header(None)):
        auth(authorization)
        return ok(summary(int(rng.choice(GEMS))) if len(GEMS) else None)

    @api.get("/demo")
    def demo(authorization: str | None = Header(None)):
        """A random 4-10 game library with random hours (Streamlit 'Try a demo profile')."""
        auth(authorization)
        picks = rng.choice(DEMO, size=min(len(DEMO), int(rng.integers(4, 11))), replace=False)
        hours = lambda: float(rng.choice([rng.integers(3, 31), rng.integers(20, 151), rng.integers(100, 601)]))
        return ok([{**summary(int(a)), "hours": hours()} for a in picks])

    @api.get("/game/{app_id}")
    def game(app_id: int, authorization: str | None = Header(None)):
        auth(authorization)
        s = summary(app_id)
        if s is None:
            raise HTTPException(404, "game not in catalog")
        g = G.loc[app_id]
        d = DET.loc[app_id]
        st = stats.loc[app_id]
        pg = st.pg
        peers = ref[ref.pg == pg] if pg else ref
        return ok({**s, "release_date": g.release_date, "developer": None if pd.isna(d.developer) else d.developer,
                   "publisher": None if pd.isna(d.publisher) else d.publisher,
                   "description": None if pd.isna(d.description) else d.description,
                   "steam_rating_label": None if pd.isna(d.rating) else d.rating,
                   "positive": g.steam_positive, "negative": g.steam_negative,
                   "owners_lo": d.owners_lo, "owners_hi": d.owners_hi,
                   "primary_genre": pg or "all", "n_genre": len(peers), "n_all": len(ref),
                   "metrics": {c: metric(c, st[c], peers) for c in stats.columns if c != "pg"}})

    @api.get("/player")
    def player(q: str = "", authorization: str | None = Header(None)):
        auth(authorization)
        sid = R.resolve_steamid(q)
        prof, lib = R.fetch_profile(sid)
        if lib is None:
            raise R.SteamError("This profile's game details are private. Profile > Edit Profile > Privacy Settings > "
                               "set 'Game details' to Public, or add games by hand.")
        games_out = []
        for a, pt, name in sorted(lib, key=lambda x: -x[1]):
            s = summary(a) or {"app_id": a, "title": name, "tags": [], "raw_tags": [], "genres": [], "image": None,
                                "in_catalog": False, "in_model": False}
            games_out.append({**s, "title": s.get("title") or name, "playtime_min": pt, "hours": round(pt / 60, 1)})
        return ok({"steamid": sid, "name": prof.get("name"), "avatar": prof.get("avatar"), "url": prof.get("url"),
                   "cc": (prof.get("country") or "us").lower(), "n_owned": len(lib),
                   "n_played": sum(1 for _, pt, _ in lib if pt > 0), "games": games_out})

    @api.post("/recommend")
    def recommend(body: dict = Body(...), authorization: str | None = Header(None)):
        auth(authorization)
        items = body.get("library") or []
        if not isinstance(items, list) or not items:
            raise R.SteamError("Add at least one game.")
        lib = []
        for it in items[:MAX_LIBRARY]:
            try:
                a = int(it["app_id"])
            except (KeyError, TypeError, ValueError):
                continue
            h = it.get("hours")
            if h is None:                     # no hours given: treat as loved (upstream for_favorites rule)
                pt = float(rec.q[a][90]) if a in rec.q else 6000.0
            else:
                pt = max(0.0, float(h)) * 60
            lib.append((a, pt))
        owned = [int(a) for a in (body.get("owned") or []) if str(a).isdigit()][:5000]
        mp = body.get("max_price")
        return ok(run(lib, owned + [a for a, _ in lib], n=body.get("n"),
                      max_price=None if mp in (None, "") else float(mp), cc=str(body.get("cc") or "us").lower()[:2],
                      include_explicit=bool(body.get("include_explicit"))))

    @api.get("/recommend/profile")
    def recommend_profile(q: str = "", max_price: float | None = Query(None), adult: int = 0, n: int = 10,
                          authorization: str | None = Header(None)):
        """Same flow and messages as upstream for_profile, but through run() so cards carry contrib + flags."""
        auth(authorization)
        sid = R.resolve_steamid(q)
        prof, lib = R.fetch_profile(sid)
        who = f"Found the account '{prof['name']}'. " if prof.get("name") else ""
        if lib is None:
            raise R.SteamError(who + "Its game details are private. Profile > Edit Profile > Privacy Settings > "
                               "set 'Game details' to Public, or add games by hand.")
        played = [(a, pt) for a, pt, _ in lib if pt > 0]
        if not played:
            raise R.SteamError(who + "It shows no playtime. In Privacy Settings, untick 'Always keep my total "
                               "playtime private', or add games by hand.")
        out = run(played, [a for a, _, _ in lib], n=n, max_price=max_price,
                  cc=(prof.get("country") or "us").lower(), include_explicit=adult == 1)
        out["player"] = {"name": prof.get("name"), "avatar": prof.get("avatar")}
        return ok(out)

    # ---- shared recommend path: upstream recommend() + per-card contrib and flags (as in Steamapp_Juan_Model)
    def run(lib, owned, n=None, max_price=None, cc="us", include_explicit=False):
        played = [(a, pt) for a, pt in lib if pt > 0]
        n = max(1, min(int(n or 10), MAX_N))
        out = rec.recommend(played, owned=owned, n=n, max_price=max_price, cc=cc, include_explicit=include_explicit)
        if out is None:
            raise R.SteamError("None of these games are in the model's data yet. Try adding more popular titles.")
        rows = [(rec.idx[a], a) for a, _ in played if a in rec.idx]
        tp = [(rec.tpos[a], pt) for a, pt in played if a in rec.tpos]
        short_ix = [i for i, pt in tp if pt < 120]
        kept_ix = [i for i, pt in tp if pt >= 120]

        def prof(ix):
            p = np.asarray(rec.T[ix].sum(0)).ravel()
            nn = np.linalg.norm(p)
            return p / nn if nn else p

        ps, pk = (prof(short_ix), prof(kept_ix)) if short_ix and kept_ix else (None, None)
        for card in out["for_you"] + out["try_new"]:
            j = rec.idx[card["app_id"]]
            c = rec.B[[r[0] for r in rows], j].astype(np.float32)
            card["contrib"] = [(rec.titles.get(rows[k][1], str(rows[k][1])), float(c[k]))
                               for k in np.argsort(-c)[:4] if c[k] > 0]
            fl = []
            t = rec.trow[j]
            if ps is not None and t >= 0:
                g = rec.T[t]
                if g.sum() > 0 and (g @ ps)[0] > (g @ pk)[0]:
                    fl.append("quit")
            if rec.flag_short[j]:
                fl.append("short")
            if rec.flag_neg[j]:
                fl.append("neg")
            card["flags"] = fl
            s = summary(card["app_id"]) or {}
            card["raw_tags"], card["genres"] = s.get("raw_tags", []), s.get("genres", [])
        out["n_candidates"] = int((rec.candidate & (True if include_explicit else ~rec.explicit)).sum())
        return out

    return api


# ---------------------------------------------------------------------------------------------- Modal
@app.cls(
    image=image,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("steam-ease-api")],
    cpu=2,
    memory=4096,
    scaledown_window=300,
    timeout=120,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=8)
class Api:
    @modal.enter(snap=True)
    def load(self):
        self.R = load_recommender(f"{REMOTE}/pipe")

    @modal.asgi_app()
    def web(self):
        return build_api(self.R)


# ---------------------------------------------------------------------------------------------- local run
def __getattr__(name):                     # `uvicorn modal_api:local_app` builds it lazily from the local clone
    if name == "local_app":
        return build_api(load_recommender(str(PIPE)))
    raise AttributeError(name)
