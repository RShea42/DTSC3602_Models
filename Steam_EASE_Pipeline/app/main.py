"""FastAPI app: Juan's app/server.py routes and behaviour, served by the sklearn pipeline.

  local:  PUBLIC_BASE_URL=http://127.0.0.1:8000  uvicorn app.main:app --port 8000
  Modal:  modal_app.py (sets PUBLIC_BASE_URL to the deployed URL and SESSION_BACKEND=modal)

Same routes, query parameters and JSON as Juan's server, so his index.html works unchanged:
  /            the page               /login, /auth, /logout   Steam sign-in (OpenID 2.0)
  /api/me      signed-in player       /api/recommend?q=        profile link / ID / name
  /api/favorites?ids=                 /api/search?q=
  options on recommend routes: max_price=<number>, adult=1
Steam IDs are never logged (request logging is off for that reason, as in Juan's server).
"""
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from . import engine
from .sessions import TTL, SessionStore

HERE = Path(__file__).resolve().parent
PAGE = (HERE / "index.html").read_text(encoding="utf-8")
OPENID = "https://steamcommunity.com/openid/login"
COOKIE = "ngf"
BASE = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")   # where Steam sends players back

@asynccontextmanager
async def lifespan(_):
    engine.get()                                     # load the model once per container, before the first request
    yield


app = FastAPI(title="EASE-safe Steam recommender", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
sessions = SessionStore()


def verify_steam_login(q, base=None):
    """Return the verified SteamID64 from Steam's OpenID reply, or None. (Juan's check, unchanged.)"""
    base = base or BASE
    if q.get("openid.mode") != "id_res" or q.get("openid.op_endpoint") != OPENID:
        return None
    if not q.get("openid.return_to", "").startswith(base + "/auth"):
        return None
    m = re.fullmatch(r"https://steamcommunity\.com/openid/id/(7656\d{13})", q.get("openid.claimed_id", ""))
    if not m:
        return None
    data = {k: v for k, v in q.items() if k.startswith("openid.")}
    data["openid.mode"] = "check_authentication"
    try:
        r = requests.post(OPENID, data=data, timeout=15)
    except requests.RequestException:
        return None
    return m.group(1) if "is_valid:true" in r.text else None


def _cookie(resp, token, max_age):
    resp.set_cookie(COOKIE, token, max_age=max_age, httponly=True, samesite="lax", path="/", secure=BASE.startswith("https://"))
    return resp


def _opts(request):
    q = request.query_params
    return {"max_price": float(q["max_price"]) if q.get("max_price") else None,
            "include_explicit": q.get("adult") == "1"}   # "Show adult games": off unless the player ticks it


def _run(fn):
    """Juan's error contract: SteamError -> 400 with its message; anything else -> 500 without details."""
    try:
        return JSONResponse(fn())
    except engine.SteamError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Something went wrong: {type(e).__name__}"}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def page():
    return PAGE


@app.get("/login")
def login():
    return RedirectResponse(OPENID + "?" + urlencode({
        "openid.ns": "http://specs.openid.net/auth/2.0", "openid.mode": "checkid_setup",
        "openid.return_to": BASE + "/auth", "openid.realm": BASE + "/",
        "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
        "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select"}), status_code=302)


@app.get("/auth")
def auth(request: Request):
    sid = verify_steam_login(dict(request.query_params))
    if not sid:
        return RedirectResponse("/#signin-failed", status_code=302)
    return _cookie(RedirectResponse("/", status_code=302), sessions.create(sid), TTL)


@app.get("/logout")
def logout(request: Request):
    sessions.delete(request.cookies.get(COOKIE))
    return _cookie(RedirectResponse("/", status_code=302), "", 0)


@app.get("/api/me")
def me(request: Request):
    sid = sessions.get(request.cookies.get(COOKIE))
    if not sid:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    return _run(lambda: engine.for_profile(sid, **_opts(request)))


@app.get("/api/recommend")
def recommend(request: Request, q: str = ""):
    return _run(lambda: engine.for_profile(q, **_opts(request)))


@app.get("/api/favorites")
def favorites(request: Request, ids: str = ""):
    return _run(lambda: engine.for_favorites([int(x) for x in ids.split(",") if x.strip().isdigit()], **_opts(request)))


@app.get("/api/search")
def search(q: str = ""):
    return _run(lambda: engine.get().search(q))
