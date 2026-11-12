"""Local web page for the recommender (standard library only).

  PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python app/server.py   ->  http://127.0.0.1:8765

Listens on this computer only. Steam IDs are used for the lookup and never stored or logged.

"Sign in through Steam" uses Steam's OpenID 2.0 sign-in: the player signs in on Steam's own page (we never see a
password), Steam sends back their Steam ID, and we confirm that answer directly with Steam before trusting it.
The ID is then held in memory against a random cookie for 2 hours, never written to disk or put in a URL.
"""
import json
import re
import secrets
import sys
import time
from urllib.parse import urlencode

import requests
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from recommender import SteamError, for_favorites, for_profile, get  # noqa: E402

PAGE = (Path(__file__).resolve().parent / "index.html").read_bytes()
PORT = 8765
OPENID = "https://steamcommunity.com/openid/login"
SESSIONS = {}                                  # cookie token -> (steamid, expires_at); memory only
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}


def verify_steam_login(q, base):
    """Return the verified SteamID64 from Steam's OpenID reply, or None."""
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


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):          # no request logging: URLs carry Steam IDs
        pass

    def _redirect(self, location, cookie=None):
        self.send_response(302)
        self.send_header("Location", location)
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _session(self):
        m = re.search(r"(?:^|;\s*)ngf=([A-Za-z0-9_-]+)", self.headers.get("Cookie", ""))
        v = SESSIONS.get(m.group(1)) if m else None
        if v and v[1] > time.time():
            return m.group(1), v[0]
        return (m.group(1) if m else None), None

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        mp = float(q["max_price"]) if q.get("max_price") else None
        ex = q.get("adult") == "1"            # "Show adult games": off unless the player ticks it
        try:
            if u.path == "/":
                return self._send(200, PAGE, "text/html")
            host = self.headers.get("Host", "")
            base = f"http://{host if host in ALLOWED_HOSTS else f'127.0.0.1:{PORT}'}"
            if u.path == "/login":
                return self._redirect(OPENID + "?" + urlencode({
                    "openid.ns": "http://specs.openid.net/auth/2.0", "openid.mode": "checkid_setup",
                    "openid.return_to": base + "/auth", "openid.realm": base + "/",
                    "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
                    "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select"}))
            if u.path == "/auth":
                sid = verify_steam_login(q, base)
                if not sid:
                    return self._redirect("/#signin-failed")
                token = secrets.token_urlsafe(24)
                SESSIONS[token] = (sid, time.time() + 7200)
                return self._redirect("/", f"ngf={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=7200")
            if u.path == "/logout":
                token, _ = self._session()
                SESSIONS.pop(token, None)
                return self._redirect("/", "ngf=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0")
            if u.path == "/api/me":
                _, sid = self._session()
                if not sid:
                    return self._send(401, {"error": "not signed in"})
                return self._send(200, for_profile(sid, max_price=mp, include_explicit=ex))
            if u.path == "/api/recommend":
                return self._send(200, for_profile(q.get("q", ""), max_price=mp, include_explicit=ex))
            if u.path == "/api/favorites":
                ids = [int(x) for x in q.get("ids", "").split(",") if x.strip().isdigit()]
                return self._send(200, for_favorites(ids, max_price=mp, include_explicit=ex))
            if u.path == "/api/search":
                return self._send(200, get().search(q.get("q", "")))
            return self._send(404, {"error": "not found"})
        except SteamError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"Something went wrong: {type(e).__name__}"})


if __name__ == "__main__":
    print("loading model...", flush=True)
    get()
    print(f"ready on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
