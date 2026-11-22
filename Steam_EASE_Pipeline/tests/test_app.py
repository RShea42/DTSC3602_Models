"""FastAPI routes, Steam sign-in and sessions (seconds; the model is replaced by a stand-in, Steam is simulated)."""
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from app import engine, main
from app.sessions import SessionStore

SID = "76561197960287930"
REPLY = {"openid.mode": "id_res", "openid.op_endpoint": main.OPENID, "openid.return_to": main.BASE + "/auth",
         "openid.claimed_id": f"https://steamcommunity.com/openid/id/{SID}", "openid.sig": "x"}


class FakeSteam:
    def __init__(self, valid):
        self.valid, self.calls = valid, 0

    def __call__(self, url, data, timeout):
        self.calls += 1
        assert url == main.OPENID and data["openid.mode"] == "check_authentication"
        return type("R", (), {"text": f"ns:http://specs.openid.net/auth/2.0\nis_valid:{'true' if self.valid else 'false'}\n"})()


@pytest.fixture
def client(monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "get", lambda: type("E", (), {"search": lambda self, q: [{"app_id": 1, "title": q}]})())
    monkeypatch.setattr(engine, "for_profile", lambda who, **kw: calls.append((who, kw)) or {"for_you": [], "who": who})
    monkeypatch.setattr(main, "sessions", SessionStore("memory"))
    with TestClient(main.app, base_url=main.BASE, follow_redirects=False) as c:
        c.calls = calls
        yield c


def sign_in(client, monkeypatch, valid=True, reply=REPLY):
    steam = FakeSteam(valid)
    monkeypatch.setattr(main.requests, "post", steam)
    return client.get("/auth", params=reply), steam


def test_page_is_juans(client):
    r = client.get("/")
    assert r.status_code == 200 and "/api/me" in r.text and "/login" in r.text


def test_login_sends_player_to_steam_with_our_return_address(client):
    r = client.get("/login")
    assert r.status_code == 302 and r.headers["location"].startswith(main.OPENID)
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q["openid.return_to"] == [main.BASE + "/auth"] and q["openid.realm"] == [main.BASE + "/"]


def test_valid_sign_in_creates_session_and_me_uses_it(client, monkeypatch):
    r, steam = sign_in(client, monkeypatch)
    assert r.status_code == 302 and r.headers["location"] == "/" and steam.calls == 1
    assert "httponly" in r.headers["set-cookie"].lower() and SID not in r.headers["set-cookie"]   # only a ticket
    me = client.get("/api/me")
    assert me.status_code == 200 and client.calls[-1][0] == SID


def test_forged_reply_is_rejected(client, monkeypatch):
    r, steam = sign_in(client, monkeypatch, valid=False)            # Steam says the signature is not genuine
    assert r.headers["location"] == "/#signin-failed" and "set-cookie" not in r.headers and steam.calls == 1
    assert client.get("/api/me").status_code == 401


@pytest.mark.parametrize("field,value", [("openid.return_to", "https://evil.example/auth"),
                                         ("openid.op_endpoint", "https://evil.example/openid"),
                                         ("openid.claimed_id", "https://evil.example/openid/id/76561197960287930")])
def test_tampered_reply_rejected_without_asking_steam(client, monkeypatch, field, value):
    r, steam = sign_in(client, monkeypatch, reply={**REPLY, field: value})
    assert r.headers["location"] == "/#signin-failed" and steam.calls == 0


def test_logout_ends_session_everywhere(client, monkeypatch):
    sign_in(client, monkeypatch)
    token = client.cookies.get(main.COOKIE)
    client.get("/logout")
    assert main.sessions.get(token) is None                          # gone server-side, not just in this browser
    client.cookies.set(main.COOKIE, token)                           # a copied ticket no longer works
    assert client.get("/api/me").status_code == 401


def test_expired_session_is_refused_and_swept():
    s = SessionStore("memory")
    t = s.create(SID)
    s.d[t] = (SID, time.time() - 1)
    assert s.get(t) is None and t not in s.d
    s.d["old"] = (SID, time.time() - 1)
    s.create(SID)
    assert s.sweep() == 1 and len(s.d) == 1


def test_options_and_errors_follow_juans_contract(client, monkeypatch):
    client.get("/api/recommend", params={"q": "someone", "max_price": "15", "adult": "1"})
    assert client.calls[-1] == ("someone", {"max_price": 15.0, "include_explicit": True})
    client.get("/api/recommend", params={"q": "someone"})
    assert client.calls[-1][1] == {"max_price": None, "include_explicit": False}

    def private(who, **kw):
        raise engine.SteamError("Its game details are private.")
    monkeypatch.setattr(engine, "for_profile", private)
    r = client.get("/api/recommend", params={"q": "x"})
    assert r.status_code == 400 and r.json() == {"error": "Its game details are private."}
    r = client.get("/api/recommend", params={"q": "x", "max_price": "cheap"})
    assert r.status_code == 500 and r.json() == {"error": "Something went wrong: ValueError"}


def test_search(client):
    assert client.get("/api/search", params={"q": "hades"}).json() == [{"app_id": 1, "title": "hades"}]
