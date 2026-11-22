"""Slow (needs serve_files/ from build_model.py): the API on the real model, in-process (no server needed).

/predict must equal plain scikit-learn pipe.predict on the saved pipeline (the same check m5_verify.py runs against
the deployed copy). Run before every deploy:  pytest -m slow
"""
from pathlib import Path

import joblib
import pandas as pd
import pytest

from paths import SERVE

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not (SERVE / "ease_safe_pipeline.joblib").exists(),
                                                   reason="needs serve_files/ (python build_model.py)")]
LIBS = [
    [(1145360, 3000), (367520, 1200)],                                   # Hades, Hollow Knight
    [(570, 50000), (730, 20000), (440, 0)],                              # Dota 2, CS, TF2 owned but unplayed
    [(292030, 9000), (489830, 4000), (377160, 100), (1091500, 30)],      # Witcher 3, Skyrim SE, Fallout 4, Cyberpunk
]


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def pipe():
    p = joblib.load(SERVE / "ease_safe_pipeline.joblib")
    p.set_params(ease__app_catalog_path=str(SERVE / "games.parquet"), ease__catalog_path=str(SERVE / "team_eval_games.parquet"),
                 safe__catalog_path=str(SERVE / "team_eval_games.parquet"), ease__n_candidates=600, safe__head=50, safe__k=10)
    return p


@pytest.mark.parametrize("lib", LIBS, ids=["roguelikes", "multiplayer", "rpgs"])
def test_predict_equals_saved_pipeline(client, pipe, lib):
    body = {"library": [{"app_id": a, "playtime_min": t} for a, t in lib], "k": 10}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    got = [(p["app_id"], p["score"]) for p in r.json()["picks"]]
    X = pd.DataFrame([(0, a, t, t > 0) for a, t in lib], columns=["player_id", "app_id", "playtime_min", "played"])
    want = [(int(a), float(s)) for a, s in zip(*pipe.predict(X)[["app_id", "score"]].to_numpy().T)]
    assert got == want and len(got) == 10
    assert not {a for a, _ in got} & {a for a, _ in lib}                  # never recommends what they own


def test_health_reports_juans_defaults(client):
    m = client.get("/health").json()["model"]
    assert (m["ease__lam"], m["encode__min_users"], m["safe__a_quit"], m["safe__b_short"], m["safe__c_negative"]) == (100, 100, 0.5, 1, 10)
    assert m["ease__serve_filters"] is True


def test_bad_requests_are_rejected(client):
    assert client.post("/predict", json={"library": []}).status_code == 422
    assert client.post("/predict", json={"library": [{"app_id": 1, "playtime_min": -5}]}).status_code == 422
    assert client.post("/recommend", json={"library": [{"app_id": 999999999, "playtime_min": 60}]}).status_code == 400
