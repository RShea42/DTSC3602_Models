"""Slow (about 1 min, needs ease_files/ and serve_files/): the app engine must return exactly what Juan's app returns.

Juan's model/recommender.py vs app/engine.py on 150 real validation libraries, for: default, a price limit,
adult games on, and the favourite-games fallback. Both get the same weights (Juan's ease.npz) and the same
deterministic stand-in for live store prices, so any difference is in the code path, not the model or the store.
(The pipeline's retrained weights equal Juan's to 99.994% bit-for-bit; that is checked in m1_check.py.)
Run: pytest -m slow
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent.parent
from paths import EASE_FILES as FILES, JUAN  # noqa: E402
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (HERE / "serve_files").exists() or not JUAN.exists(), reason="needs serve_files/ and Juan's repo")]


def fake_prices(app_ids, cc="us"):
    """Deterministic store: every 7th game not sold, every 11th free, others $1-$60."""
    out = {}
    for a in app_ids:
        a = int(a)
        if a % 7 == 0:
            out[a] = {"final": None, "initial": None, "discount": 0, "text": None}
        else:
            p = 0.0 if a % 11 == 0 else float(a % 60 + 1)
            out[a] = {"final": p, "initial": p, "discount": 0, "text": f"${p:.2f}"}
    return out


@pytest.fixture(scope="module")
def both():
    os.environ["STEAMREC_ARTIFACTS"] = str(FILES / "model" / "artifacts")
    os.environ["STEAMREC_DATA"] = str(FILES / "data_clean")
    sys.path.insert(0, str(JUAN / "model"))
    sys.path.insert(0, str(JUAN / "team_eval"))
    import recommender as juan_mod
    from app import engine as our_mod
    juan_mod.live_prices = fake_prices
    our_mod.live_prices = fake_prices
    juan = juan_mod.Recommender()
    ours = our_mod.Recommender(HERE / "serve_files")
    ours.B = ours.pipe["ease"].B_ = juan.B                       # same weights on both sides
    ours.pipe["ease"]._Bf = None
    return juan, ours


@pytest.fixture(scope="module")
def libraries():
    val = pd.read_parquet(FILES / "data_clean" / "team_eval" / "validation.parquet")
    v = val[~val.hidden_80_20]
    pids = np.random.default_rng(3).choice(np.sort(v[v.played].player_id.unique()), 150, replace=False)
    out = []
    for _, lib in v[v.player_id.isin(pids)].groupby("player_id"):
        played = [(int(a), int(p)) for a, p, pl in zip(lib.app_id, lib.playtime_min, lib.played) if pl and p > 0]
        out.append((played, [int(a) for a in lib.app_id]))
    return out


@pytest.mark.parametrize("opts", [{}, {"max_price": 10.0}, {"include_explicit": True}], ids=["default", "max_price", "adult"])
def test_same_output_as_juans_app(both, libraries, opts):
    juan, ours = both
    same = 0
    for played, owned in libraries:
        a = juan.recommend(played, owned=owned, **opts)
        b = ours.recommend(played, owned=owned, **opts)
        assert a == b, f"differs for a library of {len(owned)} games"
        same += 1
    assert same == len(libraries)


def test_favourites_fallback_matches(both):
    juan, ours = both
    for favs in ([570, 730], [1145360, 367520, 504230], [292030]):     # Dota 2 + CS; Hades, Hollow Knight, Celeste; Witcher 3
        lib = [(a, float(juan.q[a][90]) if a in juan.q else 6000) for a in favs]
        assert juan.recommend(lib, owned=favs) == ours.recommend(lib, owned=favs)
