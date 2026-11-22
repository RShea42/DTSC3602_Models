"""Fast tests on a tiny made-up catalog (seconds). The full-data check is m1_check.py."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from steamrec import EASEScorer, LibraryEncoder, make_pipeline
from steamrec.data import holdout_input

TAGS = [["Action", "Shooter"], ["Action", "RPG"], ["Puzzle"], ["Puzzle", "Casual"], ["RPG"], ["Shooter"], ["Racing"], ["Horror"]]


@pytest.fixture
def catalog(tmp_path):
    g = pd.DataFrame({
        "app_id": np.arange(1, 9), "tags": TAGS, "genres": [[]] * 8,
        "recommendable": [True] * 7 + [False],                       # game 8 never recommendable
        "median_playtime_min": [600, 600, 60, 600, 600, 600, 600, 600],   # game 3 is a short game
        "rating": ["Positive"] * 5 + ["Very Negative"] + ["Positive"] * 2})  # game 6 negative
    p = tmp_path / "games.parquet"
    g.to_parquet(p)
    return p


@pytest.fixture
def train():
    rng = np.random.default_rng(0)
    rows = [(u, a, int(rng.integers(30, 3000))) for u in range(300) for a in range(1, 9) if rng.random() < 0.5]
    return pd.DataFrame(rows, columns=["player_id", "app_id", "playtime_min"])


def test_vocab_respects_min_users(train):
    enc = LibraryEncoder(min_users=10_000).fit(train)
    assert len(enc.app_ids_) == 0
    enc = LibraryEncoder(min_users=1).fit(train)
    assert list(enc.app_ids_) == list(range(1, 9))


def test_ease_matches_closed_form(train):
    L = LibraryEncoder(min_users=1).fit_transform(train)
    B = EASEScorer(lam=5.0, dtype="float64").fit(L).B_
    X = L.V.toarray().astype(np.float64)
    P = np.linalg.inv(X.T @ X + 5.0 * np.eye(X.shape[1]))
    ref = np.eye(len(P)) - P / np.diag(P)
    assert np.allclose(B, ref) and np.all(np.diag(B) == 0)


def test_never_recommends_owned_or_unrecommendable(catalog, train):
    pipe = make_pipeline(catalog, serve_filters=False, lam=5.0, min_users=1, k=10).fit(train)
    lib = pd.DataFrame({"player_id": [1, 1, 1], "app_id": [1, 2, 4], "playtime_min": [500, 500, 0], "played": [True, True, False]})
    picks = pipe.predict(lib)
    assert not set(picks.app_id) & {1, 2, 4}                          # owned, played or not
    assert 8 not in set(picks.app_id)                                 # not recommendable
    assert list(picks["rank"]) == list(range(1, len(picks) + 1))


def test_flags_push_games_down(catalog, train):
    pipe = make_pipeline(catalog, serve_filters=False, lam=5.0, min_users=1, a_quit=0, b_short=0, c_negative=0).fit(train)
    lib = pd.DataFrame({"player_id": [1, 1], "app_id": [1, 2], "playtime_min": [500, 500]})
    plain = pipe.predict(lib).app_id.tolist()
    safe = pipe.set_params(safe__c_negative=10, safe__b_short=10).predict(lib).app_id.tolist()
    assert set(plain) == set(safe)                                    # same candidates, re-ordered
    assert safe[-2:] and set(safe[-2:]) == {3, 6}                     # short (3) and negative (6) sink to the bottom


def test_player_without_known_games_gets_nothing(catalog, train):
    pipe = make_pipeline(catalog, serve_filters=False, lam=5.0, min_users=1).fit(train)
    assert pipe.predict(pd.DataFrame({"player_id": [9], "app_id": [999], "playtime_min": [100]})).empty


def test_sklearn_contract(catalog, train):
    pipe = make_pipeline(catalog, serve_filters=False, lam=5.0, min_users=1)
    assert pipe.get_params()["ease__lam"] == 5.0
    c = clone(pipe).set_params(ease__lam=50.0)                        # what GridSearchCV does
    assert c.get_params()["ease__lam"] == 50.0 and pipe.get_params()["ease__lam"] == 5.0
    assert not hasattr(c["ease"], "B_")                               # clone is unfitted


def test_holdout_input_hides_hidden_rows():
    val = pd.DataFrame({"player_id": [1, 1, 1], "app_id": [1, 2, 3], "playtime_min": [5, 6, 0],
                        "played": [True, True, False], "hidden_80_20": [False, True, False]})
    assert holdout_input(val, "80_20").app_id.tolist() == [1, 3]


@pytest.fixture
def app_catalog(tmp_path):
    """Juan's app catalog columns. Game 2: 50% of 100 reviews (low-rated). Game 4: 10 reviews (too few).
    Game 5: explicit. Game 7: no Steam counts, falls back to positive_ratio 90 / user_reviews 500 (kept)."""
    g = pd.DataFrame({"app_id": np.arange(1, 9), "recommendable": [True] * 7 + [False],
                      "steam_positive": [900, 50, 900, 8, 900, 900, None, 900],
                      "steam_negative": [100, 50, 100, 2, 100, 100, None, 100],
                      "positive_ratio": [90] * 8, "user_reviews": [1000, 100, 1000, 10, 1000, 1000, 500, 1000],
                      "explicit": [False, False, False, False, True, False, False, False]})
    p = tmp_path / "app_games.parquet"
    g.to_parquet(p)
    return p


def test_default_is_juans_app_filters(catalog, app_catalog, train):
    pipe = make_pipeline(catalog, app_catalog, lam=5.0, min_users=1, a_quit=0, b_short=0, c_negative=0).fit(train)
    assert pipe.get_params()["ease__serve_filters"] is True           # default = what Juan's app serves
    lib = pd.DataFrame({"player_id": [1], "app_id": [1], "playtime_min": [500]})
    assert set(pipe.predict(lib).app_id) == {3, 6, 7}                 # 2 low-rated, 4 few reviews, 5 explicit, 8 not recommendable
    assert 5 in set(pipe.set_params(ease__include_explicit=True).predict(lib).app_id)
    evalmode = pipe.set_params(ease__serve_filters=False, ease__include_explicit=False).predict(lib)
    assert set(evalmode.app_id) == {2, 3, 4, 5, 6, 7}                 # Juan's evaluation: recommendable only


def test_serve_filters_need_app_catalog(catalog, train):
    pipe = make_pipeline(catalog, lam=5.0, min_users=1).fit(train)
    with pytest.raises(ValueError, match="app_catalog_path"):
        pipe.predict(pd.DataFrame({"player_id": [1], "app_id": [1], "playtime_min": [500]}))
