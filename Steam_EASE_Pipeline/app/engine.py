"""The app's recommendation engine: Juan's model/recommender.py with its scoring replaced by the sklearn pipeline.

Kept from Juan unchanged (same code, same behaviour): Steam lookups (resolve_steamid, fetch_profile), live store
prices, the max-price rule, "Try something new" (top 400 re-ranked by z(score) + 4 x tag distance), "because"
explanations, cards, search, the favourite-games fallback.
Replaced: taste / candidate mask / EASE scores / EASE-safe re-rank now come from the fitted pipeline in app mode
(serve_filters=True, 600 candidates, top 50 re-ranked), which tests/test_app.py proves gives the same output.
Differences from Juan's file: config from environment variables (Modal) instead of his local paths and .env; the
model object is built from the pipeline instead of ease.npz.
"""
import copy
import os
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from steamrec.pipeline import app_candidates

STEAM = "https://api.steampowered.com"
FILES = Path(os.environ.get("STEAMREC_FILES", Path(__file__).resolve().parent.parent / "serve_files"))
D = Path(os.environ.get("STEAMREC_DATA", FILES))       # Juan's name for his data folder; read by Peyton's modal_api.py


def _key():
    """Steam Web API key: STEAM_API_KEY (or STEAM_API_KEY2, the name in the Modal secret 'steam-api-key')."""
    k = os.environ.get("STEAM_API_KEY") or os.environ.get("STEAM_API_KEY2")
    if not k:
        raise RuntimeError("set STEAM_API_KEY (locally) or attach the Modal secret 'steam-api-key'")
    return k


# ------------------------------------------------------------------ Steam lookups (Juan, unchanged)
class SteamError(Exception):
    pass


def resolve_steamid(text):
    """Accepts a 17-digit SteamID64, a profile URL (/profiles/<id> or /id/<name>), or a custom profile name."""
    t = (text or "").strip().rstrip("/")
    m = re.search(r"/profiles/(\d{17})", t)
    if m:
        return m.group(1)
    if re.fullmatch(r"7656\d{13}", t):
        return t
    m = re.search(r"/id/([^/?#]+)", t)
    name = m.group(1) if m else t
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,64}", name):
        raise SteamError("That doesn't look like a Steam ID, profile link or custom profile name.")
    r = requests.get(f"{STEAM}/ISteamUser/ResolveVanityURL/v1/", params={"key": _key(), "vanityurl": name}, timeout=20).json()
    sid = (r.get("response") or {}).get("steamid")
    if not sid:
        raise SteamError(f"No Steam profile found for '{name}'. Try the profile link or the 17-digit Steam ID.")
    return sid


def fetch_profile(steamid):
    """Returns (summary dict, library list of (app_id, playtime_min, name)). Library is None when game details are private."""
    s = requests.get(f"{STEAM}/ISteamUser/GetPlayerSummaries/v2/", params={"key": _key(), "steamids": steamid}, timeout=20).json()
    p = ((s.get("response") or {}).get("players") or [{}])[0]
    summary = {"name": p.get("personaname"), "url": p.get("profileurl"), "avatar": p.get("avatarmedium"), "public": p.get("communityvisibilitystate") == 3,
               "country": p.get("loccountrycode")}
    r = requests.get(f"{STEAM}/IPlayerService/GetOwnedGames/v1/",
                     params={"key": _key(), "steamid": steamid, "include_appinfo": 1, "include_played_free_games": 1},
                     timeout=30).json()
    games = (r.get("response") or {}).get("games")
    lib = None if games is None else [(int(g["appid"]), int(g.get("playtime_forever", 0)), g.get("name")) for g in games]
    return summary, lib


# tags that describe format or business model, not taste: never shown as "new for you" or used for the taste profile
BROAD = {"Singleplayer", "Multiplayer", "Indie", "Free to Play", "Early Access", "Great Soundtrack", "Co-op", "Online Co-Op",
         "PvP", "Massively Multiplayer", "Controller", "Moddable", "Classic", "Remake", "Remaster", "Female Protagonist",
         "Steam Achievements", "Family Friendly", "Masterpiece", "Addictive", "Replay Value", "Difficult", "Short",
         "Beautiful", "Funny", "Cute", "Colorful", "Relaxing", "Utilities", "Software"}
NOT_NEW = {"3D", "2D", "Third Person", "First-Person", "Atmospheric", "Story Rich", "Open World", "Sandbox", "Fantasy",
           "Sci-fi", "Shooter", "Puzzle", "Platformer", "Pixel Graphics", "Retro", "Violent", "Gore", "Dark", "Exploration",
           "Character Customization", "Multiple Endings", "Choices Matter", "Anime", "Realistic", "Comedy",
           "Casual", "Action", "Adventure", "Simulation", "Strategy", "RPG"}
BROAD = {t.lower() for t in BROAD}
NOT_NEW = {t.lower() for t in NOT_NEW}


def live_prices(app_ids, cc="us"):
    """Current Steam store price for each app (one request for the batch). {app_id: {final, initial, discount, text}}."""
    out = {}
    for k in range(0, len(app_ids), 50):
        chunk = [str(a) for a in app_ids[k:k + 50]]
        try:
            r = requests.get("https://store.steampowered.com/api/appdetails",
                             params={"appids": ",".join(chunk), "filters": "price_overview", "cc": cc}, timeout=15).json()
        except (requests.RequestException, ValueError):
            continue
        for a in chunk:
            v = (r or {}).get(a) or {}
            if not v.get("success"):
                continue
            p = v.get("data") or {}
            po = p.get("price_overview") if isinstance(p, dict) else None
            out[int(a)] = ({"final": po["final"] / 100, "initial": po["initial"] / 100, "discount": po.get("discount_percent", 0),
                            "text": po.get("final_formatted")} if po else {"final": None, "initial": None, "discount": 0, "text": None})  # free or not sold
    return out


# ------------------------------------------------------------------ model
class Recommender:
    def __init__(self, files=FILES):
        files = Path(files)
        pipe = joblib.load(files / "ease_safe_pipeline.joblib")
        # app mode, Juan's live settings: his candidate filters, EASE's top 600 kept, top 50 re-ranked, whole order out
        pipe.set_params(ease__serve_filters=True, ease__app_catalog_path=str(files / "games.parquet"),
                        ease__catalog_path=str(files / "team_eval_games.parquet"), ease__n_candidates=600,
                        safe__catalog_path=str(files / "team_eval_games.parquet"), safe__head=50, safe__k=None)
        self.pipe = pipe
        self.B = pipe["ease"].B_                          # float16, rows = played game, columns = candidate game
        self.app_ids = pipe["encode"].app_ids_
        self.idx = {int(a): i for i, a in enumerate(self.app_ids)}
        it = pd.read_parquet(files / "items.parquet")
        self.q = {int(a): np.asarray(q) for a, q in zip(it.app_id, it.q) if q is not None}
        g = pd.read_parquet(files / "games.parquet", columns=["app_id", "title", "tags", "genres", "price_regular", "recommendable",
                                                              "header_image", "release_date", "steam_positive", "steam_negative",
                                                              "positive_ratio", "user_reviews", "adult", "kaggle_price_regular", "price_source", "explicit", "content_level"])
        g = g.set_index("app_id")
        self.games = g
        pos, neg = g.steam_positive.fillna(0), g.steam_negative.fillna(0)
        n = pos + neg
        self.rating = ((pos / n.where(n > 0)).fillna(g.positive_ratio / 100)).reindex(self.app_ids).to_numpy()
        self.n_reviews = n.where(n > 0, g.user_reviews).reindex(self.app_ids).fillna(0).to_numpy()
        # regular-price estimate for when the store can't be reached: Steam regular price (G24), else the older sources
        self.est_price = g.price_regular.where(g.price_source.eq("itad_steam_regular"),   # rule G24
                                               g[["price_regular", "kaggle_price_regular"]].max(axis=1)).reindex(self.app_ids).to_numpy(float)
        # user tags; ~11% of games have none in our sources, so fall back to store genres
        self.tags = {int(a): [x for x in (list(t) if t is not None and len(t) else list(ge) if ge is not None else [])
                              if x.lower() not in BROAD][:5] for a, t, ge in zip(g.index, g.tags, g.genres)}
        self.titles = g.title.to_dict()
        self.search_titles = g[g.recommendable | g.index.isin(list(self.idx))].title.dropna()
        for allow in (False, True):                       # warm the filter cache so requests don't pay for it
            self._scorer(allow).allowed()
        # ---- Juan's Recommender attributes that other code reads directly (e.g. Peyton's modal_api.py run(): per-card
        # flags and contributions). Same names, same values as Juan's __init__, taken from the fitted pipeline.
        safe = pipe["safe"]
        self.T = safe.T_                                  # tag matrix over the team catalog (tag_matrix, as Juan)
        self.tpos = {int(a): int(i) for a, i in safe.gpos_.items()}
        self.trow = np.array([self.tpos.get(int(a), -1) for a in self.app_ids])
        known = self.trow >= 0
        self.flag_short = np.where(known, safe.short_[np.maximum(self.trow, 0)], False)
        self.flag_neg = np.where(known, safe.neg_[np.maximum(self.trow, 0)], False)
        self.candidate = app_candidates(self.app_ids, str(files / "games.parquet"), include_explicit=True)
        self.explicit = g.explicit.reindex(self.app_ids).fillna(False).to_numpy(bool)

    def _scorer(self, include_explicit):
        """The pipeline's EASE step with this request's adult-content choice. A shallow copy shares the weights and
        never changes the shared model, so concurrent requests can't interfere."""
        e = copy.copy(self.pipe["ease"])
        e.include_explicit = include_explicit
        return e

    def ordered(self, library, owned, include_explicit):
        """Pipeline: EASE-safe order of up to 600 candidates -> (indices into app_ids, score array over app_ids)."""
        lib = dict(library)
        rows = [(0, a, lib.get(a, 0), lib.get(a, 0) > 0) for a in set(owned or []) | set(lib)]
        X = pd.DataFrame(rows, columns=["player_id", "app_id", "playtime_min", "played"])
        L = self._scorer(include_explicit).transform(self.pipe["encode"].transform(X))
        r = self.pipe["safe"].predict(L)
        s = np.full(len(self.app_ids), -np.inf, np.float32)
        j = np.array([self.idx[int(a)] for a in r.app_id], dtype=np.int64)
        s[j] = r.score.to_numpy(np.float32)
        return list(j), s

    # ---- enjoyment of an owned game
    def enjoyment(self, app_id, playtime_min):
        q = self.q.get(app_id)
        if q is None:
            return 0.5 if playtime_min >= 120 else 0.25
        return float(np.searchsorted(q, playtime_min, side="right")) / len(q)

    def taste_rows(self, library):
        """Played games the model knows: [(idx, 1, app_id, enjoyment)] (Juan's taste(), scores now from the pipeline)."""
        rows = []
        for a, pt in library:
            i = self.idx.get(a)
            if i is None or pt <= 0:
                continue
            rows.append((i, 1.0, a, self.enjoyment(a, pt)))
        return rows

    def _why(self, j, rows, n=2):
        contrib = self.B[[r[0] for r in rows], j].astype(np.float32)   # how much each played game pushed this pick up
        top = np.argsort(-contrib)[:n]
        return [self.titles.get(rows[k][2], str(rows[k][2])) for k in top if contrib[k] > 0]

    def _card(self, j, score, rows, prices, extra=None):
        a = int(self.app_ids[j])
        g = self.games.loc[a] if a in self.games.index else None
        lp = prices.get(a)
        if lp and lp["final"] is not None:
            price = {"price": lp["final"], "price_initial": lp["initial"], "discount": lp["discount"], "price_text": lp["text"]}
        elif lp:                                    # store lists no price: free if our catalog says so, else check the store
            free = self.est_price[j] == 0
            price = {"price": 0.0 if free else None, "price_initial": None, "discount": 0, "price_text": "Free" if free else "See store"}
        else:                                       # store unreachable: last known regular price, marked approximate
            est = None if np.isnan(self.est_price[j]) else float(self.est_price[j])
            price = {"price": est, "price_initial": est, "discount": 0,
                     "price_text": None if est is None else ("Free" if est == 0 else f"~${est:.2f}")}
        card = {"app_id": a, "title": self.titles.get(a, str(a)), "score": round(float(score), 4),
                "because": self._why(j, rows), "tags": self.tags.get(a, [])[:4], **price,
                "rating": None if np.isnan(self.rating[j]) else round(float(self.rating[j]) * 100),
                "reviews": int(self.n_reviews[j]),
                "image": None if g is None or pd.isna(g.header_image) else g.header_image,
                "store_url": f"https://store.steampowered.com/app/{a}/",
                "content_level": None if g is None else g.content_level}
        card.update(extra or {})
        return card

    def recommend(self, library, owned=None, n=10, max_price=None, cc="us", include_explicit=False):
        """library: (app_id, playtime_min) pairs; owned: every owned app_id (also excluded from picks).
        max_price is checked against the live store price (in the store currency for country cc)."""
        rows = self.taste_rows(library)
        if not rows:
            return None
        order, s = self.ordered(library, owned, include_explicit)      # EASE-safe "For you", from the pipeline
        # ---- from here on: Juan's recommend(), unchanged
        prices = {}
        if max_price is not None:
            head = order[:150]
            prices = live_prices([int(self.app_ids[j]) for j in head], cc)
            def ok(j):
                lp = prices.get(int(self.app_ids[j]))
                v = lp["final"] if lp and lp["final"] is not None else self.est_price[j]
                return not (v > max_price)
            order = [j for j in head if ok(j)] + [j for j in order[150:] if not (self.est_price[j] > max_price)]
        top = order[:n + 10]                        # spares, in case some turn out not to be on sale

        # taste profile over tags, weighted by enjoyment, for the "Try something new" row
        prof = {}
        for _, _, a, g in rows:
            for t in self.tags.get(a, [])[:3]:
                prof[t] = prof.get(t, 0) + g
        tot = sum(prof.values()) or 1
        prof = {t: v / tot for t, v in prof.items()}
        top_tags = [t for t in sorted(prof, key=prof.get, reverse=True) if t.lower() not in NOT_NEW][:5] or sorted(prof, key=prof.get, reverse=True)[:5]
        # soft rerank of the top 400: z(score) + 4 x tag distance (1 - cosine to the profile). Chosen post hoc on
        # validation players (eval/diagnosis_report.md): 93% of For you's NDCG@10 with wider tags than the old filter.
        new = []
        if prof:
            pool = [j for j in order[:400] if j not in set(top[:n])]          # never repeat a For you pick
            norm = np.sqrt(sum(v * v for v in prof.values()))
            def dist(j):
                tg = self.tags.get(int(self.app_ids[j]), [])[:3]
                return 1 - sum(prof.get(t, 0) for t in tg) / (np.sqrt(len(tg)) * norm) if tg else 0.0
            if pool:
                sc = np.array([s[j] for j in pool])
                z = (sc - sc.mean()) / (sc.std() + 1e-9)
                d = np.array([dist(j) for j in pool])
                for k in np.argsort(-(z + 4.0 * d), kind="stable")[:n + 5]:
                    j = pool[k]
                    fresh = sorted((t for t in self.tags.get(int(self.app_ids[j]), [])[:3] if t.lower() not in NOT_NEW),
                                   key=lambda t: prof.get(t, 0))
                    new.append((j, fresh[0] if fresh and prof.get(fresh[0], 0) < 0.05 else None))
        need = [int(self.app_ids[j]) for j in top + [j for j, _ in new] if int(self.app_ids[j]) not in prices]
        prices.update(live_prices(need, cc))
        if prices:                                  # drop delisted / unbuyable games (only when the store answered)
            def buyable(j):
                lp = prices.get(int(self.app_ids[j]))
                return lp is not None and (lp["final"] is not None or self.est_price[j] == 0)
            top = [j for j in top if buyable(j)]
            new = [(j, t) for j, t in new if buyable(j)]
        top = top[:n]
        new = [(j, t) for j, t in new if j not in set(top)][:n]
        played = sorted(rows, key=lambda r: -r[3])
        return {"for_you": [self._card(j, s[j], rows, prices) for j in top],
                "try_new": [self._card(j, s[j], rows, prices, {"new_tag": t}) for j, t in new], "top_tags": top_tags,
                "n_owned": len(set(owned or []) | {a for a, _ in library}), "n_used": len(rows),
                "favorites": [self.titles.get(r[2], str(r[2])) for r in played[:5]]}

    def search(self, text, n=8):
        t = text.strip().lower()
        if len(t) < 2:
            return []
        hit = self.search_titles[self.search_titles.str.lower().str.contains(re.escape(t))]
        hit = hit.loc[sorted(hit.index, key=lambda a: (-self.n_reviews[self.idx[a]] if a in self.idx else 0))][:n * 2]
        hit = hit[~hit.str.lower().duplicated()][:n]           # duplicate catalog rows: keep the most-reviewed
        return [{"app_id": int(a), "title": ti} for a, ti in hit.items()]


@lru_cache(maxsize=1)
def get():
    return Recommender()


def for_profile(text, n=10, max_price=None, include_explicit=False):
    """Full pipeline for a Steam ID / link / name. Raises SteamError with a user-facing message."""
    sid = resolve_steamid(text)
    summary, lib = fetch_profile(sid)
    who = f"Found the account '{summary['name']}' ({summary['url']}). " if summary.get("name") else ""
    if lib is None:
        raise SteamError(who + "Its game details are private. If that's not you: a Steam display name isn't unique, so paste "
                         "your profile link instead (Steam > your name > Profile, then copy the address). If it is you: "
                         "Profile > Edit Profile > Privacy Settings > set 'Game details' to Public. Or pick a few favourite games below.")
    played = [(a, pt) for a, pt, _ in lib if pt > 0]
    if not played:
        raise SteamError(who + "It shows no playtime. In Privacy Settings, untick 'Always keep my total playtime private', "
                         "or pick a few favourite games below.")
    out = get().recommend(played, owned=[a for a, _, _ in lib], n=n, max_price=max_price, cc=(summary["country"] or "us").lower(),
                          include_explicit=include_explicit)
    if out is None:
        raise SteamError("None of the games on this profile are in our data yet. Pick a few favourite games instead.")
    out["player"] = {"name": summary["name"], "avatar": summary["avatar"]}
    return out


def for_favorites(app_ids, n=10, max_price=None, include_explicit=False):
    """Fallback for private or empty profiles: treat each picked game as one the player loved."""
    r = get()
    lib = [(int(a), float(r.q[int(a)][90]) if int(a) in r.q else 6000) for a in app_ids]
    out = r.recommend(lib, owned=[a for a, _ in lib], n=n, max_price=max_price, include_explicit=include_explicit)
    if out is None:
        raise SteamError("None of those games are in our data yet. Try more popular titles.")
    return out
