"""Recommendation engine: Steam library in, two rows of explained picks out.

Used by model/recommend.py (command line) and app/server.py (local web page).
- Enjoyment of each owned game = the player's playtime percentile among that game's players (stored quantiles).
- Scores come from EASE (model/build_ease.py), the model the evaluation scored: each played game adds its row of
  item-to-item weights. EASE only knows games with >= 100 training players, so newer or niche games are not scored.
- "For you" is EASE-safe: EASE's top 50 re-ranked by z(score) - a x like_quit - b x short_game - c x negative_reviews,
  the rule and settings scored in team_eval/ (flags from team_eval/evaluate.py, settings from
  team_eval/rerank_choice.json), so the app and the reported results use the same definitions.
- "Try something new": the top 400 reranked by z(score) + 4 x tag distance from the player's profile.
  Every pick names the owned games that pushed it up most.
"""
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
import os  # noqa: E402
# Data folders; set STEAMREC_ARTIFACTS / STEAMREC_DATA to use your own copies (defaults: Juan's machine)
A = Path(os.environ.get("STEAMREC_ARTIFACTS", ROOT / "model" / "artifacts"))
D = Path(os.environ.get("STEAMREC_DATA", "C:/Users/juant/data_clean"))
STEAM = "https://api.steampowered.com"
sys.path.insert(0, str(ROOT / "team_eval"))
from evaluate import NEG, tag_matrix  # noqa: E402   the team's flag definitions, shared with the scoring
SAFE = json.loads((ROOT / "team_eval" / "rerank_choice.json").read_text())   # a_quit, b_short, c_negative


def _key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("STEAM_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("STEAM_API_KEY missing from .env")


# ------------------------------------------------------------------ Steam lookups
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
    def __init__(self):
        z = np.load(A / "ease.npz")
        self.B = z["B"]                                   # float16, rows = played game, columns = candidate game
        self.app_ids = z["app_ids"]
        self.idx = {int(a): i for i, a in enumerate(self.app_ids)}
        it = pd.read_parquet(A / "items.parquet")
        self.q = {int(a): np.asarray(q) for a, q in zip(it.app_id, it.q) if q is not None}
        g = pd.read_parquet(D / "games.parquet", columns=["app_id", "title", "tags", "genres", "price_regular", "recommendable",
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
        rec = g.recommendable.reindex(self.app_ids).fillna(False).to_numpy(bool)
        # actionable picks only: on sale in the store, not adult, and not poorly rated
        self.candidate = rec & ~((self.rating < 0.65) & (self.n_reviews >= 50)) & (self.n_reviews >= 20)
        # sexually explicit games (rule G25): kept, but only shown when the player opts in
        self.explicit = g.explicit.reindex(self.app_ids).fillna(False).to_numpy(bool)
        # user tags; ~11% of games have none in our sources, so fall back to store genres
        self.tags = {int(a): [x for x in (list(t) if t is not None and len(t) else list(ge) if ge is not None else [])
                              if x.lower() not in BROAD][:5] for a, t, ge in zip(g.index, g.tags, g.genres)}
        self.titles = g.title.to_dict()
        self.search_titles = g[g.recommendable | g.index.isin(list(self.idx))].title.dropna()
        # EASE-safe flags, built exactly as the team scoring builds them (team_eval/games.parquet, same catalog)
        fg = pd.read_parquet(D / "team_eval" / "games.parquet", columns=["app_id", "tags", "genres", "rating", "median_playtime_min"])
        self.T = tag_matrix(fg).tocsr()
        self.tpos = {int(a): i for i, a in enumerate(fg.app_id)}
        self.trow = np.array([self.tpos.get(int(a), -1) for a in self.app_ids])
        fl = fg.set_index("app_id").reindex(self.app_ids)
        self.flag_short = (fl.median_playtime_min <= 120).to_numpy()          # unknown median -> not flagged
        self.flag_neg = fl.rating.isin(NEG).to_numpy()

    # ---- EASE-safe: push flagged games down within EASE's top 50 (team_eval/rerank.py, same rule)
    def safe_rerank(self, order, s, library, k=50):
        head = list(order[:k])
        if len(head) < 2:
            return list(order)
        sc = s[head].astype(np.float64)
        z = (sc - sc.mean()) / (sc.std(ddof=1) + 1e-9)
        played = [(self.tpos[a], pt) for a, pt in library if pt > 0 and a in self.tpos]
        quit = np.zeros(len(head), bool)
        short_ix = [i for i, pt in played if pt < 120]
        kept_ix = [i for i, pt in played if pt >= 120]
        if short_ix and kept_ix:                   # like quit: closer to the under-2-hour games than to the kept ones
            def prof(ix):
                p = np.asarray(self.T[ix].sum(0)).ravel()
                n = np.linalg.norm(p)
                return p / n if n else p
            rows = self.trow[head]
            G = self.T[np.maximum(rows, 0)]
            has_tags = (rows >= 0) & (np.asarray(G.sum(1)).ravel() > 0)
            quit = has_tags & (G @ prof(short_ix) > G @ prof(kept_ix))
        new = z - SAFE["a_quit"] * quit - SAFE["b_short"] * self.flag_short[head] - SAFE["c_negative"] * self.flag_neg[head]
        return [head[i] for i in np.argsort(-new, kind="stable")] + list(order[k:])

    # ---- enjoyment of an owned game
    def enjoyment(self, app_id, playtime_min):
        q = self.q.get(app_id)
        if q is None:
            return 0.5 if playtime_min >= 120 else 0.25
        return float(np.searchsorted(q, playtime_min, side="right")) / len(q)

    def taste(self, library):
        """library: list of (app_id, playtime_min). Returns (scores over app_ids, used rows [(idx, 1, app_id, g)]).
        EASE input is binary (played or not), as in the evaluation; g only weights the tag profile."""
        rows = []
        for a, pt in library:
            i = self.idx.get(a)
            if i is None or pt <= 0:
                continue
            rows.append((i, 1.0, a, self.enjoyment(a, pt)))
        if not rows:
            return None, []
        return self.B[[r[0] for r in rows]].astype(np.float32).sum(0), rows

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
        s, rows = self.taste(library)
        if s is None:
            return None
        mask = self.candidate.copy()
        if not include_explicit:
            mask &= ~self.explicit
        for a in set(owned or []) | {a for a, _ in library}:
            i = self.idx.get(a)
            if i is not None:
                mask[i] = False
        s = np.where(mask, s, -np.inf)
        order = [j for j in np.argsort(-s)[:600] if np.isfinite(s[j])]
        order = self.safe_rerank(order, s, library)          # EASE-safe "For you"
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


if __name__ == "__main__":
    print(json.dumps(get().search("hades"), indent=1))
