"""Steam game recommender: paste a profile, react to a 5-card first look, get a refined list of 15.

Run locally:  streamlit run recommender_app.py
Needs artifacts/ (python features.py) and a Steam Web API key (see steam_api.py).
Nothing a visitor does is saved anywhere: libraries are fetched, used and kept only in this
session's memory.
"""
from __future__ import annotations

import functools
import html
import html as htmllib
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pandas as pd
import streamlit as st

from features import load_features
from recommend import RecConfig, first_look, refine

st.set_page_config(page_title="Steam Game Finder", page_icon="🎮", layout="centered")

STORE = "https://store.steampowered.com/app/{}"
# Never put "<" inside this block, even in a CSS comment: Streamlit's HTML sanitiser then drops
# the whole <style> element and every custom style disappears. (Paragraph sizes below need
# !important because Streamlit sets its own size on paragraphs.)
CSS = """
<style>
:root { --card-bg: #f4f6f9; --card-border: #d9dee6; --muted: #5b6573; --chip: #e3e8ef; }
@media (prefers-color-scheme: dark) {
  :root { --card-bg: #1b2330; --card-border: #2c3747; --muted: #9aa6b6; --chip: #2a3546; }
}
@keyframes slide-up { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: none; } }
.card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px;
        overflow: hidden; animation: slide-up .35s ease-out; }
.card img { width: 100%; display: block; }
.card .body { padding: 14px 18px 16px; }
.card h2 { margin: 0 0 4px; font-size: 1.5rem; }
.card .meta { color: var(--muted); font-size: .9rem; margin-bottom: 10px; }
.card h2 a { color: inherit; text-decoration: none; }
.card h2 a:hover { text-decoration: underline; }
.card a.store { display: inline-block; margin-top: 8px; font-size: .85rem; }
.card p.blurb { font-size: 1.02rem !important; line-height: 1.45; margin: 0 0 12px; }
.card .why { font-size: .95rem; margin-top: 10px; }
.chip { display: inline-block; background: var(--chip); border-radius: 999px; padding: 2px 10px;
        margin: 0 6px 6px 0; font-size: .8rem; }
.row { display: flex; gap: 14px; align-items: center; padding: 8px 0;
       border-bottom: 1px solid var(--card-border); }
.row img { width: 184px; border-radius: 6px; flex-shrink: 0; }
.row .rank { font-size: 1.2rem; font-weight: 600; width: 2ch; text-align: right; flex-shrink: 0; }
.row a { font-weight: 600; font-size: 1.05rem; }
.row .meta { color: var(--muted); font-size: .85rem; }
details.why { margin-top: 6px; line-height: 1.4; }
details.why summary { font-size: .78rem !important; }
details.why p, details.why p b { font-size: .68rem !important; }
details.why summary { cursor: pointer; color: var(--muted); list-style: none; user-select: none; }
details.why summary::-webkit-details-marker { display: none; }
details.why summary::before { content: "▸"; display: inline-block; width: 1.1em; }
details.why[open] summary::before { content: "▾"; }
details.why p { margin: 6px 0 0 1.1em; }
</style>
"""


DEFAULTS = RecConfig()
# RecConfig setting -> (label, min, max, step, explanation) for the Advanced settings sliders.
ADVANCED = {
    "quality_weight": ("Favor well-reviewed, popular games", 0.0, 1.0, 0.05,
                       "Share of each score from review score and review count. Higher = safer, "
                       "better-known picks; above ~0.6 lists turn generic."),
    "dislike_weight": ("Avoid games like the ones you barely played", 0.0, 1.5, 0.25,
                       "How strongly games you launched but dropped push similar games down."),
    "liked_count": ("Games that define your taste", 5, 40, 5,
                    "How many of your most-played games build your taste profile."),
    "max_franchise_picks": ("Max picks from series you own", 0, 15, 1,
                            "e.g. Fallout: New Vegas when you own Fallout 3. 0 = none."),
    "max_per_anchor": ("Variety (picks per liked game, per round)", 1, 5, 1,
                       "Lower = the list spreads across more of your tastes."),
    "min_reviews": ("Minimum Steam reviews", 10, 1000, 10,
                    "Games with fewer reviews are never recommended."),
}


def reset_advanced():
    for name in ADVANCED:
        st.session_state[f"adv_{name}"] = getattr(DEFAULTS, name)


@st.cache_resource(show_spinner="Loading the game catalog...")
def features():
    return load_features()


@st.cache_data(ttl="1h", show_spinner=False, max_entries=200)
def fetch_library(profile: str) -> dict[int, int]:
    from steam_api import owned_games     # held in memory for an hour, never written to disk
    return owned_games(profile)


@functools.lru_cache(maxsize=5000)     # server memory only; failures raise, so they aren't cached
def _fetch_store(appid: int) -> tuple[str, str]:
    """(blurb, price) from the live Steam store (US prices). Price is "" when the store page
    exists but lists no price (delisted, unreleased, or only sold in a bundle)."""
    url = (f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english&cc=us"
           "&filters=basic,price_overview")
    req = urllib.request.Request(url, headers={"User-Agent": "steam-recommender-class-project"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.load(resp)[str(appid)]
    info = data.get("data", {}) if data.get("success") else {}
    return _blurb(info.get("short_description", "")), _price(info)


def _price(info: dict) -> str:
    po = info.get("price_overview")
    if po:
        off = po.get("discount_percent", 0)
        return po["final_formatted"] + (f" ({off}% off)" if off else "")
    return "Free" if info.get("is_free") else ""


def _blurb(text: str) -> str:
    text = " ".join(re.sub(r"<[^>]+>", " ", htmllib.unescape(text)).split())
    # Keep it a quick read: whole sentences up to ~220 characters (always at least the first, and
    # a second if the first is under 80 characters and says little); if that runs very long, cut
    # at a word and add an ellipsis.
    sentences, kept = re.split(r"(?<=[.!?])\s+", text), ""
    for s in sentences:
        if kept and len(kept) >= 80 and len(kept) + len(s) > 220:
            break
        kept = f"{kept} {s}".strip()
    if len(kept) > 300:
        kept = kept[:300].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return kept


def store_info(appids) -> dict[int, tuple[str, str]]:
    """{appid: (blurb, price)} from the live store, fetched in parallel. A failed fetch gives
    ("", None): no blurb, and the price falls back to the catalog's."""
    def one(appid):
        try:
            return int(appid), _fetch_store(int(appid))
        except Exception:
            return int(appid), ("", None)
    with ThreadPoolExecutor(max_workers=5) as pool:
        return dict(pool.map(one, appids))


def details(appid: int) -> pd.Series:
    f = features()
    return f.games.iloc[f.row_of[appid]]


def meta_line(g: pd.Series, store: dict | None = None) -> str:
    live = (store or {}).get(int(g.appid), ("", None))[1]
    if live is None:                       # store unreachable: use the (possibly stale) catalog
        live = "Free" if g.price == 0 else f"${g.price:.2f}"
    # No store price: free-to-play conversions (e.g. Rocket League) aren't flagged is_free, but the
    # catalog has them at 0; anything else is delisted, unreleased, or bundle-only.
    price = live or ("Free" if g.price == 0 else "Price not listed")
    reviews = f"{g.positive_review_pct:.0f}% positive of {int(g.total_reviews):,} reviews"
    return f"{price} · {reviews} · {int(g.release_year)}"


def join_names(names) -> str:
    names = [f"<b>{html.escape(n)}</b>" for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f"{',' if len(names) > 2 else ''} and {names[-1]}"


def shared_tags(appid: int, aligned_appids: tuple[int, ...], limit: int = 4) -> list[str]:
    """Tags of the pick that its aligned library games also carry, most widely shared first.
    (A collapsed franchise is represented by its most-played game.)"""
    f = features()
    pick = details(appid).tag_list.split(",")[:12]
    others = [set(details(a).tag_list.split(",")[:12]) for a in aligned_appids if a in f.row_of]
    counts = {t: sum(t in o for o in others) for t in pick}
    return [t for t in sorted(pick, key=lambda t: -counts[t]) if counts[t]][:limit]


def why_html(x: pd.Series, reactions: dict[int, bool] | None = None,
             swiped_names: set[str] = frozenset()) -> str:
    """The collapsible 'Why we picked this' explanation for one recommended game."""
    parts = []
    if x.aligned_with:
        parts.append(f"This game aligns well with {join_names(x.aligned_with)}.")
        tags = shared_tags(int(x.appid), x.aligned_appids)
        if tags:
            parts.append(f"In common: {html.escape(', '.join(tags))}.")
    if x.because in swiped_names and not (reactions or {}).get(int(x.appid)):
        parts.append(f"It's also close to <b>{html.escape(x.because)}</b>, which you were "
                     "interested in.")
    if not parts:
        parts.append("This pick comes from your first-look reactions.")
    return f'<details class="why"><summary>Why we picked this</summary><p>{" ".join(parts)}</p></details>'


def reset():
    for k in ("stage", "pool", "deck", "card", "reactions", "final", "error", "hidden",
              "hidden_groups", "store"):
        st.session_state.pop(k, None)


def hidden_games(library: dict[int, dict], hide_top: int, hide_recent: int) -> dict[str, list[int]]:
    """Test mode: the most-played games (by total hours) and/or the most recently played ones
    (games with any playtime, newest last session first)."""
    groups = {}
    if hide_top:
        by_time = sorted(library, key=lambda a: library[a]["minutes"], reverse=True)
        groups["most played"] = by_time[:hide_top]
    if hide_recent:
        played = [a for a in library if library[a]["minutes"] > 0 and library[a]["last_played"] > 0]
        groups["recently played"] = sorted(played, key=lambda a: library[a]["last_played"],
                                           reverse=True)[:hide_recent]
    return groups


def start(profile: str, cfg: RecConfig, hide_top: int = 0, hide_recent: int = 0):
    """Fetch the library and build the pool and deck, or record a friendly error.
    Test mode: hide some games from the library, and reveal at the end whether the
    recommendations found them anyway."""
    reset()
    try:
        library = fetch_library(profile.strip())
        groups = hidden_games(library, hide_top, hide_recent)
        hidden = list(dict.fromkeys(a for ids in groups.values() for a in ids))   # each game once
        library = {a: v for a, v in library.items() if a not in set(hidden)}
        played = sum(v["minutes"] > 0 for v in library.values())
        if played < 3:
            raise ValueError(f"Only {played} played game(s) on this profile; I need at least 3 to "
                             "work out your taste.")
        pool, deck = first_look(features(), library, cfg)
    except ValueError as e:                      # private profile, unknown profile, tiny library
        st.session_state.error = str(e)
        return
    except (urllib.error.URLError, TimeoutError):
        st.session_state.error = "Couldn't reach Steam just now. Try again in a minute."
        return
    except SystemExit:                           # steam_api couldn't find an API key
        st.session_state.error = "The app isn't configured with a Steam API key."
        return
    st.session_state.update(stage="deck", library=library, pool=pool, deck=deck, card=0,
                            reactions={}, cfg=cfg, hidden=hidden, hidden_groups=groups,
                            store=store_info(deck.appid.tolist()))


def react(card: int, interested: bool):
    """Record a swipe for the card the player actually saw. Pressing a key twice quickly sends a
    second click aimed at the same card before the page has redrawn; that stale click is ignored
    (it used to answer the next, unseen card, or crash after the last one)."""
    s = st.session_state
    if s.get("stage") != "deck" or card != s.card or card >= len(s.deck):
        return
    # Only quick, plain assignments here. The slow final-list step runs in the page body (see
    # "Screens"): a fast extra click can cut a callback off halfway, which used to leave the app
    # past the last card but not yet on the results screen.
    s.reactions = {**s.reactions, int(s.deck.appid.iat[card]): interested}
    s.card = card + 1


# ---- Screens ----------------------------------------------------------------------------------

st.html(CSS)
state = st.session_state
# All cards answered: build the final list here, not in the click handler. If another click
# interrupts this, the next run simply does it again.
if state.get("stage") == "deck" and state.card >= len(state.deck):
    with st.spinner("Building your list..."):
        state.final = refine(features(), state.library, state.pool, state.deck, state.reactions,
                             state.cfg)
    state.stage = "results"
stage = state.get("stage", "input")

with st.sidebar:
    st.header("Settings")
    hide_adult = st.toggle("Hide adult content", value=True)
    recency = st.select_slider(
        "Favor recently played games", options=["Off", "3 months", "6 months", "1 year", "2 years"],
        value="6 months", disabled=stage != "input",
        help="How fast old playtime fades: a game's hours count half after this long since you "
             "last played it (never less than a quarter). Hours in the last two weeks get a boost.")
    locked = stage != "input"
    with st.expander("Advanced settings"):
        st.caption("Change these before entering a profile. Defaults come from blind testing.")
        for name in ADVANCED:      # defaults live in session state so "Reset" can restore them
            st.session_state.setdefault(f"adv_{name}", getattr(DEFAULTS, name))
        advanced = {name: st.slider(label, lo, hi, step=step, key=f"adv_{name}", help=tip,
                                    disabled=locked)
                    for name, (label, lo, hi, step, tip) in ADVANCED.items()}
        st.button("Reset to defaults", on_click=reset_advanced, disabled=locked)
    st.subheader("Test mode")
    st.caption("Hide some of your games, then see if the recommendations find them anyway. "
               "Tick either or both.")
    hide_top = st.checkbox("Hide my most-played games", disabled=locked)
    top_n = st.number_input("How many (by total hours)", 1, 30, 10, disabled=locked or not hide_top)
    hide_recent = st.checkbox("Hide my recently played games", disabled=locked)
    recent_n = st.number_input("How many (most recent first)", 1, 30, 10,
                               disabled=locked or not hide_recent)
    st.caption("Your library is read from Steam for this visit only. Nothing is stored.")
    if stage != "input":
        st.button("Start over", on_click=reset, width="stretch")

if stage == "input":
    st.title("🎮 Steam Game Finder")
    st.write("Paste your Steam profile and I'll look at what you play, show you five games that "
             "capture your taste, then build a list of 15 from your reactions.")
    with st.form("profile"):
        profile = st.text_input("Steam profile",
                                placeholder="https://steamcommunity.com/id/yourname  or  76561198...")
        go = st.form_submit_button("Show my first look", type="primary")
    if go and profile.strip():
        with st.spinner("Reading your library..."):
            half_life = {"Off": 0, "3 months": 91, "6 months": 182, "1 year": 365, "2 years": 730}[recency]
            cfg = replace(RecConfig(), **advanced, allow_adult=not hide_adult,
                          recency_half_life_days=half_life,
                          recent_2weeks_boost=0 if recency == "Off" else RecConfig.recent_2weeks_boost)
            start(profile, cfg, hide_top=int(top_n) if hide_top else 0,
                  hide_recent=int(recent_n) if hide_recent else 0)
        st.rerun()
    if state.get("error"):
        st.error(state.error)
    st.caption("Your profile's **Game details** privacy setting must be Public.")

elif stage == "deck":
    deck, i = state.deck, state.card
    card = deck.iloc[i]
    g = details(int(card.appid))
    if state.hidden:
        what = " and ".join(f"{len(ids)} {name}" for name, ids in state.hidden_groups.items())
        st.info(f"Test mode: your {what} games are hidden ({len(state.hidden)} in total). "
                "React honestly; the end screen shows whether they came back.")
    st.subheader(f"First look · {i + 1} of {len(deck)}")
    st.progress(i / len(deck))
    chips = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in g.tag_list.split(",")[:6])
    st.html(f"""
      <div class="card" id="card-{i}">
        <img src="{html.escape(str(g.header_image))}" alt="">
        <div class="body">
          <h2><a href="{STORE.format(int(card.appid))}" target="_blank">{html.escape(g['name'])}</a></h2>
          <div class="meta">{html.escape(meta_line(g, state.store))}</div>
          {f'<p class="blurb">{html.escape(blurb)}</p>' if (blurb := state.store.get(int(card.appid), ("",))[0]) else ''}
          {chips}
          <div><a class="store" href="{STORE.format(int(card.appid))}" target="_blank">View on Steam ↗</a></div>
        </div>
      </div>""")
    up, down = st.columns(2)
    up.button("▲ Interested", type="primary", width="stretch", shortcut="Up",
              on_click=react, args=(i, True), key=f"up-{i}")
    down.button("▼ Not interested", width="stretch", shortcut="Down",
                on_click=react, args=(i, False), key=f"down-{i}")
    st.caption("Use the ↑ and ↓ arrow keys to swipe.")

elif stage == "results":
    st.title("Your 15")
    liked = sum(state.reactions.values())
    st.caption(f"Built from your library and your reactions ({liked} interested, "
               f"{len(state.reactions) - liked} not). Click a game to open it on Steam.")
    f = features()
    hidden_families = set(f.games.family_id.iloc[f.rows(state.hidden)]) if state.hidden else set()
    if state.hidden:
        def found(df, families):
            return set(df.family_id[df.family_id.isin(families)])
        lines = []
        for group, ids in state.hidden_groups.items():
            families = set(f.games.family_id.iloc[f.rows(ids)])
            before = len(found(state.pool.head(state.cfg.n), families))
            after = len(found(state.final, families))
            lines.append(f"- **{group.capitalize()}** ({len(ids)} hidden): **{after}** made the "
                         f"final 15 (the top 15 before your swipes had {before})")
        st.success("**Test mode results:** hidden games that came back, or another game in their "
                   "series:\n" + "\n".join(lines))
        with st.expander("Which games were hidden?"):
            for group, ids in state.hidden_groups.items():
                names = [f.games.name.iat[r] for r in f.rows(ids)]
                note = (f" ({len(ids) - len(names)} not in the catalog, so they can't be found)"
                        if len(names) < len(ids) else "")
                st.markdown(f"**{group.capitalize()}:** {', '.join(names)}{note}")
    rows = []
    swiped_up = {n for a, n in zip(state.deck.appid, state.deck.name) if state.reactions.get(int(a))}
    prices = store_info(state.final.appid.tolist())     # cached per game, so reruns are instant
    for _, x in state.final.iterrows():
        g = details(int(x.appid))
        rows.append(f"""
          <div class="row">
            <div class="rank">{x['rank']}</div>
            <img src="{html.escape(str(g.header_image))}" alt="">
            <div>
              <a href="{STORE.format(int(x.appid))}" target="_blank">{html.escape(g['name'])}</a>
              <div class="meta">{html.escape(meta_line(g, prices))}</div>
              {why_html(x, state.reactions, swiped_up)}
              {'<div class="meta">🎯 <b>One of your hidden favorites (or its series)</b></div>'
               if x.family_id in hidden_families else ''}
            </div>
          </div>""")
    st.html("".join(rows))
