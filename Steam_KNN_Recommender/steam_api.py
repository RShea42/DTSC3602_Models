"""Fetch a player's full Steam library (every owned game with playtime) from the Steam Web API.

The key comes from STEAM_API_KEY2 in the environment or .streamlit/secrets.toml (this project's,
then ~/Steamapp's). Set STEAM_API_KEY_NAME to read a differently named entry.
Get one at https://steamcommunity.com/dev/apikey. Never commit it.

The profile's "Game details" privacy setting must be Public, or Steam returns an empty library.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
API = "https://api.steampowered.com"


def api_key() -> str:
    # Which entry holds the key. The project uses STEAM_API_KEY2; set STEAM_API_KEY_NAME to use another.
    name = os.environ.get("STEAM_API_KEY_NAME", "STEAM_API_KEY2")
    if os.environ.get(name):
        return os.environ[name]
    # This project's secrets, then the Steamapp project's (where upload_games_cleaned.py also looks).
    for secrets in (HERE / ".streamlit" / "secrets.toml", Path.home() / "Steamapp" / ".streamlit" / "secrets.toml"):
        if secrets.exists():
            key = tomllib.loads(secrets.read_text(encoding="utf-8-sig")).get(name)
            if key:
                return key
    sys.exit(f"Set {name} in the environment or a .streamlit/secrets.toml")


def _get(path: str, **params) -> dict:
    url = f"{API}/{path}/?{urllib.parse.urlencode({'key': api_key(), 'format': 'json', **params})}"
    req = urllib.request.Request(url, headers={"User-Agent": "steam-recommender-class-project"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def resolve_steamid(user: str) -> str:
    """Accepts a SteamID64, a profile URL (/profiles/<id> or /id/<name>), or a custom URL name."""
    user = user.strip().rstrip("/")
    m = re.search(r"/profiles/(\d{17})$", user) or re.fullmatch(r"(\d{17})", user)
    if m:
        return m.group(1)
    vanity = re.sub(r"^.*/id/", "", user)
    data = _get("ISteamUser/ResolveVanityURL/v1", vanityurl=vanity)["response"]
    if data.get("success") != 1:
        raise ValueError(f"Couldn't find a Steam profile for {user!r}")
    return data["steamid"]


def owned_games(user: str) -> dict[int, dict]:
    """Every game the user owns, including unplayed ones:
    {appid: {"minutes": total, "minutes_2weeks": last two weeks, "last_played": unix time or 0}}.
    Steam keeps no longer playtime history than the two-week figure."""
    steamid = resolve_steamid(user)
    data = _get("IPlayerService/GetOwnedGames/v1", steamid=steamid,
                include_played_free_games=1, include_appinfo=0)["response"]
    games = data.get("games")
    if not games:
        raise ValueError("Steam returned no games. Is the profile's 'Game details' setting Public?")
    return {int(g["appid"]): {"minutes": int(g.get("playtime_forever", 0)),
                              "minutes_2weeks": int(g.get("playtime_2weeks", 0)),
                              "last_played": int(g.get("rtime_last_played", 0))}
            for g in games}
