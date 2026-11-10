# Steam Game Finder (DS 2302 group project)

Give it a Steam profile; it reads the player's library and recommends 15 games from a catalog of
~126k Steam games. Live: https://rshea42--steam-game-finder-finder.us-east.modal.run

## What's in this folder

| Part | Files |
|---|---|
| Model | `features.py` (catalog features), `recommend.py` (KNN recommender), `steam_api.py` (library fetch) |
| Execution | `recommender_app.py` (Streamlit app), `modal_recommender.py` (hosting), `requirements.txt`, `artifacts/` (prebuilt features, so the model runs without the 400 MB catalog CSV) |
| Evaluation | `litmus.py`, `tune.py` |
| Supporting | `Data Dictionary.docx`, `upload_games_cleaned.py` (loads the cleaned catalog into Supabase) |

Not included: the catalog CSVs (over GitHub's 100 MB limit; the catalog lives in Supabase),
secrets, cached player libraries, and the EDA notebook/dashboard (they live in the project repo).

## Project status

### Done
- **Catalog features** (`features.py`): TF-IDF tag vectors (vote-order weighted) + genres; game
  families (editions, sequels, DLC-style duplicate listings); adult/software/non-game filters;
  expansion detection (needs the CSV's "About the game" text).
- **Recommender** (`recommend.py`): content-based KNN. Liked = most-played games (recency
  weighted: last-played half-life + 2-week boost), disliked = briefly played games; score blends
  taste match with review quality. List shaping: one game per family, variety across "because"
  anchors, max 3 picks from a series the player owns, expansions replaced by their base game.
- **First look**: 5 diverse cards from a 40-game pool; swipe up/down (arrow keys) refines the final 15.
- **Web app** (`recommender_app.py`, hosted with `modal_recommender.py`): store blurbs on cards,
  live Steam prices, "Why we picked this" (up to 3 library games + shared tags), adult filter on
  by default, recency setting, advanced sliders, test mode (hide most-played / recent games).
- **Evaluation tools**: `litmus.py` (blind test: hide your top games, see if they come back) and
  `tune.py` (sweep settings). Defaults were chosen from these (e.g. quality weight 0.5).
- **EDA**: notebook + dashboard (`app.py`, hosted with `modal_app.py`).

### In progress
- **Model comparison**: the group is comparing this model against two others.

### Tuning notes (manual testing on the live app)
- **Blind test, 30 hidden, no recency.** Hiding the 30 most-played games (test mode) with the
  recency setting Off gave surprisingly solid results: 4 hidden games came back. Release date
  seemed not to matter; Left 4 Dead 2 (2009) was the top recommendation.
- **A settings combination that gave quality results** (Advanced settings; code defaults in brackets):

  | Slider | Setting | Value | Default |
  |---|---|---|---|
  | Favor well-reviewed, popular games | `quality_weight` | 0 | 0.5 |
  | Avoid games like the ones you barely played | `dislike_weight` | 1.5 (slider max) | 0.5 |
  | Games that define your taste | `liked_count` | 15 | 15 |
  | Max picks from series you own | `max_franchise_picks` | 3 | 3 |
  | Variety (picks per liked game, per round) | `max_per_anchor` | 2 | 2 |
  | Minimum Steam reviews | `min_reviews` | 10 (slider min) | 50 |

  This contrasts with `tune.py`'s blind-test sweep, which favored `quality_weight` 0.5. Worth
  checking with `tune.py`/`litmus.py` (e.g. `--hide 30`) before changing defaults.

### Not started / ideas for later
1. **Fresh catalog data**: the known limitation, see below.
2. **Group mode**: compare several libraries to find games everyone can play together (the
   catalog already flags online/local co-op, PvP and LAN).
3. **Save swipes to the database**: learn from what people actually like (kept out of the prototype).
4. **True 6-month playtime**: Steam only reports lifetime and last-2-weeks playtime; real history
   needs periodic library snapshots in the database.
5. **Learned ranker**: a per-user Random Forest / Logistic Regression on top of KNN (original plan).
6. **Limit franchise weight** (`collapse_franchises`): built, off by default; it cost blind-test
   hits on recently played games.
7. **LLM layer**: set aside for now.
8. **Price cache expiry**: live prices are cached until the server restarts.

### Known limitation: stale catalog data
The source dataset ([Kaggle: fronkongames/steam-games-dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset),
v58, 125,855 games) is built by [Steam-Games-Scraper](https://github.com/FronkonGames/Steam-Games-Scraper),
which only fetches app IDs not already in the dataset and permanently skips games that were
unreleased when first seen (`notreleased.json`). So review counts and prices are frozen at first
capture, and recent releases are under-represented: 76% of 2025 games and all 2026 games show 0
reviews (e.g. PEAK, Mewgenics), which hides them from recommendations. Re-downloading the dataset
gives an identical file.

Planned fix after the project:
- **Review refresh**: re-fetch review counts from Steam's reviews API (games since 2024 first).
- **New-game scrape**: run the scraper with `notreleased.json` deleted, or fetch missing app IDs
  directly. New games also need tags (from SteamSpy) to be recommendable.

## Running it

Python 3.12; `pip install -r requirements.txt`. Secrets live in `.streamlit/secrets.toml`
(never committed): `SUPABASE_DB_URL` and the Steam Web API key (`STEAM_API_KEY2`).

```
python features.py --csv games_cleaned.csv   # build artifacts/ (or no --csv to read Supabase)
streamlit run recommender_app.py             # the recommender, locally
python litmus.py --help                      # blind test
python tune.py --library libraries/<steamid>.json
modal deploy modal_recommender.py            # publish (rebuild artifacts first)
```

Catalog refresh: `python upload_games_cleaned.py --dry-run`, then without `--dry-run` to replace
the `games_cleaned` table in Supabase. Then rebuild `artifacts/` and redeploy.
