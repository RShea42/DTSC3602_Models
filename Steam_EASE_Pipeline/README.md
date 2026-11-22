# Steam_EASE_Pipeline

Juan's **EASE-safe** recommender (`../Steam_EASE_Safe_Recommender`) rebuilt as a fitted **scikit-learn pipeline**,
served with **FastAPI on Modal**, with local and deployed outputs verified to match. Same model, same settings, same
recommendations as Juan's code (checked pick-for-pick); nothing in Juan's folder was changed.

**Live API:** https://rshea42--steam-ease-safe-api.us-east.modal.run  (interactive docs: `/docs`; open, no token)

| Course requirement | Where |
|---|---|
| Save a fitted scikit-learn pipeline with a meaningful custom transformer | `steamrec/pipeline.py`: `LibraryEncoder` and `EASEScorer` are custom transformers that learn from data in `fit` (the game vocabulary; EASE's 12,934 x 12,934 item weights). `build_model.py` fits and saves it with joblib. |
| Serve it with FastAPI on Modal | `app/api.py` (FastAPI), `modal_app.py` (Modal; loads the saved pipeline from a Volume, never retrains) |
| Verify matching local and deployed outputs | `m5_verify.py` -> `m5_out/verify_report.md`: **200/200** real libraries give identical picks, order and scores from the saved pipeline, the local API and Modal |

## How it relates to TasteForge

taste-forge.com (branch `Steamapp_Vercel` of zcrawfo6/Steamapp) already runs Juan's model through **Peyton's own
API** (`sirjacket42--steam-ease-api…`, source `Juan_EASE/modal_api.py`, not in a repo). This project does **not**
change that. It is a separate, auxiliary service. To make the live site run on this pipeline, see
[Swapping the pipeline into the live API](#swapping-the-pipeline-into-the-live-api).

## The API

All requests take the player's library; callers that already have it (TasteForge does, from its Steam sign-in) need no
Steam key. `playtime_min` is lifetime minutes; `0` = owned but never played (never recommended, not used as taste).

```bash
curl -X POST https://rshea42--steam-ease-safe-api.us-east.modal.run/predict \
  -H "Content-Type: application/json" \
  -d '{"library": [{"app_id": 1145360, "playtime_min": 3000}, {"app_id": 367520, "playtime_min": 1200}], "k": 5}'
```

| Route | In | Out |
|---|---|---|
| `POST /predict` | `library`, `k` (1-600, default 10), `include_explicit` | `picks: [{rank, app_id, title, score}]`: the pipeline's output only. Deterministic (no live data). |
| `POST /recommend` | `library`, `n` (per row), `max_price`, `country`, `include_explicit` | Juan's full app response: `for_you` and `try_new` cards (title, live price, rating, reviews, tags, image, store link, `because` = owned games that drove the pick), `top_tags`, `favorites` |
| `GET /health` | | model settings in use |

Timing: first request after ~1 min idle wakes a container (**~8-10 s**, cold start by choice: free when idle); then
~1 s for `/predict`, ~2 s for `/recommend` (live Steam store prices).

## How it works

```
library rows ─> LibraryEncoder ─> EASEScorer ──────────────> SafeReranker ─> ranked picks
                (fit: which games   (fit: EASE weights;        (Juan's re-rank: push down games
                 the model knows)    predict: score, filter,    like ones you quit, short games,
                                     drop owned, top N)         negative reviews)
```

- **EASE**: item-to-item weights learned from which games 11M players played together; a player's scores are the sum
  of the rows for games they played. `lam` (100) damps weights backed by few players.
- **Filters** (`ease__serve_filters`): `True` (default) = Juan's live app rules (hide games rated under 65% with 50+
  reviews, under 20 reviews, and sexually explicit unless opted in). `False` = Juan's evaluation setup (any
  recommendable game), used to reproduce his scores.
- `app/engine.py` is Juan's `model/recommender.py` with scoring from the pipeline; everything else (prices, "Try
  something new", explanations) is his code unchanged. `tests/test_app_parity.py` proves identical output.

## Folder map

| Path | What |
|---|---|
| `steamrec/` | the pipeline: `pipeline.py` (3 steps + `make_pipeline`), `data.py`, `flags.py` (Juan's flag rules) |
| `app/api.py` | **the deployed API** |
| `app/engine.py` | Juan's app logic on top of the pipeline |
| `app/main.py`, `sessions.py`, `index.html` | local demo page with Steam sign-in (Juan's page, repaired). **Not deployed.** |
| `build_model.py` | train + assemble `serve_files/` |
| `modal_app.py` | Modal deployment |
| `m0_*.py` … `m5_verify.py` | the checks behind every claim here; reports in `m*_out/` |
| `paths.py` | where data and Juan's folder are (override with env vars) |
| `split_for_bucket.py` | package the trained model for the Supabase bucket |
| `start_local.ps1` | start the demo page (:8000) and API (:8001) locally |

## Setup

Python 3.12. The versions are pinned on purpose: a saved scikit-learn pipeline only loads reliably with the versions
it was saved with.
```bash
python -m venv .venv && .venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt                      # requirements.txt alone = what Modal installs
```

### Data
From the Supabase Storage bucket **`team-eval`** into one folder, `ease_files/`, next to the repo (or set
`STEAMREC_EASE_FILES`):
```
ease_files/data_clean/games.parquet                <- ease_safe/catalog_games.parquet (renamed)
ease_files/data_clean/team_eval/games.parquet      <- games.parquet (bucket root)
ease_files/data_clean/team_eval/validation.parquet <- validation.parquet (checks only)
ease_files/data_clean/team_eval/train/train_full_part01..09.parquet   (training data, 36M rows)
ease_files/model/artifacts/items.parquet           <- ease_safe/items.parquet
ease_files/model/artifacts/ease.npz                <- ease_safe/ (8 parts; Juan's weights, checks only)
```
Shortcut: if the trained pipeline has been uploaded (`ease_safe/pipeline/`, see its README.txt), rejoin it into
`serve_files/` with the three catalog files and skip training.

## Common tasks

| Task | Command | Time |
|---|---|---|
| Train + build `serve_files/` | `python build_model.py` | ~1 min, **~11 GB RAM** |
| Fast tests | `pytest` | seconds |
| Real-model tests (run before every deploy) | `pytest -m slow` | ~6 min |
| Run locally | `powershell -File start_local.ps1`, or `uvicorn app.api:app --port 8001` | |
| Upload model to Modal (once, or after rebuilding) | `modal volume create steam-ease-safe-files` then `modal volume put steam-ease-safe-files serve_files/ /` | ~1 min |
| Deploy | `modal deploy modal_app.py` | ~30 s |
| Verify local = deployed | `python m5_verify.py --deployed <url>` (local API running on :8001) | ~5 min |
| Undo a deploy | `modal app history steam-ease-safe`, then `modal app rollback steam-ease-safe <version>` | seconds |

Always name the version when rolling back (without one, Modal goes to the previous version). **v1 crash-looped**
(fixed in v2); **v2 is the version verified 200/200** and live at handoff.

**Windows (Git Bash) gotchas:** set `MSYS_NO_PATHCONV=1` before `modal volume put` (otherwise `/` becomes
`C:/Program Files/Git/`), and `PYTHONIOENCODING=utf-8` (Modal's ✓ crashes the cp1252 console). Set `PYTHONUTF8=1`
for the scripts (game titles with ™/®).

**Another workspace:** the live URL is tied to the `rshea42` Modal account. To host it elsewhere, run the upload and
deploy commands logged in to that account; the URL changes to `https://<workspace>--steam-ease-safe-api…`.

## Results (all reproducible with the scripts above)

| Check | Result |
|---|---|
| M0: Juan's scores from the shared files | NDCG@10 **0.2704** (all 8,112 validation players, 80/20) and **0.2769** (his 1,000-player sample; README 0.277), to 4 decimals |
| M0: retrain from `train_full` | 49 s; 99.994% of weights bit-identical to Juan's `ease.npz`; EASE-safe lists 100% identical |
| M1: as a pipeline | 1,201,010 / 1,201,010 picks identical to Juan's evaluation lists |
| M1: app mode | 499/499 players get Juan's live-app top 10; app-mode NDCG@10 0.2663 (the app's filters hide some enjoyed games) |
| M2: tuning (report only) | `lam` flat from 100 to 300 (best +0.0005, within noise); on held-out players no setting beats Juan's. Tuned re-rank only swaps short-game flags for like-quit flags. **Juan's settings kept.** (`m2_out/report.md`) |
| M4: app engine | identical output to Juan's app on 150 libraries x (default, price limit, adult on) + favourites fallback |
| M5: local vs Modal | 200/200 identical (`m5_out/verify_report.md`) |

## Decisions

- **Juan's choices are the defaults**, even where we disagreed: `lam=100`, `min_users=100`, re-rank a/b/c =
  0.5/1/10, his live-app filters. Tuning results are reported, not applied.
- Exact score ties (common with one-game libraries) are ordered like the Juan code each mode copies: app mode like his
  app, evaluation mode like his evaluation.
- API **open** (no token): read-only, `/predict` makes no outside calls; easy for graders. To protect it:
  `@modal.asgi_app(requires_proxy_auth=True)` and callers send `Modal-Key` / `Modal-Secret` headers.
- **Cold start** (no always-on container). **No demo page deployed**; Steam sign-in stays local only.

## Swapping the pipeline into the live API

Plan for making taste-forge.com run on this pipeline. Needs Peyton's `Juan_EASE/modal_api.py` and his OK (his account).
1. Record what his API returns now for ~200 test libraries ("before").
2. In a **copy** of his API (new Modal app name), replace the inline EASE scoring with `app/engine.py`'s
   `Recommender.ordered()` (or the pipeline directly). Leave every other route alone.
3. Same 200 libraries: copy vs original must match exactly. His recommender is an **adapted** copy of Juan's (branch
   `Steamapp_Juan_Model/ease/recommender.py`: no Steam lookups, extra per-pick flags/contributions), so match his
   version, not the one tested here.
4. Point a local copy of the site (`MODAL_API_URL`) at the copy and click through every page.
5. Peyton redeploys his real API with the change. Undo: `modal app rollback <his app>`.

## Known limitations

- Catalog reviews and prices are stale after mid-2025 (the source Kaggle dataset never refreshes old rows).
- EASE only scores games with >= 100 training players: 12,934 games, almost none released after 2023.
- Training players are mostly from Kaggle reviews (10.9M of 11.0M); McAuley and 2026 libraries are ~56k.

## Fixes to Juan's page (`app/index.html`)
Juan's committed `app/index.html` (repo) is missing the profile-link / max-price form (his text, CSS and server expect
it) and has two lines spliced into the favourites search (`setTimeout$(…)`: it errors when typing). Both are repaired
here only; his folder is unchanged.
