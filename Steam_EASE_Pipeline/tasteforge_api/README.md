# tasteforge_api: the live TasteForge API, now serving the scikit-learn pipeline

`modal_api.py` is the Modal app **`steam-ease-api`** (Peyton's, account `sirjacket42`) behind
[taste-forge.com](https://taste-forge.com). Since 2026-11-23 (Modal version **v5**) it serves this folder's parent,
the fitted scikit-learn pipeline (`../steamrec`, `../app/engine.py`), instead of Juan's `recommender.py` + `ease.npz`.
Every route and response format is unchanged; the site needed no change.

**Live URL:** `https://sirjacket42--steam-ease-api-api-web.us-east.modal.run` (private: every route except `/health`
needs `Authorization: Bearer <API_TOKEN>`; only the Vercel server has it, so `/docs` is off). The public, open API
for grading is the team's `https://rshea42--steam-ease-safe-api.us-east.modal.run` (`/docs`), see `../README.md`.

| File | What |
|---|---|
| `modal_api.py` | the deployed app: FastAPI routes (list at the top of the file) around `app/engine.py`; loads the saved pipeline from the Modal Volume, never retrains |
| `verify_deployed.py` | local vs deployed check (below) |
| `verify_out/` | its inputs, raw responses and report for the live app; `verify_report_test_app.md` = the same check on the test deploy, with the pre-swap live API as reference |
| `compare_report.md` | `../peyton_swap/compare.py` re-run with the pipeline rebuilt on Peyton's machine |

## Rebuild and deploy

From `Steam_EASE_Pipeline/`, with the pinned environment and data from `../README.md` ("Setup", "Data"):
```bash
python build_model.py                                             # ~2 min -> serve_files/ (pipeline .joblib + 3 catalog files)
pytest && pytest -m slow                                          # 29 tests
modal volume put steam-ease-artifacts serve_files/ /pipeline      # Git Bash: MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8
```
The Volume's `/data` folder holds the site's catalog extras (`games.parquet`, `team_eval/games.parquet`,
`games_slim.csv`), uploaded once. Secret: `modal secret create steam-ease-api STEAM_API_KEY=<key> API_TOKEN=<token>`.

Then from this folder:
```bash
MODAL_APP_NAME=steam-ease-api-test modal deploy modal_api.py      # test copy first
python verify_deployed.py --deployed <test url> --reference <live url>
modal deploy modal_api.py                                          # live
python verify_deployed.py --deployed https://sirjacket42--steam-ease-api-api-web.us-east.modal.run
```
`verify_deployed.py` needs the token (`API_TOKEN` env var or a gitignored `.api_token` file) and a `local_data/`
folder that mirrors the Volume's `/data`.

Undo: `modal app history steam-ease-api`, then `modal app rollback steam-ease-api v4` (v4 = Juan's recipe, before the
swap). After a deploy, a container from the old version can keep answering until it idles out (5 min);
`modal container list` / `modal container stop -y <id>` forces the switch.

## Results (2026-11-23)

| Check | Result |
|---|---|
| Tests (`pytest`, `pytest -m slow`) with the rebuilt pipeline | 20/20 + 9/9 pass |
| `compare.py`: this API with the pipeline vs with Juan's recipe (local, 371 requests, all routes) | Juan's weights: **371/371** identical. Rebuilt pipeline: **369/371**; the 2 differ only in one card's displayed score, 4th decimal (0.391 vs 0.3909). No-token request refused (401) by all. |
| **Local vs deployed, live app** (200 real libraries, `POST /recommend`) | **200/200 identical** responses (picks, order, scores, full JSON). `verify_out/verify_report.md` |
| Local vs test deploy, with pre-swap live as reference | test deploy 200/200 identical; pre-swap live 199/200 (one card score 0.3951 vs 0.395) |
