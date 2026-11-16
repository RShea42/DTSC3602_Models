# Steam Game Recommender: EASE-safe (candidate model, Juan)

Give it a Steam profile; it reads the player's library and recommends games that **players with the same taste
went on to enjoy**, while steering away from games the player is likely to quit. Two rows: **For you** and
**Try something new** (games outside the player's usual tags). Every pick says which of the player's games pushed it up.

**Model: EASE-safe** = EASE + a flag-aware re-rank.
1. **EASE** (Embarrassingly Shallow Autoencoder, Steck 2019) is a collaborative-filtering model. It learns one
   item-to-item weight matrix, in closed form, from which games players played together. A player's scores are the
   sum of the rows for the games they played. One setting: regularization λ = 100.
2. **Safe re-rank:** EASE's top 50 are re-scored as
   `z(EASE score) − 0.5 × like_quit − 1 × short_game − 10 × negative_reviews`, using the team's three soft-fail flags:
   - `like_quit`: the game's tags are closer to games the player quit (under 2 h) than to games they kept playing
   - `short_game`: the game's median playtime is 2 h or less
   - `negative_reviews`: Steam rating of Mostly Negative or worse

   `z(EASE score)` is the score standardized within the player's top 50. Flagged games move down but are not removed. The weights were picked on the 75/25 holdout by a rule fixed
   in advance: the fewest flags while keeping at least 98% of EASE's accuracy (`team_eval/rerank_grid.md`).

## Head-to-head on the shared team data

The **Baseline (TasteForge)** is compared against two candidates, **EASE-safe** and Rob's **KNN**, on the same 1,000
randomly chosen validation players from the shared package in Supabase (bucket `team-eval`). None of them are in the
training data. Every model ran from its own team's code (KNN from `Steam_KNN_Recommender`, the Baseline from
zcrawfo6/Steamapp branch `peyton-dev`) and every list was scored by the shared `team_eval/evaluate.py`.
**MostPopular** (the games most often enjoyed by everyone) is a popularity reference, not a candidate.

Each player's library is split: the model sees the visible part (80% here) and must find the hidden part.
**NDCG@10** measures how many hidden games the player enjoyed appear in the top 10, with more credit near the top
(0 = none found, 1 = perfect).

**80/20 holdout** (75/25 and 50/50 show the same order; full tables in `team_eval/results_all_models.md`):

| Team test | **EASE-safe** | KNN | MostPopular | Baseline (TasteForge) |
|---|---:|---:|---:|---:|
| **1. Hidden games found:** NDCG@10 | **0.277** | 0.072 | 0.035 | 0.002 |
| **1. Hidden games found:** players with at least one hidden game in the top 10 | **76%** | 42% | 33% | 1% |
| **2. Playtime match:** picks within 2x of the player's average playtime | 34% | **35%** | 20% | 15% |
| **3. Soft fails:** picks with any flag | 21% | 18% | 17% | **16%** |
| **3. Soft fails:** picks with 2 or more flags | 1.3% | 0.8% | **0%** | 0.7% |
| **3. Regret:** of picks the player went on to play, share they quit within 2 h | 16% | **9%** | 11% | — |

- **Test 1:** EASE-safe scores about 4x KNN on NDCG@10 (0.277 vs 0.072) and finds at least one hidden game for 76%
  of players vs 42%. KNN's NDCG@10 is 0.21 below EASE-safe's (95% bootstrap CI 0.19 to 0.23). This holds on all three
  holdouts.
- **Test 2:** about a tie between EASE-safe and KNN; both are far better than MostPopular and the Baseline.
- **Test 3:** KNN flags slightly fewer picks (18% vs 21%) and has less regret. EASE-safe adds the same idea through
  its like-quit flag and closes most of the gap, while still finding about 4x more games.
- **Baseline (TasteForge):** in its newest version it scores the full catalog, averages tag matches and ignores
  review counts, so its lists fill with obscure games that have about a dozen reviews. Its regret is left blank
  because it found almost no hidden games.
- **Edge cases:** KNN returned no list for 2 of 3,000 runs, both tiny libraries at 50/50 that the scoring skips, so
  they don't affect any number above. KNN also recommended a few games outside the shared catalog (0.5% of picks).

## On Rob's blind test (`litmus.py`)

Rob's own test on the same 300 libraries for every model (validation players, 20+ played games each). Each player's
10 most-played games are hidden. KNN ran through Rob's code with its default settings and recency off (these
libraries have no last-played dates). Details: `team_eval/litmus_all_models.md`.

| Hidden games found (of 10), average per player | **EASE-safe** | KNN | Popularity only | Baseline (TasteForge) |
|---|---:|---:|---:|---:|
| in the top 15 | **4.93** | 2.13 | 2.01 | 0.01 |
| in the top 50 | **6.71** | 3.53 | 3.23 | 0.08 |
| in the top 100 | **7.54** | 4.45 | 4.27 | 0.19 |

On its own test KNN barely beats the popularity-only ranking (2.13 vs 2.01 in the top 15); EASE-safe more than doubles
it. KNN's hits count any edition in a game's family, which slightly favors it; most libraries are from 2017.

## Earlier: our own final test (EASE family)

Before the shared package existed, every model family was scored on a separate held-out test set of 11,647 players
(`eval/eval_report.md`, protocol frozen first in `eval/PROTOCOL.md`). EASE beat item-kNN by 49% and ALS by 42% on
NDCG@10 (0.207 vs 0.140 and 0.146), and beat MostPopular for every data source and library size.
Hybrid-EASE (EASE + game descriptions) tied EASE (+0.0008). That test set is now retired, so EASE-safe was
tuned and compared on validation players only.

## Keeping it accurate over time

- **New games:** EASE can only recommend games that 100+ players in its training data have played (12,934 games).
  The training libraries are mostly from 2017–2023, so new releases are missing until people play them.
  Hybrid-EASE (`eval/models.py`, 75% EASE + 25% match on store descriptions) does **not** fix this as tuned: only
  0.14% of its picks on the shared holdouts (769 of 566,000) were games EASE can't score, because the description
  match alone can't outscore EASE's top picks. The planned fix is to **reserve 1–2 spots per list** for the
  best-matching newer games, ranked by the description match. That needs its own test before it ships.
- **Retraining:** EASE retrains in minutes (`model/build_ease.py`). Adding fresh Steam libraries keeps it current.
- **Fresh catalog:** the soft-fail flags read review ratings and median playtimes. Refresh them (SteamSpy refill)
  so the flags don't go stale.
- **One definition everywhere:** the app imports its flags from `team_eval/evaluate.py` and its weights from
  `team_eval/rerank_choice.json`, the same files used for scoring. Checked: the app reproduces the scored
  EASE-safe top-10 lists exactly for 300 of 300 sampled players.

## What's in this folder

| Part | Files |
|---|---|
| Model | `model/build_ease.py` (trains EASE), `model/build_model.py` (item stats), `model/recommender.py` (EASE scores, safe re-rank, both rows, "why" text, Steam library fetch) |
| Execution | `model/recommend.py` (command line), `app/server.py` + `app/index.html` (local web page with Steam sign-in) |
| Team comparison | `team_eval/evaluate.py` (shared scoring), `run_ours.py` (our top-10 lists), `rerank.py` + `rerank_choice.json` + `rerank_grid.md` (EASE-safe), `rob_on_team_eval.py` + `results_vs_rob.md` (head-to-head), `litmus_compare.*` (Rob's blind test), `results.md` (all validation players) |
| Our evaluation | `eval/PROTOCOL.md`, `eval/harness.py`, `eval/models.py` (all six model families), `eval/data.py`, `eval/tune.py`, `eval/run_test.py`, `eval/layer3.py`, reports `eval/*.md`, `eval/run1/` |

Not included: the trained weights (`ease.npz` is 327 MB, over GitHub's 100 MB limit; rebuild below),
the datasets, and secrets.

## Running it

Python 3.12; `pip install -r requirements.txt`.

**Just running the app (no retraining):** the trained weights and the two catalogs are too big for GitHub, so they
are in Supabase Storage: bucket `team-eval`, folder `ease_safe/` (plus `games.parquet` at the bucket root).
The model is split into 8 parts (50 MB file limit); `ease_safe/README.txt` has the command to rejoin them, a
checksum, and where each file goes. Then point the app at the folders:

```
$env:STEAMREC_ARTIFACTS = "<unzip folder>/model/artifacts"
$env:STEAMREC_DATA      = "<unzip folder>/data_clean"
python model/recommend.py <steam id or profile link>
```

The app also needs a `.env` file in this folder with `STEAM_API_KEY=<your own key>`
(https://steamcommunity.com/dev/apikey).

**Re-running the D3 comparison** (all four models on the shared validation players). Needs the shared package
from Supabase Storage (bucket `team-eval`) in one folder, a clone of this repo (for Rob's KNN in
`Steam_KNN_Recommender/`) and a clone of [zcrawfo6/Steamapp](https://github.com/zcrawfo6/Steamapp) on branch
`peyton-dev` (for TasteForge). Folder paths are constants at the top of each script; set them to yours.

```
python team_eval/evaluate.py --data <team_eval folder> <team_eval folder>/recs_ease_safe.parquet <team_eval folder>/recs_mostpopular.parquet
python team_eval/rob_on_team_eval.py 1000 2       # KNN on 1,000 sampled players (~30 min; saves progress, rerun resumes)
python team_eval/tasteforge_on_team_eval.py       # TasteForge + the all-model table -> results_all_models.md
python team_eval/litmus_compare.py 300            # Rob's blind test: KNN and EASE on 300 libraries
python team_eval/litmus_more.py                   # same 300 libraries: EASE-safe and TasteForge -> litmus_all_models.md
python team_eval/error_analysis.py                # NDCG@10 by library size, source, popular vs niche -> error_analysis.md
```

The saved outputs (`results_all_models.md`, `litmus_all_models.md`, `error_analysis.md`) are committed, so the
numbers in the D3 note can be checked without re-running.

**Rebuilding from scratch:** data paths for the evaluation are set near the top of `eval/data.py`
(currently `C:/Users/juant/...`); change them to yours.

```
# first: the cleaned tables and the train/validation/test player split. These come from the main project's
# cleaning scripts (clean/build_clean.py, clean/reserve_test.py), which are not in this folder.
python eval/data.py               # evaluation data -> eval/cache/
python model/build_model.py       # item stats -> model/artifacts/items.parquet
python model/build_ease.py        # train EASE (~5 GB RAM) -> model/artifacts/ease.npz
python model/recommend.py <steam id or profile link>   # recommendations in the terminal
python app/server.py              # web page at http://127.0.0.1:8765
python team_eval/run_ours.py      # EASE top-10 lists on the shared holdouts
python team_eval/rerank.py        # EASE-safe lists from EASE's top 50
python team_eval/evaluate.py --data <team_eval folder> <team_eval folder>/recs_ease_safe.parquet recs_yours.parquet
```
