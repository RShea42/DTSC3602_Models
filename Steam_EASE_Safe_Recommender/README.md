# Steam Game Recommender: EASE-safe (candidate model, Juan)

Give it a Steam profile; it reads the player's library and recommends games that **players with the same taste
went on to enjoy**, while steering away from games the player is likely to quit. Two rows: **For you** and
**Try something new** (games outside the player's usual tags). Every pick says which of the player's games pushed it up.

**Model: EASE-safe** = EASE + a flag-aware re-rank.
1. **EASE** (Embarrassingly Shallow Autoencoder, Steck 2019) is a collaborative-filtering model. It learns one
   item-to-item weight matrix, in closed form, from which games players played together. A player's scores are the
   sum of the rows for the games they played. One setting: regularization λ = 100.
2. **Safe re-rank:** EASE's top 50 are re-scored as
   `z(EASE score) − 0.5 × like_quit − 1 × short_game − 10 × negative_reviews`. These are the team's three soft-fail
   flags: like the player's under-2-hour games, median playtime ≤ 2 h, Mostly Negative or worse.
   Flagged games move down but are not removed. The weights were picked on the 75/25 holdout by a rule fixed
   in advance: the fewest flags while keeping at least 98% of EASE's accuracy (`team_eval/rerank_grid.md`).

## Head-to-head on the shared team data

Same 1,000 randomly chosen validation players for every model, from the shared package in Supabase
(bucket `team-eval`). None of them are in the training data. Scored by the shared `team_eval/evaluate.py`.
Rob's KNN was run through his own code from this repo (`Steam_KNN_Recommender`).

**80/20 holdout** (75/25 and 50/50 show the same pattern; full tables in `team_eval/results_vs_rob.md`):

| Team test | **EASE-safe** | EASE | Rob's KNN | MostPopular |
|---|---:|---:|---:|---:|
| **1. Hidden games found:** NDCG@10 | **0.277** | 0.282 | 0.072 | 0.035 |
| **1. Hidden games found:** players with at least one hidden game in the top 10 | **76%** | 75% | 42% | 33% |
| **2. Playtime match:** picks within 2x of the player's average playtime | 34% | 33% | 35% | 20% |
| **3. Soft fails:** picks with any flag | 21% | 31% | 18% | 17% |
| **3. Soft fails:** picks with 2 or more flags | 1.3% | 4.5% | 0.8% | 0% |
| **3. Soft fails:** found picks the player quit under 2 h | 16% | 18% | 9% | 11% |

- **Test 1:** EASE-safe finds about 4x more hidden games than Rob's KNN (NDCG@10 −0.21 for KNN vs EASE,
  95% CI −0.23 to −0.19). This holds on all three holdouts.
- **Test 2:** about a tie with KNN. Both are far better than MostPopular.
- **Test 3:** KNN flags slightly fewer picks (18% vs 21%) and has less regret. This is its strength, since it
  avoids games like the ones a player barely touched. EASE-safe adds the same idea through the like-quit flag and
  closes most of the gap, while still finding about 4x more games.
- **Edge cases:** KNN returned no list for 2 of 3,000 runs. Both were tiny libraries at 50/50 that the scoring
  skips (fewer than 5 visible games), so they don't affect any number above. One was a library with no played game
  in its catalog. The other had no played games at all, which raises a pandas error in `recommend.py` instead of a
  clear message. KNN also recommended a few games outside the shared catalog (0.5% of picks), which can't count as hits.

All ~8,000 validation players, EASE-safe vs EASE (`team_eval/results.md`): NDCG@10 0.270 vs 0.276 at 80/20,
0.274 vs 0.280 at 75/25, 0.313 vs 0.316 at 50/50. Picks with any flag drop from 32% to 21%.

## On Rob's blind test (`litmus.py`)

Rob's own test, run with his code, on the same 300 libraries for his KNN and for EASE (validation players,
20+ played games each). Each player's 10 most-played games are hidden.

| Hidden games found (of 10), average per player | Rob KNN | EASE | Popularity (Rob's baseline) |
|---|---:|---:|---:|
| in the top 15 | 2.13 | **4.88** | 2.01 |
| in the top 50 | 3.53 | **6.71** | 3.23 |
| in the top 100 | 4.45 | **7.54** | 4.27 |

This test was run on plain EASE, before the safe re-rank. The re-rank only reorders EASE's top 50 and
changes NDCG@10 by about 2%. Caveats: recency is off because these libraries have no last-played dates;
Rob's default settings are used; his hits count any edition in a game's family, which slightly favors KNN; most
libraries are from 2017. Details: `team_eval/litmus_compare.md`.

## Earlier: our own final test (EASE family)

Before the shared package existed, every model family was scored on a separate held-out test set of 11,647 players
(`eval/eval_report.md`, protocol frozen first in `eval/PROTOCOL.md`). EASE beat item-kNN by 49% and ALS by 42% on
NDCG@10 (0.207 vs 0.140 and 0.146), and beat MostPopular for every data source and library size.
Hybrid-EASE (EASE + game descriptions) tied EASE (+0.0008). That test set is now retired, so EASE-safe was
tuned and compared on validation players only.

## Keeping it accurate over time

- **New games:** EASE can only recommend games that 100+ players in its training data have played (12,934 games).
  The training libraries are mostly from 2017–2023, so new releases are missing until people play them.
  The planned fix is **Hybrid-EASE-safe**: the description-based part of Hybrid-EASE (`eval/models.py`) scores
  new games from their store text. It ties EASE on our tests, which can't measure this benefit.
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

Python 3.12; `pip install -r requirements.txt`. Data paths are set near the top of `eval/data.py` and
`model/recommender.py` (currently `C:/Users/juant/...`); change them to yours.
The app also reads the shared package's `games.parquet` from `<data_clean>/team_eval/` (download it from Supabase).

```
# first: cleaned tables + train/validation/test split from the project's cleaning step
python eval/data.py               # evaluation data -> eval/cache/
python model/build_model.py       # item stats -> model/artifacts/items.parquet
python model/build_ease.py        # train EASE (~5 GB RAM) -> model/artifacts/ease.npz
python model/recommend.py <steam id or profile link>   # recommendations in the terminal
python app/server.py              # web page at http://127.0.0.1:8765
python team_eval/run_ours.py      # top-10 lists on the shared holdouts, then team_eval/rerank.py for EASE-safe
python team_eval/evaluate.py --data <team_eval folder> recs_ease_safe.parquet recs_yours.parquet
```
