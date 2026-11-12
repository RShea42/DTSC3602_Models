# Evaluation protocol (frozen 2026-11-02, before any test result)

Every rule below was fixed before the test set was opened. Tuning happens on validation players only. The test set is
scored **once**, at the end, for every model. Any later change is logged at the bottom with the reason.

## 1. Data split (Layers 2 and 3)
- **Unit:** a player within a source: McAuley 2017 full libraries, Steam 2026 full libraries (our collection), Kaggle
  reviewers (partial libraries, secondary slice).
- **Rule:** `sha256("steamrec-split-v1|<source>|<player>") mod 10`: 0 = test, 1 = validation, 2-9 = train.
- Players already examined (the 2,400 from the first offline check and the label-pilot players) are never test; if
  the rule puts them in test they move to validation.
- **Hidden games:** for test and validation players, each played game is hidden when
  `sha256("steamrec-split-v1|<source>|<player>|<app_id>") mod 5 == 0` (about 20%). The rest is visible history.
- **Test players are removed from every model's training data.** Statistics such as popularity, playtime percentiles
  and similarities are computed on train players only.
- Files: `data_clean/splits/` (players.parquet, hidden_games.parquet, manifest.json with SHA-256 of each file).

## 2. What counts as a hit
- Enjoyment `g` = the player's playtime percentile among that game's **train** players in the same source
  (label_research.md round 6).
- **Hit:** a hidden game with g >= 0.5. **Graded gain:** 2 if g >= 0.8, 1 if 0.5 <= g < 0.8, 0 otherwise.
- Owned-but-never-played games are neither hits nor misses: they are excluded from targets (but still owned, so never
  recommended).

## 3. Candidate pool
- Every recommendable game (game, on sale, released, not adult; about 111k), minus every game the player owns.
- Full-catalog ranking. **No sampled negatives.** Games a model cannot score get its lowest score (ties broken by
  app id, the same way for every model).

## 4. Players included
- Main results: test players with >= 5 visible played games and >= 1 hidden enjoyed game.
- Players with 1-4 visible played games are reported as their own row (leave-the-visible-in, same scoring).
- Players with no hidden enjoyed game are excluded (counts reported before and after every filter).

## 5. Models (Layer 2)
MostPopular (count of enjoyed train interactions), item-kNN (cosine), EASE (closed form, games with >= 50 train
players), ALS (implicit, confidence 1 + alpha * g), Hybrid (ALS + content similarity from game descriptions).
Every model gets the same tuning budget (30 settings or fewer, grid or random) chosen by validation NDCG@10 only.
New players are always folded in from their visible history (the same path the live app uses).

## 6. Metrics and statistics (Layer 2)
- **Headline:** NDCG@10 with binary gain, macro-averaged over players.
- Secondary: graded NDCG@10, Recall@20 (divided by min(20, hits)), long-tail Recall@20 (tail = games outside the head
  that covers 80% of train interactions), Coverage@10, average recommendation popularity.
- Per-player NDCG@10 vectors; each model vs the hybrid: paired t-test and randomization test, 95% bootstrap CI of the
  mean difference (players resampled), Holm correction across the comparisons.
- Results by source (2017, 2026, Kaggle) and by library size (1-4, 5-20, 21-100, 100+ visible games). The chosen model
  must beat MostPopular in every bucket.

## 7. Layer 3
- **Regret@10:** among top-10 picks that appear in the player's hidden games and were played, the share played under
  120 minutes. Owned-never-played picks reported separately. Regret avoided = MostPopular regret - ours, with bootstrap
  CI.
- **Exploration:** mean tag distance of picks from the player's tag profile; serendipity vs MostPopular (hits in our
  list that are not in the popularity top 10, divided by the size of that set); both rows ("For you" and "Try
  something new"). "Try something new" must raise tag distance while keeping NDCG@10 within 20% of "For you".
- **Value per dollar (heuristic):** median enjoyed hours per dollar of regular price over hits; free-to-play reported
  separately (share still played past 2 h and 10 h). Requires the corrected regular-price column.

## 8. Changes after freezing
1. **2026-11-02, before any test result: Kaggle players are subsampled for scoring.** Kaggle has 2.7 M validation and
   test players; scoring all of them is impractical on this machine. Rule: keep a Kaggle player when
   `sha256("steamrec-kaggle-sub-v1|<player>") mod 100` < 9 (test) or < 3 (validation). McAuley and Steam 2026 players
   are scored in full. Kaggle remains a secondary slice.
2. **2026-11-02, before any test result: targets are recommendable hits only.** A hidden enjoyed game that can't be
   recommended (delisted, adult, not a game) can't be produced by any model, so counting it would only lower every
   model's score equally. Players with no recommendable hidden hit are excluded. Tuning uses up to 8 settings per model
   (within the 30-setting cap in section 5), the same for every model.
3. **2026-11-02, before any test result: a second hybrid, built on EASE.** On validation, EASE (NDCG@10 0.276) is well
   ahead of ALS (best 0.188 after 5 of 8 settings), so a hybrid built only on ALS may not be the team's best model.
   Rule: "Hybrid-EASE" = w * z(EASE) + (1 - w) * z(content), tuned on the same 7 values of w as the ALS hybrid, on
   validation players only. Whichever hybrid has the higher validation NDCG@10 is "ours": it is compared against every
   other model in the statistics (Holm over 5 comparisons), and the "Try something new" row and Layer 3 are built from
   it. Both hybrids are reported on the test set. The test set is still scored once.
4. **2026-11-03, after the first test run: a second, disclosed run with the completed catalog.** The first run
   (2026-11-03, archived unchanged in `eval/run1/`) used a content side that covered only 38,248 of 111,167
   recommendable games (Kaggle text, no 2024-2026 releases) and a catalog where 66% of recommendable games had no tags.
   Run 2 changes only the content inputs: (a) SteamSpy tags fill the empty tags (cleaning rule G22) and regular prices
   replace Oct 2026 crawl prices (G23); (b) every recommendable game is embedded from the cleaned catalog (title,
   genres, tags, description; same gte-small model; `model/embed_catalog.py`). Splits, label, metrics, statistics,
   the subsampling rule, tuning grids and the "Try something new" margin (20%) are unchanged. Models that do not use
   content (MostPopular, item-kNN, EASE, ALS) see identical inputs, so their validation tuning is reused, not repeated;
   both hybrids are retuned on validation only. The test set is then scored a second time. Both runs are reported;
   run 1 stays the pre-registered result and run 2 is labelled as a follow-up that used the test set twice.
