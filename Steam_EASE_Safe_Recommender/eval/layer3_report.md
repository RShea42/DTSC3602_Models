# Layer 3: value to the player (test set, 5+ visible games)

> **Run 2** (PROTOCOL section 8, change 4): completed tags and full-catalog content embeddings; the test set's second use, disclosed. Run 1, the pre-registered result: `eval/run1/`.

Computed from the saved test lists (`eval/run_test.py`); the test set was not scored again.

| model                     |   players with a played pick |   Regret@10 |   tag distance |   serendipity vs MostPopular |   median hours per $ (paid hits) |   free picks played > 2 h |   free picks played > 10 h | regret avoided vs MostPopular (pp)   |
|:--------------------------|-----------------------------:|------------:|---------------:|-----------------------------:|---------------------------------:|--------------------------:|---------------------------:|:-------------------------------------|
| MostPopular               |                         2757 |      0.0786 |         0.6216 |                     nan      |                           7.5544 |                    0.8114 |                     0.6302 | nan                                  |
| item-kNN                  |                         5626 |      0.1547 |         0.5916 |                       0.0577 |                           2.4446 |                    0.6803 |                     0.3984 | -7.9 (-9.4 to -6.4), n=1994          |
| EASE                      |                         6932 |      0.1318 |         0.5592 |                       0.0867 |                           2.76   |                    0.7195 |                     0.461  | -5.3 (-6.4 to -4.4), n=2474          |
| ALS                       |                         5861 |      0.139  |         0.5726 |                       0.0611 |                           2.3209 |                    0.6907 |                     0.4261 | -8.2 (-9.7 to -6.6), n=1977          |
| Hybrid                    |                         5882 |      0.1437 |         0.5734 |                       0.0611 |                           2.2688 |                    0.6813 |                     0.4183 | -8.9 (-10.5 to -7.3), n=1982         |
| Hybrid-EASE               |                         6924 |      0.1319 |         0.5592 |                       0.0871 |                           2.7394 |                    0.7186 |                     0.4576 | -5.5 (-6.5 to -4.5), n=2463          |
| Hybrid: Try something new |                         5686 |      0.1799 |         0.6896 |                       0.0596 |                           2.0354 |                    0.6878 |                     0.4314 | -10.8 (-12.2 to -9.3), n=2095        |

## Try something new vs For you (Hybrid-EASE)

- NDCG@10: For you 0.2083, Try something new 0.1405 (67% of For you). Pre-declared margin: within 20% -> **NOT met**.
- Tag distance: +0.130 for Try something new (95% CI +0.129 to +0.132) -> **raises exploration**.

## Notes

- Regret uses only picks the player actually went on to play (their hidden games), so it measures quitting early, not ignoring a pick.
- Value per dollar is a heuristic: it uses regular price, not what the player paid, and lifetime hours.
- Unplayed owned games can't appear in lists (owned games are excluded), so the 'owned but never launched' case of the framework does not arise offline; the user study covers it.
