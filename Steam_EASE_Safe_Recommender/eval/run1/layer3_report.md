# Layer 3: value to the player (test set, 5+ visible games)

Computed from the saved test lists (`eval/run_test.py`); the test set was not scored again.

| model                     |   players with a played pick |   Regret@10 |   tag distance |   serendipity vs MostPopular |   median hours per $ (paid hits) |   free picks played > 2 h |   free picks played > 10 h | regret avoided vs MostPopular (pp)   |
|:--------------------------|-----------------------------:|------------:|---------------:|-----------------------------:|---------------------------------:|--------------------------:|---------------------------:|:-------------------------------------|
| MostPopular               |                         2757 |      0.0786 |         0.6159 |                     nan      |                           7.5544 |                    0.8114 |                     0.6302 | nan                                  |
| item-kNN                  |                         5626 |      0.1547 |         0.5759 |                       0.0577 |                           2.5457 |                    0.6804 |                     0.3987 | -7.9 (-9.4 to -6.4), n=1994          |
| EASE                      |                         6932 |      0.1318 |         0.5548 |                       0.0867 |                           2.8    |                    0.7195 |                     0.461  | -5.3 (-6.4 to -4.4), n=2474          |
| ALS                       |                         5861 |      0.139  |         0.5694 |                       0.0611 |                           2.3547 |                    0.6907 |                     0.4261 | -8.2 (-9.7 to -6.6), n=1977          |
| Hybrid                    |                         5869 |      0.1419 |         0.5737 |                       0.0614 |                           2.343  |                    0.6859 |                     0.4285 | -8.6 (-10.2 to -7.0), n=1981         |
| Hybrid-EASE               |                         6935 |      0.132  |         0.556  |                       0.0869 |                           2.7966 |                    0.7206 |                     0.4619 | -5.4 (-6.4 to -4.4), n=2473          |
| Hybrid: Try something new |                         5645 |      0.1873 |         0.6891 |                       0.0582 |                           1.8897 |                    0.6836 |                     0.428  | -11.8 (-13.3 to -10.3), n=2079       |

## Try something new vs For you (Hybrid-EASE)

- NDCG@10: For you 0.2078, Try something new 0.1374 (66% of For you). Pre-declared margin: within 20% -> **NOT met**.
- Tag distance: +0.133 for Try something new (95% CI +0.131 to +0.135) -> **raises exploration**.

## Notes

- Regret uses only picks the player actually went on to play (their hidden games), so it measures quitting early, not ignoring a pick.
- Value per dollar is a heuristic: it uses regular price, not what the player paid, and lifetime hours.
- Unplayed owned games can't appear in lists (owned games are excluded), so the 'owned but never launched' case of the framework does not arise offline; the user study covers it.
