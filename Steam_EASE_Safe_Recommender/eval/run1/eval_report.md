# Layer 2: offline evaluation on the test set (scored once)

Protocol: `eval/PROTOCOL.md`. Settings chosen on validation only: `eval/tuning.md`. Ours = **Hybrid-EASE** (higher validation NDCG@10 of the two hybrids: {'Hybrid': 0.1883479823119256, 'Hybrid-EASE': 0.2765248466657698}). Test players: 30,977 with at least one recommendable hidden enjoyed game; main results use the 11,647 with 5+ visible games.

## Main results (5+ visible games)

| model                     |   players |   NDCG@10 |   graded NDCG@10 |   Recall@20 |   long-tail Recall@20 |   Coverage@10 |   avg popularity of picks |
|:--------------------------|----------:|----------:|-----------------:|------------:|----------------------:|--------------:|--------------------------:|
| MostPopular               |     11647 |    0.0325 |           0.03   |      0.1009 |                0      |        0.0003 |                  183698   |
| item-kNN                  |     11647 |    0.1396 |           0.1318 |      0.2534 |                0.0918 |        0.0479 |                   33345.2 |
| EASE                      |     11647 |    0.2074 |           0.1946 |      0.3408 |                0.0871 |        0.0317 |                   73843.2 |
| ALS                       |     11647 |    0.1464 |           0.1401 |      0.2593 |                0.0031 |        0.0118 |                   46274.2 |
| Hybrid                    |     11647 |    0.1476 |           0.1412 |      0.2606 |                0.004  |        0.0123 |                   45605.4 |
| Hybrid-EASE               |     11647 |    0.2078 |           0.195  |      0.3397 |                0.0865 |        0.032  |                   73573.2 |
| Hybrid: Try something new |     11647 |    0.1374 |           0.1273 |      0.216  |                0.0648 |        0.0341 |                   58248.6 |

## Statistics: Hybrid-EASE vs each model (NDCG@10, paired over players)

| Hybrid-EASE vs   |   mean difference |   relative | 95% CI             |   t-test p |   randomization p |   Holm-adjusted p |
|:-----------------|------------------:|-----------:|:-------------------|-----------:|------------------:|------------------:|
| MostPopular      |            0.1753 |     5.3939 | +0.1706 to +0.1799 |      0     |            0      |             0     |
| item-kNN         |            0.0682 |     0.4886 | +0.0646 to +0.0720 |      0     |            0      |             0     |
| ALS              |            0.0614 |     0.4193 | +0.0575 to +0.0651 |      0     |            0      |             0     |
| Hybrid           |            0.0601 |     0.4073 | +0.0562 to +0.0641 |      0     |            0      |             0     |
| EASE             |            0.0003 |     0.0015 | -0.0001 to +0.0008 |      0.168 |            0.1676 |             0.168 |

## NDCG@10 by source (5+ visible games)

| model                     |   kaggle |   mcauley |   steam2026 |
|:--------------------------|---------:|----------:|------------:|
| MostPopular               |   0.0229 |    0.0464 |      0.0651 |
| item-kNN                  |   0.0869 |    0.2211 |      0.1972 |
| EASE                      |   0.1236 |    0.3343 |      0.3694 |
| ALS                       |   0.0998 |    0.2183 |      0.2009 |
| Hybrid                    |   0.1012 |    0.2191 |      0.2093 |
| Hybrid-EASE               |   0.1238 |    0.3346 |      0.3742 |
| Hybrid: Try something new |   0.0741 |    0.2317 |      0.2952 |

## NDCG@10 by visible library size (all test players)

| model                     |    1-4 |   5-20 |   21-100 |   100+ |
|:--------------------------|-------:|-------:|---------:|-------:|
| MostPopular               | 0.0313 | 0.029  |   0.037  | 0.0476 |
| item-kNN                  | 0.0725 | 0.1075 |   0.184  | 0.2617 |
| EASE                      | 0.1048 | 0.1578 |   0.2748 | 0.4062 |
| ALS                       | 0.0796 | 0.1142 |   0.1918 | 0.2638 |
| Hybrid                    | 0.0799 | 0.1156 |   0.1926 | 0.2655 |
| Hybrid-EASE               | 0.105  | 0.1579 |   0.2755 | 0.4082 |
| Hybrid: Try something new | 0.0819 | 0.0975 |   0.1862 | 0.333  |

Hybrid-EASE beats MostPopular in every library-size bucket: **yes**.


Players per source and bucket:

| src       |   1-4 |   5-20 |   21-100 |   100+ |
|:----------|------:|-------:|---------:|-------:|
| kaggle    | 11049 |   5991 |     1022 |     31 |
| mcauley   |   260 |   1333 |     2695 |    391 |
| steam2026 |     2 |      7 |       44 |    133 |
