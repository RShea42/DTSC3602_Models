# Layer 2: offline evaluation on the test set (run 2: second, disclosed scoring)

> **Run 2** (PROTOCOL section 8, change 4): completed tags and full-catalog content embeddings; the test set's second use, disclosed. Run 1, the pre-registered result: `eval/run1/`.

Protocol: `eval/PROTOCOL.md`. Settings chosen on validation only: `eval/tuning.md`. Ours = **Hybrid-EASE** (higher validation NDCG@10 of the two hybrids: {'Hybrid': 0.18872911042533055, 'Hybrid-EASE': 0.27714344258837353}). Test players: 30,977 with at least one recommendable hidden enjoyed game; main results use the 11,647 with 5+ visible games.

## Main results (5+ visible games)

| model                     |   players |   NDCG@10 |   graded NDCG@10 |   Recall@20 |   long-tail Recall@20 |   Coverage@10 |   avg popularity of picks |
|:--------------------------|----------:|----------:|-----------------:|------------:|----------------------:|--------------:|--------------------------:|
| MostPopular               |     11647 |    0.0325 |           0.03   |      0.1009 |                0      |        0.0003 |                  183698   |
| item-kNN                  |     11647 |    0.1396 |           0.1318 |      0.2534 |                0.0918 |        0.0479 |                   33345.2 |
| EASE                      |     11647 |    0.2074 |           0.1946 |      0.3408 |                0.0871 |        0.0317 |                   73843.2 |
| ALS                       |     11647 |    0.1464 |           0.1401 |      0.2593 |                0.0031 |        0.0118 |                   46274.2 |
| Hybrid                    |     11647 |    0.1467 |           0.1404 |      0.2604 |                0.0047 |        0.0124 |                   44465.1 |
| Hybrid-EASE               |     11647 |    0.2083 |           0.1955 |      0.3406 |                0.0881 |        0.0322 |                   72858.9 |
| Hybrid: Try something new |     11647 |    0.1405 |           0.1302 |      0.2211 |                0.0686 |        0.0345 |                   57766.1 |

## Statistics: Hybrid-EASE vs each model (NDCG@10, paired over players)

| Hybrid-EASE vs   |   mean difference |   relative | 95% CI             |   t-test p |   randomization p |   Holm-adjusted p |
|:-----------------|------------------:|-----------:|:-------------------|-----------:|------------------:|------------------:|
| MostPopular      |            0.1758 |     5.4095 | +0.1712 to +0.1805 |     0      |            0      |            0      |
| item-kNN         |            0.0687 |     0.4922 | +0.0650 to +0.0725 |     0      |            0      |            0      |
| ALS              |            0.0619 |     0.4228 | +0.0579 to +0.0655 |     0      |            0      |            0      |
| Hybrid           |            0.0616 |     0.4199 | +0.0576 to +0.0655 |     0      |            0      |            0      |
| EASE             |            0.0008 |     0.004  | +0.0003 to +0.0013 |     0.0015 |            0.0008 |            0.0015 |

## NDCG@10 by source (5+ visible games)

| model                     |   kaggle |   mcauley |   steam2026 |
|:--------------------------|---------:|----------:|------------:|
| MostPopular               |   0.0229 |    0.0464 |      0.0651 |
| item-kNN                  |   0.0869 |    0.2211 |      0.1972 |
| EASE                      |   0.1236 |    0.3343 |      0.3694 |
| ALS                       |   0.0998 |    0.2183 |      0.2009 |
| Hybrid                    |   0.1003 |    0.2183 |      0.2005 |
| Hybrid-EASE               |   0.1241 |    0.3354 |      0.3758 |
| Hybrid: Try something new |   0.0738 |    0.2403 |      0.2967 |

## NDCG@10 by visible library size (all test players)

| model                     |    1-4 |   5-20 |   21-100 |   100+ |
|:--------------------------|-------:|-------:|---------:|-------:|
| MostPopular               | 0.0313 | 0.029  |   0.037  | 0.0476 |
| item-kNN                  | 0.0725 | 0.1075 |   0.184  | 0.2617 |
| EASE                      | 0.1048 | 0.1578 |   0.2748 | 0.4062 |
| ALS                       | 0.0796 | 0.1142 |   0.1918 | 0.2638 |
| Hybrid                    | 0.0802 | 0.1148 |   0.1915 | 0.2641 |
| Hybrid-EASE               | 0.1051 | 0.1581 |   0.2762 | 0.4102 |
| Hybrid: Try something new | 0.0824 | 0.0982 |   0.1936 | 0.339  |

Hybrid-EASE beats MostPopular in every library-size bucket: **yes**.


Players per source and bucket:

| src       |   1-4 |   5-20 |   21-100 |   100+ |
|:----------|------:|-------:|---------:|-------:|
| kaggle    | 11049 |   5991 |     1022 |     31 |
| mcauley   |   260 |   1333 |     2695 |    391 |
| steam2026 |     2 |      7 |       44 |    133 |
