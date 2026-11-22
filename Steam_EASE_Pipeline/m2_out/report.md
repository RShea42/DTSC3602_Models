# M2 tuning report (report only: defaults stay Juan's)

Players: tune 34,029 / check 14,588 (fixed hash of player_id). Model trained on train_full parts.
Scored with Juan's `team_eval/evaluate.score()`, evaluation setup.

## Stage 0: rule check

Juan's a/b/c rule on all validation players, 75/25, lam=100 picks **(0.5, 1.0, 10.0)**; Juan's rerank_choice.json: (0.5, 1.0, 10.0). Match: the rule is applied as Juan applied it.

## Stage 1: lam (tune players, 80/20, Juan's a/b/c)

|      lam |   NDCG@10 |   HitRate@10 (any) |   Any flag % |   Regret % |   Coverage@10 |   diff vs 100 |   fold std |   folds better |
|---------:|----------:|-------------------:|-------------:|-----------:|--------------:|--------------:|-----------:|---------------:|
|  30.0000 |    0.2679 |             0.7429 |      21.0707 |    18.0126 |        0.0210 |       -0.0004 |     0.0016 |         1.0000 |
| 100.0000 |    0.2682 |             0.7431 |      20.9170 |    17.7667 |        0.0198 |        0.0000 |     0.0000 |         0.0000 |
| 150.0000 |    0.2683 |             0.7420 |      20.7385 |    17.6174 |        0.0191 |        0.0001 |     0.0003 |         2.0000 |
| 200.0000 |    0.2687 |             0.7413 |      20.6466 |    17.4829 |        0.0185 |        0.0004 |     0.0006 |         4.0000 |
| 250.0000 |    0.2687 |             0.7412 |      20.6042 |    17.4734 |        0.0182 |        0.0005 |     0.0011 |         3.0000 |
| 300.0000 |    0.2683 |             0.7401 |      20.5707 |    17.4033 |        0.0179 |        0.0000 |     0.0014 |         2.0000 |
| 500.0000 |    0.2673 |             0.7406 |      20.3728 |    17.2867 |        0.0167 |       -0.0009 |     0.0017 |         2.0000 |

Best by NDCG@10: **lam = 250** (Juan's default: 100). 'folds better' = of 5 GroupKFold player folds, how many favour it over 100.

## Stage 2: a/b/c with Juan's rule (tune players, 75/25)

- lam 100: rule picks a/b/c = **(0.5, 1.0, 10.0)** (plain EASE NDCG@10 0.2789); full grid in m2_out/grid_lam100.csv
- lam 250: rule picks a/b/c = **(1.0, 0.0, 10.0)** (plain EASE NDCG@10 0.2785); full grid in m2_out/grid_lam250.csv

## Stage 3: check players (never used above), Juan's defaults vs tuned

Juan: lam 100, a/b/c (0.5, 1.0, 10.0).  Tuned: lam 250, a/b/c (1.0, 0.0, 10.0).

|   holdout |   players |   Juan NDCG@10 |   tuned NDCG@10 |    diff | 95% CI             |   Juan any flag % |   tuned any flag % |
|----------:|----------:|---------------:|----------------:|--------:|:-------------------|------------------:|-------------------:|
|     80_20 |      2452 |         0.2753 |          0.2750 | -0.0003 | -0.0026 to +0.0020 |           21.2643 |            19.2781 |
|     75_25 |      2526 |         0.2764 |          0.2763 | -0.0001 | -0.0022 to +0.0020 |           21.7775 |            19.8100 |
|     50_50 |      2393 |         0.3121 |          0.3116 | -0.0005 | -0.0026 to +0.0016 |           23.6858 |            21.5880 |
