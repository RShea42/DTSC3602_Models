# D3 error analysis (shared head-to-head, 1,000 validation players)

## Overall NDCG@10 (check: equals evaluate.py)

|   holdout |   EASE-safe |   EASE |   KNN (Rob) |   TasteForge |   MostPopular |
|----------:|------------:|-------:|------------:|-------------:|--------------:|
|     50_50 |      0.3279 | 0.3318 |      0.1163 |       0.0026 |        0.0631 |
|     75_25 |      0.2757 | 0.2811 |      0.073  |       0.0021 |        0.0413 |
|     80_20 |      0.277  | 0.2823 |      0.0717 |       0.0019 |        0.0349 |

## NDCG@10 by visible library size (80/20)

| size         |   EASE-safe |   EASE |   KNN (Rob) |   TasteForge |   MostPopular |   players |
|:-------------|------------:|-------:|------------:|-------------:|--------------:|----------:|
| 5-20 games   |      0.2145 | 0.2126 |      0.0745 |       0.0038 |        0.0287 |       426 |
| 21-100 games |      0.3038 | 0.3126 |      0.0658 |       0.0006 |        0.0375 |       471 |
| 100+ games   |      0.4124 | 0.4318 |      0.0873 |       0      |        0.049  |       103 |

## NDCG@10 by data source (80/20)

| src       |   EASE-safe |   EASE |   KNN (Rob) |   TasteForge |   MostPopular |   players |
|:----------|------------:|-------:|------------:|-------------:|--------------:|----------:|
| kaggle    |      0.1388 | 0.1388 |      0.0467 |       0.001  |        0.0218 |       299 |
| mcauley   |      0.3346 | 0.3411 |      0.083  |       0.0024 |        0.04   |       660 |
| steam2026 |      0.3556 | 0.3816 |      0.0731 |       0      |        0.0484 |        41 |

## Share of hidden enjoyed games found in the top 10, popular vs niche (80/20)

Popular = top 500 games by training players (2546 hidden games); niche = all others (4279).

| model       |   popular hidden games found % |   niche hidden games found % |
|:------------|-------------------------------:|-----------------------------:|
| EASE-safe   |                           43.3 |                          5.3 |
| EASE        |                           43.5 |                          6.1 |
| KNN (Rob)   |                           13.7 |                          0.4 |
| TasteForge  |                            0.3 |                          0   |
| MostPopular |                            8.4 |                          0   |

## Where EASE-safe finds nothing (80/20)

337 of 1000 players (34%) get no hidden enjoyed game in the top 10.

| | players with no hit | all players |
|---|---:|---:|
| median visible games | 12 | 26 |
| share of their hidden games that are niche | 57% | 63% |
| share from Kaggle (reviews, not full libraries) | 60% | 30% |
