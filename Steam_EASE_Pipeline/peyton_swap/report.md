# Peyton's API, Juan's recipe vs the scikit-learn pipeline (local)

374 requests per copy + 1 without a token. Store prices: one fixed fake store for all copies.

## swap A: 374 / 374 identical; no-token request refused (401) like the original: True

| route              |   requests |   identical |
|:-------------------|-----------:|------------:|
| /health            |          1 |           1 |
| /search            |          6 |           6 |
| /games             |          1 |           1 |
| /game/{id}         |         11 |          11 |
| /gem               |          5 |           5 |
| /demo              |          5 |           5 |
| /recommend         |        342 |         342 |
| /recommend/profile |          2 |           2 |
| /player            |          1 |           1 |

## swap B: 372 / 374 identical; no-token request refused (401) like the original: True

| route              |   requests |   identical |
|:-------------------|-----------:|------------:|
| /health            |          1 |           1 |
| /search            |          6 |           6 |
| /games             |          1 |           1 |
| /game/{id}         |         11 |          11 |
| /gem               |          5 |           5 |
| /demo              |          5 |           5 |
| /recommend         |        342 |         340 |
| /recommend/profile |          2 |           2 |
| /player            |          1 |           1 |

Differences (first field that differs):

- `/recommend` .for_you[2].score: '0.391' vs '0.3909'; everything else identical (same games, order, prices, flags, explanations)
- `/recommend` .for_you[2].score: '0.391' vs '0.3909'; everything else identical (same games, order, prices, flags, explanations)

Status codes from the original (to show error paths were exercised): {200: 368, 400: 5, 404: 1}
