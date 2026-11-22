# Team comparison on validation players

## Holdout 80/20 (visible/hidden)

|                     |   sample1000_ease_safe |   sample1000_ease |
|:--------------------|-----------------------:|------------------:|
| players             |              1000.0000 |         1000.0000 |
| picks per player    |                10.0000 |           10.0000 |
| owned picks %       |                 0.0000 |            0.0000 |
| not recommendable % |                 0.0000 |            0.0000 |
| NDCG@10             |                 0.2769 |            0.2823 |
| HitRate@10          |                 0.6630 |            0.6530 |
| HitRate@10 (any)    |                 0.7630 |            0.7520 |
| Coverage@10         |                 0.0095 |            0.0102 |
| Popularity pct      |                99.7536 |           99.7185 |
| Tag distance        |                 0.4908 |            0.4978 |
| Playtime gap        |                 0.5390 |            0.5374 |
| Within 2x %         |                35.0859 |           35.5096 |
| Playtime known %    |                96.6200 |           96.6500 |
| Playtime gap (avg)  |                 0.5529 |            0.5566 |
| Within 2x (avg) %   |                33.5645 |           33.2023 |
| Regret %            |                16.3562 |           18.0771 |
| Flag: negative %    |                 0.0000 |            0.1200 |
| Flag: short %       |                 3.0000 |            8.3700 |
| Flag: like quit %   |                19.1300 |           27.3400 |
| Any flag %          |                20.8800 |           31.3100 |
| 2+ flags %          |                 1.2500 |            4.5200 |

| model           |   NDCG@10 minus sample1000_ease_safe | 95% CI             |
|:----------------|-------------------------------------:|:-------------------|
| sample1000_ease |                              +0.0054 | +0.0016 to +0.0092 |

## Holdout 75/25 (visible/hidden)

|                     |   sample1000_ease_safe |   sample1000_ease |
|:--------------------|-----------------------:|------------------:|
| players             |               870.0000 |          870.0000 |
| picks per player    |                10.0000 |           10.0000 |
| owned picks %       |                 0.0000 |            0.0000 |
| not recommendable % |                 0.0000 |            0.0000 |
| NDCG@10             |                 0.2757 |            0.2811 |
| HitRate@10          |                 0.6989 |            0.6920 |
| HitRate@10 (any)    |                 0.8391 |            0.8379 |
| Coverage@10         |                 0.0090 |            0.0095 |
| Popularity pct      |                99.7398 |           99.7111 |
| Tag distance        |                 0.4907 |            0.4999 |
| Playtime gap        |                 0.5345 |            0.5336 |
| Within 2x %         |                36.1005 |           36.3159 |
| Playtime known %    |                96.0920 |           96.2184 |
| Playtime gap (avg)  |                 0.5619 |            0.5649 |
| Within 2x (avg) %   |                31.7584 |           31.7286 |
| Regret %            |                18.0191 |           20.7901 |
| Flag: negative %    |                 0.0000 |            0.1609 |
| Flag: short %       |                 3.0230 |            8.7816 |
| Flag: like quit %   |                20.2299 |           28.4828 |
| Any flag %          |                22.0575 |           32.8046 |
| 2+ flags %          |                 1.1954 |            4.5977 |

| model           |   NDCG@10 minus sample1000_ease_safe | 95% CI             |
|:----------------|-------------------------------------:|:-------------------|
| sample1000_ease |                              +0.0055 | +0.0020 to +0.0090 |

## Holdout 50/50 (visible/hidden)

|                     |   sample1000_ease_safe |   sample1000_ease |
|:--------------------|-----------------------:|------------------:|
| players             |               873.0000 |          873.0000 |
| picks per player    |                10.0000 |           10.0000 |
| owned picks %       |                 0.0000 |            0.0000 |
| not recommendable % |                 0.0000 |            0.0000 |
| NDCG@10             |                 0.3280 |            0.3319 |
| HitRate@10          |                 0.8213 |            0.8282 |
| HitRate@10 (any)    |                 0.9084 |            0.9072 |
| Coverage@10         |                 0.0090 |            0.0098 |
| Popularity pct      |                99.7279 |           99.6926 |
| Tag distance        |                 0.5012 |            0.5094 |
| Playtime gap        |                 0.5531 |            0.5511 |
| Within 2x %         |                34.9797 |           35.1303 |
| Playtime known %    |                95.7847 |           95.8305 |
| Playtime gap (avg)  |                 0.5895 |            0.5908 |
| Within 2x (avg) %   |                31.4398 |           31.1499 |
| Regret %            |                16.3749 |           18.5216 |
| Flag: negative %    |                 0.0000 |            0.1489 |
| Flag: short %       |                 3.0584 |            8.6254 |
| Flag: like quit %   |                20.4926 |           28.9576 |
| Any flag %          |                22.3024 |           33.4135 |
| 2+ flags %          |                 1.2486 |            4.3070 |

| model           |   NDCG@10 minus sample1000_ease_safe | 95% CI             |
|:----------------|-------------------------------------:|:-------------------|
| sample1000_ease |                              +0.0039 | +0.0006 to +0.0072 |
