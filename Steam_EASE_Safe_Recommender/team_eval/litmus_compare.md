# Rob's litmus test on both models (300 players, 10 most-played games hidden, recency off)

| | Rob KNN | EASE | Popularity (Rob's baseline) |
|---|---:|---:|---:|
| hidden games the model can recommend at all (of 10) | 8.4 | 9.1 | |
| avg hidden games found in top 15 | 2.13 | 4.88 | 2.01 |
| players with 1+ found in top 15 | 92% | 99% | 91% |
| avg hidden games found in top 50 | 3.53 | 6.71 | 3.23 |
| players with 1+ found in top 50 | 99% | 100% | 97% |
| avg hidden games found in top 100 | 4.45 | 7.54 | 4.27 |
| players with 1+ found in top 100 | 100% | 100% | 99% |
| top 15, mcauley only (284 players) | 2.16 | 4.97 | 2.05 |
| top 15, steam2026 only (16 players) | 1.62 | 3.38 | 1.38 |

EASE minus Rob, hidden games found in top 15: +2.75 (95% CI +2.55 to +2.95)
