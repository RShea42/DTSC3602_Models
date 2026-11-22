# M5: local vs deployed outputs

200 real libraries (validation players, 80/20 visible games; inputs in m5_out/requests.json), top 10 picks each.
Players with no played game the model knows get no picks (counted as matching when every source returns none): 0.

| compared with saved pipeline   |   players |   same picks, same order |   max score difference | identical scores   |
|:-------------------------------|----------:|-------------------------:|-----------------------:|:-------------------|
| local FastAPI                  |       200 |                      200 |                      0 | True               |
| deployed on Modal              |       200 |                      200 |                      0 | True               |

## Timing

|                   |   first request s |   median request s |   total s |
|:------------------|------------------:|-------------------:|----------:|
| local FastAPI     |              0.56 |              0.576 |     115.4 |
| deployed on Modal |             10.43 |              1.029 |     217.8 |

**Result: PASS**
