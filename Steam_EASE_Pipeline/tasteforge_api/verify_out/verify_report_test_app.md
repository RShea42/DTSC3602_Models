# TasteForge API: local vs deployed (scikit-learn pipeline)

200 real player libraries (the team's M5 inputs, played games only, `n=10`), `POST /recommend`;
3990 recommendation cards per source. Inputs: `verify_out/requests.json`; raw responses: `verify_out/responses_*.json`.

- deployed: `https://sirjacket42--steam-ease-api-test-api-web.us-east.modal.run`
- reference: `https://sirjacket42--steam-ease-api-api-web.us-east.modal.run`

| comparison | requests | same status | same picks (app, order, score) | same JSON except prices | identical JSON |
|:--|--:|--:|--:|--:|--:|
| local vs deployed | 200 | 200 | 200 | 200 | 200 |
| local vs reference | 200 | 200 | 199 | 199 | 199 |

| source | first request s | median s |
|:--|--:|--:|
| local | 1.60 | 1.66 |
| deployed | 13.99 | 1.13 |
| reference | 0.71 | 0.30 |

**Result (local vs deployed picks): PASS**
