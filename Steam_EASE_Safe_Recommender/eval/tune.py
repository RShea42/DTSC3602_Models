"""Tune every model on VALIDATION players only (eval/PROTOCOL.md section 5). Same budget for each: up to 8 settings.
Selection metric: mean NDCG@10 over validation players with >= 5 visible games (all sources pooled).
Run: OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python eval/tune.py [model ...]
Out: eval/tuning.md, eval/cache/best_params.json, eval/cache/tuning_rows.jsonl (resumable)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import evaluate, load, train_matrix  # noqa: E402
from models import ALS, EASE, Hybrid, ItemKNN, MostPopular  # noqa: E402

C = HERE / "cache"
U, rec, val = load("validation")
val = [p for p in val if p["n_visible"] >= 5]
X = train_matrix()
print(f"validation players {len(val):,} " + str(pd.Series([p['src'] for p in val]).value_counts().to_dict()), flush=True)

GRID = {
    "MostPopular": [{}],
    "item-kNN": [dict(k=k, weight=w) for w in ("g", "binary") for k in (50, 100, 200, 400)],
    "EASE": [dict(lam=l) for l in (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)],   # games with >= 100 train players (memory)
    "ALS": [dict(factors=f, alpha=a, reg=r) for f, a, r in
            [(64, 20, 0.05), (128, 20, 0.05), (64, 10, 0.05), (64, 40, 0.05), (128, 40, 0.1), (256, 20, 0.1), (128, 10, 0.01), (64, 20, 0.5)]],
    "Hybrid": [dict(w=w) for w in (0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.98)],
    "Hybrid-EASE": [dict(w=w) for w in (0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.98)],   # PROTOCOL section 8, change 3
}
_bases = {}


def content_matrix():
    E = np.zeros((len(U), 384), np.float32)
    ids = np.load(HERE.parent / "model" / "artifacts" / "game_embedding_ids.npy")
    emb = np.load(HERE.parent / "model" / "artifacts" / "game_embeddings.npy")
    pos = pd.Series(np.arange(len(U)), index=U).reindex(ids)
    ok = pos.notna().to_numpy()
    E[pos[ok].astype(int).to_numpy()] = emb[ok]
    return E
done_f = C / "tuning_rows.jsonl"
done = [json.loads(l) for l in done_f.open()] if done_f.exists() else []
seen = {(d["model"], json.dumps(d["params"], sort_keys=True)) for d in done}


def record(name, params, df, secs):
    row = {"model": name, "params": params, "ndcg10": float(df.ndcg10.mean()), "recall20": float(df.recall20.mean()),
           "by_src": df.groupby("src").ndcg10.mean().round(4).to_dict(), "seconds": round(secs)}
    with done_f.open("a") as f:
        f.write(json.dumps(row) + "\n")
    done.append(row); print(row, flush=True)


def best(name):
    r = [d for d in done if d["model"] == name]
    return max(r, key=lambda d: d["ndcg10"])["params"] if r else None


which = sys.argv[1:] or list(GRID)
for name in which:
    for params in GRID[name]:
        key = (name, json.dumps(params, sort_keys=True))
        if key in seen:
            continue
        t = time.time()
        if name in ("Hybrid", "Hybrid-EASE"):
            base_name, base_cls = ("ALS", ALS) if name == "Hybrid" else ("EASE", EASE)
            bp = best(base_name)
            if bp is None:
                raise SystemExit(f"tune {base_name} before {name}")
            if base_name not in _bases:
                _bases[base_name] = base_cls(**bp).fit(X)
            if "E" not in _bases:
                _bases["E"] = content_matrix()
            m = Hybrid(_bases[base_name], _bases["E"], **params)
        else:
            m = {"MostPopular": MostPopular, "item-kNN": ItemKNN, "EASE": EASE, "ALS": ALS}[name](**params).fit(X)
        record(name, params, evaluate(m, val, rec, keep_lists=False), time.time() - t)

bp = {n: best(n) for n in GRID if best(n) is not None}
json.dump(bp, open(C / "best_params.json", "w"), indent=2)
t = pd.DataFrame([{"model": d["model"], "params": json.dumps(d["params"]), "NDCG@10": d["ndcg10"], "Recall@20": d["recall20"],
                   **{f"NDCG@10 {k}": v for k, v in d["by_src"].items()}, "seconds": d["seconds"]} for d in done])
(HERE / "tuning.md").write_text("# Tuning on validation players (>= 5 visible games)\n\nSelection: highest NDCG@10. "
                                "Test players never used.\n\n" + t.round(4).to_markdown(index=False) +
                                "\n\nChosen:\n\n```json\n" + json.dumps(bp, indent=2) + "\n```\n", encoding="utf-8")
print(json.dumps(bp, indent=1))
