"""Write our models' top-10 lists for the team holdouts, in the shared recs format (team_eval/evaluate.py).

Same fitted settings as Run 2 (eval/cache/best_params.json), trained on all train players, validation players only.
Out: data_clean/team_eval/recs_<model>.parquet for MostPopular, EASE, Hybrid-EASE (same EASE fit) and ALS (the live app's model type).
Run alone (EASE needs ~6 GB):
  OPENBLAS_NUM_THREADS=4 PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python team_eval/run_ours.py [models...]
"""
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from harness import train_matrix  # noqa: E402
from models import ALS, EASE, Hybrid, MostPopular  # noqa: E402

C = ROOT / "eval" / "cache"
OUT = Path("C:/Users/juant/data_clean/team_eval")
HOLD = ["80_20", "75_25", "50_50"]
best = json.load(open(C / "best_params.json"))
U = np.load(C / "universe.npy")
rec = np.load(C / "recommendable.npy")
players = {h: pickle.load(open(C / f"team_{h}_players.pkl", "rb")) for h in HOLD}
want = sys.argv[1:] or ["MostPopular", "EASE", "ALS"]
t0 = time.time()


def top10(model, name, k_keep=10):
    """Top-10 recs file; with k_keep > 10 also saves the top k_keep with scores for re-ranking (eval/cache)."""
    rows = []
    for h in HOLD:
        P = players[h]
        for k in range(0, len(P), 256):
            batch = P[k:k + 256]
            S = model.score(batch)
            S[:, ~rec] = -np.inf
            for r, p in enumerate(batch):
                s = S[r]
                s[np.setdiff1d(p["owned_u"], p["hid_u"])] = -np.inf
                top = np.argpartition(-s, k_keep)[:k_keep]
                top = top[np.argsort(-s[top], kind="stable")]
                top = top[np.isfinite(s[top])]
                rows += [(p["player_id"], h, i + 1, int(U[t]), float(s[t])) for i, t in enumerate(top)]
    df = pd.DataFrame(rows, columns=["player_id", "holdout", "rank", "app_id", "score"])
    tag = name.replace('-', '_').lower()
    if k_keep > 10:
        df.to_parquet(C / f"team_top{k_keep}_{tag}.parquet", index=False)
    df[df["rank"] <= 10].drop(columns="score").to_parquet(OUT / f"recs_{tag}.parquet", index=False)
    print(f"{name}: {len(df):,} picks ({time.time() - t0:.0f}s)", flush=True)


X = train_matrix()
if "MostPopular" in want:
    top10(MostPopular().fit(X), "MostPopular")
if "EASE" in want:                                   # one EASE fit -> EASE and Hybrid-EASE lists
    ids = np.load(ROOT / "model" / "artifacts" / "game_embedding_ids.npy")
    emb = np.load(ROOT / "model" / "artifacts" / "game_embeddings.npy")
    E = np.zeros((len(U), emb.shape[1]), np.float32)
    pos = pd.Series(np.arange(len(U)), index=U).reindex(ids); ok = pos.notna().to_numpy()
    E[pos[ok].astype(int).to_numpy()] = emb[ok]
    del emb
    ease = EASE(**best["EASE"]).fit(X)
    top10(ease, "EASE", k_keep=50)
    if "noHybrid" in want:
        sys.exit(0)
    top10(Hybrid(ease, E, **best["Hybrid-EASE"]), "Hybrid-EASE")
    del ease, E
    gc.collect()
if "ALS" in want:
    top10(ALS(**best["ALS"]).fit(X), "ALS")
print("done")
