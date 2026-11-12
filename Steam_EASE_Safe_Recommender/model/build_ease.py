"""Build the live app's model: EASE, the exact model and settings we evaluated (eval/models.py, eval/cache/best_params.json).

Trained on the evaluation's train players (eval/cache/train_g.npz), so the app serves the same model the reports score.
EASE scores games with >= 100 train players (12,934 games, 11,773 recommendable); newer or niche games are not scored.
Out: model/artifacts/ease.npz with B (item-to-item weights, float16) and app_ids (one per row/column of B).
Run alone (~5 GB of memory):
  OPENBLAS_NUM_THREADS=4 PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python model/build_ease.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from harness import train_matrix  # noqa: E402
from models import EASE  # noqa: E402

C = ROOT / "eval" / "cache"
t0 = time.time()
params = json.load(open(C / "best_params.json"))["EASE"]
U = np.load(C / "universe.npy")
m = EASE(**params).fit(train_matrix())
B = m.B.astype(np.float16)
np.savez(ROOT / "model" / "artifacts" / "ease.npz", B=B, app_ids=U[m.vocab].astype(np.int64),
         params=np.array([params["lam"], m.min_users], dtype=np.float64))
print(f"EASE lam={params['lam']}, min_users={m.min_users}: {len(m.vocab):,} games, B {B.nbytes / 1e6:.0f} MB, {time.time() - t0:.0f}s")
