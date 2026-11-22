"""M0 part 2: retrain EASE from the team-eval train parts and compare with Juan's saved ease.npz.

Same math as eval/models.py EASE (binary played matrix, Gram + lam on the diagonal, inverse, B = P / -diag(P)),
with the settings stored in ease.npz (lam=100, min_users=100). Saves m0_out/ease_refit.npz (same format) and prints
fit time plus how close the weights are.
"""
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from paths import EASE_FILES
HERE = Path(__file__).resolve().parent
FILES = EASE_FILES
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

saved = np.load(FILES / "model" / "artifacts" / "ease.npz")
lam, min_users = saved["params"]
log(f"saved model: lam={lam}, min_users={min_users}")

t = pd.concat([pd.read_parquet(f, columns=["player_id", "app_id"])
               for f in sorted(glob.glob(str(FILES / "data_clean" / "team_eval" / "train" / "*.parquet")))], ignore_index=True)
n = t.groupby("app_id").player_id.nunique()
vocab = np.sort(n[n >= min_users].index.to_numpy())
t = t[t.app_id.isin(vocab)]
col = pd.Series(np.arange(len(vocab)), index=vocab)
users = pd.factorize(t.player_id)[0]
X = sp.csr_matrix((np.ones(len(t), np.float32), (users, col[t.app_id].to_numpy())), shape=(users.max() + 1, len(vocab)))
del t
log(f"train matrix {X.shape}, {X.nnz:,} played rows")

f0 = time.time()
G = (X.T @ X).toarray().astype(np.float64)
G[np.diag_indices_from(G)] += lam
P = np.linalg.inv(G)
B = P / (-np.diag(P))
np.fill_diagonal(B, 0)
log(f"EASE fit took {time.time() - f0:.0f}s")

assert np.array_equal(vocab, saved["app_ids"]), "game order differs from saved model"
Bs = saved["B"].astype(np.float32)
B16 = B.astype(np.float16)
d = np.abs(B16.astype(np.float32) - Bs)
log(f"max |diff| {d.max():.2e}, mean |diff| {d.mean():.2e}, exactly equal after float16: {np.mean(B16 == saved['B']) * 100:.4f}%")
log(f"weight scale for reference: mean |B| {np.abs(Bs).mean():.2e}")
np.savez(HERE / "m0_out" / "ease_refit.npz", B=B16, app_ids=vocab.astype(np.int64), params=saved["params"])
log("saved m0_out/ease_refit.npz")
