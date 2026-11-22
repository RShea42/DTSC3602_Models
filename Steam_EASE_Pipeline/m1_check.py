"""M1 pass/fail: the scikit-learn pipeline, trained from the train parts, must reproduce M0's lists and scores.

1. fit the pipeline on train_full parts (real training, ~1 min)
2. save with joblib, load it back (what Modal will do)
3. predict all three holdouts; compare pick-by-pick with M0's lists from Juan's saved model (m0_out/)
4. write recs for Juan's evaluate.py (m1_out/recs_ease_safe.parquet, plus plain EASE via a=b=c=0)
Run: PYTHONUTF8=1 <ease-pipeline venv python> m1_check.py
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from steamrec import make_pipeline
from steamrec.data import load_train, predict_holdouts

from paths import ARTIFACTS, TE
HERE = Path(__file__).resolve().parent
OUT = HERE / "m1_out"
OUT.mkdir(exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

pipe = make_pipeline(TE / "games.parquet", serve_filters=False)   # Juan's evaluation setup
train = load_train(TE)
log(f"train rows {len(train):,}")
pipe.fit(train)
del train
log(f"fitted: {len(pipe['encode'].app_ids_):,} games")

saved = np.load(ARTIFACTS / "ease.npz")
same_w = np.mean(pipe["ease"].B_ == saved["B"]) * 100
log(f"weights identical to Juan's ease.npz: {same_w:.4f}%")

joblib.dump(pipe, OUT / "ease_safe_pipeline.joblib")
size = (OUT / "ease_safe_pipeline.joblib").stat().st_size / 1e6
pipe = joblib.load(OUT / "ease_safe_pipeline.joblib")
log(f"saved + reloaded pipeline ({size:.0f} MB)")

val = pd.read_parquet(TE / "validation.parquet")
ok = True
for name, params in [("ease_safe", {}), ("ease", {"safe__a_quit": 0, "safe__b_short": 0, "safe__c_negative": 0})]:
    pipe.set_params(**({"safe__a_quit": 0.5, "safe__b_short": 1.0, "safe__c_negative": 10.0} | params))
    recs = predict_holdouts(pipe, val)
    recs.to_parquet(OUT / f"recs_{name}.parquet", index=False)
    ref = pd.read_parquet(HERE / "m0_out" / f"recs_{name}.parquet")
    m = ref.merge(recs, on=["player_id", "holdout", "rank"], how="outer", suffixes=("_m0", "_m1"))
    same = m.app_id_m0 == m.app_id_m1
    log(f"{name}: {len(recs):,} picks vs M0 {len(ref):,}; identical {same.mean() * 100:.3f}%; "
        f"players differing {m[~same].player_id.nunique()}")
    ok &= len(recs) == len(ref) and same.mean() > 0.9999
print("M1 PARITY:", "PASS" if ok else "FAIL")
