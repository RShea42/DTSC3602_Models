"""M2: tune EASE's lam and the safety re-rank (a, b, c). Report only: the pipeline's defaults stay Juan's.

Players: validation players split by a fixed hash into TUNE (70%) and CHECK (30%). Everything is chosen on TUNE;
CHECK is scored once at the end. The model itself is always trained on the train_full parts (other players).
Scoring: Juan's team_eval/evaluate.score(), evaluation setup (serve_filters=False), as his numbers were scored.

Stage 0  Rule check: Juan's a/b/c rule on ALL players, 75/25, lam=100 must pick his 0.5 / 1 / 10.
Stage 1  lam in LAMS with Juan's a/b/c, chosen by NDCG@10 on TUNE 80/20 (Juan chose lam by NDCG).
Stage 2  Juan's a/b/c grid + rule on TUNE 75/25, at lam=100 and at the stage-1 pick.
Stage 3  CHECK players, all holdouts: Juan's defaults vs the tuned settings, paired bootstrap 95% CI.
Spread:  GroupKFold(5) over TUNE players: fold-to-fold std of the NDCG difference vs Juan's lam.
Out: m2_out/ (candidate cache per lam, report.md). Run: PYTHONUTF8=1 <ease-pipeline python> m2_tune.py
"""
import gc
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from steamrec import make_pipeline
from steamrec.data import HOLDOUTS, holdout_input, load_train

from paths import JUAN, TE
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(JUAN / "team_eval"))
from evaluate import score, tag_matrix  # noqa: E402

OUT = HERE / "m2_out"
OUT.mkdir(exist_ok=True)
LAMS = [30.0, 100.0, 150.0, 200.0, 250.0, 300.0, 500.0]
JUAN_ABC = (0.5, 1.0, 10.0)
GRID = list(itertools.product([0, 0.25, 0.5, 1, 2, 4], [0, 0.25, 0.5, 1, 2], [0, 10]))   # team_eval/rerank.py
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
report = []
md = lambda s="": report.append(s)

games = pd.read_parquet(TE / "games.parquet")
T = tag_matrix(games).tocsr()
val = pd.read_parquet(TE / "validation.parquet")
pids = np.sort(val.player_id.unique())
is_tune = (pids.astype(np.uint64) * np.uint64(2654435761) % np.uint64(2**32)) % np.uint64(100) < np.uint64(70)
TUNE, CHECK = set(pids[is_tune]), set(pids[~is_tune])
val_tune, val_check = val[val.player_id.isin(TUNE)], val[val.player_id.isin(CHECK)]
src = val.drop_duplicates("player_id").set_index("player_id").src
log(f"players: tune {len(TUNE):,}, check {len(CHECK):,}; tune share by source "
    + str((src[src.index.isin(TUNE)].value_counts() / src.value_counts()).round(3).to_dict()))

pipe = make_pipeline(TE / "games.parquet", serve_filters=False)
safe = pipe["safe"].fit()


# ------------------------------------------------------------------ candidates (top 50) per lam, cached
def flagged(lam):
    """Flagged top-50 candidates for every holdout at this lam (one EASE fit)."""
    f = OUT / f"cand_lam{int(lam)}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    pipe.set_params(ease__lam=lam)
    pipe[:-1].fit(load_train(TE))
    out = []
    for h in HOLDOUTS:
        L = pipe[:-1].transform(holdout_input(val, h))
        out.append(safe.flags(L).assign(holdout=h))
    c = pd.concat(out, ignore_index=True)
    c.to_parquet(f, index=False)
    pipe["ease"].B_ = pipe["ease"]._Bf = None
    gc.collect()
    return c


def recs(c, h, abc):
    safe.set_params(a_quit=abc[0], b_short=abc[1], c_negative=abc[2])
    return safe.rerank(c[c.holdout == h]).assign(holdout=h)


def ev(c, h, abc, v):
    o, per_player = score(recs(c, h, abc), v, games, T, h)
    return o, per_player


def rule(c, h, v):
    """Juan's rule: lowest Any-flag share among settings with NDCG@10 >= 98% of plain EASE (a=b=c=0)."""
    rows = []
    for abc in GRID:
        o, _ = ev(c, h, abc, v)
        rows.append({"a": abc[0], "b": abc[1], "c": abc[2], "NDCG@10": o["NDCG@10"], "Any flag %": o["Any flag %"]})
    g = pd.DataFrame(rows)
    base = g[(g.a == 0) & (g.b == 0) & (g.c == 0)].iloc[0]["NDCG@10"]
    ok = g[g["NDCG@10"] >= 0.98 * base]
    best = ok.sort_values(["Any flag %", "NDCG@10"], ascending=[True, False]).iloc[0]
    return (float(best.a), float(best.b), float(best.c)), g, base


cands = {}
for lam in LAMS:
    cands[lam] = flagged(lam)
    log(f"lam {lam:g}: candidates ready")

# ------------------------------------------------------------------ stage 0: rule check
pick0, grid0, base0 = rule(cands[100.0], "75_25", val)
log(f"stage 0: Juan's rule on all players picks a/b/c = {pick0} (Juan: {JUAN_ABC})")
md("# M2 tuning report (report only: defaults stay Juan's)\n")
md(f"Players: tune {len(TUNE):,} / check {len(CHECK):,} (fixed hash of player_id). Model trained on train_full parts.")
md("Scored with Juan's `team_eval/evaluate.score()`, evaluation setup.\n")
md("## Stage 0: rule check\n")
md(f"Juan's a/b/c rule on all validation players, 75/25, lam=100 picks **{pick0}**; Juan's rerank_choice.json: {JUAN_ABC}. "
   + ("Match: the rule is applied as Juan applied it." if pick0 == JUAN_ABC else "**MISMATCH: investigate before trusting stage 2.**"))

# ------------------------------------------------------------------ stage 1: lam
gkf = GroupKFold(n_splits=5)
rows, per = [], {}
for lam in LAMS:
    o, pp = ev(cands[lam], "80_20", JUAN_ABC, val_tune)
    per[lam] = pp
    rows.append({"lam": lam, "NDCG@10": o["NDCG@10"], "HitRate@10 (any)": o["HitRate@10 (any)"], "Any flag %": o["Any flag %"],
                 "Regret %": o["Regret %"], "Coverage@10": o["Coverage@10"]})
s1 = pd.DataFrame(rows)
ref = per[100.0]
spread = []
for lam in LAMS:
    d = (per[lam] - ref).dropna()
    folds = [d.iloc[te].mean() for _, te in gkf.split(d, groups=d.index)]
    spread.append({"lam": lam, "diff vs 100": d.mean(), "fold std": np.std(folds, ddof=1), "folds better": sum(f > 0 for f in folds)})
s1 = s1.merge(pd.DataFrame(spread), on="lam")
best_lam = float(s1.sort_values("NDCG@10", ascending=False).iloc[0].lam)
log(f"stage 1: best lam on tune players = {best_lam:g}")
md("\n## Stage 1: lam (tune players, 80/20, Juan's a/b/c)\n")
md(s1.to_markdown(index=False, floatfmt=".4f"))
md(f"\nBest by NDCG@10: **lam = {best_lam:g}** (Juan's default: 100). 'folds better' = of 5 GroupKFold player folds, how many favour it over 100.")

# ------------------------------------------------------------------ stage 2: a/b/c
md("\n## Stage 2: a/b/c with Juan's rule (tune players, 75/25)\n")
picks = {}
for lam in sorted({100.0, best_lam}):
    p, g, base = rule(cands[lam], "75_25", val_tune)
    picks[lam] = p
    g.sort_values("Any flag %").to_csv(OUT / f"grid_lam{int(lam)}.csv", index=False)
    md(f"- lam {lam:g}: rule picks a/b/c = **{p}** (plain EASE NDCG@10 {base:.4f}); full grid in m2_out/grid_lam{int(lam)}.csv")
    log(f"stage 2: lam {lam:g} -> a/b/c {p}")
tuned = (best_lam, picks[best_lam])

# ------------------------------------------------------------------ stage 3: check players
md("\n## Stage 3: check players (never used above), Juan's defaults vs tuned\n")
md(f"Juan: lam 100, a/b/c {JUAN_ABC}.  Tuned: lam {tuned[0]:g}, a/b/c {tuned[1]}.\n")
rows = []
rng = np.random.default_rng(0)
for h in HOLDOUTS:
    oj, pj = ev(cands[100.0], h, JUAN_ABC, val_check)
    ot, pt = ev(cands[tuned[0]], h, tuned[1], val_check)
    d = (pt - pj).dropna().to_numpy()
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    rows.append({"holdout": h, "players": oj["players"], "Juan NDCG@10": oj["NDCG@10"], "tuned NDCG@10": ot["NDCG@10"],
                 "diff": d.mean(), "95% CI": f"{np.percentile(boots, 2.5):+.4f} to {np.percentile(boots, 97.5):+.4f}",
                 "Juan any flag %": oj["Any flag %"], "tuned any flag %": ot["Any flag %"]})
s3 = pd.DataFrame(rows)
md(s3.to_markdown(index=False, floatfmt=".4f"))
(OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
log("wrote m2_out/report.md")
print("\n".join(report))
