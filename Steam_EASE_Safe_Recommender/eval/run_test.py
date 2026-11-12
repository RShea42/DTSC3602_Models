"""Score the TEST set once with every tuned model (eval/PROTOCOL.md). Refuses to run a second time.

Fits each model on train players with its validation-chosen settings, folds in each test player's visible games,
ranks the full recommendable catalog, and scores the hidden enjoyed games. Also builds the hybrid's
"Try something new" row (same rule as the live app) for Layer 3. Statistics: hybrid vs each other model,
paired t-test + randomization test + 95% bootstrap CI over players, Holm-corrected.
Run: OPENBLAS_NUM_THREADS=4 PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python eval/run_test.py
Out: eval/eval_report.md, eval/cache/test_results.parquet (per player x model, with top-10 lists)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import bucket, coverage, evaluate, load, train_matrix  # noqa: E402
from models import ALS, EASE, Hybrid, ItemKNN, MostPopular  # noqa: E402

C = HERE / "cache"
LOCK = C / "test_results.parquet"
RUN2 = C / "run2.lock"   # PROTOCOL section 8, change 4: one disclosed second scoring, run 1 archived in eval/run1/
if LOCK.exists():
    if os.environ.get("STEAMREC_RUN2") != "1" or RUN2.exists() or not (HERE / "run1" / "cache" / "test_results.parquet").exists():
        raise SystemExit("The test set has already been scored. Results: eval/eval_report.md")
    RUN2.write_text("run 2 started\n")
best = json.load(open(C / "best_params.json"))
need = {"MostPopular", "item-kNN", "EASE", "ALS", "Hybrid", "Hybrid-EASE"}
rows = [json.loads(l) for l in open(C / "tuning_rows.jsonl")]
val = {h: max(r["ndcg10"] for r in rows if r["model"] == h) for h in ("Hybrid", "Hybrid-EASE") if any(r["model"] == h for r in rows)}
if set(best) != need:
    raise SystemExit(f"tuning incomplete: have {sorted(best)}")

U, rec, test = load("test")
X = train_matrix()
OURS = max(val, key=val.get)                        # PROTOCOL section 8, change 3: the validation winner is "ours"
(C / "ours.txt").write_text(OURS)
print(f"test players {len(test):,}; ours = {OURS} (validation {val})", flush=True)

# content embeddings + tags over the universe (hybrid content side; "Try something new")
ids = np.load(HERE.parent / "model" / "artifacts" / "game_embedding_ids.npy")
emb = np.load(HERE.parent / "model" / "artifacts" / "game_embeddings.npy")
E = np.zeros((len(U), emb.shape[1]), np.float32)
pos = pd.Series(np.arange(len(U)), index=U).reindex(ids); ok = pos.notna().to_numpy()
E[pos[ok].astype(int).to_numpy()] = emb[ok]
g = pd.read_parquet("C:/Users/juant/data_clean/games.parquet", columns=["app_id", "tags", "genres"]).set_index("app_id").reindex(U)
TAGS = [list(t)[:3] if t is not None and len(t) else (list(ge)[:3] if ge is not None else []) for t, ge in zip(g.tags, g.genres)]
np.save(C / "universe_tags.npy", np.array(["|".join(t) for t in TAGS], dtype=object), allow_pickle=True)

results, t0 = [], time.time()
bases = {}
for name, cls in [("MostPopular", MostPopular), ("item-kNN", ItemKNN), ("EASE", EASE), ("ALS", ALS)]:
    m = cls(**best[name]).fit(X)
    bases[name] = m
    d = evaluate(m, test, rec); d["model"] = name; results.append(d)
    print(f"{name}: NDCG@10 {d[d.n_visible >= 5].ndcg10.mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)
hybs = {"Hybrid": Hybrid(bases["ALS"], E, **best["Hybrid"]), "Hybrid-EASE": Hybrid(bases["EASE"], E, **best["Hybrid-EASE"])}
for name, h in hybs.items():
    d = evaluate(h, test, rec); d["model"] = name; results.append(d)
    print(f"{name}: NDCG@10 {d[d.n_visible >= 5].ndcg10.mean():.4f}", flush=True)
hyb = hybs[OURS]


# ---- "Try something new": the hybrid's top 400, kept only when its tags are new to the player (live-app rule)
class TryNew:
    def score(self, batch):
        S = hyb.score(batch)
        for r, p in enumerate(batch):
            prof = {}
            for u, gg in zip(p["vis_u"], p["vis_g"]):
                for t in TAGS[u]:
                    prof[t] = prof.get(t, 0) + gg
            tot = sum(prof.values()) or 1
            s = S[r].copy(); s[~rec] = -np.inf; s[np.setdiff1d(p["owned_u"], p["hid_u"])] = -np.inf
            order = np.argsort(-s)[:400]
            new = [j for j in order if TAGS[j] and any(prof.get(t, 0) / tot < 0.02 for t in TAGS[j])]
            keep = np.zeros(S.shape[1], bool); keep[new] = True
            S[r, ~keep] = -np.inf
            S[r, keep] = S[r, keep] + 1000         # keep the hybrid order among the kept games
        return S


d = evaluate(TryNew(), test, rec); d["model"] = "Hybrid: Try something new"; results.append(d)
res = pd.concat(results, ignore_index=True)
res["bucket"] = bucket(res.n_visible)
res.to_parquet(LOCK)                                # from here on, the test set counts as used

# ------------------------------------------------------------------ report
main = res[res.n_visible >= 5]
models = ["MostPopular", "item-kNN", "EASE", "ALS", "Hybrid", "Hybrid-EASE"]
summ = []
for m in models + ["Hybrid: Try something new"]:
    x = main[main.model == m]
    summ.append({"model": m, "players": len(x), "NDCG@10": x.ndcg10.mean(), "graded NDCG@10": x.ndcg10_graded.mean(),
                 "Recall@20": x.recall20.mean(), "long-tail Recall@20": x.tail_recall20.mean(),
                 "Coverage@10": coverage(x, rec), "avg popularity of picks": x.arp10.mean()})
summ = pd.DataFrame(summ)

piv = main.pivot_table(index=["src", "user"], columns="model", values="ndcg10")
rng = np.random.default_rng(0)
tests = []
for m in [x for x in models if x != OURS]:
    diff = (piv[OURS] - piv[m]).dropna().to_numpy()
    t_p = stats.ttest_rel(piv[OURS], piv[m], nan_policy="omit").pvalue
    flips = rng.choice([-1, 1], size=(10000, len(diff)))
    r_p = (np.abs((flips * diff).mean(1)) >= abs(diff.mean())).mean()
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    tests.append({f"{OURS} vs": m, "mean difference": diff.mean(), "relative": diff.mean() / piv[m].mean(),
                  "95% CI": f"{np.percentile(boots, 2.5):+.4f} to {np.percentile(boots, 97.5):+.4f}",
                  "t-test p": t_p, "randomization p": r_p})
tests = pd.DataFrame(tests).sort_values("t-test p").reset_index(drop=True)
k = len(tests)
tests["Holm-adjusted p"] = np.maximum.accumulate([min(1, p * (k - i)) for i, p in enumerate(tests["t-test p"])])

by_src = res[res.n_visible >= 5].pivot_table(index="model", columns="src", values="ndcg10").reindex(models + ["Hybrid: Try something new"])
by_bucket = res.pivot_table(index="model", columns="bucket", values="ndcg10", observed=True).reindex(models + ["Hybrid: Try something new"])
counts = res[res.model == OURS].groupby(["src", "bucket"], observed=True).size().unstack(fill_value=0)
beats = (by_bucket.loc[OURS] > by_bucket.loc["MostPopular"]).all()

md = ["# Layer 2: offline evaluation on the test set (scored once)\n",
      f"Protocol: `eval/PROTOCOL.md`. Settings chosen on validation only: `eval/tuning.md`. Ours = **{OURS}** "
      f"(higher validation NDCG@10 of the two hybrids: {val}). Test players: "
      f"{res[res.model == OURS].shape[0]:,} with at least one recommendable hidden enjoyed game; main results use the "
      f"{(res[res.model == OURS].n_visible >= 5).sum():,} with 5+ visible games.\n",
      "## Main results (5+ visible games)\n", summ.round(4).to_markdown(index=False),
      f"\n## Statistics: {OURS} vs each model (NDCG@10, paired over players)\n", tests.round(4).to_markdown(index=False),
      "\n## NDCG@10 by source (5+ visible games)\n", by_src.round(4).to_markdown(),
      "\n## NDCG@10 by visible library size (all test players)\n", by_bucket.round(4).to_markdown(),
      f"\n{OURS} beats MostPopular in every library-size bucket: **{'yes' if beats else 'NO'}**.\n",
      "\nPlayers per source and bucket:\n", counts.to_markdown()]
(HERE / "eval_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
