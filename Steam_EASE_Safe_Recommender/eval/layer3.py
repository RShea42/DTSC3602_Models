"""Layer 3: value to the player (eval/PROTOCOL.md section 7). Reads the saved test top-10 lists; no new scoring.

Regret@10   among top-10 picks the player went on to play (hidden played games), share played < 120 min.
            Regret avoided = MostPopular regret - model regret (bootstrap CI over players).
            Owned-never-played games can't be picked (owned games are excluded), so that case does not arise.
Exploration tag distance of picks from the player's tag profile (1 - cosine); serendipity vs MostPopular
            = hits in a list that MostPopular's list does not contain / size of that difference.
            "Try something new" must raise tag distance and keep NDCG@10 within 20% of "For you" (the hybrid).
Value       hours played per dollar of regular price over hit picks (median); free games separately:
            share of picked-and-played free games played past 2 h and 10 h. Regular price = max(2023 regular,
            2026 crawl) because the cleaned price column can hold sale prices (data note, concern 6).
Run: PYTHONUTF8=1 C:/Users/juant/.venvs/steamrec-ml/Scripts/python eval/layer3.py
Out: eval/layer3_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import load  # noqa: E402

C = HERE / "cache"
res = pd.read_parquet(C / "test_results.parquet")
res = res[res.n_visible >= 5]
U, rec, test = load("test")
P = {(p["src"], p["user"]): p for p in test}
TAGS = [t.split("|") if t else [] for t in np.load(C / "universe_tags.npy", allow_pickle=True)]
g = pd.read_parquet("C:/Users/juant/data_clean/games.parquet", columns=["app_id", "price_regular", "kaggle_price_regular"])
price = g.set_index("app_id")[["price_regular", "kaggle_price_regular"]].max(axis=1).reindex(U).to_numpy()
rng = np.random.default_rng(0)
MODELS = ["MostPopular", "item-kNN", "EASE", "ALS", "Hybrid", "Hybrid-EASE", "Hybrid: Try something new"]
OURS = (C / "ours.txt").read_text().strip()     # validation winner (PROTOCOL section 8, change 3)


def tagvec(items, weights=None):
    v = {}
    for i, it in enumerate(items):
        for t in TAGS[it]:
            v[t] = v.get(t, 0) + (1 if weights is None else weights[i])
    return v


def cos(a, b):
    num = sum(a[k] * b.get(k, 0) for k in a)
    den = np.sqrt(sum(x * x for x in a.values())) * np.sqrt(sum(x * x for x in b.values()))
    return num / den if den else 0.0


rows = []
pop_lists = {(r.src, r.user): set(r.top10) for r in res[res.model == "MostPopular"].itertuples()}
for r in res.itertuples():
    p = P[(r.src, r.user)]
    top = list(r.top10)
    pt = dict(zip(p["hid_u"].tolist(), p["hid_pt"].tolist()))
    played = [t for t in top if t in pt]
    hits = set(p["hit_u"].tolist())
    prof = tagvec(p["vis_u"], p["vis_g"])
    dist = np.mean([1 - cos(tagvec([t]), prof) for t in top if TAGS[t]]) if top else np.nan
    diff = [t for t in top if t not in pop_lists[(r.src, r.user)]]
    hp = [t for t in top if t in hits]
    paid = [pt[t] / 60 / price[t] for t in hp if price[t] > 0]
    free = [pt[t] for t in played if price[t] == 0]
    rows.append({"model": r.model, "src": r.src, "user": r.user, "ndcg10": r.ndcg10,
                 "n_played_picks": len(played), "regret": np.mean([pt[t] < 120 for t in played]) if played else np.nan,
                 "tag_distance": dist, "serendipity": (len([t for t in diff if t in hits]) / len(diff)) if diff else np.nan,
                 "hours_per_dollar": paid, "free_played_min": free})
L = pd.DataFrame(rows)


def boot_diff(a, b):
    """Mean of a - b over players who have both values, with a 95% bootstrap CI."""
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    d = (j.a - j.b).to_numpy()
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)


key = ["src", "user"]
reg = L.pivot_table(index=key, columns="model", values="regret")
tab = []
for m in MODELS:
    x = L[L.model == m]
    hpd = np.concatenate([np.array(v, float) for v in x.hours_per_dollar]) if len(x) else np.array([])
    fr = np.concatenate([np.array(v, float) for v in x.free_played_min]) if len(x) else np.array([])
    row = {"model": m, "players with a played pick": int(x.regret.notna().sum()), "Regret@10": x.regret.mean(),
           "tag distance": x.tag_distance.mean(), "serendipity vs MostPopular": x.serendipity.mean(),
           "median hours per $ (paid hits)": np.median(hpd) if len(hpd) else np.nan,
           "free picks played > 2 h": (fr >= 120).mean() if len(fr) else np.nan,
           "free picks played > 10 h": (fr >= 600).mean() if len(fr) else np.nan}
    if m != "MostPopular" and m in reg:
        mu, lo, hi, n = boot_diff(reg["MostPopular"], reg[m])
        row["regret avoided vs MostPopular (pp)"] = f"{100 * mu:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f}), n={n}"
    tab.append(row)
tab = pd.DataFrame(tab)

fy, tn = L[L.model == OURS].set_index(key), L[L.model == "Hybrid: Try something new"].set_index(key)
nd_fy, nd_tn = fy.ndcg10.mean(), tn.ndcg10.mean()
dmu, dlo, dhi, _ = boot_diff(tn.tag_distance, fy.tag_distance)
within = nd_tn >= 0.8 * nd_fy
md = ["# Layer 3: value to the player (test set, 5+ visible games)\n",
      "Computed from the saved test lists (`eval/run_test.py`); the test set was not scored again.\n",
      tab.round(4).to_markdown(index=False),
      f"\n## Try something new vs For you ({OURS})\n",
      f"- NDCG@10: For you {nd_fy:.4f}, Try something new {nd_tn:.4f} ({nd_tn / nd_fy:.0%} of For you). "
      f"Pre-declared margin: within 20% -> **{'met' if within else 'NOT met'}**.",
      f"- Tag distance: +{dmu:.3f} for Try something new (95% CI {dlo:+.3f} to {dhi:+.3f}) -> "
      f"**{'raises exploration' if dlo > 0 else 'no clear increase'}**.",
      "\n## Notes\n",
      "- Regret uses only picks the player actually went on to play (their hidden games), so it measures quitting early, "
      "not ignoring a pick.",
      "- Value per dollar is a heuristic: it uses regular price, not what the player paid, and lifetime hours.",
      "- Unplayed owned games can't appear in lists (owned games are excluded), so the 'owned but never launched' case "
      "of the framework does not arise offline; the user study covers it."]
(HERE / "layer3_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
