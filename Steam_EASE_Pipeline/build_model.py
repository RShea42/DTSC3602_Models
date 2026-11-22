"""Train the pipeline with Juan's defaults and assemble serve_files/ (everything the app and Modal need).

  serve_files/ease_safe_pipeline.joblib   fitted pipeline (trained here from the team-eval train_full parts)
  serve_files/games.parquet               Juan's app catalog (data_clean/games.parquet): app filters, cards, search
  serve_files/team_eval_games.parquet     the team catalog: safety flags, as scored
  serve_files/items.parquet               playtime quantiles (enjoyment for "Try something new" and favourites)
Run: PYTHONUTF8=1 <ease-pipeline python> build_model.py   (~1.5 min)
"""
import shutil
import time
from pathlib import Path

import joblib

from steamrec import make_pipeline
from steamrec.data import load_train

from paths import EASE_FILES, SERVE, TE
HERE = Path(__file__).resolve().parent
FILES = EASE_FILES
OUT = SERVE
OUT.mkdir(exist_ok=True)
t0 = time.time()

shutil.copy2(FILES / "data_clean" / "games.parquet", OUT / "games.parquet")
shutil.copy2(TE / "games.parquet", OUT / "team_eval_games.parquet")
shutil.copy2(FILES / "model" / "artifacts" / "items.parquet", OUT / "items.parquet")
pipe = make_pipeline(OUT / "team_eval_games.parquet", OUT / "games.parquet")      # every default is Juan's
pipe.fit(load_train(TE))
joblib.dump(pipe, OUT / "ease_safe_pipeline.joblib")
print(f"built serve_files/ in {time.time() - t0:.0f}s: "
      + ", ".join(f"{p.name} {p.stat().st_size / 1e6:.0f} MB" for p in sorted(OUT.iterdir())))
