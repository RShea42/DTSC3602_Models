"""Where the data and Juan's code are. Every script imports from here, so a new machine changes one place.

Override with environment variables; otherwise the first folder that exists is used:
  STEAMREC_EASE_FILES   the downloaded data (README: "Data"): data_clean/, model/artifacts/
                        default: ../ease_files or ../../ease_files
  STEAMREC_JUAN_REPO    Juan's Steam_EASE_Safe_Recommender folder (evaluator, original app) for the checks
                        default: the sibling folder in this repo
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _pick(env, *candidates):
    if os.environ.get(env):
        return Path(os.environ[env])
    return next((c for c in candidates if c.exists()), candidates[0])


EASE_FILES = _pick("STEAMREC_EASE_FILES", HERE.parent / "ease_files", HERE.parent.parent / "ease_files")
JUAN = _pick("STEAMREC_JUAN_REPO", HERE.parent / "Steam_EASE_Safe_Recommender")
DATA = EASE_FILES / "data_clean"                     # Juan's app catalog: data_clean/games.parquet
TE = DATA / "team_eval"                              # team package: games, validation, train/ parts
ARTIFACTS = EASE_FILES / "model" / "artifacts"       # Juan's ease.npz, items.parquet
SERVE = HERE / "serve_files"                         # what build_model.py makes and Modal serves
