"""Loading the team-eval package and turning a holdout into pipeline input."""
import glob
from pathlib import Path

import pandas as pd

HOLDOUTS = ["80_20", "75_25", "50_50"]


def load_train(team_eval_dir):
    """All train_full parts: played rows of training players (player_id, src, app_id, playtime_min, g)."""
    parts = sorted(glob.glob(str(Path(team_eval_dir) / "train" / "train_full_part*.parquet")))
    if not parts:
        raise FileNotFoundError(f"no train_full_part*.parquet in {Path(team_eval_dir) / 'train'}")
    return pd.concat([pd.read_parquet(p, columns=["player_id", "app_id", "playtime_min"]) for p in parts], ignore_index=True)


def holdout_input(val, holdout):
    """Pipeline input for one holdout: every non-hidden row (owned; played ones are the taste input).
    Hidden games are left out, so they can be recommended and count as hits."""
    v = val[~val[f"hidden_{holdout}"]]
    return v[["player_id", "app_id", "playtime_min", "played"]]


def predict_holdouts(pipe, val, holdouts=HOLDOUTS):
    """Top-k lists for every holdout, in the team's recs format (player_id, holdout, rank, app_id)."""
    out = [pipe.predict(holdout_input(val, h)).assign(holdout=h) for h in holdouts]
    return pd.concat(out, ignore_index=True)[["player_id", "holdout", "rank", "app_id"]]
