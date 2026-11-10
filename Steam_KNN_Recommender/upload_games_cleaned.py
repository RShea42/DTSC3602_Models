"""Load games_cleaned.csv into its own Supabase table (public.games_cleaned).

Never touches public.games, which the TasteForge dashboard reads.

Usage:
  python upload_games_cleaned.py --dry-run          # show the table layout, no database connection
  python upload_games_cleaned.py                    # create/replace games_cleaned and load it
  python upload_games_cleaned.py --keep-big-columns # also load Screenshots and About the game

SUPABASE_DB_URL comes from the environment, or from Steamapp/.streamlit/secrets.toml.
"""
import argparse, io, os, re, sys, time, tomllib
from pathlib import Path

import pandas as pd
import psycopg2

HERE = Path(__file__).parent
CSV = HERE / "games_cleaned.csv"
SECRETS = Path.home() / "Steamapp" / ".streamlit" / "secrets.toml"
TABLE = "public.games_cleaned"
BIG_COLUMNS = ["Screenshots", "About the game"]  # ~320 MB of the 400 MB file
CHUNK = 20_000


def snake(name: str) -> str:
    """'Positive review %' -> 'positive_review_pct'"""
    s = name.strip().lower().replace("%", "pct")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def pg_type(dtype) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "bigint"
    if pd.api.types.is_float_dtype(dtype):
        return "double precision"
    return "text"


def db_url() -> str:
    if os.environ.get("SUPABASE_DB_URL"):
        return os.environ["SUPABASE_DB_URL"]
    if SECRETS.exists():
        url = tomllib.loads(SECRETS.read_text(encoding="utf-8-sig")).get("SUPABASE_DB_URL")
        if url:
            return url
    sys.exit(f"Set SUPABASE_DB_URL in the environment or in {SECRETS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-big-columns", action="store_true")
    args = ap.parse_args()

    drop = [] if args.keep_big_columns else BIG_COLUMNS
    # Read every row once to pick column types, so a blank late in the file can't break the load.
    sample = pd.read_csv(CSV, usecols=lambda c: c not in drop)
    if sample["AppID"].duplicated().any():
        sys.exit(f"{sample['AppID'].duplicated().sum()} duplicate AppIDs; fix those before loading.")
    cols = {c: snake(c) for c in sample.columns}
    ddl = ",\n  ".join(f'"{cols[c]}" {pg_type(sample[c].dtype)}' for c in sample.columns)
    create = f"drop table if exists {TABLE};\ncreate table {TABLE} (\n  {ddl}\n);"
    print(create)
    if args.dry_run:
        return

    conn = psycopg2.connect(db_url())
    start, total = time.time(), 0
    with conn, conn.cursor() as cur:
        cur.execute(create)
        col_list = ",".join(f'"{v}"' for v in cols.values())
        for chunk in pd.read_csv(CSV, chunksize=CHUNK, dtype={c: sample[c].dtype for c in sample.columns
                                                               if pg_type(sample[c].dtype) == "text"}):
            chunk = chunk.drop(columns=drop)
            buf = io.StringIO()
            chunk.to_csv(buf, index=False, header=False)
            buf.seek(0)
            cur.copy_expert(f"copy {TABLE} ({col_list}) from stdin with csv", buf)
            total += len(chunk)
            print(f"  {total:,} rows ({time.time() - start:.0f}s)")
        cur.execute(f"alter table {TABLE} add primary key (appid)")
        cur.execute(f"select count(*), pg_size_pretty(pg_total_relation_size('{TABLE}')) from {TABLE}")
        n, size = cur.fetchone()
        cur.execute("select pg_size_pretty(pg_database_size(current_database()))")
        print(f"Loaded {n:,} rows into {TABLE} ({size}); whole database is now {cur.fetchone()[0]}.")


if __name__ == "__main__":
    main()
