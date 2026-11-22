"""Package the trained pipeline for the Supabase bucket (files there are limited to 50 MB).

Writes bucket_parts/: ease_safe_pipeline.joblib.part01.. (45 MB each), plus README.txt with the checksum and the
rejoin command. Upload the folder's contents to team-eval/ease_safe/pipeline/. Only the .joblib is new: the other
serve files (both catalogs, items.parquet) are already in the bucket (see README.md, "Data").
Run after build_model.py:  python split_for_bucket.py
"""
import hashlib

from paths import HERE, SERVE

SRC = SERVE / "ease_safe_pipeline.joblib"
OUT = HERE / "bucket_parts"
OUT.mkdir(exist_ok=True)
CHUNK = 45 * 1024 * 1024

data = SRC.read_bytes()
sha = hashlib.sha256(data).hexdigest().upper()
names = []
for i in range(0, len(data), CHUNK):
    name = f"{SRC.name}.part{i // CHUNK + 1:02d}"
    (OUT / name).write_bytes(data[i:i + CHUNK])
    names.append(name)
(OUT / "README.txt").write_text(f"""Trained EASE-safe scikit-learn pipeline (Steam_EASE_Pipeline in RShea42/DTSC3602_Models)

{len(names)} parts of {SRC.name} ({len(data) / 1e6:.0f} MB). To use it:
1. Download all parts into one folder and rejoin:
     PowerShell:   cmd /c copy /b {'+'.join(names)} {SRC.name}
     Mac/Linux:    cat {SRC.name}.part* > {SRC.name}
2. Check it (must match exactly; if not, do not load it - a .joblib file can run code when loaded):
     (Get-FileHash {SRC.name} -Algorithm SHA256).Hash
     {sha}
3. Put it in Steam_EASE_Pipeline/serve_files/ with the three catalog files (README.md, "Data").
Needs the pinned versions in requirements.txt (scikit-learn 1.9.1, numpy 2.5.4, ...) to load.
""", encoding="utf-8")
print(f"{len(names)} parts in bucket_parts/, SHA-256 {sha}")
