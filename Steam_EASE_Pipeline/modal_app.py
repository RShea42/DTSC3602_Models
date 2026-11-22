"""Deploy the model API (app/api.py) on Modal.

The fitted pipeline and catalogs live in a Modal Volume (uploaded once from serve_files/, never retrained on Modal);
the image pins the same library versions as the local environment (requirements.txt), so local and deployed
outputs can match exactly (m5_verify.py).

One-time:  modal volume create steam-ease-safe-files
           modal volume put steam-ease-safe-files serve_files/ /
Deploy:    modal deploy modal_app.py      ->  https://<workspace>--steam-ease-safe-api.modal.run
Cold start: no container runs while idle; the first request after a quiet spell loads the model.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# Read only where the image is built (your machine). Modal also imports this file inside the container, where
# requirements.txt isn't present and the image is already built, so the list isn't needed there.
REQ = HERE / "requirements.txt"
PINNED = [line.strip() for line in REQ.read_text().splitlines()
          if line.strip() and not line.startswith("#")] if REQ.exists() else []

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PINNED)                       # requirements.txt: the same versions as local
         .env({"STEAMREC_FILES": "/files", "PYTHONUTF8": "1"})
         .add_local_python_source("steamrec", "app"))
files = modal.Volume.from_name("steam-ease-safe-files")
app = modal.App("steam-ease-safe")


@app.function(image=image, volumes={"/files": files}, memory=4096, timeout=120)
@modal.asgi_app()
def api():
    from app.api import app as fastapi_app
    return fastapi_app
