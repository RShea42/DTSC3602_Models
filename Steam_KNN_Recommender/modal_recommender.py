"""Host the Steam Game Finder (recommender_app.py) on Modal, separately from the EDA dashboard.

One-time: put the Steam Web API key in a Modal secret (the value never goes into git):
    modal secret create steam-api-key STEAM_API_KEY2=<your key>
Before each deploy, make sure artifacts/ is current:
    python features.py --csv games_cleaned.csv
Deploy:
    modal deploy modal_recommender.py
"""
import shlex
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
PORT = 8000
CODE = ["recommender_app.py", "recommend.py", "features.py", "steam_api.py"]

image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
    str(HERE / "requirements.txt"))
for name in CODE:
    image = image.add_local_file(HERE / name, f"/root/{name}")
image = image.add_local_dir(HERE / "artifacts", "/root/artifacts")   # built by features.py

app = modal.App("steam-game-finder", image=image)


@app.function(
    secrets=[modal.Secret.from_name("steam-api-key")],
    memory=2048,                 # the catalog features take ~1 GB once loaded
    scaledown_window=15 * 60,    # stay warm 15 min after the last visitor
    timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=100)   # one container serves every visitor
@modal.web_server(PORT, startup_timeout=120)
def finder():
    cmd = (f"streamlit run /root/recommender_app.py --server.port {PORT} --server.address 0.0.0.0 "
           "--server.headless true --server.enableCORS false --server.enableXsrfProtection false "
           "--browser.gatherUsageStats false")
    subprocess.Popen(shlex.split(cmd), cwd="/root")
