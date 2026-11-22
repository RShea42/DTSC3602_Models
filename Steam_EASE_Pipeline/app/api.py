"""Model API: the fitted scikit-learn pipeline served with FastAPI (deployed on Modal by modal_app.py).

  POST /predict     library in -> the pipeline's ranked picks with scores. Deterministic: no live data, so local and
                    deployed outputs can be compared exactly (m5_verify.py).
  POST /recommend   library in -> Juan's full app response (cards, "Try something new", live store prices). For apps
                    such as TasteForge, which already have the player's library from their own Steam sign-in.
  GET  /health      model loaded? which settings?
  GET  /docs        interactive documentation (FastAPI)

Local: uvicorn app.api:app --port 8001     (STEAMREC_FILES=serve_files by default)
Nothing here needs a Steam API key: callers send the library. Requests are not logged.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import engine


class Game(BaseModel):
    app_id: int = Field(..., description="Steam app id")
    playtime_min: float = Field(0, ge=0, description="lifetime minutes played; 0 = owned but not played")


class LibraryRequest(BaseModel):
    library: list[Game] = Field(..., min_length=1, max_length=50_000, description="the player's games (played and unplayed)")
    include_explicit: bool = Field(False, description="allow sexually explicit games (off in Juan's app unless the player opts in)")


class PredictRequest(LibraryRequest):
    k: int = Field(10, ge=1, le=600, description="how many picks")


class RecommendRequest(LibraryRequest):
    n: int = Field(10, ge=1, le=50, description="picks per row")
    max_price: float | None = Field(None, ge=0, description="price limit, checked against the live store price")
    country: str = Field("us", pattern=r"^[a-zA-Z]{2}$", description="store country for prices")


class Pick(BaseModel):
    rank: int
    app_id: int
    title: str
    score: float


class PredictResponse(BaseModel):
    picks: list[Pick]
    n_played_known: int = Field(description="played games the model knows (its taste input)")
    model: dict


def predict(req: PredictRequest):
    """The pipeline's output for one library: EASE-safe order (app mode, Juan's defaults), top k."""
    r = engine.get()
    library = [(g.app_id, g.playtime_min) for g in req.library if g.playtime_min > 0]
    owned = [g.app_id for g in req.library]
    rows = r.taste_rows(library)
    picks = []
    if rows:
        order, s = r.ordered(library, owned, req.include_explicit)
        picks = [Pick(rank=i + 1, app_id=int(r.app_ids[j]), title=str(r.titles.get(int(r.app_ids[j]), r.app_ids[j])), score=float(s[j]))
                 for i, j in enumerate(order[:req.k])]
    return PredictResponse(picks=picks, n_played_known=len(rows), model=model_info())


def model_info():
    p = engine.get().pipe.get_params()
    return {"pipeline": "LibraryEncoder -> EASEScorer -> SafeReranker", "games": int(len(engine.get().app_ids)),
            **{k: p[k] for k in ["encode__min_users", "ease__lam", "ease__serve_filters", "safe__a_quit", "safe__b_short", "safe__c_negative"]}}


@asynccontextmanager
async def lifespan(_):
    engine.get()                     # load the saved pipeline once per container
    yield


app = FastAPI(title="EASE-safe Steam recommender API",
              description="Juan's EASE-safe model as a fitted scikit-learn pipeline. Send a Steam library, get games to play next.",
              version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": model_info()}


@app.post("/predict", response_model=PredictResponse)
def predict_route(req: PredictRequest):
    return predict(req)


@app.post("/recommend")
def recommend_route(req: RecommendRequest):
    library = [(g.app_id, g.playtime_min) for g in req.library if g.playtime_min > 0]
    out = engine.get().recommend(library, owned=[g.app_id for g in req.library], n=req.n, max_price=req.max_price,
                                 cc=req.country.lower(), include_explicit=req.include_explicit)
    if out is None:
        raise HTTPException(400, "None of the played games in this library are in our data yet.")
    return out
