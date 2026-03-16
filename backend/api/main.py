"""Cloud Run serving container — FastAPI with live-feature transparency."""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.ml.inference.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

_predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model from GCS once when the container starts."""
    log.info("Container starting — loading model …")
    try:
        _predictor.load()
        log.info("Model loaded. Ready to serve.")
    except Exception as exc:
        log.error(f"Failed to load model during startup: {exc}")
        log.warning("Container will start without a model. Predictions will fail until a model is loaded.")
    yield


app = FastAPI(
    title="Market Sentiment Predictor",
    description="Predicts next-day stock direction from live market/news features and returns the exact feature snapshot used for inference.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class Instance(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Container health-check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict next-day price direction for one or more tickers.

    Example request body:
    ```json
    {
      "instances": [
        {"ticker": "AAPL"}
      ]
    }
    ```
    """
    predictions = []
    for inst in req.instances:
        try:
            result = _predictor.predict(inst.ticker)
        except ValueError as exc:
            # Ticker not found in BQ, bad input, etc.
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            log.exception(f"Prediction failed for ticker={inst.ticker}: {exc}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
        predictions.append(result)
        _predictor.register_new_ticker_async(result["ticker"])

    return PredictResponse(predictions=predictions)
