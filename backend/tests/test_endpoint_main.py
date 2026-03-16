import importlib
import sys
import types

import pytest


def _load_endpoint_module(monkeypatch):
    predictor_module = types.ModuleType("backend.ml.inference.predictor")

    class DummyPredictor:
        def load(self):
            return None

        def predict(self, ticker):
            return {"ticker": ticker, "prediction": "Up"}

        def register_new_ticker_async(self, ticker):
            return None

    predictor_module.Predictor = DummyPredictor
    monkeypatch.setitem(sys.modules, "backend.ml.inference.predictor", predictor_module)
    sys.modules.pop("backend.api.main", None)
    return importlib.import_module("backend.api.main")


def test_health_returns_ok(monkeypatch):
    module = _load_endpoint_module(monkeypatch)

    assert module.health() == {"status": "ok"}


def test_predict_returns_predictions(monkeypatch):
    module = _load_endpoint_module(monkeypatch)
    calls = []

    def fake_predict(ticker):
        calls.append(ticker)
        return {"ticker": ticker, "prediction": "Up", "confidence": 0.91}

    monkeypatch.setattr(module._predictor, "predict", fake_predict)

    response = module.predict(
        module.PredictRequest(instances=[module.Instance(ticker="AAPL"), module.Instance(ticker="MSFT")])
    )

    assert response.predictions == [
        {"ticker": "AAPL", "prediction": "Up", "confidence": 0.91},
        {"ticker": "MSFT", "prediction": "Up", "confidence": 0.91},
    ]
    assert calls == ["AAPL", "MSFT"]


def test_predict_translates_value_error_to_422(monkeypatch):
    module = _load_endpoint_module(monkeypatch)

    def fake_predict(_ticker):
        raise ValueError("Ticker is required.")

    monkeypatch.setattr(module._predictor, "predict", fake_predict)

    with pytest.raises(module.HTTPException) as exc_info:
        module.predict(module.PredictRequest(instances=[module.Instance(ticker="")]))

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Ticker is required."


def test_predict_translates_unexpected_error_to_500(monkeypatch):
    module = _load_endpoint_module(monkeypatch)

    def fake_predict(_ticker):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(module._predictor, "predict", fake_predict)

    with pytest.raises(module.HTTPException) as exc_info:
        module.predict(module.PredictRequest(instances=[module.Instance(ticker="AAPL")]))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Prediction error: backend unavailable"
