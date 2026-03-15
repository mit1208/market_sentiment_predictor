import numpy as np

from backend.ml.shared.live_data import _trace


def test_trace_coerces_numpy_scalar_values():
    payload = _trace(np.array([101.5]), "yahoo", "history.close")

    assert payload == {
        "value": 101.5,
        "provider": "yahoo",
        "raw_field": "history.close",
        "fallback_used": False,
    }


def test_trace_drops_non_scalar_numpy_values():
    assert _trace(np.array([101.5, 102.0]), "yahoo", "history.close") is None
