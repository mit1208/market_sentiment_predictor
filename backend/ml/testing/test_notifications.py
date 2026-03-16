import requests as _requests_module

from backend.ml.shared.notifications import (
    build_feature_ranking_notification_payload,
    maybe_send_feature_ranking_webhook,
    maybe_send_model_training_notification,
)


def _sample_contract():
    return {
        "run_id": "fr-20260314",
        "candidate_feature_version": "cand-v1",
        "candidate_feature_count": 54,
        "selected_feature_names": ["gap_pct", "volume_vs_avg_10d", "gemini_sentiment_avg"],
        "selected_feature_count": 3,
        "review_status": "PENDING_REVIEW",
        "execution_status": "COMPLETED",
        "report_uri": "gs://bucket/report.md",
        "contract_uri": "gs://bucket/selected_features.json",
        "generated_at": "2026-03-14T14:00:00+00:00",
        "completed_at": "2026-03-14T14:30:00+00:00",
        "automl": {"training_job_resource_name": "projects/x/locations/us/trainingPipelines/y"},
    }


class _OkResponse:
    status_code = 200
    is_redirect = False
    headers: dict = {}

    def raise_for_status(self):
        return None


class _RedirectResponse:
    status_code = 302
    is_redirect = True
    headers = {"Location": "https://redirected.example.test/webhook"}

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, calls, responses):
        self._calls = calls
        self._responses = iter(responses)

    def post(self, url, *, json, timeout, allow_redirects):
        self._calls.append({"url": url, "json": json, "timeout": timeout})
        return next(self._responses)


def test_build_feature_ranking_notification_payload_has_text_and_metadata():
    payload = build_feature_ranking_notification_payload(_sample_contract())

    assert payload["event_type"] == "feature_ranking_ready_for_review"
    assert "AutoML feature ranking completed" in payload["text"]
    assert payload["selected_feature_count"] == 3
    assert payload["selected_feature_names"] == ["gap_pct", "volume_vs_avg_10d", "gemini_sentiment_avg"]


def test_maybe_send_feature_ranking_webhook_marks_contract_sent(monkeypatch):
    calls = []
    fake_session = _FakeSession(calls, [_OkResponse()])
    monkeypatch.setattr("backend.ml.shared.notifications.requests.Session", lambda: fake_session)

    notified, updated = maybe_send_feature_ranking_webhook(
        webhook_url="https://example.test/webhook",
        contract=_sample_contract(),
    )

    assert notified is True
    assert len(calls) == 1
    assert calls[0]["url"] == "https://example.test/webhook"
    assert calls[0]["json"]["run_id"] == "fr-20260314"
    assert updated["notification"]["status"] == "SENT"


def test_maybe_send_feature_ranking_webhook_follows_redirect_with_post(monkeypatch):
    calls = []
    fake_session = _FakeSession(calls, [_RedirectResponse(), _OkResponse()])
    monkeypatch.setattr("backend.ml.shared.notifications.requests.Session", lambda: fake_session)

    notified, updated = maybe_send_feature_ranking_webhook(
        webhook_url="https://example.test/webhook",
        contract=_sample_contract(),
    )

    assert notified is True
    assert len(calls) == 2
    assert calls[0]["url"] == "https://example.test/webhook"
    assert calls[1]["url"] == "https://redirected.example.test/webhook"
    assert calls[1]["json"]["run_id"] == "fr-20260314"
    assert updated["notification"]["status"] == "SENT"


def test_maybe_send_feature_ranking_webhook_skips_duplicate_notifications(monkeypatch):
    contract = _sample_contract()
    contract["notification"] = {"status": "SENT", "sent_at": "2026-03-14T15:00:00+00:00"}

    class _FailSession:
        def post(self, *_args, **_kwargs):
            raise AssertionError("webhook should not be called twice")

    monkeypatch.setattr("backend.ml.shared.notifications.requests.Session", _FailSession)
    notified, updated = maybe_send_feature_ranking_webhook(
        webhook_url="https://example.test/webhook",
        contract=contract,
    )

    assert notified is False
    assert updated["notification"]["status"] == "SENT"


def test_maybe_send_model_training_notification(monkeypatch):
    calls = []
    fake_session = _FakeSession(calls, [_OkResponse()])
    monkeypatch.setattr("backend.ml.shared.notifications.requests.Session", lambda: fake_session)

    manifest = {
        "model_name": "test_model",
        "accuracy": 0.55,
        "holdout_accuracy": 0.54,
        "promoted": True,
        "artifact_uri": "gs://bucket/model.joblib",
    }
    sent = maybe_send_model_training_notification(
        webhook_url="https://example.test/webhook",
        manifest=manifest,
    )

    assert sent is True
    assert len(calls) == 1
    assert "PROMOTED" in calls[0]["json"]["text"]
    assert calls[0]["json"]["accuracy"] == 0.55
