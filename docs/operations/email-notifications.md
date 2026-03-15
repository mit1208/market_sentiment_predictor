# Email Notifications

This project already sends an HTTP webhook after AutoML feature ranking is complete and ready for review. The simplest way to turn that into email is a Google Apps Script web app.

The relay template is here:

- [feature-rank-email-relay.gs](/Users/mitpatel/Desktop/code-exam/market_sentiment_predictor/docs/operations/feature-rank-email-relay.gs)

It is preconfigured to email:

- `mitbpatel128@gmail.com`

## Setup

1. Open [script.google.com](https://script.google.com/) while signed in as `mitbpatel128@gmail.com`.
2. Create a new Apps Script project.
3. Replace the default file contents with the contents of [feature-rank-email-relay.gs](/Users/mitpatel/Desktop/code-exam/market_sentiment_predictor/docs/operations/feature-rank-email-relay.gs).
4. Change `EXPECTED_TOKEN` to a random secret string.
5. Click `Deploy` -> `New deployment`.
6. Choose `Web app`.
7. Set:
   - Execute as: `Me`
   - Who has access: `Anyone`
8. Deploy and authorize `MailApp` when prompted.
9. Copy the web app URL.
10. Set the repo env var:

```bash
FEATURE_RANK_NOTIFICATION_WEBHOOK="https://script.google.com/macros/s/YOUR_DEPLOYMENT_ID/exec?token=YOUR_RANDOM_SECRET"
```

11. Redeploy the pipeline jobs:

```bash
make deploy-pipeline
```

## What Happens

This project sends notifications for two key ML pipeline events:

1. **AutoML feature ranking completes**: When `make finalize-feature-ranking` finishes, it notifies that features are ready for PR-style review.
2. **Model training completes**: When `make train` finishes, it notifies with the final accuracy and holdout metrics, and whether the model was promoted to live.

The Apps Script relay converts these JSON payloads into emails and sends them to `mitpatel128@gmail.com`.

The email includes:

- run ID
- candidate feature version
- candidate and selected feature counts
- selected feature names
- review status
- report URI
- contract URI

## How To Verify

1. Run:

```bash
make finalize-feature-ranking
```

2. Check Gmail inbox for a subject like:

```text
[Market Sentiment] AutoML feature ranking ready: fr-...
```

3. Check Cloud Run logs for the finalize job:

```bash
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="market-sentiment-feature-rank-finalize"' --limit=50
```

4. Check BigQuery notification status:

```sql
SELECT
  run_id,
  execution_status,
  review_status,
  JSON_VALUE(contract_json, '$.notification.status') AS notification_status,
  JSON_VALUE(contract_json, '$.notification.sent_at') AS notification_sent_at,
  JSON_VALUE(contract_json, '$.notification.error_message') AS notification_error
FROM `mkt-sent-friend-260313-225751.market_sentiment_predictor.feature_selection_runs`
ORDER BY generated_at DESC
LIMIT 10;
```

## Notes

- Apps Script sends email as the Google account that owns the script.
- The webhook token is the only authentication on this simple relay, so use a strong random value.
- If you redeploy the Apps Script and get a new URL, update `FEATURE_RANK_NOTIFICATION_WEBHOOK` and rerun `make deploy-pipeline`.
