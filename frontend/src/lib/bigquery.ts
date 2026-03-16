import { BigQuery } from "@google-cloud/bigquery";

const bigquery = new BigQuery({
  projectId: process.env.GCP_PROJECT_ID,
});

export async function getFreshSnapshots(tickers: string[], ttlSeconds: number) {
  if (tickers.length === 0) return [];

  const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";
  const query = `
    SELECT
      request_id,
      ticker,
      snapshot_timestamp,
      current_price,
      day_change_pct,
      headline_count,
      sentiment_score,
      feature_snapshot,
      model_artifact_uri
    FROM \`${process.env.GCP_PROJECT_ID}.${datasetId}.live_feature_snapshots\`
    WHERE ticker IN UNNEST(@tickers)
      AND snapshot_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @ttl SECOND)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY snapshot_timestamp DESC, created_at DESC) = 1
  `;

  const options = {
    query,
    params: { tickers, ttl: ttlSeconds },
  };

  const [rows] = await bigquery.query(options);
  return rows;
}

export async function logUserPrediction(email: string, requestId: string, ticker: string) {
  const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";
  const tableId = "user_prediction_log";

  const row = {
    user_email: email,
    request_id: requestId,
    ticker: ticker,
    predicted_at: bigquery.timestamp(new Date()),
  };

  await bigquery.dataset(datasetId).table(tableId).insert(row);
}

export async function getUserHistory(email: string) {
  const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";
  const query = `
    SELECT
      l.user_email,
      l.ticker,
      l.predicted_at,
      s.snapshot_timestamp,
      s.current_price,
      s.day_change_pct,
      s.headline_count,
      s.sentiment_score,
      s.feature_snapshot,
      s.model_artifact_uri,
      COALESCE(s.request_id, l.request_id) AS request_id
    FROM \`${process.env.GCP_PROJECT_ID}.${datasetId}.user_prediction_log\` l
    LEFT JOIN \`${process.env.GCP_PROJECT_ID}.${datasetId}.live_feature_snapshots\` s
      ON l.request_id = s.request_id
      AND l.ticker = s.ticker
    WHERE l.user_email = @user_email
    QUALIFY ROW_NUMBER() OVER (PARTITION BY l.request_id, l.ticker ORDER BY s.created_at DESC) = 1
    ORDER BY l.predicted_at DESC
    LIMIT 50
  `;

  const options = {
    query,
    params: { user_email: email },
  };

  const [rows] = await bigquery.query(options);
  return rows;
}
