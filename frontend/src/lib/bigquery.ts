import { BigQuery } from "@google-cloud/bigquery";

function loadCredentialsFromEnv(): { client_email?: string; private_key?: string } | undefined {
  const raw = process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON;
  if (!raw) return undefined;
  try {
    return JSON.parse(raw) as { client_email?: string; private_key?: string };
  } catch {
    throw new Error("GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON.");
  }
}

export function getBigQueryClient(): BigQuery {
  const projectId = process.env.GCP_PROJECT_ID;
  const credentials = loadCredentialsFromEnv();
  return new BigQuery({
    projectId,
    credentials,
  });
}

const bigquery = getBigQueryClient();

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
    WITH latest_logs AS (
      SELECT
        user_email,
        ticker,
        request_id,
        predicted_at
      FROM \`${process.env.GCP_PROJECT_ID}.${datasetId}.user_prediction_log\`
      WHERE user_email = @user_email
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY user_email, ticker
        ORDER BY predicted_at DESC, request_id DESC
      ) = 1
    ),
    latest_snapshots AS (
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
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY request_id, ticker
        ORDER BY snapshot_timestamp DESC, created_at DESC
      ) = 1
    ),
    latest_swarm AS (
      SELECT
        deliberation_id,
        request_id,
        ticker,
        final_direction,
        final_confidence,
        consensus,
        created_at
      FROM \`${process.env.GCP_PROJECT_ID}.${datasetId}.swarm_deliberation_summary\`
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY request_id, ticker
        ORDER BY created_at DESC
      ) = 1
    )
    SELECT
      l.user_email,
      l.ticker,
      l.predicted_at,
      snap.snapshot_timestamp,
      snap.current_price,
      snap.day_change_pct,
      snap.headline_count,
      snap.sentiment_score,
      snap.feature_snapshot,
      snap.model_artifact_uri,
      COALESCE(snap.request_id, l.request_id) AS request_id,
      swarm.deliberation_id AS swarm_deliberation_id,
      swarm.final_direction AS swarm_final_direction,
      swarm.final_confidence AS swarm_final_confidence,
      swarm.consensus AS swarm_consensus,
      swarm.created_at AS swarm_created_at
    FROM latest_logs l
    LEFT JOIN latest_snapshots snap
      ON l.request_id = snap.request_id
      AND l.ticker = snap.ticker
    LEFT JOIN latest_swarm swarm
      ON l.request_id = swarm.request_id
      AND l.ticker = swarm.ticker
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

export async function getDailyUsage(
  email: string,
): Promise<{ prediction_count: number; used_tickers: string[] }> {
  const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";
  const query = `
    SELECT ARRAY_AGG(DISTINCT ticker ORDER BY ticker) AS used_tickers
    FROM \`${process.env.GCP_PROJECT_ID}.${datasetId}.user_prediction_log\`
    WHERE user_email = @user_email
      AND DATE(predicted_at) = CURRENT_DATE()
  `;

  const [rows] = await bigquery.query({
    query,
    params: { user_email: email },
  });

  const row = rows[0] as { used_tickers?: unknown } | undefined;
  const usedTickers = Array.isArray(row?.used_tickers)
    ? row.used_tickers
        .map((value) => (typeof value === "string" ? value.trim().toUpperCase() : ""))
        .filter((value): value is string => Boolean(value))
    : [];

  return {
    prediction_count: usedTickers.length,
    used_tickers: usedTickers,
  };
}
