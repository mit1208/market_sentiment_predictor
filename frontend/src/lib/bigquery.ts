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
    FROM \`${process.env.GCP_PROJECT_ID}.${datasetId}.user_prediction_log\` l
    LEFT JOIN \`${process.env.GCP_PROJECT_ID}.${datasetId}.live_feature_snapshots\` snap
      ON l.request_id = snap.request_id
      AND l.ticker = snap.ticker
    LEFT JOIN (
      SELECT
        deliberation_id,
        request_id,
        ticker,
        final_direction,
        final_confidence,
        consensus,
        created_at
      FROM \`${process.env.GCP_PROJECT_ID}.${datasetId}.swarm_deliberation_summary\`
      QUALIFY ROW_NUMBER() OVER (PARTITION BY request_id, ticker ORDER BY created_at DESC) = 1
    ) swarm
      ON l.request_id = swarm.request_id
      AND l.ticker = swarm.ticker
    WHERE l.user_email = @user_email
    QUALIFY ROW_NUMBER() OVER (PARTITION BY l.request_id, l.ticker ORDER BY snap.created_at DESC) = 1
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

export async function getDailyUsage(email: string): Promise<{ prediction_count: number }> {
  const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";
  const query = `
    SELECT COALESCE(prediction_count, 0) AS prediction_count
    FROM \`${process.env.GCP_PROJECT_ID}.${datasetId}.user_daily_usage\`
    WHERE user_email = @user_email
      AND usage_date = CURRENT_DATE()
    LIMIT 1
  `;

  const [rows] = await bigquery.query({
    query,
    params: { user_email: email },
  });

  if (rows.length === 0) {
    return { prediction_count: 0 };
  }

  return { prediction_count: Number(rows[0].prediction_count) || 0 };
}

export async function incrementDailyUsage(email: string, count: number = 1): Promise<number> {
  const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";
  const query = `
    MERGE \`${process.env.GCP_PROJECT_ID}.${datasetId}.user_daily_usage\` AS target
    USING (SELECT @user_email AS user_email, CURRENT_DATE() AS usage_date) AS source
      ON target.user_email = source.user_email AND target.usage_date = source.usage_date
    WHEN MATCHED THEN
      UPDATE SET
        prediction_count = target.prediction_count + @count,
        last_predicted_at = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
      INSERT (user_email, usage_date, prediction_count, last_predicted_at)
      VALUES (source.user_email, source.usage_date, @count, CURRENT_TIMESTAMP());
  `;

  await bigquery.query({
    query,
    params: { user_email: email, count },
  });

  const usage = await getDailyUsage(email);
  return usage.prediction_count;
}
