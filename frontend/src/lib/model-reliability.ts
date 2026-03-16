import { BigQuery } from "@google-cloud/bigquery";

export interface ModelReliabilityFold {
  fold: number;
  accuracy: number | null;
  auc: number | null;
  train_start_date?: string;
  train_end_date?: string;
  validation_start_date?: string;
  validation_end_date?: string;
  train_rows?: number;
  validation_rows?: number;
}

export interface ModelReliabilityTopFeature {
  feature: string;
  importance: number;
}

export interface ModelReliabilitySnapshot {
  trainingDate: string;
  version: string;
  modelType: string;
  trainingStartDate: string;
  trainingEndDate: string;
  trainingSamples: number;
  featureCount: number;
  deploymentStatus: string;
  promotionGateMetric: string;
  promotionGateValue: number | null;
  validationStrategy: string;
  nSplits: number | null;
  cvAccuracyMean: number | null;
  cvAccuracyStd: number | null;
  cvAucMean: number | null;
  cvAucStd: number | null;
  holdoutStartDate: string;
  holdoutEndDate: string;
  holdoutAccuracy: number | null;
  holdoutAuc: number | null;
  holdoutPrecision: number | null;
  holdoutRecall: number | null;
  holdoutF1: number | null;
  notes: string;
  topFeatures: ModelReliabilityTopFeature[];
  folds: ModelReliabilityFold[];
}

const PROJECT_ID =
  process.env.GCP_PROJECT_ID ??
  process.env.GOOGLE_CLOUD_PROJECT ??
  "mkt-sent-friend-260313-225751";
const DATASET_ID =
  process.env.BQ_DATASET ??
  process.env.BQ_DATASET_ID ??
  "market_sentiment_predictor";
const BIGQUERY_LOCATION = process.env.BQ_LOCATION ?? "US";

const MODEL_RELIABILITY_QUERY = `
  SELECT
    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S UTC', training_date) AS training_date,
    version,
    model_type,
    CAST(training_start_date AS STRING) AS training_start_date,
    CAST(training_end_date AS STRING) AS training_end_date,
    CAST(training_samples AS INT64) AS training_samples,
    CAST(feature_count AS INT64) AS feature_count,
    notes,
    performance_metrics,
    validation_metrics,
    test_metrics,
    TO_JSON_STRING(
      ARRAY(
        SELECT AS STRUCT feature, CAST(importance AS FLOAT64) AS importance
        FROM UNNEST(feature_importance)
        WHERE CAST(importance AS FLOAT64) > 0
        ORDER BY CAST(importance AS FLOAT64) DESC
        LIMIT 10
      )
    ) AS top_features_json
  FROM \`${PROJECT_ID}.${DATASET_ID}.model_metadata\`
  ORDER BY training_date DESC
  LIMIT 1
`;

function parseIntValue(value: string | null | undefined): number | null {
  if (!value) {
    return null;
  }

  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

export async function fetchModelReliabilitySnapshot(): Promise<ModelReliabilitySnapshot> {
  const bigquery = new BigQuery({ projectId: PROJECT_ID });
  const [rows] = await bigquery.query({
    query: MODEL_RELIABILITY_QUERY,
    useLegacySql: false,
    location: BIGQUERY_LOCATION,
  });

  const row = rows?.[0] as Record<string, string | null> | undefined;
  if (!row) {
    throw new Error("No model metadata rows found in BigQuery.");
  }

  const performanceMetrics = JSON.parse(row.performance_metrics ?? "{}") as {
    deployment_status?: string;
    promotion_gate_metric?: string;
    promotion_gate_value?: number;
  };
  const validationMetrics = JSON.parse(row.validation_metrics ?? "{}") as {
    strategy?: string;
    n_splits?: number;
    summary?: {
      accuracy_mean?: number;
      accuracy_std?: number;
      auc_mean?: number;
      auc_std?: number;
    };
    folds?: ModelReliabilityFold[];
  };
  const testMetrics = JSON.parse(row.test_metrics ?? "{}") as {
    holdout_start_date?: string;
    holdout_end_date?: string;
    accuracy?: number;
    auc?: number;
    precision?: number;
    recall?: number;
    f1?: number;
  };
  const topFeatures = JSON.parse(
    row.top_features_json ?? "[]",
  ) as ModelReliabilityTopFeature[];

  return {
    trainingDate: row.training_date ?? "n/a",
    version: row.version ?? "n/a",
    modelType: row.model_type ?? "n/a",
    trainingStartDate: row.training_start_date ?? "n/a",
    trainingEndDate: row.training_end_date ?? "n/a",
    trainingSamples: parseIntValue(row.training_samples) ?? 0,
    featureCount: parseIntValue(row.feature_count) ?? 0,
    deploymentStatus: performanceMetrics.deployment_status ?? "UNKNOWN",
    promotionGateMetric: performanceMetrics.promotion_gate_metric ?? "n/a",
    promotionGateValue: performanceMetrics.promotion_gate_value ?? null,
    validationStrategy: validationMetrics.strategy ?? "n/a",
    nSplits: validationMetrics.n_splits ?? null,
    cvAccuracyMean: validationMetrics.summary?.accuracy_mean ?? null,
    cvAccuracyStd: validationMetrics.summary?.accuracy_std ?? null,
    cvAucMean: validationMetrics.summary?.auc_mean ?? null,
    cvAucStd: validationMetrics.summary?.auc_std ?? null,
    holdoutStartDate: testMetrics.holdout_start_date ?? "n/a",
    holdoutEndDate: testMetrics.holdout_end_date ?? "n/a",
    holdoutAccuracy: testMetrics.accuracy ?? null,
    holdoutAuc: testMetrics.auc ?? null,
    holdoutPrecision: testMetrics.precision ?? null,
    holdoutRecall: testMetrics.recall ?? null,
    holdoutF1: testMetrics.f1 ?? null,
    notes: row.notes ?? "",
    topFeatures,
    folds: validationMetrics.folds ?? [],
  };
}
