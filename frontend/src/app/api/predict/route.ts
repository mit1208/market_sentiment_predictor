import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/lib/auth";
import { getFreshSnapshots, logUserPrediction, getDailyUsage, incrementDailyUsage } from "@/lib/bigquery";

export const runtime = "nodejs";

const LIVE_SNAPSHOT_FRESH_TTL_SECONDS = parseInt(process.env.LIVE_SNAPSHOT_FRESH_TTL_SECONDS || "3600", 10);
const FREE_DAILY_LIMIT = 3;

interface FeatureSnapshotPayload {
  company_name?: string | null;
  quote_currency?: string | null;
  [key: string]: unknown;
}

interface PredictionPayload {
  request_id: string;
  ticker: string;
  prediction: string;
  confidence: number;
  prob_up: number;
  as_of: string;
  predicted_at: string;
  feature_snapshot: FeatureSnapshotPayload;
  model_artifact_uri?: string | null;
  [key: string]: unknown;
}

interface PredictResponsePayload {
  predictions?: PredictionPayload[];
  [key: string]: unknown;
}

interface YahooQuoteResult {
  symbol?: string;
  shortName?: string;
  longName?: string;
  displayName?: string;
  currency?: string;
}

interface YahooQuoteResponse {
  quoteResponse?: {
    result?: YahooQuoteResult[];
  };
}

function orderPredictionsByRequestedTickers(
  predictions: PredictionPayload[],
  requestedTickers: string[],
): PredictionPayload[] {
  const rank = new Map(requestedTickers.map((ticker, index) => [ticker, index] as const));

  return [...predictions].sort((left, right) => {
    const leftRank = rank.get(left.ticker) ?? Number.MAX_SAFE_INTEGER;
    const rightRank = rank.get(right.ticker) ?? Number.MAX_SAFE_INTEGER;
    return leftRank - rightRank || left.ticker.localeCompare(right.ticker);
  });
}

function getPredictUrl(): string {
  return process.env.PREDICT_API_URL ?? "";
}

function getUpstreamErrorMessage(payload: unknown, status: number): string {
  if (payload && typeof payload === "object") {
    const obj = payload as Record<string, unknown>;
    if (typeof obj.error === "string" && obj.error) return obj.error;
    if (typeof obj.detail === "string" && obj.detail) return obj.detail;
    if (Array.isArray(obj.detail) && obj.detail.length > 0) {
      const first = obj.detail[0];
      if (first && typeof first === "object" && "msg" in first && typeof (first as { msg: unknown }).msg === "string")
        return (first as { msg: string }).msg;
    }
  }
  return `Prediction request failed with HTTP ${status}.`;
}

function normalizeOptionalText(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }

  const normalized = value.trim();
  return normalized ? normalized : null;
}

async function fetchTickerProfiles(
  tickers: string[],
): Promise<Record<string, { companyName: string | null; quoteCurrency: string | null }>> {
  const normalizedTickers = [...new Set(
    tickers
      .map((ticker) => normalizeOptionalText(ticker)?.toUpperCase() ?? null)
      .filter((ticker): ticker is string => Boolean(ticker)),
  )];

  if (normalizedTickers.length === 0) {
    return {};
  }

  try {
    const response = await fetch(
      `https://query1.finance.yahoo.com/v7/finance/quote?symbols=${encodeURIComponent(normalizedTickers.join(","))}`,
      {
        cache: "no-store",
        headers: {
          "Content-Type": "application/json",
        },
      },
    );

    if (!response.ok) {
      return {};
    }

    const payload = (await response.json()) as YahooQuoteResponse;
    const results = payload.quoteResponse?.result ?? [];

    return Object.fromEntries(
      results
        .map((item) => {
          const symbol = normalizeOptionalText(item.symbol)?.toUpperCase();
          if (!symbol) {
            return null;
          }

          const companyName =
            normalizeOptionalText(item.longName) ??
            normalizeOptionalText(item.shortName) ??
            normalizeOptionalText(item.displayName);
          const quoteCurrency = normalizeOptionalText(item.currency)?.toUpperCase() ?? null;

          return [symbol, { companyName, quoteCurrency }] as const;
        })
        .filter(
          (
            entry,
          ): entry is readonly [string, { companyName: string | null; quoteCurrency: string | null }] =>
            Boolean(entry),
        ),
    );
  } catch {
    return {};
  }
}

async function enrichPredictionPayload(payload: PredictResponsePayload): Promise<PredictResponsePayload> {
  if (!payload || typeof payload !== "object") {
    return payload;
  }

  if (!Array.isArray(payload.predictions) || payload.predictions.length === 0) {
    return payload;
  }

  const tickersNeedingProfiles = payload.predictions
    .map((prediction) => {
      const ticker = normalizeOptionalText(prediction.ticker)?.toUpperCase();
      if (!ticker) {
        return null;
      }

      const featureSnapshot =
        prediction.feature_snapshot && typeof prediction.feature_snapshot === "object"
          ? prediction.feature_snapshot
          : null;
      const hasCompanyName = Boolean(normalizeOptionalText(featureSnapshot?.company_name));
      const hasQuoteCurrency = Boolean(normalizeOptionalText(featureSnapshot?.quote_currency));

      return hasCompanyName && hasQuoteCurrency ? null : ticker;
    })
    .filter((ticker): ticker is string => Boolean(ticker));

  if (tickersNeedingProfiles.length === 0) {
    return payload;
  }

  const profiles = await fetchTickerProfiles(tickersNeedingProfiles);
  if (Object.keys(profiles).length === 0) {
    return payload;
  }

  return {
    ...payload,
    predictions: payload.predictions.map((prediction) => {
      const ticker = normalizeOptionalText(prediction.ticker)?.toUpperCase();
      if (!ticker || !profiles[ticker]) {
        return prediction;
      }

      const featureSnapshot =
        prediction.feature_snapshot && typeof prediction.feature_snapshot === "object"
          ? prediction.feature_snapshot
          : {};
      const profile = profiles[ticker];

      return {
        ...prediction,
        feature_snapshot: {
          ...featureSnapshot,
          company_name:
            normalizeOptionalText(featureSnapshot.company_name) ?? profile.companyName ?? null,
          quote_currency:
            normalizeOptionalText(featureSnapshot.quote_currency)?.toUpperCase() ??
            profile.quoteCurrency ??
            null,
        },
      };
    }),
  };
}

export async function POST(req: NextRequest) {
  const session = await getServerSession(authOptions);
  const predictUrl = getPredictUrl();

  // ── Auth gate: require sign-in ──
  if (!session?.user?.email) {
    return NextResponse.json(
      {
        error: "Sign in to run predictions. Free users get 3 predictions per day.",
        require_auth: true,
      },
      { status: 401 },
    );
  }
  
  if (!predictUrl) {
    return NextResponse.json(
      {
        error: "Prediction service is not configured. Set PREDICT_API_URL in frontend/.env.local.",
      },
      { status: 500 },
    );
  }

  let body: { instances?: { ticker: string }[]; tickers?: string[] };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  // Support both { instances: [{ ticker: "AAPL" }] } and { tickers: ["AAPL"] } formats
  const requestedTickers: string[] = body.instances
    ? body.instances
        .map((inst) => inst.ticker?.trim().toUpperCase())
        .filter((t): t is string => Boolean(t))
    : body.tickers ?? [];

  if (requestedTickers.length === 0) {
    return NextResponse.json({ predictions: [] });
  }

  // ── Rate limit: free tier = 3 predictions/day ──
  const userEmail = session.user.email;
  try {
    const usage = await getDailyUsage(userEmail);
    if (usage.prediction_count >= FREE_DAILY_LIMIT) {
      return NextResponse.json(
        {
          error: `You've used all ${FREE_DAILY_LIMIT} free predictions today. Upgrade to Pro for unlimited access.`,
          limit_reached: true,
          prediction_count: usage.prediction_count,
          daily_limit: FREE_DAILY_LIMIT,
          remaining: 0,
        },
        { status: 429 },
      );
    }
  } catch (err) {
    // Fail open — don't block predictions if usage check fails
    console.error("Usage check failed, proceeding:", err);
  }


  // 1. Check for fresh snapshots in BigQuery
  type BQSnapshot = Record<string, unknown>;
  let freshSnapshots: BQSnapshot[] = [];
  try {
    freshSnapshots = await getFreshSnapshots(requestedTickers, LIVE_SNAPSHOT_FRESH_TTL_SECONDS);
  } catch (error) {
    console.error("Failed to fetch fresh snapshots from BigQuery:", error);
  }

  const freshTickerMap = new Map(freshSnapshots.map((s) => [s.ticker as string, s]));
  const staleTickers = requestedTickers.filter((ticker) => !freshTickerMap.has(ticker));

  const finalPredictions: PredictionPayload[] = [];

  // Map fresh snapshots to the expected prediction format
  freshSnapshots.forEach((snapshot) => {
    let featureSnapshot: Record<string, unknown> = {};
    if (typeof snapshot.feature_snapshot === "string") {
      try {
        featureSnapshot = JSON.parse(snapshot.feature_snapshot) as Record<string, unknown>;
      } catch (e) {
        console.error("Failed to parse cached feature_snapshot:", e);
      }
    } else if (snapshot.feature_snapshot && typeof snapshot.feature_snapshot === "object") {
      featureSnapshot = snapshot.feature_snapshot as Record<string, unknown>;
    }

    // If the snapshot is missing prediction fields, we treat it as stale so we can re-predict and enrich it
    if (!featureSnapshot.prediction) {
      freshTickerMap.delete(snapshot.ticker as string);
      staleTickers.push(snapshot.ticker as string);
      return;
    }

    // Unwrap BQ timestamp object { value: "…" } before computing age
    const rawTs = snapshot.snapshot_timestamp;
    const tsStr =
      rawTs && typeof rawTs === "object" && "value" in rawTs
        ? String((rawTs as { value: unknown }).value)
        : rawTs instanceof Date
        ? rawTs.toISOString()
        : String(rawTs);
    const snapshotAgeSeconds = Math.max(
      0,
      Math.floor((Date.now() - new Date(tsStr).getTime()) / 1000),
    );

    // Override cache_status so the frontend shows "BQ cache" not "Live API"
    const fs: Record<string, unknown> = {
      ...featureSnapshot,
      cache_status: {
        source_mode: "bigquery_cache",
        reason: "frontend_ttl_cache",
        cache_hit: true,
        stale_served: false,
        quota_exhausted: false,
        snapshot_timestamp: tsStr,
        snapshot_age_seconds: snapshotAgeSeconds,
        snapshot_age_minutes: Math.round((snapshotAgeSeconds / 60) * 10) / 10,
        fresh_ttl_seconds: LIVE_SNAPSHOT_FRESH_TTL_SECONDS,
        fresh_ttl_minutes: Math.round((LIVE_SNAPSHOT_FRESH_TTL_SECONDS / 60) * 10) / 10,
      },
    };

    finalPredictions.push({
      request_id: snapshot.request_id as string,
      ticker: snapshot.ticker as string,
      prediction: (fs.prediction as string) || "Unknown",
      confidence: (fs.confidence as number) || 0,
      prob_up: (fs.prob_up as number) || 0,
      as_of: tsStr,
      predicted_at: tsStr,
      feature_snapshot: fs as FeatureSnapshotPayload,
      model_artifact_uri: (snapshot.model_artifact_uri as string) ?? null,
    });
  });

  // 2. Fetch stale tickers from FastAPI (forward in the instances format it expects)
  if (staleTickers.length > 0) {
    const headers = { "Content-Type": "application/json" };
    try {
      const upstream = await fetch(predictUrl, {
        method: "POST",
        headers,
        body: JSON.stringify({ instances: staleTickers.map((t) => ({ ticker: t })) }),
        cache: "no-store",
      });

      if (upstream.ok) {
        const payload = (await upstream.json()) as PredictResponsePayload;
        const enriched = await enrichPredictionPayload(payload);
        if (enriched.predictions) {
          finalPredictions.push(...enriched.predictions);
        }
      } else {
        const payload = await upstream.json();
        const message = getUpstreamErrorMessage(payload, upstream.status);
        // If some succeeded (from cache), maybe we still return what we have? 
        // For simplicity, if FastAPI fails and we have no cache, we fail.
        if (finalPredictions.length === 0) {
          return NextResponse.json({ error: message, status: upstream.status }, { status: upstream.status });
        }
      }
    } catch (error) {
      console.error("FastAPI call failed:", error);
      if (finalPredictions.length === 0) {
        return NextResponse.json({ error: "Unable to reach prediction service." }, { status: 502 });
      }
    }
  }

  // 3. Log results to BigQuery for authenticated users
  const orderedPredictions = orderPredictionsByRequestedTickers(finalPredictions, requestedTickers);

  try {
    const rowsToLog = orderedPredictions.filter(
      (p) => typeof p.request_id === "string" && p.request_id && typeof p.ticker === "string" && p.ticker,
    );
    const skipped = orderedPredictions.length - rowsToLog.length;
    if (skipped > 0) {
      console.warn(`Skipping ${skipped} prediction logs due to missing request_id or ticker.`);
    }
    await Promise.all(
      rowsToLog.map((p) => logUserPrediction(userEmail, p.request_id, p.ticker))
    );
  } catch (error) {
    console.error("Failed to log predictions to BigQuery:", error);
  }

  // 4. Increment daily usage counter
  try {
    await incrementDailyUsage(userEmail);
  } catch (error) {
    console.error("Failed to increment daily usage:", error);
  }

  return NextResponse.json({ predictions: orderedPredictions });
}
