import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

interface FeatureSnapshotPayload {
  company_name?: string | null;
  quote_currency?: string | null;
  [key: string]: unknown;
}

interface PredictionPayload {
  ticker?: string;
  feature_snapshot?: FeatureSnapshotPayload | null;
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

function getPredictUrl(): string {
  return process.env.PREDICT_API_URL ?? "";
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

async function enrichPredictionPayload(payload: unknown): Promise<unknown> {
  if (!payload || typeof payload !== "object") {
    return payload;
  }

  const responsePayload = payload as PredictResponsePayload;
  if (!Array.isArray(responsePayload.predictions) || responsePayload.predictions.length === 0) {
    return payload;
  }

  const tickersNeedingProfiles = responsePayload.predictions
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
    ...responsePayload,
    predictions: responsePayload.predictions.map((prediction) => {
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
  const predictUrl = getPredictUrl();
  if (!predictUrl) {
    return NextResponse.json(
      {
        error:
          "Prediction service is not configured. Set PREDICT_API_URL in frontend/.env.local.",
      },
      { status: 500 },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid request body." },
      { status: 400 },
    );
  }

  const headers = {
    "Content-Type": "application/json",
  };

  try {
    const upstream = await fetch(predictUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const contentType = upstream.headers.get("content-type") ?? "";
    const payload = contentType.includes("application/json")
      ? await upstream.json()
      : await upstream.text();

    if (!upstream.ok) {
      const message =
        typeof payload === "object" &&
        payload &&
        "error" in payload &&
        typeof payload.error === "string"
          ? payload.error
          : `Prediction request failed with HTTP ${upstream.status}.`;

      return NextResponse.json(
        {
          error: message,
          status: upstream.status,
        },
        { status: upstream.status },
      );
    }

    const enrichedPayload = await enrichPredictionPayload(payload);
    return NextResponse.json(enrichedPayload, { status: upstream.status });
  } catch (error: unknown) {
    const details = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        error:
          "Unable to reach the prediction service. Check your network connection, endpoint configuration, and Google Cloud access.",
        details,
      },
      { status: 502 },
    );
  }
}
