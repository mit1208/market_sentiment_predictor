import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

interface YahooChartMeta {
  currency?: string;
  symbol?: string;
  regularMarketPrice?: number;
  previousClose?: number;
}

interface YahooChartResult {
  meta?: YahooChartMeta;
  timestamp?: number[];
  indicators?: {
    quote?: Array<{
      close?: (number | null)[];
      open?: (number | null)[];
      high?: (number | null)[];
      low?: (number | null)[];
      volume?: (number | null)[];
    }>;
  };
}

interface YahooChartResponse {
  chart?: {
    result?: YahooChartResult[];
    error?: { code: string; description: string } | null;
  };
}

export interface ChartPoint {
  date: string;
  close: number;
}

export interface ChartResponse {
  ticker: string;
  currency: string | null;
  points: ChartPoint[];
}

export async function GET(req: NextRequest): Promise<NextResponse> {
  const ticker = req.nextUrl.searchParams.get("ticker")?.trim().toUpperCase();
  if (!ticker) {
    return NextResponse.json({ error: "ticker is required" }, { status: 400 });
  }

  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1d&range=1y&includePrePost=false`;
    const res = await fetch(url, {
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
    });

    if (!res.ok) {
      return NextResponse.json({ error: `Yahoo Finance returned ${res.status}` }, { status: 502 });
    }

    const payload = (await res.json()) as YahooChartResponse;
    const result = payload.chart?.result?.[0];

    if (!result) {
      const errMsg = payload.chart?.error?.description ?? "No data returned";
      return NextResponse.json({ error: errMsg }, { status: 404 });
    }

    const timestamps = result.timestamp ?? [];
    const closes = result.indicators?.quote?.[0]?.close ?? [];
    const currency = result.meta?.currency ?? null;

    const points: ChartPoint[] = timestamps
      .map((ts, i) => {
        const close = closes[i];
        if (close === null || close === undefined || !isFinite(close)) return null;
        const date = new Date(ts * 1000).toISOString().slice(0, 10);
        return { date, close: Math.round(close * 100) / 100 };
      })
      .filter((p): p is ChartPoint => p !== null);

    return NextResponse.json({ ticker, currency, points } satisfies ChartResponse);
  } catch (err) {
    console.error("Chart fetch error:", err);
    return NextResponse.json({ error: "Failed to fetch chart data" }, { status: 500 });
  }
}
