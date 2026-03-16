import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/lib/auth";
import { getUserHistory } from "@/lib/bigquery";

export const runtime = "nodejs";

interface ParsedFeatureSnapshot {
  prediction?: string;
  confidence?: number;
  prob_up?: number;
  [key: string]: unknown;
}

type HistoryRow = Record<string, unknown> & {
  ticker?: string | null;
  predicted_at?: unknown;
  snapshot_timestamp?: unknown;
};

function normalizeTimestamp(value: unknown): string | null {
  if (!value) return null;
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "string") return value;
  if (typeof value === "object" && "value" in value) {
    const raw = (value as { value?: unknown }).value;
    if (typeof raw === "string") return raw;
  }
  return null;
}

function getHistoryTimestampMs(row: HistoryRow): number {
  const predictedAt = normalizeTimestamp(row.predicted_at);
  const snapshotAt = normalizeTimestamp(row.snapshot_timestamp);
  const candidate = predictedAt ?? snapshotAt;
  if (!candidate) return 0;
  const ms = new Date(candidate).getTime();
  return Number.isNaN(ms) ? 0 : ms;
}

export async function GET(_req: NextRequest) {
  const session = await getServerSession(authOptions);

  const userEmail = session?.user?.email;
  if (!userEmail) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const history = await getUserHistory(userEmail);
    
    // Parse the feature_snapshot JSON strings
    const formattedHistory = history.map((row: Record<string, unknown>) => {
      let featureSnapshot: ParsedFeatureSnapshot = {};
      const raw = row.feature_snapshot;
      if (typeof raw === "string") {
        try {
          featureSnapshot = JSON.parse(raw) as ParsedFeatureSnapshot;
        } catch (e) {
          console.error("Failed to parse history feature_snapshot:", e);
        }
      } else if (raw && typeof raw === "object") {
        featureSnapshot = raw as ParsedFeatureSnapshot;
      }
      
      const predictedAt = normalizeTimestamp(row.predicted_at);
      const snapshotAt = normalizeTimestamp(row.snapshot_timestamp);

      return {
        ...row,
        feature_snapshot: featureSnapshot,
        predicted_at: predictedAt,
        snapshot_timestamp: snapshotAt,
        prediction: featureSnapshot.prediction || "Unknown",
        confidence: featureSnapshot.confidence || 0,
        prob_up: featureSnapshot.prob_up || 0,
      } satisfies HistoryRow & Record<string, unknown>;
    }) as HistoryRow[];

    const dedupedHistory = new Map<string, HistoryRow>();
    for (const row of formattedHistory) {
      const ticker = String(row.ticker ?? "");
      if (!ticker) continue;
      const existing = dedupedHistory.get(ticker);
      if (!existing || getHistoryTimestampMs(row) > getHistoryTimestampMs(existing)) {
        dedupedHistory.set(ticker, row);
      }
    }

    const orderedHistory = Array.from(dedupedHistory.values())
      .sort((a, b) => getHistoryTimestampMs(b) - getHistoryTimestampMs(a))
      .slice(0, 50);

    return NextResponse.json({ history: orderedHistory });
  } catch (error) {
    console.error("Failed to fetch user history:", error);
    return NextResponse.json({ error: "Failed to fetch history" }, { status: 500 });
  }
}
