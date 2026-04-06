import { NextResponse } from "next/server";
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

export async function GET() {
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
      const swarmCreatedAt = normalizeTimestamp(row.swarm_created_at);

      return {
        ...row,
        feature_snapshot: featureSnapshot,
        predicted_at: predictedAt,
        snapshot_timestamp: snapshotAt,
        swarm_created_at: swarmCreatedAt,
        prediction: featureSnapshot.prediction || "Unknown",
        confidence: featureSnapshot.confidence || 0,
        prob_up: featureSnapshot.prob_up || 0,
      };
    });

    return NextResponse.json({ history: formattedHistory });
  } catch (error) {
    console.error("Failed to fetch user history:", error);
    return NextResponse.json({ error: "Failed to fetch history" }, { status: 500 });
  }
}
