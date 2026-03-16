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
      
      return {
        ...row,
        feature_snapshot: featureSnapshot,
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
