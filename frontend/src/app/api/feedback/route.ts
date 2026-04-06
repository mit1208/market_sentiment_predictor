import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { getBigQueryClient } from "@/lib/bigquery";
import { randomUUID } from "crypto";

const VALID_FEEDBACK_TYPES = ["prediction_rating", "market_sentiment", "general"] as const;
const VALID_DIRECTIONS = ["Up", "Down"] as const;
const VALID_SENTIMENTS = ["bullish", "bearish", "neutral"] as const;

function isValidDirection(value: unknown): value is "Up" | "Down" {
  return typeof value === "string" && (VALID_DIRECTIONS as readonly string[]).includes(value);
}

function isValidSentiment(value: unknown): value is "bullish" | "bearish" | "neutral" {
  return typeof value === "string" && (VALID_SENTIMENTS as readonly string[]).includes(value);
}

function isValidRating(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 5;
}

export async function POST(request: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.email) {
    return NextResponse.json({ error: "Sign in to submit feedback." }, { status: 401 });
  }

  let body: Record<string, unknown>;
  try {
    body = (await request.json()) as Record<string, unknown>;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const feedbackType = body.feedback_type;
  if (
    typeof feedbackType !== "string" ||
    !(VALID_FEEDBACK_TYPES as readonly string[]).includes(feedbackType)
  ) {
    return NextResponse.json(
      { error: `feedback_type must be one of: ${VALID_FEEDBACK_TYPES.join(", ")}` },
      { status: 400 },
    );
  }

  const ticker = typeof body.ticker === "string" ? body.ticker.trim().toUpperCase() : null;
  if (!ticker) {
    return NextResponse.json({ error: "ticker is required." }, { status: 400 });
  }

  const feedbackId = randomUUID();
  const bigquery = getBigQueryClient();
  const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";

  const row: Record<string, unknown> = {
    feedback_id: feedbackId,
    user_email: session.user.email,
    ticker,
    request_id: typeof body.request_id === "string" ? body.request_id : null,
    feedback_type: feedbackType,
    model_prediction: isValidDirection(body.model_prediction) ? body.model_prediction : null,
    user_prediction: isValidDirection(body.user_prediction) ? body.user_prediction : null,
    confidence_rating: isValidRating(body.confidence_rating) ? body.confidence_rating : null,
    accuracy_rating: isValidRating(body.accuracy_rating) ? body.accuracy_rating : null,
    sentiment_label: isValidSentiment(body.sentiment_label) ? body.sentiment_label : null,
    comment:
      typeof body.comment === "string" && body.comment.trim().length > 0
        ? body.comment.trim().slice(0, 2000)
        : null,
    submitted_at: bigquery.timestamp(new Date()),
    predicted_at:
      typeof body.predicted_at === "string" ? bigquery.timestamp(new Date(body.predicted_at)) : null,
    model_confidence:
      typeof body.model_confidence === "number" && Number.isFinite(body.model_confidence)
        ? body.model_confidence
        : null,
  };

  try {
    await bigquery.dataset(datasetId).table("user_feedback").insert(row);
  } catch (err: unknown) {
    console.error("BigQuery insert error:", err);
    return NextResponse.json({ error: "Failed to save feedback." }, { status: 500 });
  }

  return NextResponse.json({
    feedback_id: feedbackId,
    message: "Feedback submitted successfully.",
  });
}

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user?.email) {
    return NextResponse.json({ error: "Sign in to view feedback." }, { status: 401 });
  }

  const bigquery = getBigQueryClient();
  const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";

  try {
    const query = `
      SELECT
        feedback_id,
        ticker,
        request_id,
        feedback_type,
        model_prediction,
        user_prediction,
        confidence_rating,
        accuracy_rating,
        sentiment_label,
        comment,
        submitted_at,
        predicted_at,
        model_confidence
      FROM \`${process.env.GCP_PROJECT_ID}.${datasetId}.user_feedback\`
      WHERE user_email = @user_email
      ORDER BY submitted_at DESC
      LIMIT 50
    `;

    const [rows] = await bigquery.query({
      query,
      params: { user_email: session.user.email },
    });

    return NextResponse.json({ feedback: rows });
  } catch (err: unknown) {
    console.error("BigQuery query error:", err);
    return NextResponse.json({ error: "Failed to load feedback." }, { status: 500 });
  }
}
