import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { getDailyUsage } from "@/lib/bigquery";

export const runtime = "nodejs";

const FREE_DAILY_LIMIT = 3;

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user?.email) {
    return NextResponse.json({
      signed_in: false,
      prediction_count: 0,
      used_tickers: [],
      daily_limit: FREE_DAILY_LIMIT,
      remaining: FREE_DAILY_LIMIT,
      limit_reached: false,
    });
  }

  try {
    const usage = await getDailyUsage(session.user.email);
    const remaining = Math.max(0, FREE_DAILY_LIMIT - usage.prediction_count);

    return NextResponse.json({
      signed_in: true,
      prediction_count: usage.prediction_count,
      used_tickers: usage.used_tickers,
      daily_limit: FREE_DAILY_LIMIT,
      remaining,
      limit_reached: remaining <= 0,
    });
  } catch (err: unknown) {
    console.error("Failed to fetch usage:", err);
    // Fail open — don't block predictions if usage check fails
    return NextResponse.json({
      signed_in: true,
      prediction_count: 0,
      used_tickers: [],
      daily_limit: FREE_DAILY_LIMIT,
      remaining: FREE_DAILY_LIMIT,
      limit_reached: false,
    });
  }
}
