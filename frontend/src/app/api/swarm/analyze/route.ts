import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const DEFAULT_SWARM_API_URL = "https://swarm-service-xxm6sesglq-uc.a.run.app";
const FREE_TICKER_LIMIT = 3;

function getSwarmUrl(path: string): string {
  const baseUrl = process.env.SWARM_API_URL?.trim() || DEFAULT_SWARM_API_URL;
  if (!baseUrl) {
    return "";
  }
  return `${baseUrl.replace(/\/$/, "")}${path}`;
}

export async function POST(req: NextRequest) {
  const swarmUrl = getSwarmUrl("/swarm/analyze");
  if (!swarmUrl) {
    return NextResponse.json(
      { error: "Swarm service is not configured." },
      { status: 500 },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  let upstreamBody: unknown = body;
  if (body && typeof body === "object" && !Array.isArray(body)) {
    const payload = { ...(body as Record<string, unknown>) };
    const submittedTickers = Array.isArray(payload.tickers)
      ? payload.tickers.filter((ticker): ticker is string => typeof ticker === "string" && ticker.trim().length > 0)
      : [];

    if (submittedTickers.length > FREE_TICKER_LIMIT) {
      return NextResponse.json(
        {
          error: `Deep Analysis is limited to ${FREE_TICKER_LIMIT} tickers at a time.`,
          ticker_limit: FREE_TICKER_LIMIT,
        },
        { status: 403 },
      );
    }

    delete payload.tickers;
    upstreamBody = payload;
  }

  const upstream = await fetch(swarmUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(upstreamBody),
    cache: "no-store",
  });

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text();
    return NextResponse.json(
      { error: text || `Swarm analyze failed with HTTP ${upstream.status}.` },
      { status: upstream.status || 502 },
    );
  }

  const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>();
  const writer = writable.getWriter();
  const reader = upstream.body.getReader();

  void (async () => {
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (value) {
          await writer.write(value);
        }
      }
    } catch (error) {
      console.error("Swarm SSE proxy failed", error);
    } finally {
      await writer.close();
      reader.releaseLock();
    }
  })();

  return new Response(readable, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "Content-Encoding": "none",
      "X-Accel-Buffering": "no",
    },
  });
}
