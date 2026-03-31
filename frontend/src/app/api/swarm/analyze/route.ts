import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const DEFAULT_SWARM_API_URL = "https://swarm-service-749244369851.us-central1.run.app";

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

  const upstream = await fetch(swarmUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    cache: "no-store",
  });

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text();
    return NextResponse.json(
      { error: text || `Swarm analyze failed with HTTP ${upstream.status}.` },
      { status: upstream.status || 502 },
    );
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
