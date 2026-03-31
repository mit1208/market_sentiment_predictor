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
  const swarmUrl = getSwarmUrl("/swarm/interview");
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
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });

  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
