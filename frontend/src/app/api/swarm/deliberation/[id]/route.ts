import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

function getSwarmUrl(path: string): string {
  const baseUrl = process.env.SWARM_API_URL?.trim();
  if (!baseUrl) {
    return "";
  }
  return `${baseUrl.replace(/\/$/, "")}${path}`;
}

export async function GET(
  _req: NextRequest,
  context: { params: Promise<{ id: string }> },
) {
  const swarmBase = process.env.SWARM_API_URL?.trim();
  if (!swarmBase) {
    return NextResponse.json(
      { error: "Swarm service is not configured. Set SWARM_API_URL in frontend/.env.local." },
      { status: 500 },
    );
  }

  const { id } = await context.params;
  const upstream = await fetch(getSwarmUrl(`/swarm/deliberation/${encodeURIComponent(id)}`), {
    cache: "no-store",
  });
  const text = await upstream.text();

  return new Response(text, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
