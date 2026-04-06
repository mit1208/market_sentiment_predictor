import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_SWARM_API_URL = "https://swarm-service-xxm6sesglq-uc.a.run.app";

function getSwarmUrl(path: string): string {
  const baseUrl = process.env.SWARM_API_URL?.trim() || DEFAULT_SWARM_API_URL;
  if (!baseUrl) {
    return "";
  }
  return `${baseUrl.replace(/\/$/, "")}${path}`;
}

export async function GET(
  _req: NextRequest,
  context: { params: Promise<{ id: string }> },
) {
  const swarmBase = process.env.SWARM_API_URL?.trim() || DEFAULT_SWARM_API_URL;
  if (!swarmBase) {
    return NextResponse.json(
      { error: "Swarm service is not configured." },
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
