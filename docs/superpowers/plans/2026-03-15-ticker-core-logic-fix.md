# Ticker Core Logic Fix — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 5 bugs that break the ticker prediction flow — auth session always null, wrong BQ dataset, user_id/user_email mismatch, BQ cache hits mislabelled "Live API", and stale migration SQL.

**Architecture:** Adopt approach B (user_email, no users lookup table). Export `authOptions` from the NextAuth route so API routes can call `getServerSession(authOptions)`. `lib/bigquery.ts` is the single source of truth for all BQ queries; routes import from it. When the Next.js predict route serves a ticker from BQ TTL cache, it injects a `cache_status` override so the frontend shows "BQ cache" not "Live API".

**Tech Stack:** Next.js 16, NextAuth v4 (JWT strategy, Google OAuth), `@google-cloud/bigquery` v8, TypeScript 5. Verification: `npm run build` (type-check + build) from `frontend/`, curl for integration smoke tests.

---

## Chunk 1: Auth + BigQuery library

### Task 1: Fix `auth/[...nextauth]/route.ts` — export authOptions, simplify callbacks

**Files:**
- Modify: `frontend/src/app/api/auth/[...nextauth]/route.ts`

**What to know:**
- NextAuth v4 wraps config in `NextAuth(authOptions)` which returns a handler. Other routes need the config object, not the handler, to call `getServerSession(authOptions)`.
- With JWT strategy + Google OAuth, `token.email` is populated automatically from the Google profile — no custom jwt callback needed at all.
- The `session` callback currently injects `token.userId` onto `session.user.id`. After simplification, `token.userId` won't exist, so remove this callback entirely. NextAuth's default `session` callback correctly forwards `session.user.email` from the token.
- The `events.signIn` handler and the `jwt` callback call `getOrCreateUser()` which requires a `users` table that doesn't exist. Remove both.

- [ ] **Step 1: Rewrite the file**

Replace `frontend/src/app/api/auth/[...nextauth]/route.ts` with:

```ts
import NextAuth, { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";

export const authOptions: NextAuthOptions = {
  debug: false,
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  session: {
    strategy: "jwt",
  },
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: build succeeds with no type errors in the auth route. If it fails on `NextAuthOptions` not being exported from `next-auth`, use `import type { NextAuthOptions } from "next-auth"`.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/api/auth/\[...nextauth\]/route.ts
git commit -m "fix(auth): export authOptions, remove getOrCreateUser callbacks"
```

---

### Task 2: Fix `lib/bigquery.ts` — dataset default, user_email approach, remove dead code

**Files:**
- Modify: `frontend/src/lib/bigquery.ts`

**What to know:**
- The BigQuery constructor reads `GCP_PROJECT_ID` from env. The dataset fallback `"market_sentiment"` is wrong — the actual deployed dataset is `"market_sentiment_predictor"`.
- `getOrCreateUser()` and `logUserSignIn()` can be deleted entirely — no callers remain after Task 1.
- `logUserPrediction` currently writes `{ user_id, request_id, ticker, predicted_at }`. The BQ table has `user_email` as the column name (per `infra/terraform/user_prediction_log.json`). Change the parameter name to `email` and the field name to `user_email`.
- `getUserHistory` currently SELECTs `l.user_id` and filters `WHERE l.user_id = @user_id`. Both must change to `user_email`.

- [ ] **Step 1: Fix dataset default**

In `frontend/src/lib/bigquery.ts`, every function that reads the dataset uses a local `const datasetId = process.env.BQ_DATASET || "market_sentiment"`. Change all three occurrences to:

```ts
const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";
```

There are 5 functions that each define `datasetId` locally: `getOrCreateUser`, `getFreshSnapshots`, `logUserPrediction`, `logUserSignIn`, `getUserHistory`. Since `getOrCreateUser` and `logUserSignIn` will be deleted, fix the remaining three: `getFreshSnapshots`, `logUserPrediction`, `getUserHistory`.

> **Sequencing note:** Task 1 removes the `import { logUserSignIn, getOrCreateUser }` line from `route.ts`. Task 2 deletes those functions from `bigquery.ts`. Apply Task 1 **before** Task 2 — reversing the order will cause a TypeScript build failure between commits (missing export error).

- [ ] **Step 2: Remove `getOrCreateUser` and `logUserSignIn`**

Delete the entire `getOrCreateUser` function (lines 8–33) and the entire `logUserSignIn` function (lines 79–90). No callers remain after Task 1.

- [ ] **Step 3: Fix `logUserPrediction` signature and field name**

Change:
```ts
export async function logUserPrediction(userId: string, requestId: string, ticker: string) {
  const datasetId = process.env.BQ_DATASET || "market_sentiment";
  const tableId = "user_prediction_log";

  const row = {
    user_id: userId,
    request_id: requestId,
    ticker: ticker,
    predicted_at: bigquery.timestamp(new Date()),
  };
```

To:
```ts
export async function logUserPrediction(email: string, requestId: string, ticker: string) {
  const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";
  const tableId = "user_prediction_log";

  const row = {
    user_email: email,
    request_id: requestId,
    ticker: ticker,
    predicted_at: bigquery.timestamp(new Date()),
  };
```

- [ ] **Step 4: Fix `getUserHistory` — SELECT list, clause order, WHERE, and params**

The current query has a pre-existing SQL bug: `QUALIFY` appears before `WHERE`, which is invalid in BigQuery (`WHERE` must precede `QUALIFY`). Fix this alongside the user_email changes. Replace the entire function body (lines 92–124 of `bigquery.ts`):

```ts
export async function getUserHistory(email: string) {
  const datasetId = process.env.BQ_DATASET || "market_sentiment_predictor";
  const query = `
    SELECT
      l.user_email,
      l.ticker,
      l.predicted_at,
      s.snapshot_timestamp,
      s.current_price,
      s.day_change_pct,
      s.headline_count,
      s.sentiment_score,
      s.feature_snapshot,
      s.model_artifact_uri,
      s.request_id
    FROM \`${process.env.GCP_PROJECT_ID}.${datasetId}.user_prediction_log\` l
    JOIN \`${process.env.GCP_PROJECT_ID}.${datasetId}.live_feature_snapshots\` s
      ON l.request_id = s.request_id
      AND l.ticker = s.ticker
    WHERE l.user_email = @user_email
    QUALIFY ROW_NUMBER() OVER (PARTITION BY l.request_id, l.ticker ORDER BY s.created_at DESC) = 1
    ORDER BY l.predicted_at DESC
    LIMIT 50
  `;

  const options = {
    query,
    params: { user_email: email },
  };

  const [rows] = await bigquery.query(options);
  return rows;
}
```

Key changes from current code:
- Signature: `userId: string` → `email: string`
- Dataset default: `"market_sentiment"` → `"market_sentiment_predictor"`
- SELECT: `l.user_id` → `l.user_email`
- Clause order: `WHERE` moved before `QUALIFY` (fixes pre-existing BQ syntax bug)
- WHERE: `l.user_id = @user_id` → `l.user_email = @user_email`
- Params: `{ user_id: userId }` → `{ user_email: email }`

- [ ] **Step 5: Verify TypeScript compiles**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: no errors referencing `bigquery.ts`. If there are import errors (e.g., `getOrCreateUser` still imported somewhere), the build output will show which file — fix those imports.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/bigquery.ts
git commit -m "fix(bq): use user_email, fix dataset default, remove dead user-id functions"
```

---

## Chunk 2: Route fixes + migration

### Task 3: Fix `predict/route.ts` — session, cache_status injection, email logging

**Files:**
- Modify: `frontend/src/app/api/predict/route.ts`

**What to know:**
- `getServerSession()` with no arguments always returns `null` in NextAuth v4. Must pass the exported `authOptions`.
- When serving a ticker from the frontend's BQ TTL cache (lines 266–287), the `feature_snapshot` stored in BQ was written with `cache_status.source_mode = "live_api"` (that was the truth when it was first computed). When we re-serve it, we must overwrite `cache_status` to reflect that THIS response came from the frontend cache, not a live call.
- The BQ client returns `snapshot_timestamp` as `{ value: "2026-03-15T…" }` (not a native Date). Unwrap it before passing to `new Date()`.
- Existing logging block reads `session?.user?.id` — after Task 1 simplification, the session no longer has `.id`. Use `session?.user?.email`.
- BQ cache hits do NOT need `enrichPredictionPayload` — company_name/currency are already stored in the snapshot's `feature_snapshot` JSON from the original FastAPI call.

- [ ] **Step 1: Add authOptions import and fix getServerSession**

At the top of `frontend/src/app/api/predict/route.ts`, change:
```ts
import { getServerSession } from "next-auth/next";
import { getFreshSnapshots, logUserPrediction } from "@/lib/bigquery";
```
To:
```ts
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/app/api/auth/[...nextauth]/route";
import { getFreshSnapshots, logUserPrediction } from "@/lib/bigquery";
```

Then at the top of the `POST` handler, change:
```ts
  const session = await getServerSession();
```
To:
```ts
  const session = await getServerSession(authOptions);
```

- [ ] **Step 2: Inject cache_status override for BQ cache hits**

In the `freshSnapshots.forEach` block (around line 245), replace the section that builds `finalPredictions.push({…})` for BQ hits. Currently the feature_snapshot is passed as-is (`feature_snapshot: fs as FeatureSnapshotPayload`). Change the block to compute snapshot age and override `cache_status`:

```ts
  freshSnapshots.forEach((snapshot) => {
    let featureSnapshot: Record<string, unknown> = {};
    if (typeof snapshot.feature_snapshot === "string") {
      try {
        featureSnapshot = JSON.parse(snapshot.feature_snapshot) as Record<string, unknown>;
      } catch (e) {
        console.error("Failed to parse cached feature_snapshot:", e);
      }
    } else if (snapshot.feature_snapshot && typeof snapshot.feature_snapshot === "object") {
      featureSnapshot = snapshot.feature_snapshot as Record<string, unknown>;
    }

    if (!featureSnapshot.prediction) {
      freshTickerMap.delete(snapshot.ticker as string);
      staleTickers.push(snapshot.ticker as string);
      return;
    }

    // Unwrap BQ timestamp object { value: "…" } before computing age
    const rawTs = snapshot.snapshot_timestamp;
    const tsStr =
      rawTs && typeof rawTs === "object" && "value" in rawTs
        ? String((rawTs as { value: unknown }).value)
        : rawTs instanceof Date
        ? rawTs.toISOString()
        : String(rawTs);
    const snapshotAgeSeconds = Math.max(
      0,
      Math.floor((Date.now() - new Date(tsStr).getTime()) / 1000),
    );

    const fs: Record<string, unknown> = {
      ...featureSnapshot,
      cache_status: {
        source_mode: "bigquery_cache",
        reason: "frontend_ttl_cache",
        cache_hit: true,
        stale_served: false,
        quota_exhausted: false,
        snapshot_timestamp: tsStr,
        snapshot_age_seconds: snapshotAgeSeconds,
        snapshot_age_minutes: Math.round((snapshotAgeSeconds / 60) * 10) / 10,
        fresh_ttl_seconds: LIVE_SNAPSHOT_FRESH_TTL_SECONDS,
        fresh_ttl_minutes: Math.round((LIVE_SNAPSHOT_FRESH_TTL_SECONDS / 60) * 10) / 10,
      },
    };

    finalPredictions.push({
      request_id: snapshot.request_id as string,
      ticker: snapshot.ticker as string,
      prediction: (fs.prediction as string) || "Unknown",
      confidence: (fs.confidence as number) || 0,
      prob_up: (fs.prob_up as number) || 0,
      as_of: tsStr,
      predicted_at: tsStr,
      feature_snapshot: fs as FeatureSnapshotPayload,
      model_artifact_uri: (snapshot.model_artifact_uri as string) ?? null,
    });
  });
```

- [ ] **Step 3: Fix logging block to use email**

Change:
```ts
  const userId = (session?.user as { id?: string })?.id;
  if (userId) {
    Promise.all(
      finalPredictions.map((p) => logUserPrediction(userId, p.request_id, p.ticker))
    ).catch((error) => {
      console.error("Failed to log predictions to BigQuery in background:", error);
    });
  }
```

To:
```ts
  const userEmail = session?.user?.email ?? null;
  if (userEmail) {
    Promise.all(
      finalPredictions.map((p) => logUserPrediction(userEmail, p.request_id, p.ticker))
    ).catch((error) => {
      console.error("Failed to log predictions to BigQuery in background:", error);
    });
  }
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: no errors. Common failure: TypeScript may warn that `session?.user?.email` could be `undefined` not `null` — use `?? null` or `|| null` cast as shown above.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/api/predict/route.ts
git commit -m "fix(predict): getServerSession(authOptions), inject BQ cache_status, use email for logging"
```

---

### Task 4: Fix `history/route.ts` — session and email

**Files:**
- Modify: `frontend/src/app/api/history/route.ts`

- [ ] **Step 1: Add authOptions import and fix getServerSession**

Change:
```ts
import { getServerSession } from "next-auth/next";
import { getUserHistory } from "@/lib/bigquery";
```
To:
```ts
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/app/api/auth/[...nextauth]/route";
import { getUserHistory } from "@/lib/bigquery";
```

- [ ] **Step 2: Fix session extraction and getUserHistory call**

Change:
```ts
  const userId = (session?.user as { id?: string })?.id;
  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const history = await getUserHistory(userId);
```
To:
```ts
  const userEmail = session?.user?.email;
  if (!userEmail) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const history = await getUserHistory(userEmail);
```

- [ ] **Step 3: Verify TypeScript compiles clean**

```bash
cd frontend && npm run build 2>&1 | tail -20
```

Expected: clean build. If there are any remaining references to `userId` in the file, the TS compiler will flag them.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/api/history/route.ts
git commit -m "fix(history): getServerSession(authOptions), use email for user lookup"
```

---

### Task 5: Fix migration SQL

**Files:**
- Modify: `backend/db/migrations/20260315110000_init_user_logging.sql`

**What to know:**
- The old file created 3 tables (`users`, `user_prediction_log` with `user_id`, `user_signin_log`). None have been applied to production (not in `terraform.tfstate`).
- Approach B only needs one table: `user_prediction_log` with `user_email`.
- The Terraform resource in `main.tf` (line 127) and its schema file `infra/terraform/user_prediction_log.json` already define the correct `user_email` column. The migration SQL just needs to match.

- [ ] **Step 1: Replace migration SQL**

Replace the entire content of `backend/db/migrations/20260315110000_init_user_logging.sql` with:

```sql
-- migrate:up

CREATE TABLE IF NOT EXISTS `user_prediction_log` (
  user_email   STRING    NOT NULL,
  request_id   STRING    NOT NULL,
  ticker       STRING    NOT NULL,
  predicted_at TIMESTAMP NOT NULL
)
PARTITION BY DATE(predicted_at)
CLUSTER BY user_email, ticker;

-- migrate:down

DROP TABLE IF EXISTS `user_prediction_log`;
```

- [ ] **Step 2: Commit**

```bash
git add backend/db/migrations/20260315110000_init_user_logging.sql
git commit -m "fix(migration): simplify to single user_prediction_log with user_email (approach B)"
```

---

## Chunk 3: Integration smoke tests

### Task 6: Integration smoke tests

**What to know:**
- Run the Next.js dev server locally. The frontend proxies prediction requests to the FastAPI Cloud Run service at the URL in `NEXTAUTH_URL` / `PREDICT_API_URL` from `.env.local`.
- Tests 1–3 don't require Google login. Tests 4–5 require a valid Google OAuth session (test manually in browser).
- The BQ cache label test requires a ticker that was already predicted (has a snapshot in `live_feature_snapshots` less than 1 hour old). Run the same ticker twice — the second call should show "BQ cache".

- [ ] **Step 1: Start dev server**

```bash
cd frontend && npm run dev
```

Expected: server starts on `http://localhost:3000`, no startup errors.

- [ ] **Step 2: Smoke test — unauthenticated prediction (ticker with no BQ cache)**

```bash
curl -s -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"ticker":"AAPL"}]}' | jq '.predictions[0] | {ticker, prediction, source_mode: .feature_snapshot.cache_status.source_mode}'
```

Expected output:
```json
{
  "ticker": "AAPL",
  "prediction": "Up" or "Down",
  "source_mode": "live_api" or "bigquery_cache"
}
```
(First call will be `live_api` if no recent snapshot exists; `bigquery_cache` if one does.)

- [ ] **Step 3: Smoke test — BQ cache label on second call**

Run the same ticker again within 1 hour:

```bash
curl -s -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"ticker":"AAPL"}]}' | jq '.predictions[0].feature_snapshot.cache_status'
```

Expected: `source_mode` is `"bigquery_cache"`, `reason` is `"frontend_ttl_cache"`, `cache_hit` is `true`, `snapshot_age_seconds` is a positive number.

- [ ] **Step 4: Smoke test — unauthenticated history returns 401**

```bash
curl -s http://localhost:3000/api/history | jq .
```

Expected:
```json
{ "error": "Unauthorized" }
```
HTTP status 401.

- [ ] **Step 5: Smoke test — login and history (manual browser test)**

1. Open `http://localhost:3000` in a browser
2. Click "Sign in with Google" — complete the OAuth flow
3. Submit a prediction for any ticker
4. Navigate to the history section — the prediction you just made should appear

- [ ] **Step 6: Verify BQ row written correctly (optional — requires GCP access)**

In BigQuery console, run:
```sql
SELECT user_email, ticker, request_id, predicted_at
FROM `mkt-sent-friend-260313-225751.market_sentiment_predictor.user_prediction_log`
ORDER BY predicted_at DESC
LIMIT 5
```

Expected: a row with your Google account email, the ticker you predicted, and a recent timestamp.

- [ ] **Step 7: Commit all integration test notes (if any notes were captured)**

```bash
git add -p  # stage only intended changes
git commit -m "test(integration): smoke tests pass for BQ cache label and auth flow"
```

---

## Environment prerequisites

Before running integration tests, confirm `frontend/.env.local` has:

```
PREDICT_API_URL=<Cloud Run predict endpoint>
GOOGLE_CLIENT_ID=<from Google Cloud Console>
GOOGLE_CLIENT_SECRET=<from Google Cloud Console>
NEXTAUTH_SECRET=<any random string>
NEXTAUTH_URL=http://localhost:3000
GCP_PROJECT_ID=mkt-sent-friend-260313-225751
BQ_DATASET=market_sentiment_predictor
```

Also confirm the `user_prediction_log` table exists in BigQuery (apply Terraform if not):
```bash
cd infra/terraform && terraform plan && terraform apply
```
