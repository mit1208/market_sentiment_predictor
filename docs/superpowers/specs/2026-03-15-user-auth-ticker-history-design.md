# User Auth & Ticker History

## Summary

Add Google sign-in to the Next.js frontend via NextAuth.js and persist each user's prediction history by linking their email to existing `live_feature_snapshots` rows through a thin BigQuery join table. Reduce redundant backend/API calls by checking for fresh cached snapshots before hitting the FastAPI prediction service.

## Auth: NextAuth.js + Google Provider

- Add `next-auth` with the Google OAuth provider to the Next.js app.
- Session stored in a cookie (JWT strategy) — no session DB needed.
- UI: top-right "Sign in with Google" button; avatar + sign-out when authenticated.
- Unauthenticated users retain full prediction desk access — login is only required to save and view history.
- The FastAPI backend remains auth-unaware; user identity stays in the Next.js layer.

### Required env vars (frontend)

| Variable | Purpose |
|----------|---------|
| `NEXTAUTH_SECRET` | JWT signing secret |
| `NEXTAUTH_URL` | Canonical app URL |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |

## Storage: `user_prediction_log` join table

A thin table that links a user's email to existing `live_feature_snapshots` rows. No prediction data is duplicated.

### Schema

```sql
CREATE TABLE IF NOT EXISTS `user_prediction_log` (
  user_email    STRING    NOT NULL,
  request_id    STRING    NOT NULL,
  ticker        STRING    NOT NULL,
  predicted_at  TIMESTAMP NOT NULL
)
PARTITION BY DATE(predicted_at)
CLUSTER BY user_email, ticker;
```

### Write path

After the Next.js `/api/predict` route receives a successful response from the FastAPI backend, it writes one row per ticker to `user_prediction_log` with the user's email (from the NextAuth session) and the `request_id` from the response. If the user is not logged in, no row is written — prediction still works.

### Read path

A new Next.js API route `/api/history` queries:

```sql
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
  s.model_artifact_uri
FROM user_prediction_log l
JOIN live_feature_snapshots s
  ON l.request_id = s.request_id
  AND l.ticker = s.ticker
WHERE l.user_email = @user_email
ORDER BY l.predicted_at DESC
LIMIT 50
```

## Smart call reduction

Before forwarding tickers to the FastAPI backend, the Next.js `/api/predict` route checks `live_feature_snapshots` for each ticker:

1. Query for the most recent snapshot per ticker where `snapshot_timestamp` is within the configured TTL (default: reuse the existing `LIVE_SNAPSHOT_FRESH_TTL_SECONDS`, 1 hour).
2. Tickers with a fresh snapshot are returned directly from BigQuery — no FastAPI call.
3. Only stale/missing tickers are sent to the backend.
4. Results are merged into a single response, transparent to the frontend.

This reduces both FastAPI endpoint calls and upstream Finnhub/Yahoo API calls.

## Frontend UX

### Auth controls

- Top-right of the page: "Sign in with Google" button (logged out) or avatar + email + "Sign out" (logged in).
- Minimal footprint — does not change the prediction desk layout.

### History section

- Appears below the prediction desk when the user is logged in.
- Simple table: ticker, prediction direction, probability, date.
- Sorted by most recent first, limited to 50 rows.
- Clicking a row expands to show the full snapshot details (reuses existing result card components).
- Hidden when not logged in.

## Components changed

| File | Change |
|------|--------|
| `frontend/package.json` | Add `next-auth` dependency |
| `frontend/src/app/api/auth/[...nextauth]/route.ts` | New — NextAuth config with Google provider |
| `frontend/src/app/api/predict/route.ts` | Add session check, write to `user_prediction_log`, smart cache check |
| `frontend/src/app/api/history/route.ts` | New — fetch user's prediction history from BQ |
| `frontend/src/app/page.tsx` | Add auth UI (sign in/out), history section |
| `frontend/src/app/layout.tsx` | Wrap with NextAuth `SessionProvider` |
| `backend/db/migrations/2026031XXXXXX_add_user_prediction_log.sql` | New — create `user_prediction_log` table |
| `infra/terraform/main.tf` | Add `user_prediction_log` table resource |
| `.env.example` | Add NextAuth + Google OAuth env vars |

## Out of scope

- Role-based access control or admin features.
- Prediction history analytics or charts.
- Cross-user sharing of predictions.
- Changes to the FastAPI backend auth model.
