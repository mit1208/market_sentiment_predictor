# Swarm Agent Prediction Layer — Design Spec

**Date**: 2026-03-29
**Status**: Draft
**Inspired by**: [MiroFish](https://github.com/666ghj/MiroFish) multi-agent prediction engine

## Overview

Add a MiroFish-style multi-agent deliberation layer as a post-ensemble validation system for the existing ML prediction pipeline. When a user requests "Deep Analysis," the ML model predicts first, then a swarm of 4 specialized analyst agents debate, challenge, and validate the prediction before returning a final verdict with full transparency.

## Decision Context

- The ML ensemble (LGB+XGB) remains the primary prediction engine
- Agents do NOT replace the model — they validate, adjust confidence, and can override direction only by supermajority
- No knowledge graph needed — structured financial data is already available in feature snapshots
- Gemini 2.5 Flash for all agent calls (cost-effective)
- Agent memory stored in BigQuery (no Zep Cloud dependency)

## MiroFish Pipeline Mapping

| MiroFish Stage | Our Stage | Implementation |
|---|---|---|
| Knowledge Graph | Skipped | Data already structured in feature snapshots |
| Environment Setup | Stage 1: Agent Personas | Cached personas in BigQuery, weekly recalibration |
| Simulation | Stage 2: Deliberation | 2-round structured debate via Gemini Flash |
| Report Generation | Stage 3: Synthesis | Deterministic Python moderator (0 LLM tokens) |
| Interactive | Stage 4: Ask the Analysts | On-demand Gemini calls from frontend |

## Architecture

```
Frontend (Next.js)
  ├─ "Quick Predict" → /api/predict → ML ensemble only (existing)
  └─ "Deep Analysis" → /api/predict → ML ensemble
                      → POST swarm-service/analyze
                          ├─ Stage 1: Load cached personas from BQ
                          ├─ Stage 2: Round 1 (4 parallel Gemini) → SSE
                          │           Round 2 (4 parallel Gemini) → SSE
                          ├─ Stage 3: Deterministic moderator (Python)
                          └─ Stage 4: On-demand /swarm/interview
```

### New Infrastructure

- **1 Cloud Run service**: `swarm-service` (FastAPI, Python 3.11)
- **2 BigQuery tables**: `swarm_agent_personas`, `swarm_deliberation_rounds`
- **1 Cloud Scheduler job**: Weekly persona recalibration
- **SSE streaming**: Swarm service → Frontend for real-time debate visibility

### User Modes

- **Quick Predict**: Existing ML-only flow. No swarm. $0 extra cost.
- **Deep Analysis**: ML prediction + swarm deliberation. User sees full agent debate in real-time. ~$0.001 per prediction.

---

## Stage 1: Agent Persona Generation

### Agent Archetypes

| Agent | Feature Subset | Personality |
|---|---|---|
| **Sentiment Analyst** | gemini_sentiment_avg/max/min, gemini_magnitude_avg, headline_count, has_earnings_news, has_analyst_news, sentiment_dispersion, sentiment_market_alignment + raw news headlines | Narrative-driven, weighs market mood heavily, sensitive to news momentum |
| **Macro Economist** | yield_curve_spread, fed_funds_rate, fed_funds_change_3m, credit_spread, dxy_change_5d, spy_change_pct, spy_gap_pct, vix_level, vix_change_pct | Top-down thinker, focuses on macro regime, skeptical of short-term noise |
| **Risk Manager** | vix_level, vix_change_pct, volume_vs_avg_10d, intraday_range_pct, volatility_regime, day_range_position, gap_pct | Conservative, focuses on downside risk, flags unusual volatility |
| **Contrarian** | All features the other 3 agents receive + the ML model's prediction + direction | Deliberately stress-tests consensus, argues the opposite case, only agrees when evidence is overwhelming |

### Persona Storage

**Table**: `swarm_agent_personas`

| Column | Type | Description |
|---|---|---|
| agent_id | STRING | e.g., "sentiment_analyst" |
| display_name | STRING | e.g., "Sentiment Analyst" |
| system_prompt | STRING | Full Gemini system prompt (~500 tokens) |
| personality_traits | JSON | MBTI, bias profile, risk tolerance |
| calibration_stats | JSON | Recent accuracy, known biases, correction notes |
| created_at | TIMESTAMP | When persona was generated |
| updated_at | TIMESTAMP | Last recalibration |

### Persona Generation

- **When**: Weekly Cloud Scheduler job (`swarm-persona-recalibration`)
- **How**: Single Gemini Flash call per agent. Input: agent archetype definition + calibration stats from `swarm_deliberation_rounds` (last 30 days accuracy). Output: updated system_prompt incorporating self-correction instructions.
- **First run**: Seed with hardcoded base personas (no calibration data yet)
- **Token budget**: ~800 input + ~500 output per agent = ~5,200 tokens total per week. Negligible cost.

---

## Stage 2: Agent Deliberation

### Round 1 — Independent Analysis

4 parallel Gemini Flash calls. Each agent receives:

**Input** (~800 tokens per agent):
- System prompt (from persona table)
- ML prediction: `{direction: "Up", confidence: 0.70, probability: 0.70}`
- Agent-specific feature subset as compact `key: value` pairs
- For Sentiment Analyst: raw news headlines (truncated to 5 most recent)

**Required output** (structured JSON, ~200 tokens):
```json
{
  "direction": "Up|Down",
  "confidence_delta": -0.15,
  "conviction": "low|medium|high",
  "thesis": "2-3 sentence reasoning",
  "key_signals": ["signal_1", "signal_2"],
  "agrees_with_model": true
}
```

All 4 verdicts stored in `swarm_deliberation_rounds` and streamed to frontend via SSE.

### Round 2 — Reaction Round

4 parallel Gemini Flash calls. Each agent receives:

**Input** (~1,000 tokens per agent):
- System prompt
- Their own Round 1 verdict
- Compressed summary of other 3 agents (~100 tokens each): `{agent, direction, conviction, thesis}`

**Required output** (structured JSON, ~250 tokens):
```json
{
  "revised": true,
  "direction": "Up",
  "confidence_delta": -0.10,
  "conviction": "medium",
  "thesis": "Updated reasoning",
  "rebuttals": {
    "Risk Manager": "Short rebuttal to their point"
  },
  "agrees_with_model": true
}
```

Revisions streamed to frontend — user sees agents responding to each other.

### Token Budget

- Round 1: 4 × (800 input + 200 output) = ~4,000 tokens
- Round 2: 4 × (1,000 input + 250 output) = ~5,000 tokens
- **Total: ~9,000 tokens per deep prediction**
- **Cost: ~$0.001 with Gemini 2.5 Flash**

---

## Stage 3: Deterministic Moderator

**Zero LLM calls.** Pure Python function.

### Override Rules

```python
def moderate(ml_prediction, agent_verdicts):
    dissenters = [a for a in agent_verdicts if not a["agrees_with_model"]]
    high_conviction_dissenters = [a for a in dissenters if a["conviction"] == "high"]

    direction_overridden = (
        len(dissenters) >= 3 and len(high_conviction_dissenters) >= 2
    )

    # Weighted confidence adjustment
    weights = {"low": 0.5, "medium": 1.0, "high": 1.5}
    total_weight = sum(weights[a["conviction"]] for a in agent_verdicts)
    adjusted_delta = sum(
        a["confidence_delta"] * weights[a["conviction"]]
        for a in agent_verdicts
    ) / total_weight

    final_confidence = clamp(
        ml_prediction["confidence"] + adjusted_delta, 0.30, 0.95
    )

    final_direction = (
        opposite(ml_prediction["direction"])
        if direction_overridden
        else ml_prediction["direction"]
    )

    dissent_summary = None
    if dissenters:
        strongest_dissent = max(dissenters, key=lambda a: weights[a["conviction"]])
        dissent_summary = f"{strongest_dissent['display_name']}: {strongest_dissent['thesis']}"

    return {
        "final_direction": final_direction,
        "final_confidence": final_confidence,
        "direction_overridden": direction_overridden,
        "consensus": f"{4 - len(dissenters)}/4 agree",
        "dissent": dissent_summary
    }
```

### Confidence Bounds

- Floor: 0.30 (never report extreme low confidence — just say uncertain)
- Ceiling: 0.95 (never report near-certainty on stock predictions)

---

## Stage 4: Interactive — Ask the Analysts

**Zero cost until user interaction.** Available in the frontend after deep analysis.

### Endpoint

`POST /swarm/interview`

```json
// Request
{
  "deliberation_id": "delib:AAPL:20260329T143022Z",
  "agent": "risk_manager",
  "question": "Why do you think VIX is underpriced?"
}

// Response (SSE stream)
{
  "agent": "Risk Manager",
  "answer": "Current VIX at 18 is below the 30-day average of 22...",
  "turn": 1
}
```

### Context Loading

Per interview call, backend loads:
- Agent's system prompt from `swarm_agent_personas`
- Agent's Round 1 + Round 2 verdicts from `swarm_deliberation_rounds`
- The feature subset the agent analyzed
- Conversation history for this interview (capped at 5 turns)

**Token budget**: ~1,000 input + ~150 output per question.

---

## BigQuery Tables

### `swarm_agent_personas`

Stores agent configurations and calibration data. Updated weekly.

| Column | Type |
|---|---|
| agent_id | STRING (PK) |
| display_name | STRING |
| system_prompt | STRING |
| personality_traits | JSON |
| calibration_stats | JSON |
| feature_subset | JSON (list of feature names) |
| created_at | TIMESTAMP |
| updated_at | TIMESTAMP |

### `swarm_deliberation_rounds`

Stores every agent verdict for every deep prediction. Used for calibration, audit, and frontend replay.

| Column | Type |
|---|---|
| deliberation_id | STRING (PK) |
| request_id | STRING (FK to live_feature_snapshots) |
| ticker | STRING |
| agent_id | STRING |
| round | INT (1 or 2) |
| direction | STRING |
| confidence_delta | FLOAT |
| conviction | STRING |
| thesis | STRING |
| key_signals | JSON |
| rebuttals | JSON |
| agrees_with_model | BOOL |
| revised | BOOL |
| ml_direction | STRING |
| ml_confidence | FLOAT |
| actual_outcome | STRING (backfilled next day) |
| agent_was_correct | BOOL (backfilled next day) |
| created_at | TIMESTAMP |

### `swarm_deliberation_summary`

One row per deliberation. Stores the moderator verdict.

| Column | Type |
|---|---|
| deliberation_id | STRING (PK) |
| request_id | STRING |
| ticker | STRING |
| final_direction | STRING |
| final_confidence | FLOAT |
| direction_overridden | BOOL |
| consensus | STRING |
| dissent | STRING |
| actual_outcome | STRING (backfilled next day) |
| created_at | TIMESTAMP |

---

## Swarm Cloud Run Service

### Service: `swarm-service`

- **Framework**: FastAPI (Python 3.11)
- **Port**: 8081
- **Dockerfile**: `backend/swarm/Dockerfile` (built from project root)
- **Scales to zero** when no deep predictions requested

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/swarm/analyze` | Run full 4-stage deliberation, returns SSE stream |
| POST | `/swarm/interview` | Ask a specific agent a question |
| GET | `/swarm/deliberation/{id}` | Replay a past deliberation |
| GET | `/health` | Health check |

### `/swarm/analyze` Request

```json
{
  "ticker": "AAPL",
  "request_id": "req:AAPL:20260329T143022Z",
  "ml_prediction": {
    "direction": "Up",
    "confidence": 0.70,
    "probability": 0.70
  },
  "feature_snapshot": { /* full feature snapshot from ML predictor */ },
  "news_items": [ /* raw headlines */ ]
}
```

### `/swarm/analyze` SSE Response Stream

```
event: round1_verdict
data: {"agent": "Sentiment Analyst", "direction": "Up", "conviction": "high", "thesis": "..."}

event: round1_verdict
data: {"agent": "Macro Economist", "direction": "Up", "conviction": "medium", "thesis": "..."}

event: round1_verdict
data: {"agent": "Risk Manager", "direction": "Down", "conviction": "high", "thesis": "..."}

event: round1_verdict
data: {"agent": "Contrarian", "direction": "Down", "conviction": "low", "thesis": "..."}

event: round2_verdict
data: {"agent": "Sentiment Analyst", "revised": false, "thesis": "...", "rebuttals": {...}}

event: round2_verdict
data: {"agent": "Macro Economist", "revised": true, "thesis": "...", "rebuttals": {...}}

event: round2_verdict
data: {"agent": "Risk Manager", "revised": false, "thesis": "...", "rebuttals": {...}}

event: round2_verdict
data: {"agent": "Contrarian", "revised": true, "thesis": "...", "rebuttals": {...}}

event: moderator_verdict
data: {"final_direction": "Up", "final_confidence": 0.62, "direction_overridden": false, "consensus": "2/4 agree", "dissent": "Risk Manager: ..."}

event: done
data: {"deliberation_id": "delib:AAPL:20260329T143022Z"}
```

---

## Frontend Changes

### Prediction Page

User sees a toggle: **Quick Predict** | **Deep Analysis**

**Quick Predict**: Existing behavior — ML prediction card with feature snapshot.

**Deep Analysis**: After ML prediction loads:
1. "Swarm Analysis" section appears with loading state
2. Agent cards stream in as SSE events arrive:
   - Round 1: 4 cards with agent avatar, name, direction badge (Up/Down), conviction level, thesis, key signals
   - Round 2: Cards update with revision indicator, rebuttals appear as sub-cards
3. Moderator verdict card appears at bottom: final direction, adjusted confidence, consensus meter, dissent quote, override badge if applicable
4. "Chat with Analysts" button opens an interview panel — select agent, type question, see response

### Frontend API Route

`/api/swarm/analyze` — Next.js route that:
1. Receives ticker + ML prediction from the frontend
2. Calls swarm-service SSE endpoint
3. Forwards SSE events to the browser
4. Stores deliberation_id for interview follow-ups

`/api/swarm/interview` — Next.js route that proxies interview requests to swarm-service.

---

## Backend File Structure

```
backend/swarm/
├── Dockerfile
├── main.py                    # FastAPI app + SSE endpoints
├── agents/
│   ├── personas.py            # Load/cache personas from BQ
│   ├── deliberation.py        # Round 1 + Round 2 orchestration
│   ├── moderator.py           # Deterministic override logic
│   └── interview.py           # On-demand agent interview
├── gemini/
│   └── client.py              # Gemini Flash structured output calls
├── models/
│   └── schemas.py             # Pydantic models for verdicts, requests, responses
├── storage/
│   └── bigquery.py            # BQ read/write for personas + deliberation rounds
└── config.py                  # Environment variables + constants
```

---

## Weekly Persona Recalibration Job

**Cloud Scheduler**: `swarm-persona-recalibration` (runs weekly, Sunday 00:00 UTC)

**Process**:
1. Query `swarm_deliberation_rounds` for last 30 days where `actual_outcome` is populated
2. Per agent: compute accuracy, directional bias, conviction calibration
3. Gemini Flash call per agent: "Given these stats, update your analytical approach" → new system_prompt
4. Write updated rows to `swarm_agent_personas`

**Outcome backfill**: A daily Cloud Scheduler job (runs at market close + 1hr, 17:00 ET):
1. Queries `swarm_deliberation_summary` for predictions from the previous trading day
2. Fetches actual next-day close via Yahoo Finance (`yfinance`) for each ticker
3. Computes actual direction (Up/Down) from close-to-close return
4. Backfills `actual_outcome` in `swarm_deliberation_summary` and `swarm_deliberation_rounds`
5. Computes `agent_was_correct` per agent row by comparing agent's final direction to actual outcome

---

## Environment Variables (swarm-service)

```
GCP_PROJECT_ID
GCP_REGION
BQ_DATASET
GEMINI_API_KEY
GEMINI_MODEL=gemini-2.5-flash
SWARM_SERVICE_PORT=8081
PERSONA_RECALIBRATION_CRON=0 0 * * 0
MAX_INTERVIEW_TURNS=5
CONFIDENCE_FLOOR=0.30
CONFIDENCE_CEILING=0.95
SUPERMAJORITY_THRESHOLD=3
HIGH_CONVICTION_THRESHOLD=2
```

---

## Cost Analysis

| Action | Gemini Flash Calls | Tokens | Est. Cost |
|---|---|---|---|
| Quick Predict | 0 (swarm) | 0 | $0 |
| Deep Analysis | 8 | ~9,000 | ~$0.001 |
| Interview (per question) | 1 | ~1,150 | ~$0.0001 |
| Weekly recalibration | 4 | ~5,200 | ~$0.0005 |
| Daily outcome backfill | 0 | 0 | $0 (BQ query only) |

---

## Terraform Additions

- `google_cloud_run_v2_service.swarm_service` — swarm-service Cloud Run
- `google_cloud_scheduler_job.swarm_persona_recalibration` — weekly persona job
- `google_cloud_scheduler_job.swarm_outcome_backfill` — daily outcome backfill
- `google_bigquery_table.swarm_agent_personas`
- `google_bigquery_table.swarm_deliberation_rounds`
- `google_bigquery_table.swarm_deliberation_summary`
