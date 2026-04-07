"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useEffectEvent, useRef, useState } from "react";
import { useSession, signIn, signOut } from "next-auth/react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Brain,
  CheckCircle,
  Clock3,
  Database,
  Filter,
  Gauge,
  History,
  Loader2,
  LogIn,
  LogOut,
  MessageSquare,
  Newspaper,
  Plus,
  Search,
  Send,
  Shield,
  Sparkles,
  Star,
  ThumbsDown,
  ThumbsUp,
  TrendingDown,
  TrendingUp,
  User,
  X,
} from "lucide-react";
import { clsx } from "clsx";

interface FeatureItem {
  name: string;
  label: string;
  value: number | string | null;
  source: string;
  providers: string[];
  raw_fields: string[];
  fallback_used: boolean;
}

interface FeatureGroup {
  key: string;
  label: string;
  features: FeatureItem[];
}

interface NewsItem {
  headline: string;
  source: string;
  provider: string;
  url?: string | null;
  published_at?: string | null;
  sentiment_score?: number | null;
  magnitude?: number | null;
  category?: string | null;
  scored_at?: string | null;
  sentiment_label?: "positive" | "negative" | "neutral" | string | null;
}

interface CacheStatus {
  source_mode: "live_api" | "bigquery_cache" | string;
  reason: string;
  cache_hit: boolean;
  stale_served: boolean;
  quota_exhausted: boolean;
  snapshot_timestamp?: string | null;
  snapshot_age_seconds?: number | null;
  snapshot_age_minutes?: number | null;
  fresh_ttl_seconds?: number | null;
  fresh_ttl_minutes?: number | null;
  max_stale_ttl_seconds?: number | null;
  max_stale_ttl_minutes?: number | null;
  daily_live_fetch_limit?: number | null;
  daily_live_fetch_count?: number | null;
  daily_live_fetch_remaining?: number | null;
  origin_snapshot_id?: string | null;
  origin_request_id?: string | null;
}

interface FeatureSnapshot {
  as_of: string;
  company_name?: string | null;
  quote_currency?: string | null;
  feature_count: number;
  selected_feature_count?: number;
  headlines_text?: string;
  groups: FeatureGroup[];
  news_items: NewsItem[];
  cache_status?: CacheStatus | null;
  provider_summary?: {
    finnhub?: number;
    yahoo?: number;
    derived?: number;
    fallback_fields?: number;
  };
}

interface ModelFeatureImportance {
  feature: string;
  importance: number;
}

interface ModelQuality {
  score_semantics?: string;
  cv_accuracy_mean?: number | null;
  cv_auc_mean?: number | null;
  holdout_accuracy?: number | null;
  holdout_auc?: number | null;
  used_feature_count?: number;
  zero_importance_feature_count?: number;
  top_features?: ModelFeatureImportance[];
  feature_selection_source?: string | null;
  feature_selection_mode?: string | null;
  feature_selection_is_temporary?: boolean;
}

interface PredictionResult {
  request_id: string;
  ticker: string;
  prediction: "Up" | "Down";
  confidence: number;
  prob_up: number;
  as_of: string;
  predicted_at: string;
  model_artifact_uri?: string | null;
  model_trained_at?: string | null;
  model_quality?: ModelQuality | null;
  feature_snapshot: FeatureSnapshot;
}

type AnalysisMode = "quick" | "deep";
type SourceFilter = "all" | "finnhub" | "yahoo" | "derived";
type SortMode = "group" | "value" | "source";
type Exchange = "NYSE" | "TSX";
const DEFAULT_TICKERS: string[] = ["SPY", "QQQ", "AAPL"];
const FREE_TICKER_LIMIT = 3;

function isDefaultTicker(ticker: string) {
  return DEFAULT_TICKERS.includes(ticker);
}

interface SwarmVerdictEvent {
  direction: "Up" | "Down";
  conviction: "low" | "medium" | "high";
  thesis: string;
  agrees_with_model: boolean;
  public_notes?: string[];
  key_signals?: string[];
  rebuttals?: Record<string, string>;
  revised?: boolean;
}

interface SwarmAgentState {
  agentId: string;
  agent: string;
  round1?: SwarmVerdictEvent;
  round2?: SwarmVerdictEvent;
}

interface SwarmModeratorVerdict {
  deliberation_id: string;
  request_id: string;
  ticker: string;
  final_direction: "Up" | "Down";
  final_confidence: number;
  direction_overridden: boolean;
  consensus: string;
  dissent?: string | null;
}

interface InterviewTurn {
  role: "user" | "agent";
  message: string;
}

interface SwarmLogEntry {
  id: string;
  at: string;
  kind: "system" | "stream" | "interview" | "error";
  title: string;
  detail: string;
  payload?: Record<string, unknown> | null;
}

interface SwarmTickerState {
  requestId: string;
  status: "idle" | "streaming" | "done" | "error";
  deliberationId?: string;
  source?: "live" | "archive";
  agents: Record<string, SwarmAgentState>;
  moderator?: SwarmModeratorVerdict;
  error?: string;
  interviewTurns: InterviewTurn[];
  logs: SwarmLogEntry[];
}

interface HistoryItem {
  request_id: string;
  ticker: string;
  prediction: string;
  prob_up: number;
  confidence: number;
  current_price: number | null;
  headline_count: number | null;
  predicted_at: string;
  snapshot_timestamp?: string | null;
  feature_snapshot: Record<string, unknown>;
  model_artifact_uri?: string | null;
  swarm_deliberation_id?: string | null;
  swarm_final_direction?: string | null;
  swarm_final_confidence?: number | null;
  swarm_consensus?: string | null;
  swarm_created_at?: string | null;
}

type SwarmRoundView = "round1" | "round2" | "final";

interface FeatureRow extends FeatureItem {
  groupKey: string;
  groupLabel: string;
}

const compactFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const percentFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
});

function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }

  if (typeof error === "string" && error.trim()) {
    return error;
  }

  if (error && typeof error === "object") {
    const record = error as Record<string, unknown>;

    if (typeof record.message === "string" && record.message.trim()) {
      return record.message;
    }

    if (typeof record.error === "string" && record.error.trim()) {
      return record.error;
    }
  }

  return "Prediction failed.";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function getString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function getNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function getStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function sanitizeFeatureItem(value: unknown): FeatureItem | null {
  if (!isRecord(value)) {
    return null;
  }

  const name = getString(value.name);
  const label = getString(value.label) ?? name;
  const source = getString(value.source) ?? "derived";

  if (!name || !label) {
    return null;
  }

  const rawValue = value.value;
  const normalizedValue =
    typeof rawValue === "number" || typeof rawValue === "string" || rawValue === null
      ? rawValue
      : null;

  return {
    name,
    label,
    value: normalizedValue,
    source,
    providers: getStringArray(value.providers),
    raw_fields: getStringArray(value.raw_fields),
    fallback_used: value.fallback_used === true,
  };
}

function sanitizeFeatureGroups(value: unknown): FeatureGroup[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((group) => {
    if (!isRecord(group)) {
      return [];
    }

    const key = getString(group.key);
    const label = getString(group.label) ?? key;
    if (!key || !label) {
      return [];
    }

    const features = Array.isArray(group.features)
      ? group.features
        .map((feature) => sanitizeFeatureItem(feature))
        .filter((feature): feature is FeatureItem => Boolean(feature))
      : [];

    return [{ key, label, features }];
  });
}

function sanitizeNewsItems(value: unknown): NewsItem[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const headline = getString(item.headline);
    const source = getString(item.source) ?? "Unknown";
    const provider = getString(item.provider) ?? source;
    if (!headline) {
      return [];
    }

    return [{
      headline,
      source,
      provider,
      url: getString(item.url),
      published_at: getString(item.published_at),
      sentiment_score: getNumber(item.sentiment_score),
      magnitude: getNumber(item.magnitude),
      category: getString(item.category),
      scored_at: getString(item.scored_at),
      sentiment_label: getString(item.sentiment_label),
    }];
  });
}

function sanitizeTopFeatures(value: unknown): ModelFeatureImportance[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }

    const feature = getString(item.feature);
    const importance = getNumber(item.importance);
    if (!feature || importance === null) {
      return [];
    }

    return [{ feature, importance }];
  });
}

function sanitizePredictionResult(value: unknown): PredictionResult | null {
  if (!isRecord(value)) {
    return null;
  }

  const ticker = getString(value.ticker)?.toUpperCase();
  if (!ticker) {
    return null;
  }

  const featureSnapshot = isRecord(value.feature_snapshot) ? value.feature_snapshot : {};
  const groups = sanitizeFeatureGroups(featureSnapshot.groups);
  const newsItems = sanitizeNewsItems(featureSnapshot.news_items);
  const derivedFeatureCount = groups.reduce((sum, group) => sum + group.features.length, 0);
  const modelQuality = isRecord(value.model_quality) ? value.model_quality : null;

  return {
    request_id: getString(value.request_id) ?? `${ticker}-${getString(value.predicted_at) ?? "cached"}`,
    ticker,
    prediction: value.prediction === "Up" ? "Up" : "Down",
    confidence: getNumber(value.confidence) ?? 0,
    prob_up: getNumber(value.prob_up) ?? 0,
    as_of: getString(value.as_of) ?? "",
    predicted_at: getString(value.predicted_at) ?? "",
    model_artifact_uri: getString(value.model_artifact_uri),
    model_trained_at: getString(value.model_trained_at),
    model_quality: modelQuality
      ? {
        score_semantics: getString(modelQuality.score_semantics) ?? undefined,
        cv_accuracy_mean: getNumber(modelQuality.cv_accuracy_mean),
        cv_auc_mean: getNumber(modelQuality.cv_auc_mean),
        holdout_accuracy: getNumber(modelQuality.holdout_accuracy),
        holdout_auc: getNumber(modelQuality.holdout_auc),
        used_feature_count: getNumber(modelQuality.used_feature_count) ?? undefined,
        zero_importance_feature_count:
          getNumber(modelQuality.zero_importance_feature_count) ?? undefined,
        top_features: sanitizeTopFeatures(modelQuality.top_features),
        feature_selection_source: getString(modelQuality.feature_selection_source),
        feature_selection_mode: getString(modelQuality.feature_selection_mode),
        feature_selection_is_temporary: modelQuality.feature_selection_is_temporary === true,
      }
      : null,
    feature_snapshot: {
      as_of: getString(featureSnapshot.as_of) ?? "",
      company_name: getString(featureSnapshot.company_name),
      quote_currency: getString(featureSnapshot.quote_currency),
      feature_count: getNumber(featureSnapshot.feature_count) ?? derivedFeatureCount,
      selected_feature_count: getNumber(featureSnapshot.selected_feature_count) ?? undefined,
      headlines_text: getString(featureSnapshot.headlines_text) ?? undefined,
      groups,
      news_items: newsItems,
      cache_status: isRecord(featureSnapshot.cache_status)
        ? {
          source_mode: getString(featureSnapshot.cache_status.source_mode) ?? "unknown",
          reason: getString(featureSnapshot.cache_status.reason) ?? "Unavailable",
          cache_hit: featureSnapshot.cache_status.cache_hit === true,
          stale_served: featureSnapshot.cache_status.stale_served === true,
          quota_exhausted: featureSnapshot.cache_status.quota_exhausted === true,
          snapshot_timestamp: getString(featureSnapshot.cache_status.snapshot_timestamp),
          snapshot_age_seconds: getNumber(featureSnapshot.cache_status.snapshot_age_seconds),
          snapshot_age_minutes: getNumber(featureSnapshot.cache_status.snapshot_age_minutes),
          fresh_ttl_seconds: getNumber(featureSnapshot.cache_status.fresh_ttl_seconds),
          fresh_ttl_minutes: getNumber(featureSnapshot.cache_status.fresh_ttl_minutes),
          max_stale_ttl_seconds: getNumber(featureSnapshot.cache_status.max_stale_ttl_seconds),
          max_stale_ttl_minutes: getNumber(featureSnapshot.cache_status.max_stale_ttl_minutes),
          daily_live_fetch_limit: getNumber(featureSnapshot.cache_status.daily_live_fetch_limit),
          daily_live_fetch_count: getNumber(featureSnapshot.cache_status.daily_live_fetch_count),
          daily_live_fetch_remaining: getNumber(
            featureSnapshot.cache_status.daily_live_fetch_remaining,
          ),
          origin_snapshot_id: getString(featureSnapshot.cache_status.origin_snapshot_id),
          origin_request_id: getString(featureSnapshot.cache_status.origin_request_id),
        }
        : null,
      provider_summary: isRecord(featureSnapshot.provider_summary)
        ? {
          finnhub: getNumber(featureSnapshot.provider_summary.finnhub) ?? undefined,
          yahoo: getNumber(featureSnapshot.provider_summary.yahoo) ?? undefined,
          derived: getNumber(featureSnapshot.provider_summary.derived) ?? undefined,
          fallback_fields:
            getNumber(featureSnapshot.provider_summary.fallback_fields) ?? undefined,
        }
        : undefined,
    },
  };
}

function sanitizePredictionResults(value: unknown): PredictionResult[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => sanitizePredictionResult(item))
    .filter((item): item is PredictionResult => Boolean(item));
}

function sanitizeHistoryItem(value: unknown): HistoryItem | null {
  if (!isRecord(value)) {
    return null;
  }

  const ticker = getString(value.ticker)?.toUpperCase();
  if (!ticker) {
    return null;
  }

  return {
    request_id: getString(value.request_id) ?? `${ticker}-history`,
    ticker,
    prediction: getString(value.prediction) ?? "Pending",
    prob_up: getNumber(value.prob_up) ?? 0,
    confidence: getNumber(value.confidence) ?? 0,
    current_price: getNumber(value.current_price),
    headline_count: getNumber(value.headline_count),
    predicted_at: getString(value.predicted_at) ?? "",
    snapshot_timestamp: getString(value.snapshot_timestamp),
    feature_snapshot: isRecord(value.feature_snapshot) ? value.feature_snapshot : {},
    model_artifact_uri: getString(value.model_artifact_uri),
    swarm_deliberation_id: getString(value.swarm_deliberation_id),
    swarm_final_direction: getString(value.swarm_final_direction),
    swarm_final_confidence: getNumber(value.swarm_final_confidence),
    swarm_consensus: getString(value.swarm_consensus),
    swarm_created_at: getString(value.swarm_created_at),
  };
}

function sanitizeHistoryItems(value: unknown): HistoryItem[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => sanitizeHistoryItem(item))
    .filter((item): item is HistoryItem => Boolean(item));
}

function buildHistoryKey(item: Pick<HistoryItem, "request_id" | "ticker">): string {
  return `${item.request_id}:${item.ticker}`;
}

function buildSwarmReplayState(value: unknown): SwarmTickerState | null {
  if (!isRecord(value)) {
    return null;
  }

  const getReplayLogEntry = (
    eventName: string,
    payload: Record<string, unknown>,
  ): Pick<SwarmLogEntry, "kind" | "title" | "detail"> => {
    if (eventName === "round1_verdict") {
      return {
        kind: "stream",
        title: `${String(payload.agent ?? "Agent")} · Round 1`,
        detail: `${String(payload.direction ?? "Pending")} with ${String(payload.conviction ?? "unknown")} conviction. ${String(payload.thesis ?? "")}`,
      };
    }
    if (eventName === "round2_verdict") {
      return {
        kind: "stream",
        title: `${String(payload.agent ?? "Agent")} · Round 2`,
        detail: `${payload.revised ? "Revised view." : "Held position."} ${String(payload.direction ?? "Pending")} with ${String(payload.conviction ?? "unknown")} conviction. ${String(payload.thesis ?? "")}`,
      };
    }
    if (eventName === "moderator_verdict") {
      return {
        kind: "stream",
        title: "Moderator verdict",
        detail: `${String(payload.final_direction ?? "Pending")} at ${typeof payload.final_confidence === "number" ? `${((payload.final_confidence as number) * 100).toFixed(1)}%` : "unknown confidence"}. ${String(payload.consensus ?? "")}`,
      };
    }
    return {
      kind: "system",
      title: "Swarm analysis complete",
      detail: `Deliberation ${String(payload.deliberation_id ?? "")} is ready for replay and follow-up questions.`,
    };
  };

  const deliberationId = getString(value.deliberation_id) ?? undefined;
  const summary = isRecord(value.summary) ? value.summary : null;
  const rounds = Array.isArray(value.rounds) ? value.rounds.filter(isRecord) : [];
  const agents: Record<string, SwarmAgentState> = {};
  const logs: SwarmLogEntry[] = [];
  let requestId = summary ? getString(summary.request_id) ?? "" : "";

  rounds.forEach((round, index) => {
    const agentId = getString(round.agent_id) ?? `agent-${index}`;
    const agentLabel = getString(round.display_name) ?? getString(round.agent) ?? agentId;
    const agentState = agents[agentId] ?? { agentId, agent: agentLabel };
    const roundNumber = getNumber(round.round);
    const eventName = roundNumber === 2 ? "round2_verdict" : "round1_verdict";
    const payload = {
      agent_id: agentId,
      agent: agentLabel,
      direction: getString(round.direction) ?? "Up",
      conviction: getString(round.conviction) ?? "medium",
      thesis: getString(round.thesis) ?? "",
      key_signals: Array.isArray(round.key_signals) ? round.key_signals : [],
      rebuttals: isRecord(round.rebuttals) ? round.rebuttals : {},
      agrees_with_model: round.agrees_with_model === true,
      revised: round.revised === true,
    } satisfies Record<string, unknown>;

    const verdict: SwarmVerdictEvent = {
      direction: (getString(round.direction) ?? "Up") as "Up" | "Down",
      conviction: (getString(round.conviction) ?? "medium") as "low" | "medium" | "high",
      thesis: getString(round.thesis) ?? "",
      agrees_with_model: round.agrees_with_model === true,
      key_signals: Array.isArray(round.key_signals)
        ? round.key_signals.map((signal) => String(signal))
        : undefined,
      rebuttals: isRecord(round.rebuttals)
        ? Object.fromEntries(Object.entries(round.rebuttals).map(([key, val]) => [key, String(val)]))
        : undefined,
      revised: round.revised === true,
    };

    if (roundNumber === 2) {
      agentState.round2 = verdict;
    } else {
      agentState.round1 = verdict;
    }
    agents[agentId] = agentState;
    requestId ||= getString(round.request_id) ?? "";
    const logEntry = getReplayLogEntry(eventName, payload);
    logs.push({
      id: `${deliberationId ?? "delib"}-${eventName}-${agentId}-${index}`,
      at: getString(round.created_at) ?? new Date().toISOString(),
      kind: logEntry.kind,
      title: logEntry.title,
      detail: logEntry.detail,
      payload,
    });
  });

  const moderator = summary
    ? {
      deliberation_id: getString(summary.deliberation_id) ?? deliberationId ?? "",
      request_id: getString(summary.request_id) ?? requestId,
      ticker: getString(summary.ticker) ?? "",
      final_direction: (getString(summary.final_direction) ?? "Up") as "Up" | "Down",
      final_confidence: getNumber(summary.final_confidence) ?? 0,
      direction_overridden: summary.direction_overridden === true,
      consensus: getString(summary.consensus) ?? "Consensus pending",
      dissent: getString(summary.dissent),
    } satisfies SwarmModeratorVerdict
    : undefined;

  if (moderator) {
    const moderatorPayload = {
      final_direction: moderator.final_direction,
      final_confidence: moderator.final_confidence,
      consensus: moderator.consensus,
      direction_overridden: moderator.direction_overridden,
      deliberation_id: moderator.deliberation_id,
    } satisfies Record<string, unknown>;
    const moderatorLog = getReplayLogEntry("moderator_verdict", moderatorPayload);
    logs.push({
      id: `${moderator.deliberation_id}-moderator`,
      at: getString(summary?.created_at) ?? new Date().toISOString(),
      kind: moderatorLog.kind,
      title: moderatorLog.title,
      detail: moderatorLog.detail,
      payload: moderatorPayload,
    });
    const doneLog = getReplayLogEntry("done", { deliberation_id: moderator.deliberation_id });
    logs.push({
      id: `${moderator.deliberation_id}-done`,
      at: getString(summary?.created_at) ?? new Date().toISOString(),
      kind: doneLog.kind,
      title: doneLog.title,
      detail: doneLog.detail,
      payload: { deliberation_id: moderator.deliberation_id },
    });
  }

  return {
    requestId,
    status: moderator || Object.keys(agents).length > 0 ? "done" : "idle",
    deliberationId,
    source: "archive",
    agents,
    moderator,
    interviewTurns: [],
    logs,
  };
}

function flattenFeatures(groups: FeatureGroup[]): FeatureRow[] {
  return groups.flatMap((group) =>
    group.features.map((feature) => ({
      ...feature,
      groupKey: group.key,
      groupLabel: group.label,
    })),
  );
}

function getFeature(result: PredictionResult, name: string): FeatureRow | undefined {
  return flattenFeatures(result.feature_snapshot.groups).find(
    (feature) => feature.name === name,
  );
}

function getNumericFeature(result: PredictionResult, name: string): number {
  const value = getFeature(result, name)?.value;
  return typeof value === "number" ? value : 0;
}

function formatFeatureValue(feature: Pick<FeatureItem, "name" | "value">): string {
  const value = feature.value;
  if (value === null || value === undefined || value === "") {
    return "0.00";
  }

  if (typeof value !== "number") {
    return String(value);
  }

  if (
    feature.name.includes("pct") ||
    feature.name.includes("position") ||
    feature.name.startsWith("tfidf_")
  ) {
    return `${percentFormatter.format(value * 100)}%`;
  }

  if (feature.name.includes("volume")) {
    return compactFormatter.format(value);
  }

  if (feature.name.includes("sentiment")) {
    return value.toFixed(4);
  }

  if (Number.isInteger(value)) {
    return value.toLocaleString();
  }

  return value.toLocaleString(undefined, {
    maximumFractionDigits: 2,
  });
}

function formatQuotedPrice(value: number | null, currency?: string | null): string {
  if (value === null) {
    return "$0.00";
  }

  const normalizedCurrency = currency?.trim().toUpperCase();
  if (normalizedCurrency) {
    try {
      return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: normalizedCurrency,
        currencyDisplay: "code",
        maximumFractionDigits: 2,
      }).format(value);
    } catch {
      return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${normalizedCurrency}`;
    }
  }

  return `$${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

function formatSignedPercent(value: number | null): string {
  if (value === null) {
    return "0.00%";
  }

  const prefix = value > 0 ? "+" : "";
  return `${prefix}${percentFormatter.format(value * 100)}%`;
}

function formatClock(value?: string | null): string {
  if (!value) {
    return "Pending";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Pending";
  }

  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatRelativeTime(value: string | null | undefined, nowMs: number): string {
  if (!value) {
    return "Pending";
  }

  const date = new Date(value);
  const targetMs = date.getTime();
  if (Number.isNaN(targetMs)) {
    return "Pending";
  }

  const diffSeconds = Math.round((targetMs - nowMs) / 1000);
  const absSeconds = Math.abs(diffSeconds);

  const units = [
    { limit: 60, seconds: 1, label: "s" },
    { limit: 3600, seconds: 60, label: "m" },
    { limit: 86400, seconds: 3600, label: "h" },
    { limit: 604800, seconds: 86400, label: "d" },
  ];

  for (const unit of units) {
    if (absSeconds < unit.limit) {
      const amount = Math.max(1, Math.round(absSeconds / unit.seconds));
      return diffSeconds >= 0 ? `in ${amount}${unit.label}` : `${amount}${unit.label} ago`;
    }
  }

  const weeks = Math.max(1, Math.round(absSeconds / 604800));
  return diffSeconds >= 0 ? `in ${weeks}w` : `${weeks}w ago`;
}

function getFeatureValueTone(feature: FeatureRow): string {
  if (typeof feature.value !== "number") {
    return "text-slate-950";
  }

  if (
    feature.name.includes("pct") ||
    feature.name.includes("change") ||
    feature.name.includes("sentiment")
  ) {
    if (feature.value > 0) {
      return "text-emerald-700";
    }
    if (feature.value < 0) {
      return "text-rose-700";
    }
  }

  return "text-slate-950";
}

function getSourceClasses(source: string): string {
  if (source === "finnhub") {
    return "bg-teal-100 text-teal-800";
  }
  if (source === "yahoo") {
    return "bg-amber-100 text-amber-800";
  }
  if (source === "derived") {
    return "bg-slate-200 text-slate-700";
  }
  return "bg-white/80 text-slate-600";
}

function getPredictionClasses(prediction: PredictionResult["prediction"]): string {
  return prediction === "Up"
    ? "from-emerald-500 to-teal-500 text-emerald-950"
    : "from-orange-500 to-rose-500 text-rose-950";
}

function getPredictionAccent(prediction: PredictionResult["prediction"]): string {
  return prediction === "Up" ? "text-emerald-700" : "text-rose-700";
}

function getDisplayCurrency(ticker: string, currency?: string | null): string | null {
  const normalized = currency?.trim().toUpperCase();
  if (normalized) {
    return normalized;
  }

  if (/\.(TO|V|NE|CN)$/i.test(ticker)) {
    return "CAD";
  }

  return "USD";
}

function getNewsTone(sentimentScore: number | null, headlineCount: number | null) {
  if (headlineCount === null || headlineCount <= 0) {
    return {
      label: "No news",
      summary: "No recent headlines were available for this snapshot.",
      chipClasses: "border-slate-200 bg-slate-100 text-slate-700",
      panelClasses: "border-slate-200/70 bg-white/85",
      itemClasses: "border-slate-200/80 bg-white hover:border-slate-300 hover:bg-slate-50",
    };
  }

  if (sentimentScore !== null && sentimentScore >= 0.05) {
    return {
      label: "News positive",
      summary: "Headlines are leaning positive for this ticker right now.",
      chipClasses: "border-emerald-200 bg-emerald-50 text-emerald-700",
      panelClasses: "border-emerald-200/80 bg-emerald-50/70",
      itemClasses: "border-emerald-100/90 bg-white hover:border-emerald-200 hover:bg-emerald-50/40",
    };
  }

  if (sentimentScore !== null && sentimentScore <= -0.05) {
    return {
      label: "News negative",
      summary: "Headlines are leaning negative for this ticker right now.",
      chipClasses: "border-rose-200 bg-rose-50 text-rose-700",
      panelClasses: "border-rose-200/80 bg-rose-50/70",
      itemClasses: "border-rose-100/90 bg-white hover:border-rose-200 hover:bg-rose-50/40",
    };
  }

  return {
    label: "News mixed",
    summary: "Headlines look mixed rather than clearly positive or negative.",
    chipClasses: "border-amber-200 bg-amber-50 text-amber-800",
    panelClasses: "border-amber-200/80 bg-amber-50/70",
    itemClasses: "border-amber-100/90 bg-white hover:border-amber-200 hover:bg-amber-50/40",
  };
}

function getHeadlineSentimentTone(item: NewsItem) {
  const score = typeof item.sentiment_score === "number" ? item.sentiment_score : null;
  const label = (item.sentiment_label ?? "").toString().toLowerCase();

  if ((score !== null && score >= 0.05) || label === "positive") {
    return {
      label: "Positive",
      scoreLabel: score === null ? "—" : score.toFixed(2),
      chipClasses: "border-emerald-200 bg-emerald-50 text-emerald-700",
      cardClasses: "border-emerald-200/80 bg-emerald-50/35 hover:border-emerald-300 hover:bg-emerald-50/55",
      scoreClasses: "text-emerald-700",
    };
  }

  if ((score !== null && score <= -0.05) || label === "negative") {
    return {
      label: "Negative",
      scoreLabel: score === null ? "—" : score.toFixed(2),
      chipClasses: "border-rose-200 bg-rose-50 text-rose-700",
      cardClasses: "border-rose-200/80 bg-rose-50/35 hover:border-rose-300 hover:bg-rose-50/55",
      scoreClasses: "text-rose-700",
    };
  }

  return {
    label: "Neutral",
    scoreLabel: score === null ? "—" : score.toFixed(2),
    chipClasses: "border-amber-200 bg-amber-50 text-amber-800",
    cardClasses: "border-amber-200/80 bg-amber-50/35 hover:border-amber-300 hover:bg-amber-50/55",
    scoreClasses: "text-amber-800",
  };
}

function getCacheTone(cacheStatus: CacheStatus | null | undefined) {
  if (!cacheStatus) {
    return {
      label: "Mode pending",
      chipClasses: "border-slate-200 bg-slate-100 text-slate-700",
      summary: "No cache metadata was returned for this snapshot.",
    };
  }

  if (cacheStatus.source_mode === "live_api") {
    return {
      label: "Live API",
      chipClasses: "border-teal-200 bg-teal-50 text-teal-700",
      summary:
        "This prediction used live market and news inputs directly from upstream providers.",
    };
  }

  if (cacheStatus.stale_served) {
    return {
      label: "BQ cache stale",
      chipClasses: "border-amber-200 bg-amber-50 text-amber-800",
      summary:
        "This prediction reused a cached BigQuery snapshot because the ticker hit its daily live-fetch budget.",
    };
  }

  return {
    label: "BQ cache",
    chipClasses: "border-indigo-200 bg-indigo-50 text-indigo-700",
    summary:
      "This prediction reused a recent BigQuery snapshot instead of calling live providers again.",
  };
}

function normalizeHeadline(headline: string): string {
  return headline
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function getDedupedNewsItems(items: NewsItem[]): NewsItem[] {
  const sorted = [...items].sort((left, right) => {
    const leftTime = left.published_at ? new Date(left.published_at).getTime() : 0;
    const rightTime = right.published_at ? new Date(right.published_at).getTime() : 0;
    return rightTime - leftTime;
  });

  const seen = new Set<string>();
  const deduped: NewsItem[] = [];

  for (const item of sorted) {
    const key = normalizeHeadline(item.headline);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(item);
  }

  return deduped;
}

function MetricCard({
  icon,
  label,
  value,
  detail,
  accent = "text-slate-950",
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
  accent?: string;
}) {
  return (
    <div className="rounded-[26px] border border-white/70 bg-white/80 p-4 shadow-[0_18px_60px_-42px_rgba(15,23,42,0.45)] backdrop-blur">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
        {icon}
        {label}
      </div>
      <div className={clsx("mt-3 text-2xl font-semibold tracking-tight", accent)}>{value}</div>
      <p className="mt-2 text-sm leading-6 text-slate-600">{detail}</p>
    </div>
  );
}

function TopDriverCards({
  driverFeatures,
}: {
  driverFeatures: FeatureRow[];
}) {
  if (driverFeatures.length === 0) {
    return null;
  }

  return (
    <div className="rounded-[32px] border border-white/70 bg-white/80 p-5 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.45)] backdrop-blur">
      <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
        <Brain className="h-4 w-4 text-slate-500" />
        Top live drivers
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-600">
        These are the highest-importance features for the current model version, shown
        with their live values for the requested ticker.
      </p>
      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {driverFeatures.map((feature) => (
          <div
            key={feature.name}
            className="rounded-[24px] border border-slate-200/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.96),rgba(245,247,250,0.95))] p-4"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  {feature.groupLabel}
                </p>
                <h3 className="mt-2 text-base font-semibold text-slate-950">
                  {feature.label}
                </h3>
              </div>
              <span
                className={clsx(
                  "rounded-full px-2.5 py-1 text-[11px] font-medium",
                  getSourceClasses(feature.source),
                )}
              >
                {feature.source}
              </span>
            </div>
            <div className={clsx("mt-4 text-2xl font-semibold", getFeatureValueTone(feature))}>
              {formatFeatureValue(feature)}
            </div>
            <p className="mt-2 text-xs text-slate-500">{feature.raw_fields.join(", ") || "—"}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function getConvictionClasses(conviction: SwarmVerdictEvent["conviction"]): string {
  if (conviction === "high") {
    return "border-rose-200 bg-rose-50 text-rose-700";
  }
  if (conviction === "medium") {
    return "border-amber-200 bg-amber-50 text-amber-800";
  }
  return "border-slate-200 bg-slate-100 text-slate-700";
}

function formatSwarmLogTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Pending";
  }
  return date.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

function getSwarmLogTone(kind: SwarmLogEntry["kind"]): string {
  if (kind === "error") {
    return "border-rose-200 bg-rose-50/90";
  }
  if (kind === "interview") {
    return "border-amber-200 bg-amber-50/80";
  }
  if (kind === "stream") {
    return "border-teal-200 bg-teal-50/70";
  }
  return "border-slate-200 bg-slate-50/90";
}

function getSwarmStatusLabel(status: SwarmTickerState["status"] | undefined): string {
  if (status === "streaming") {
    return "In progress";
  }
  if (status === "done") {
    return "Ready";
  }
  if (status === "error") {
    return "Needs attention";
  }
  return "Standing by";
}

function getSwarmStatusDetail(status: SwarmTickerState["status"] | undefined): string {
  if (status === "streaming") {
    return "The analyst bots are reviewing price action, news tone, and historical signals now.";
  }
  if (status === "done") {
    return "You can review the debate, see the final call, and ask follow-up questions.";
  }
  if (status === "error") {
    return "The last run did not finish cleanly. You can retry without losing the visible log.";
  }
  return "Run deep analysis when you want a slower, more conversational second opinion.";
}

function getSwarmLogEmoji(kind: SwarmLogEntry["kind"]): string {
  if (kind === "error") {
    return "⚠️";
  }
  if (kind === "interview") {
    return "💬";
  }
  if (kind === "stream") {
    return "🤖";
  }
  return "🫶";
}

function SwarmAnalysisPanel({
  ticker,
  swarm,
  onRetry,
  onAsk,
  interviewAgent,
  setInterviewAgent,
  interviewQuestion,
  setInterviewQuestion,
  interviewLoading,
  variant = "live",
  archivedAt,
  nowMs,
}: {
  ticker: string;
  swarm: SwarmTickerState | null;
  onRetry?: () => void;
  onAsk?: () => void;
  interviewAgent?: string;
  setInterviewAgent?: (value: string) => void;
  interviewQuestion?: string;
  setInterviewQuestion?: (value: string) => void;
  interviewLoading?: boolean;
  variant?: "live" | "archive";
  archivedAt?: string | null;
  nowMs?: number;
}) {
  type DiscussionMessage = {
    id: string;
    speaker: string;
    role: "agent" | "moderator";
    stage: string;
    body: string;
    meta: string;
    at: string;
  };

  const [selectedRoundOverride, setSelectedRoundOverride] = useState<SwarmRoundView | null>(null);
  const agents = Object.values(swarm?.agents ?? {});
  const hasRound2 = agents.some((agent) => Boolean(agent.round2));
  const hasFinal = Boolean(swarm?.moderator);
  const round1Count = agents.filter((agent) => Boolean(agent.round1)).length;
  const round2Count = agents.filter((agent) => Boolean(agent.round2)).length;
  const selectedRound = selectedRoundOverride ?? (hasFinal ? "final" : hasRound2 ? "round2" : "round1");
  const showAskPanel =
    variant === "live" &&
    Boolean(swarm?.deliberationId && onAsk && setInterviewAgent && setInterviewQuestion);

  const discussionMessages: DiscussionMessage[] = (swarm?.logs ?? []).flatMap<DiscussionMessage>((entry) => {
    if (entry.kind !== "stream" || !entry.payload) {
      return [];
    }

    const payload = entry.payload;
    const roundLabel = entry.title.includes("Round 1")
      ? "round1"
      : entry.title.includes("Round 2")
        ? "round2"
        : entry.title === "Moderator verdict"
          ? "final"
          : null;

    if (selectedRound === "round1" && roundLabel !== "round1") {
      return [];
    }
    if (selectedRound === "round2" && roundLabel !== "round2") {
      return [];
    }
    if (selectedRound === "final" && roundLabel !== "final") {
      return [];
    }

    if (roundLabel === "final") {
      const directionSummary = `${String(payload.final_direction ?? "Pending")} at ${typeof payload.final_confidence === "number" ? `${((payload.final_confidence as number) * 100).toFixed(1)}% confidence` : "unknown confidence"}`;
      const consensusSummary = typeof payload.consensus === "string" ? payload.consensus : "";
      return [{
        id: `${entry.id}-moderator`,
        speaker: "Moderator",
        role: "moderator" as const,
        stage: "Final conclusion",
        body: [directionSummary, consensusSummary].filter(Boolean).join(". "),
        meta: payload.direction_overridden ? "Override applied" : "Consensus close",
        at: entry.at,
      }];
    }

    const speaker = String(payload.agent ?? "Agent");
    const direction = String(payload.direction ?? "Pending");
    const conviction = String(payload.conviction ?? "unknown");
    const thesis = String(payload.thesis ?? "");
    const baseMessage = {
      id: `${entry.id}-thesis`,
      speaker,
      role: "agent" as const,
      stage: roundLabel === "round1" ? "Round 1 final thought" : "Round 2 final thought",
      body: thesis,
      meta: `${direction} · ${conviction}`,
      at: entry.at,
    };

    if (roundLabel === "round1") {
      return [baseMessage];
    }

    return [baseMessage];
  });

  const renderAgentCard = (agent: SwarmAgentState, round: SwarmRoundView) => {
    if (round === "final") {
      const finalVerdict = agent.round2 ?? agent.round1;
      return (
        <div
          key={`${agent.agentId}-final`}
          className="rounded-[24px] border border-cyan-300/18 bg-[linear-gradient(180deg,rgba(255,255,255,0.10),rgba(255,255,255,0.04))] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_18px_50px_-36px_rgba(34,211,238,0.55)] backdrop-blur-md"
        >
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-200/70">
                Analyst
              </div>
              <h3 className="mt-1 text-lg font-semibold text-white">{agent.agent}</h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {finalVerdict && (
                <span className={clsx("rounded-full border px-2.5 py-1 text-[11px] font-semibold", getConvictionClasses(finalVerdict.conviction))}>
                  {finalVerdict.conviction}
                </span>
              )}
              <span
                className={clsx(
                  "rounded-full px-2.5 py-1 text-[11px] font-semibold",
                  (finalVerdict?.direction ?? "Up") === "Up"
                    ? "bg-emerald-50 text-emerald-700"
                    : "bg-rose-50 text-rose-700",
                )}
              >
                {finalVerdict?.direction ?? "Pending"}
              </span>
            </div>
          </div>

          <div className="mt-4 rounded-[18px] border border-white/10 bg-slate-950/35 p-3">
            <div className="flex items-center justify-between gap-3 text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-200/65">
              <span>Final thought</span>
              {agent.round2?.revised && <span className="text-amber-300">Changed view</span>}
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-100/85">
              {finalVerdict?.thesis ?? "Waiting for final stance…"}
            </p>
          </div>
        </div>
      );
    }

    const verdict = round === "round2" ? agent.round2 : agent.round1;
    const heading = round === "round2" ? "Round 2 final thought" : "Round 1 final thought";
    const emptyCopy = round === "round2" ? "Waiting for the rebuttal round…" : "Waiting for first-pass verdict…";

    return (
      <div
        key={`${agent.agentId}-${round}`}
        className="rounded-[24px] border border-cyan-300/18 bg-[linear-gradient(180deg,rgba(255,255,255,0.10),rgba(255,255,255,0.04))] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_18px_50px_-36px_rgba(34,211,238,0.55)] backdrop-blur-md"
      >
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-200/70">
              Analyst
            </div>
            <h3 className="mt-1 text-lg font-semibold text-white">{agent.agent}</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {verdict && (
              <span className={clsx("rounded-full border px-2.5 py-1 text-[11px] font-semibold", getConvictionClasses(verdict.conviction))}>
                {verdict.conviction}
              </span>
            )}
            <span
              className={clsx(
                "rounded-full px-2.5 py-1 text-[11px] font-semibold",
                (verdict?.direction ?? "Up") === "Up"
                  ? "bg-emerald-50 text-emerald-700"
                  : "bg-rose-50 text-rose-700",
              )}
            >
              {verdict?.direction ?? "Pending"}
            </span>
          </div>
        </div>

        <div className="mt-4 space-y-3">
          <div className="rounded-[18px] border border-white/10 bg-slate-950/35 p-3">
            <div className="flex items-center justify-between gap-3 text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-200/65">
              <span>{heading}</span>
              {round === "round2" && verdict?.revised && <span className="text-amber-300">Changed view</span>}
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-100/85">
              {verdict?.thesis ?? emptyCopy}
            </p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="relative overflow-hidden rounded-[34px] border border-cyan-400/25 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.22),transparent_34%),radial-gradient(circle_at_top_right,rgba(45,212,191,0.18),transparent_28%),linear-gradient(145deg,rgba(2,6,23,0.96),rgba(7,19,43,0.94),rgba(3,10,28,0.98))] p-5 text-white shadow-[0_34px_120px_-52px_rgba(8,145,178,0.55)]">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.06)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.05)_1px,transparent_1px)] bg-[size:32px_32px] opacity-40" />
      <div className="pointer-events-none absolute inset-x-6 top-0 h-px bg-gradient-to-r from-transparent via-cyan-300/70 to-transparent" />

      <div className="relative flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-cyan-100">
            <Brain className="h-4 w-4 text-cyan-300" />
            {variant === "archive" ? "Deep analysis archive" : "Swarm analysis"}
          </div>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-200/85">
            Four analyst bots review the ML call for {ticker}, debate it in two rounds, and then
            a moderator bot settles the final read. Pick a round to replay the discussion as it unfolds.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-100">
            {swarm?.source === "archive" ? "Archive replay" : "Live stream"}
          </span>
          <span className="rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-100">
            {getSwarmStatusLabel(swarm?.status)}
          </span>
          {archivedAt && (
            <span className="rounded-full border border-cyan-300/20 bg-white/8 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-200/90">
              saved {formatRelativeTime(archivedAt, nowMs ?? 0)}
            </span>
          )}
          {onRetry && (!swarm || swarm.status === "error") && (
            <button
              onClick={onRetry}
              className="rounded-full border border-cyan-300/30 bg-[linear-gradient(135deg,rgba(34,211,238,0.28),rgba(59,130,246,0.28))] px-4 py-2 text-sm font-semibold text-white transition hover:brightness-110"
            >
              {variant === "archive" ? "Run fresh analysis" : swarm ? "Try again" : "Run analysis"}
            </button>
          )}
        </div>
      </div>

      <div className="relative mt-4 rounded-[22px] border border-cyan-300/20 bg-white/8 px-4 py-3 backdrop-blur-md">
        <div className="flex items-start gap-3">
          <div>
            <div className="text-sm font-semibold text-white">Live cockpit status</div>
            <p className="mt-1 text-sm leading-6 text-slate-200/80">
              {swarm?.source === "archive"
                ? "This is a saved deep-analysis replay linked to an earlier prediction. You can inspect every round without triggering a new model call."
                : getSwarmStatusDetail(swarm?.status)}
            </p>
          </div>
        </div>
      </div>

      {swarm?.error && (
        <div className="mt-4 rounded-[20px] border border-rose-400/30 bg-rose-400/12 px-4 py-3 text-sm text-rose-100">
          {swarm.error}
        </div>
      )}

      <div className="relative mt-5 rounded-[24px] border border-cyan-300/18 bg-white/7 p-4 backdrop-blur-md">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              <MessageSquare className="h-4 w-4 text-cyan-300" />
              Discussion rounds
            </div>
            <p className="mt-1 text-sm text-slate-200/75">
              Switch between opening takes, rebuttal round, and the final moderator call.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedRoundOverride("round1")}
              className={clsx(
                "rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] transition",
                selectedRound === "round1"
                  ? "border-cyan-300/40 bg-cyan-400/20 text-white"
                  : "border-cyan-300/18 bg-white/6 text-slate-200/80 hover:bg-white/10",
              )}
            >
              Round 1 · {round1Count}/4
            </button>
            <button
              onClick={() => setSelectedRoundOverride("round2")}
              className={clsx(
                "rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] transition",
                selectedRound === "round2"
                  ? "border-cyan-300/40 bg-cyan-400/20 text-white"
                  : "border-cyan-300/18 bg-white/6 text-slate-200/80 hover:bg-white/10",
              )}
            >
              Round 2 · {round2Count}/4
            </button>
            <button
              onClick={() => setSelectedRoundOverride("final")}
              className={clsx(
                "rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] transition",
                selectedRound === "final"
                  ? "border-cyan-300/40 bg-cyan-400/20 text-white"
                  : "border-cyan-300/18 bg-white/6 text-slate-200/80 hover:bg-white/10",
              )}
            >
              Final call
            </button>
          </div>
        </div>

        <div className="mt-4 rounded-[22px] border border-cyan-300/15 bg-slate-950/20 p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-white">
                {variant === "archive" ? "Replay transcript" : "Live conversation"}
              </div>
              <p className="mt-1 text-sm text-slate-200/75">
                {selectedRound === "round1"
                  ? "Each agent's round 1 closing thought lands here."
                  : selectedRound === "round2"
                    ? "Each agent's round 2 closing thought lands here."
                    : "The moderator's final conclusion lands here."}
              </p>
            </div>
            <span className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-100/90">
              {discussionMessages.length} messages
            </span>
          </div>

          <div className="mt-4 max-h-[28rem] space-y-3 overflow-y-auto pr-1">
            {discussionMessages.length > 0 ? (
              <AnimatePresence initial={false}>
                {discussionMessages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.18 }}
                    className={clsx(
                      "rounded-[20px] border px-4 py-3",
                      message.role === "moderator"
                        ? "border-fuchsia-300/25 bg-fuchsia-400/10"
                        : "border-cyan-300/18 bg-white/7",
                    )}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span
                          className={clsx(
                            "rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]",
                            message.role === "moderator"
                              ? "bg-fuchsia-200/20 text-fuchsia-100"
                              : "bg-cyan-200/15 text-cyan-100",
                          )}
                        >
                          {message.speaker}
                        </span>
                        <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-300/80">
                          {message.stage}
                        </span>
                      </div>
                      <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-300/70">
                        {formatSwarmLogTime(message.at)}
                      </span>
                    </div>
                    <p className="mt-3 text-sm leading-6 text-slate-100/90">{message.body}</p>
                    <div className="mt-3 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-300/75">
                      {message.meta}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            ) : variant === "live" && swarm?.status === "streaming" ? (
              <div className="space-y-3">
                {Array.from({ length: 4 }).map((_, index) => (
                  <div key={index} className="animate-pulse rounded-[20px] border border-cyan-300/12 bg-white/6 px-4 py-3">
                    <div className="h-3 w-28 rounded bg-cyan-200/15" />
                    <div className="mt-3 h-3 w-full rounded bg-white/10" />
                    <div className="mt-2 h-3 w-5/6 rounded bg-white/10" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-[18px] border border-cyan-300/15 bg-white/6 px-4 py-3 text-sm text-slate-200/75">
                Swarm messages will appear here as soon as the selected round has content.
              </div>
            )}
          </div>
        </div>

        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          {agents.length > 0 ? (
            agents.map((agent) => renderAgentCard(agent, selectedRound))
          ) : (
            <div className="rounded-[18px] border border-cyan-300/15 bg-white/6 px-4 py-3 text-sm text-slate-200/75 lg:col-span-2">
              Swarm activity will appear here as soon as deep analysis starts.
            </div>
          )}
        </div>
      </div>

      {swarm?.status === "streaming" && Object.keys(swarm.agents).length === 0 && (
        <div className="mt-5 flex items-center gap-2 text-sm text-cyan-100/80">
          <Loader2 className="h-4 w-4 animate-spin" />
          Analyst bots are forming their first read…
        </div>
      )}

      {swarm?.moderator && selectedRound === "final" && (
        <div className="mt-5 rounded-[24px] border border-fuchsia-300/20 bg-[linear-gradient(145deg,rgba(91,33,182,0.32),rgba(15,23,42,0.96),rgba(8,47,73,0.88))] p-5 text-white shadow-[0_24px_80px_-44px_rgba(168,85,247,0.55)]">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-fuchsia-100/70">
                Final conclusion
              </div>
              <div className="mt-2 text-2xl font-semibold tracking-tight">
                {swarm.moderator.final_direction} · {(swarm.moderator.final_confidence * 100).toFixed(1)}%
              </div>
            </div>
            <div className="flex flex-wrap gap-2 text-xs font-semibold uppercase tracking-[0.16em]">
              <span className="rounded-full border border-white/15 bg-white/10 px-3 py-1">
                {swarm.moderator.consensus}
              </span>
              {swarm.moderator.direction_overridden && (
                <span className="rounded-full border border-amber-300/30 bg-amber-400/20 px-3 py-1 text-amber-100">
                  Override
                </span>
              )}
            </div>
          </div>
          {swarm.moderator.dissent && (
            <p className="mt-3 text-sm leading-6 text-white/75">{swarm.moderator.dissent}</p>
          )}
        </div>
      )}

      <div className="mt-5 rounded-[24px] border border-cyan-300/18 bg-white/7 p-4 backdrop-blur-md">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              <Activity className="h-4 w-4 text-cyan-300" />
              Live swarm log
            </div>
            <p className="mt-1 text-sm text-slate-200/75">
              The full stream stays here for audit and debugging, while the round view above keeps only the conclusions.
            </p>
          </div>
          <span className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-cyan-100/90">
            {swarm?.logs.length ?? 0} entries
          </span>
        </div>

        <div className="mt-4 max-h-[26rem] space-y-3 overflow-y-auto pr-1">
          {swarm?.logs.length ? (
            swarm.logs.map((entry) => (
              <div
                key={entry.id}
                className={clsx("rounded-[18px] border px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]", getSwarmLogTone(entry.kind))}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-950">
                    <span>{getSwarmLogEmoji(entry.kind)}</span>
                    <span>{entry.title}</span>
                  </div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {formatSwarmLogTime(entry.at)}
                  </div>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-700">{entry.detail}</p>
                {entry.payload && (
                  <details className="mt-3">
                    <summary className="cursor-pointer text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                      Raw payload
                    </summary>
                    <pre className="mt-2 overflow-x-auto rounded-[14px] bg-slate-950 px-3 py-3 text-xs leading-5 text-slate-100">
                      {JSON.stringify(entry.payload, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ))
          ) : (
            <div className="rounded-[18px] border border-cyan-300/15 bg-white/6 px-4 py-3 text-sm text-slate-200/75">
              Swarm activity will appear here as soon as deep analysis starts.
            </div>
          )}
        </div>
      </div>

      {showAskPanel && swarm?.deliberationId && (
        <div className="mt-5 rounded-[24px] border border-cyan-300/18 bg-white/7 p-4 backdrop-blur-md">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <MessageSquare className="h-4 w-4 text-cyan-300" />
            Ask the analyst bots
          </div>
          <p className="mt-2 text-sm leading-6 text-slate-200/75">
            Ask a plain-English follow-up like you would in a review meeting. The full exchange stays visible below.
          </p>
          <div className="mt-3 grid gap-3 md:grid-cols-[180px_1fr_auto]">
            <select
              value={interviewAgent ?? ""}
              onChange={(event) => setInterviewAgent?.(event.target.value)}
              className="rounded-[18px] border border-cyan-300/20 bg-slate-950/35 px-3 py-3 text-sm text-white outline-none"
            >
              {agents.map((agent) => (
                <option key={agent.agentId} value={agent.agentId}>
                  {agent.agent}
                </option>
              ))}
            </select>
            <input
              value={interviewQuestion ?? ""}
              onChange={(event) => setInterviewQuestion?.(event.target.value)}
              placeholder="Why did you challenge the model?"
              className="rounded-[18px] border border-cyan-300/20 bg-slate-950/35 px-4 py-3 text-sm text-white outline-none placeholder:text-slate-400"
            />
            <button
              onClick={onAsk}
              disabled={!interviewQuestion?.trim() || interviewLoading}
              className="rounded-[18px] border border-cyan-300/20 bg-[linear-gradient(135deg,rgba(45,212,191,0.28),rgba(59,130,246,0.30))] px-4 py-3 text-sm font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {interviewLoading ? "Asking…" : "Ask"}
            </button>
          </div>
          {swarm.interviewTurns.length > 0 && (
            <div className="mt-4 space-y-3">
              {swarm.interviewTurns.map((turn, index) => (
                <div
                  key={`${turn.role}-${index}`}
                  className={clsx(
                    "rounded-[18px] px-4 py-3 text-sm leading-6",
                    turn.role === "user"
                      ? "border border-cyan-300/18 bg-white/10 text-slate-100"
                      : "border border-teal-300/25 bg-teal-300/10 text-teal-50",
                  )}
                >
                  <span className="mr-2 font-semibold uppercase tracking-[0.14em] text-cyan-100/70">
                    {turn.role === "user" ? "🧑 You" : "🤖 Bot"}
                  </span>
                  {turn.message}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function FeatureWorkbench({
  features,
  search,
  setSearch,
  sourceFilter,
  setSourceFilter,
  sortMode,
  setSortMode,
  groupFilter,
  setGroupFilter,
  groups,
}: {
  features: FeatureRow[];
  search: string;
  setSearch: (value: string) => void;
  sourceFilter: SourceFilter;
  setSourceFilter: (value: SourceFilter) => void;
  sortMode: SortMode;
  setSortMode: (value: SortMode) => void;
  groupFilter: string;
  setGroupFilter: (value: string) => void;
  groups: FeatureGroup[];
}) {
  return (
    <div className="rounded-[32px] border border-white/70 bg-white/80 p-5 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.45)] backdrop-blur">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
            <Brain className="h-4 w-4 text-slate-500" />
            All model inputs
          </div>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
            This is the full technical input frame. Headline vector dimensions such as
            `tfidf_*` are internal numeric features, not human-readable explanations.
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-500">
            <Search className="h-4 w-4" />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search features"
              className="w-full bg-transparent text-slate-900 outline-none placeholder:text-slate-400"
            />
          </label>

          <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
              <Filter className="h-3.5 w-3.5" />
              Source
            </div>
            <div className="flex flex-wrap gap-2">
              {(["all", "finnhub", "yahoo", "derived"] as const).map((option) => (
                <button
                  key={option}
                  onClick={() => setSourceFilter(option)}
                  className={clsx(
                    "rounded-full px-2.5 py-1 text-xs font-medium transition",
                    sourceFilter === option
                      ? "bg-slate-900 text-white"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200",
                  )}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
              <BarChart3 className="h-3.5 w-3.5" />
              Sort
            </div>
            <div className="flex flex-wrap gap-2">
              {(["group", "value", "source"] as const).map((option) => (
                <button
                  key={option}
                  onClick={() => setSortMode(option)}
                  className={clsx(
                    "rounded-full px-2.5 py-1 text-xs font-medium capitalize transition",
                    sortMode === option
                      ? "bg-slate-900 text-white"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200",
                  )}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        <button
          onClick={() => setGroupFilter("all")}
          className={clsx(
            "rounded-full px-3 py-1.5 text-xs font-medium transition",
            groupFilter === "all"
              ? "bg-slate-900 text-white"
              : "bg-slate-100 text-slate-600 hover:bg-slate-200",
          )}
        >
          all groups
        </button>
        {groups.map((group) => (
          <button
            key={group.key}
            onClick={() => setGroupFilter(group.key)}
            className={clsx(
              "rounded-full px-3 py-1.5 text-xs font-medium transition",
              groupFilter === group.key
                ? "bg-slate-900 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200",
            )}
          >
            {group.label}
          </button>
        ))}
      </div>

      <div className="mt-5 grid gap-3 xl:grid-cols-2">
        {features.map((feature) => (
          <motion.div
            key={`${feature.groupKey}-${feature.name}`}
            layout
            className="rounded-[24px] border border-slate-200/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.96),rgba(245,247,250,0.95))] p-4"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    {feature.groupLabel}
                  </span>
                  <span
                    className={clsx(
                      "rounded-full px-2.5 py-1 text-[11px] font-medium",
                      getSourceClasses(feature.source),
                    )}
                  >
                    {feature.source}
                  </span>
                  {feature.fallback_used && (
                    <span className="rounded-full bg-amber-100 px-2.5 py-1 text-[11px] font-medium text-amber-800">
                      fallback
                    </span>
                  )}
                </div>
                <h3 className="mt-3 text-base font-semibold text-slate-950">
                  {feature.label}
                </h3>
              </div>
              <div className={clsx("text-right text-xl font-semibold", getFeatureValueTone(feature))}>
                {formatFeatureValue(feature)}
              </div>
            </div>

            <div className="mt-4 space-y-2 text-xs text-slate-500">
              <p>
                <span className="font-semibold uppercase tracking-[0.16em] text-slate-400">
                  providers
                </span>{" "}
                {feature.providers.join(" • ") || "—"}
              </p>
              <p>
                <span className="font-semibold uppercase tracking-[0.16em] text-slate-400">
                  raw fields
                </span>{" "}
                {feature.raw_fields.join(", ") || "—"}
              </p>
            </div>
          </motion.div>
        ))}
      </div>

      {features.length === 0 && (
        <div className="mt-5 rounded-[24px] border border-dashed border-slate-300 bg-slate-50 px-4 py-8 text-center text-sm text-slate-500">
          No features match the current search and filter combination.
        </div>
      )}
    </div>
  );
}

function NewsRail({
  result,
  nowMs,
  tone,
}: {
  result: PredictionResult;
  nowMs: number;
  tone: ReturnType<typeof getNewsTone>;
}) {
  const items = getDedupedNewsItems(result.feature_snapshot.news_items).slice(0, 10);

  return (
    <div
      className={clsx(
        "flex h-[30rem] min-h-0 flex-col rounded-[28px] border p-5 shadow-[0_22px_70px_-44px_rgba(15,23,42,0.45)] backdrop-blur",
        tone.panelClasses,
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
          <Newspaper className="h-4 w-4 text-slate-500" />
          Live news tone
        </div>
        <span
          className={clsx(
            "rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em]",
            tone.chipClasses,
          )}
        >
          {tone.label}
        </span>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <p className="text-sm leading-6 text-slate-700">{tone.summary}</p>
        <span className="rounded-full bg-white/80 px-3 py-1 text-xs font-medium text-slate-600">
          {items.length} shown / {result.feature_snapshot.news_items.length} raw
        </span>
      </div>

      <p className="mt-2 text-sm leading-6 text-slate-600">
        Duplicate headlines are collapsed here. The raw headline count is still preserved
        for the model input.
      </p>

      <div className="mt-4 min-h-0 flex-1 overflow-hidden rounded-[24px] border border-white/60 bg-white/55 p-2">
        <div className="h-full space-y-3 overflow-y-auto pr-1">
          {items.map((item, index) => {
            const sentimentTone = getHeadlineSentimentTone(item);
            return (
              <a
                key={`${item.headline}-${index}`}
                href={item.url ?? undefined}
                target={item.url ? "_blank" : undefined}
                rel={item.url ? "noreferrer" : undefined}
                className={clsx(
                  "block rounded-[22px] border px-4 py-3 transition",
                  tone.itemClasses,
                  sentimentTone.cardClasses,
                )}
              >
                <div className="flex flex-wrap items-center gap-2 text-[11px] font-medium uppercase tracking-[0.18em] text-slate-400">
                  <span>{item.provider}</span>
                  {item.source ? <span>{item.source}</span> : null}
                  {item.published_at ? <span>{formatRelativeTime(item.published_at, nowMs)}</span> : null}
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <span
                    className={clsx(
                      "rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.16em]",
                      sentimentTone.chipClasses,
                    )}
                  >
                    {sentimentTone.label}
                  </span>
                  <span className={clsx("text-xs font-semibold", sentimentTone.scoreClasses)}>
                    Gemini {sentimentTone.scoreLabel}
                  </span>
                  {item.category ? (
                    <span className="rounded-full bg-white/80 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.14em] text-slate-600">
                      {item.category}
                    </span>
                  ) : null}
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-800">{item.headline}</p>
              </a>
            );
          })}
        </div>
      </div>
    </div>
  );
}

interface ChartPoint {
  date: string;
  close: number;
}

function PriceChart({
  points,
  ticker,
  currency,
  prediction,
}: {
  points: ChartPoint[];
  ticker: string;
  currency: string | null;
  prediction: "Up" | "Down";
}) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; point: ChartPoint } | null>(null);

  if (points.length < 2) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-slate-400">
        Not enough data to render chart
      </div>
    );
  }

  const W = 900;
  const H = 320;
  const padL = 70;
  const padR = 24;
  const padT = 18;
  const padB = 52;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const prices = points.map((p) => p.close);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceRange = maxPrice - minPrice || 1;

  const xOf = (i: number) => padL + (i / (points.length - 1)) * innerW;
  const yOf = (price: number) => padT + (1 - (price - minPrice) / priceRange) * innerH;

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(i).toFixed(1)} ${yOf(p.close).toFixed(1)}`)
    .join(" ");

  const areaD =
    pathD +
    ` L ${xOf(points.length - 1).toFixed(1)} ${(padT + innerH).toFixed(1)} L ${padL.toFixed(1)} ${(padT + innerH).toFixed(1)} Z`;

  const isUp = prediction === "Up";
  const strokeColor = isUp ? "#10b981" : "#f43f5e";
  const gradientId = `chart-grad-${ticker}`;

  const yTicks = 4;
  const yStep = priceRange / yTicks;
  const yTickValues = Array.from({ length: yTicks + 1 }, (_, i) => minPrice + i * yStep);

  const xTickCount = Math.min(7, points.length);
  const xTickIndices = Array.from({ length: xTickCount }, (_, i) =>
    Math.round((i / (xTickCount - 1)) * (points.length - 1)),
  );

  const lastClose = points[points.length - 1].close;
  const firstClose = points[0].close;
  const totalReturn = ((lastClose - firstClose) / firstClose) * 100;

  const formatPrice = (v: number) =>
    currency === "USD" || !currency ? `$${v.toFixed(2)}` : `${v.toFixed(2)} ${currency}`;
  const yUnit = currency ? currency.toUpperCase() : "USD";

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const svgX = ((e.clientX - rect.left) / rect.width) * W;
    const relX = svgX - padL;
    const idx = Math.max(0, Math.min(points.length - 1, Math.round((relX / innerW) * (points.length - 1))));
    const point = points[idx];
    setTooltip({ x: xOf(idx), y: yOf(point.close), point });
  };

  return (
    <div className="relative select-none">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
          {ticker} · 1-year close price
        </span>
        <span
          className={clsx(
            "rounded-full px-3 py-1 text-xs font-semibold",
            totalReturn >= 0 ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700",
          )}
        >
          {totalReturn >= 0 ? "+" : ""}
          {totalReturn.toFixed(2)}% 1Y range
        </span>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={strokeColor} stopOpacity="0.18" />
            <stop offset="100%" stopColor={strokeColor} stopOpacity="0.02" />
          </linearGradient>
          <clipPath id={`clip-${ticker}`}>
            <rect x={padL} y={padT} width={innerW} height={innerH} />
          </clipPath>
        </defs>

        <text
          x={12}
          y={padT + innerH / 2}
          textAnchor="middle"
          fontSize="10"
          fill="#94a3b8"
          fontFamily="ui-sans-serif, sans-serif"
          transform={`rotate(-90 12 ${padT + innerH / 2})`}
        >
          {yUnit}
        </text>

        {yTickValues.map((val) => (
          <g key={val}>
            <line
              x1={padL}
              x2={padL + innerW}
              y1={yOf(val)}
              y2={yOf(val)}
              stroke="#e2e8f0"
              strokeWidth="1"
            />
            <text
              x={padL - 6}
              y={yOf(val) + 4}
              textAnchor="end"
              fontSize="10"
              fill="#94a3b8"
              fontFamily="ui-monospace, monospace"
            >
              {currency === "USD" || !currency
                ? `$${val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val.toFixed(1)}`
                : val >= 1000
                  ? `${(val / 1000).toFixed(1)}k ${currency}`
                  : `${val.toFixed(1)} ${currency}`}
            </text>
          </g>
        ))}

        {xTickIndices.map((idx) => (
          <text
            key={idx}
            x={xOf(idx)}
            y={padT + innerH + 24}
            textAnchor="middle"
            fontSize="10"
            fill="#94a3b8"
            fontFamily="ui-sans-serif, sans-serif"
          >
            {points[idx].date
              ? new Date(points[idx].date).toLocaleDateString("en-US", { month: "short", year: "2-digit" })
              : ""}
          </text>
        ))}

        <path d={areaD} fill={`url(#${gradientId})`} clipPath={`url(#clip-${ticker})`} />
        <path
          d={pathD}
          fill="none"
          stroke={strokeColor}
          strokeWidth="1.8"
          strokeLinejoin="round"
          strokeLinecap="round"
          clipPath={`url(#clip-${ticker})`}
        />

        <line
          x1={padL}
          x2={padL + innerW}
          y1={yOf(lastClose)}
          y2={yOf(lastClose)}
          stroke={strokeColor}
          strokeWidth="1"
          strokeDasharray="3 3"
          opacity="0.5"
        />

        <circle
          cx={xOf(points.length - 1)}
          cy={yOf(lastClose)}
          r="4"
          fill={strokeColor}
          stroke="white"
          strokeWidth="1.5"
        />

        {tooltip && (
          <>
            <line
              x1={tooltip.x}
              x2={tooltip.x}
              y1={padT}
              y2={padT + innerH}
              stroke="#64748b"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
            <circle cx={tooltip.x} cy={tooltip.y} r="4" fill={strokeColor} stroke="white" strokeWidth="1.5" />
            <rect
              x={Math.min(tooltip.x + 8, W - 118)}
              y={Math.max(padT, tooltip.y - 26)}
              width="110"
              height="40"
              rx="6"
              fill="rgba(15,23,42,0.88)"
            />
            <text
              x={Math.min(tooltip.x + 63, W - 63)}
              y={Math.max(padT + 15, tooltip.y - 11)}
              textAnchor="middle"
              fontSize="10"
              fill="rgba(255,255,255,0.65)"
              fontFamily="ui-sans-serif, sans-serif"
            >
              {tooltip.point.date}
            </text>
            <text
              x={Math.min(tooltip.x + 63, W - 63)}
              y={Math.max(padT + 30, tooltip.y + 4)}
              textAnchor="middle"
              fontSize="12"
              fontWeight="600"
              fill="white"
              fontFamily="ui-monospace, monospace"
            >
              {formatPrice(tooltip.point.close)}
            </text>
          </>
        )}
      </svg>
    </div>
  );
}

export default function Home() {
  const { data: session } = useSession();
  const [ticker, setTicker] = useState("");
  const [tickers, setTickers] = useState<string[]>(DEFAULT_TICKERS);
  const [exchange, setExchange] = useState<Exchange>("NYSE");
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("quick");
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [selectedResultKey, setSelectedResultKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [swarmByTicker, setSwarmByTicker] = useState<Record<string, SwarmTickerState>>({});
  const [interviewQuestion, setInterviewQuestion] = useState("");
  const [interviewAgent, setInterviewAgent] = useState("");
  const [interviewLoading, setInterviewLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [sortMode, setSortMode] = useState<SortMode>("group");
  const [groupFilter, setGroupFilter] = useState<string>("all");
  const [showAllInputs, setShowAllInputs] = useState(false);
  const [nowMs, setNowMs] = useState(Date.now());
  const [chartPoints, setChartPoints] = useState<ChartPoint[]>([]);
  const [chartCurrency, setChartCurrency] = useState<string | null>(null);
  const [chartTicker, setChartTicker] = useState<string | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [feedbackDirection, setFeedbackDirection] = useState<"Up" | "Down" | null>(null);
  const [feedbackSentiment, setFeedbackSentiment] = useState<"bullish" | "bearish" | "neutral" | null>(null);
  const [feedbackConfidence, setFeedbackConfidence] = useState<number>(3);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const [feedbackSuccess, setFeedbackSuccess] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [usageCount, setUsageCount] = useState(0);
  const [usageLimit, setUsageLimit] = useState(3);
  const [usedTickersToday, setUsedTickersToday] = useState<string[]>([]);
  const [usageLoaded, setUsageLoaded] = useState(false);
  const [archiveHistoryKey, setArchiveHistoryKey] = useState<string | null>(null);
  const [archivedSwarm, setArchivedSwarm] = useState<SwarmTickerState | null>(null);
  const [archiveLoading, setArchiveLoading] = useState(false);
  const [archiveError, setArchiveError] = useState<string | null>(null);
  const autoBootstrapped = useRef(false);
  const swarmAbortControllers = useRef<Record<string, AbortController>>({});
  const swarmArchiveLoads = useRef<Record<string, boolean>>({});
  const activeSwarmPanelRef = useRef<HTMLDivElement | null>(null);
  const totalTickerCount = tickers.length;

  function getResultKey(result: Pick<PredictionResult, "request_id" | "ticker">) {
    return `${result.request_id}:${result.ticker}`;
  }

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const storedResults = window.localStorage.getItem("predictionResults");
      const storedTickers = window.localStorage.getItem("predictionTickers");
      const storedTicker = window.localStorage.getItem("selectedTicker");

      let restoredResults: PredictionResult[] = [];
      if (storedResults) {
        const parsed = JSON.parse(storedResults);
        restoredResults = sanitizePredictionResults(parsed);
        setResults(restoredResults);
      }

      if (storedTickers) {
        const parsed = JSON.parse(storedTickers);
        if (Array.isArray(parsed)) {
          const restoredTickers = parsed
            .map((value) => (typeof value === "string" ? value.trim().toUpperCase() : ""))
            .filter((value): value is string => Boolean(value));

          if (restoredTickers.length > 0) {
            setTickers(restoredTickers);
          }
        }
      } else if (restoredResults.length > 0) {
        setTickers(restoredResults.map((result) => result.ticker));
      }

      if (storedTicker) {
        setSelectedTicker(storedTicker);
      }
    } catch {
      // Ignore malformed local storage values.
    } finally {
      setHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!hydrated || typeof window === "undefined") return;
    window.localStorage.setItem("predictionResults", JSON.stringify(results));
  }, [results, hydrated]);

  useEffect(() => {
    if (!hydrated || typeof window === "undefined") return;
    window.localStorage.setItem("predictionTickers", JSON.stringify(tickers));
  }, [tickers, hydrated]);

  useEffect(() => {
    if (!hydrated || typeof window === "undefined") return;
    if (selectedTicker) {
      window.localStorage.setItem("selectedTicker", selectedTicker);
    } else {
      window.localStorage.removeItem("selectedTicker");
    }
  }, [selectedTicker, hydrated]);

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 15000);
    return () => window.clearInterval(timer);
  }, []);

  const fetchUsage = async () => {
    try {
      const res = await fetch("/api/usage");
      if (res.ok) {
        const data = await res.json();
        setUsageCount(data.prediction_count ?? 0);
        setUsageLimit(data.daily_limit ?? 3);
        setUsedTickersToday(
          Array.isArray(data.used_tickers)
            ? data.used_tickers
                .map((value: unknown) => (typeof value === "string" ? value.trim().toUpperCase() : ""))
                .filter((value: string) => Boolean(value))
            : [],
        );
      }
    } catch {
      // silently fail
    } finally {
      setUsageLoaded(true);
    }
  };

  useEffect(() => {
    if (!hydrated) return;

    if (session?.user?.email) {
      fetchHistory();
      fetchUsage();
    } else {
      setHistory([]);
      setUsageCount(0);
      setUsedTickersToday([]);
      setUsageLoaded(true);
      setTickers((prev) => {
        const guestTickers = prev.filter((tickerItem) => isDefaultTicker(tickerItem));
        return guestTickers.length > 0 ? guestTickers : DEFAULT_TICKERS;
      });
      setResults((prev) => prev.filter((result) => isDefaultTicker(result.ticker)));
      setSelectedTicker((current) => (current && isDefaultTicker(current) ? current : null));
      setSelectedResultKey(null);
    }
  }, [hydrated, session]);

  useEffect(() => {
    const ticker = selectedTicker;
    if (!ticker) return;
    if (ticker === chartTicker) return;
    setChartLoading(true);
    setChartPoints([]);
    fetch(`/api/chart?ticker=${encodeURIComponent(ticker)}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.points) {
          setChartPoints(data.points);
          setChartCurrency(data.currency ?? null);
          setChartTicker(ticker);
        }
      })
      .catch(() => { })
      .finally(() => setChartLoading(false));
  }, [selectedTicker, chartTicker]);

  const fetchHistory = async () => {
    setLoadingHistory(true);
    try {
      const res = await fetch("/api/history");
      if (res.ok) {
        const data = await res.json();
        setHistory(sanitizeHistoryItems(data.history));
      }
    } catch (err) {
      console.error("Failed to fetch history:", err);
    } finally {
      setLoadingHistory(false);
    }
  };

  const loadSwarmArchive = async (deliberationId: string): Promise<SwarmTickerState> => {
    const response = await fetch(`/api/swarm/deliberation/${encodeURIComponent(deliberationId)}`);
    const payload = await response.json().catch(() => ({ error: "Failed to load archived deep analysis." }));
    if (!response.ok) {
      throw new Error(getErrorMessage((payload as { error?: unknown }).error));
    }

    const replayState = buildSwarmReplayState(payload);
    if (!replayState) {
      throw new Error("Archived deep analysis could not be parsed.");
    }
    return replayState;
  };

  const openArchivedSwarm = async (item: HistoryItem) => {
    if (!item.swarm_deliberation_id) {
      return;
    }

    setArchiveHistoryKey(buildHistoryKey(item));
    setArchiveLoading(true);
    setArchiveError(null);
    setArchivedSwarm(null);
    try {
      const replayState = await loadSwarmArchive(item.swarm_deliberation_id);
      setArchivedSwarm(replayState);
    } catch (err: unknown) {
      setArchivedSwarm(null);
      setArchiveError(getErrorMessage(err));
    } finally {
      setArchiveLoading(false);
    }
  };

  const hydrateActiveSwarmFromArchive = async (result: PredictionResult, deliberationId: string) => {
    const loadKey = `${result.ticker}:${result.request_id}:${deliberationId}`;
    if (swarmArchiveLoads.current[loadKey]) {
      return;
    }

    swarmArchiveLoads.current[loadKey] = true;
    try {
      const replayState = await loadSwarmArchive(deliberationId);
      setSwarmByTicker((current) => ({
        ...current,
        [result.ticker]: replayState,
      }));
    } catch {
      delete swarmArchiveLoads.current[loadKey];
    }
  };

  const appendSwarmLog = (
    tickerSymbol: string,
    requestId: string,
    entry: Omit<SwarmLogEntry, "id" | "at"> & { at?: string },
  ) => {
    upsertSwarmState(tickerSymbol, (current) => {
      const next: SwarmTickerState = current ?? {
        requestId,
        status: "streaming",
        source: "live",
        agents: {},
        interviewTurns: [],
        logs: [],
      };
      return {
        ...next,
        requestId,
        logs: [
          ...(next.logs ?? []),
          {
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            at: entry.at ?? new Date().toISOString(),
            kind: entry.kind,
            title: entry.title,
            detail: entry.detail,
            payload: entry.payload ?? null,
          },
        ],
      };
    });
  };

  const buildSwarmLogEntry = (
    eventName: string,
    payload: Record<string, unknown>,
  ): Omit<SwarmLogEntry, "id" | "at"> => {
    if (eventName === "agent_status") {
      return {
        kind: "stream",
        title: `${String(payload.agent ?? "Agent")} · ${Number(payload.round) === 2 ? "Round 2" : "Round 1"} status`,
        detail: String(payload.message ?? "Agent status update."),
        payload,
      };
    }
    if (eventName === "agent_note") {
      return {
        kind: "stream",
        title: `${String(payload.agent ?? "Agent")} · thought ${String(payload.index ?? "1")}`,
        detail: String(payload.note ?? "Agent note."),
        payload,
      };
    }
    if (eventName === "round1_verdict") {
      return {
        kind: "stream",
        title: `${String(payload.agent ?? "Agent")} · Round 1`,
        detail: `${String(payload.direction ?? "Pending")} with ${String(payload.conviction ?? "unknown")} conviction. ${String(payload.thesis ?? "")}`,
        payload,
      };
    }
    if (eventName === "round2_verdict") {
      return {
        kind: "stream",
        title: `${String(payload.agent ?? "Agent")} · Round 2`,
        detail: `${payload.revised ? "Revised view." : "Held position."} ${String(payload.direction ?? "Pending")} with ${String(payload.conviction ?? "unknown")} conviction. ${String(payload.thesis ?? "")}`,
        payload,
      };
    }
    if (eventName === "moderator_verdict") {
      return {
        kind: "stream",
        title: "Moderator verdict",
        detail: `${String(payload.final_direction ?? "Pending")} at ${typeof payload.final_confidence === "number" ? `${((payload.final_confidence as number) * 100).toFixed(1)}%` : "unknown confidence"}. ${String(payload.consensus ?? "")}`,
        payload,
      };
    }
    if (eventName === "done") {
      return {
        kind: "system",
        title: "Swarm analysis complete",
        detail: `Deliberation ${String(payload.deliberation_id ?? "")} is ready for replay and follow-up questions.`,
        payload,
      };
    }
    return {
      kind: "system",
      title: eventName,
      detail: "Received swarm event.",
      payload,
    };
  };

  const upsertSwarmState = (tickerSymbol: string, updater: (current: SwarmTickerState | null) => SwarmTickerState) => {
    setSwarmByTicker((current) => ({
      ...current,
      [tickerSymbol]: updater(current[tickerSymbol] ?? null),
    }));
  };

  const applySwarmEvent = (
    tickerSymbol: string,
    requestId: string,
    eventName: string,
    payload: Record<string, unknown>,
  ) => {
    upsertSwarmState(tickerSymbol, (current) => {
      const next: SwarmTickerState = current ?? {
        requestId,
        status: "streaming",
        source: "live",
        agents: {},
        interviewTurns: [],
        logs: [],
      };
      next.requestId = requestId;
      next.source = "live";

      if (eventName === "round1_verdict" || eventName === "round2_verdict") {
        const agentId = String(payload.agent_id ?? "");
        const agentLabel = String(payload.agent ?? agentId);
        const agentState = next.agents[agentId] ?? { agentId, agent: agentLabel };
        const verdict: SwarmVerdictEvent = {
          direction: String(payload.direction ?? "Up") as "Up" | "Down",
          conviction: String(payload.conviction ?? "medium") as "low" | "medium" | "high",
          thesis: String(payload.thesis ?? ""),
          agrees_with_model: Boolean(payload.agrees_with_model),
          public_notes: Array.isArray(payload.public_notes)
            ? payload.public_notes.map((value) => String(value))
            : undefined,
          key_signals: Array.isArray(payload.key_signals)
            ? payload.key_signals.map((value) => String(value))
            : undefined,
          rebuttals:
            payload.rebuttals && typeof payload.rebuttals === "object"
              ? Object.fromEntries(
                  Object.entries(payload.rebuttals).map(([key, value]) => [key, String(value)]),
                )
              : undefined,
          revised: typeof payload.revised === "boolean" ? payload.revised : undefined,
        };
        if (eventName === "round1_verdict") {
          agentState.round1 = verdict;
        } else {
          agentState.round2 = verdict;
        }
        next.agents = { ...next.agents, [agentId]: agentState };
      } else if (eventName === "moderator_verdict") {
        next.moderator = payload as unknown as SwarmModeratorVerdict;
      } else if (eventName === "done") {
        next.deliberationId = String(payload.deliberation_id ?? "");
        next.status = "done";
      }

      return { ...next };
    });
    appendSwarmLog(tickerSymbol, requestId, buildSwarmLogEntry(eventName, payload));
    if (eventName === "done" && session?.user?.email) {
      setTimeout(() => {
        void fetchHistory();
      }, 1200);
    }
  };

  const parseSseChunk = (
    chunk: string,
    onEvent: (eventName: string, payload: Record<string, unknown>) => void,
  ) => {
    const events = chunk.split("\n\n").filter(Boolean);
    for (const eventBlock of events) {
      let eventName = "message";
      let dataText = "";
      for (const line of eventBlock.split("\n")) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataText += line.slice(5).trim();
        }
      }
      if (!dataText) {
        continue;
      }
      try {
        onEvent(eventName, JSON.parse(dataText) as Record<string, unknown>);
      } catch {
        // Ignore malformed chunks from partial buffers.
      }
    }
  };

  const startSwarmAnalysis = async (result: PredictionResult) => {
    if (!ensureTickerRunAllowed("starting swarm analysis")) {
      return;
    }

    activeSwarmPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    setError(null);
    const existing = swarmAbortControllers.current[result.ticker];
    existing?.abort();
    const controller = new AbortController();
    swarmAbortControllers.current[result.ticker] = controller;

    setInterviewQuestion("");
    upsertSwarmState(result.ticker, (current) => ({
      requestId: result.request_id,
      status: "streaming",
      source: "live",
      agents: {},
      interviewTurns: current?.requestId === result.request_id ? current.interviewTurns : [],
      logs:
        current?.requestId === result.request_id
          ? current.logs
          : [],
      deliberationId: undefined,
      moderator: undefined,
    }));
    appendSwarmLog(result.ticker, result.request_id, {
      kind: "system",
      title: "Swarm analysis started",
      detail: `Requesting deep analysis for ${result.ticker} from the live prediction snapshot.`,
      payload: {
        ticker: result.ticker,
        request_id: result.request_id,
        ml_prediction: {
          direction: result.prediction,
          confidence: result.confidence,
          probability: result.prob_up,
        },
      },
    });

    try {
      const response = await fetch("/api/swarm/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker: result.ticker,
          tickers,
          request_id: result.request_id,
          ml_prediction: {
            direction: result.prediction,
            confidence: result.confidence,
            probability: result.prob_up,
          },
          feature_snapshot: result.feature_snapshot,
          news_items: result.feature_snapshot.news_items,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => ({ error: "Swarm analysis failed." }));
        throw new Error(getErrorMessage((payload as { error?: unknown }).error));
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          if (buffer.trim()) {
            parseSseChunk(buffer, (eventName, payload) =>
              applySwarmEvent(result.ticker, result.request_id, eventName, payload),
            );
          }
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const boundary = buffer.lastIndexOf("\n\n");
        if (boundary !== -1) {
          const complete = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          parseSseChunk(complete, (eventName, payload) =>
            applySwarmEvent(result.ticker, result.request_id, eventName, payload),
          );
        }
      }
    } catch (err: unknown) {
      if ((err as { name?: string })?.name === "AbortError") {
        return;
      }
      upsertSwarmState(result.ticker, (current) => ({
        requestId: result.request_id,
        status: "error",
        agents: current?.agents ?? {},
        moderator: current?.moderator,
        deliberationId: current?.deliberationId,
        error: getErrorMessage(err),
        interviewTurns: current?.interviewTurns ?? [],
        logs: current?.logs ?? [],
      }));
      appendSwarmLog(result.ticker, result.request_id, {
        kind: "error",
        title: "Swarm analysis failed",
        detail: getErrorMessage(err),
      });
    }
  };

  const askSwarmAgent = async () => {
    if (!activeResult || !activeSwarm?.deliberationId || !interviewAgent || !interviewQuestion.trim()) {
      return;
    }

    const question = interviewQuestion.trim();
    setInterviewLoading(true);
    upsertSwarmState(activeResult.ticker, (current) => ({
      ...(current ?? {
        requestId: activeResult.request_id,
        status: "done",
        agents: {},
        interviewTurns: [],
        logs: [],
      }),
      interviewTurns: [...(current?.interviewTurns ?? []), { role: "user", message: question }],
    }));
    appendSwarmLog(activeResult.ticker, activeResult.request_id, {
      kind: "interview",
      title: `Question for ${interviewAgent}`,
      detail: question,
    });

    try {
      const response = await fetch("/api/swarm/interview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          deliberation_id: activeSwarm.deliberationId,
          agent: interviewAgent,
          question,
          history: activeSwarm.interviewTurns.map((turn) => ({
            role: turn.role,
            message: turn.message,
          })),
        }),
      });
      const payload = (await response.json()) as { answer?: string; error?: unknown };
      if (!response.ok) {
        throw new Error(getErrorMessage(payload.error));
      }
      upsertSwarmState(activeResult.ticker, (current) => ({
        ...(current ?? {
          requestId: activeResult.request_id,
          status: "done",
          agents: {},
          interviewTurns: [],
          logs: [],
        }),
        interviewTurns: [
          ...(current?.interviewTurns ?? []),
          { role: "agent", message: payload.answer ?? "No response returned." },
        ],
      }));
      appendSwarmLog(activeResult.ticker, activeResult.request_id, {
        kind: "interview",
        title: `${interviewAgent} answered`,
        detail: payload.answer ?? "No response returned.",
        payload: { agent: interviewAgent },
      });
      setInterviewQuestion("");
    } catch (err: unknown) {
      setError(getErrorMessage(err));
      appendSwarmLog(activeResult.ticker, activeResult.request_id, {
        kind: "error",
        title: "Interview request failed",
        detail: getErrorMessage(err),
        payload: { agent: interviewAgent },
      });
    } finally {
      setInterviewLoading(false);
    }
  };

  const runPredictEffect = useEffectEvent(() => {
    void predict();
  });

  const runSwarmEffect = useEffectEvent((result: PredictionResult) => {
    void startSwarmAnalysis(result);
  });

  const runHydrateSwarmArchiveEffect = useEffectEvent((result: PredictionResult, deliberationId: string) => {
    void hydrateActiveSwarmFromArchive(result, deliberationId);
  });

  useEffect(() => {
    if (results.length === 0) {
      setSelectedTicker((current) => current ?? tickers[0] ?? null);
      setSelectedResultKey(null);
      return;
    }

    const preferred =
      (selectedResultKey
        ? results.find((result) => getResultKey(result) === selectedResultKey)
        : null) ??
      (selectedTicker
        ? results.find((result) => result.ticker === selectedTicker)
        : null) ??
      results[0] ??
      null;

    setSelectedTicker(preferred?.ticker ?? null);
    setSelectedResultKey(preferred ? getResultKey(preferred) : null);
  }, [results, tickers, selectedResultKey, selectedTicker]);

  useEffect(() => {
    if (!hydrated) return;
    if (autoBootstrapped.current) return;
    if (loading) return;
    if (results.length > 0) return;
    if (tickers.length === 0) return;
    autoBootstrapped.current = true;
    runPredictEffect();
  }, [hydrated, loading, results.length, tickers.length]);

  const activeResult =
    (selectedResultKey
      ? results.find((result) => getResultKey(result) === selectedResultKey)
      : null) ??
    (selectedTicker
      ? results.find((result) => result.ticker === selectedTicker)
      : null) ??
    results[0] ??
    null;
  const activeSwarm = activeResult ? swarmByTicker[activeResult.ticker] ?? null : null;
  const selectedArchiveItem =
    archiveHistoryKey ? history.find((item) => buildHistoryKey(item) === archiveHistoryKey) ?? null : null;
  const projectedUsageCount = session?.user?.email
    ? new Set([...usedTickersToday, ...tickers.map((tickerItem) => tickerItem.toUpperCase())]).size
    : 0;
  const wouldExceedDailyTickerLimit =
    Boolean(session?.user?.email) && usageLoaded && projectedUsageCount > usageLimit;

  useEffect(() => {
    if (analysisMode !== "deep" || !activeResult) {
      return;
    }
    const swarm = swarmByTicker[activeResult.ticker];
    if (swarm?.requestId === activeResult.request_id && (swarm.status === "streaming" || swarm.status === "done")) {
      return;
    }
    const matchingHistory = history.find(
      (item) => item.request_id === activeResult.request_id && item.ticker === activeResult.ticker,
    );
    if (matchingHistory?.swarm_deliberation_id) {
      runHydrateSwarmArchiveEffect(activeResult, matchingHistory.swarm_deliberation_id);
      return;
    }
    runSwarmEffect(activeResult);
  }, [analysisMode, activeResult, history, swarmByTicker]);

  useEffect(() => {
    if (analysisMode !== "deep" || !activeResult) {
      return;
    }

    const timer = window.setTimeout(() => {
      activeSwarmPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 120);

    return () => window.clearTimeout(timer);
  }, [analysisMode, activeResult]);

  useEffect(() => {
    const availableAgents = Object.keys(activeSwarm?.agents ?? {});
    if (availableAgents.length === 0) {
      return;
    }
    if (!interviewAgent || !availableAgents.includes(interviewAgent)) {
      setInterviewAgent(availableAgents[0]);
    }
  }, [activeSwarm, interviewAgent]);

  const activeFeatures = activeResult
    ? flattenFeatures(activeResult.feature_snapshot.groups)
    : [];

  const filteredFeatures = activeFeatures
    .filter((feature) => {
      if (sourceFilter !== "all" && feature.source !== sourceFilter) {
        return false;
      }

      if (groupFilter !== "all" && feature.groupKey !== groupFilter) {
        return false;
      }

      const query = search.trim().toLowerCase();
      if (!query) {
        return true;
      }

      return (
        feature.label.toLowerCase().includes(query) ||
        feature.name.toLowerCase().includes(query) ||
        feature.groupLabel.toLowerCase().includes(query) ||
        feature.raw_fields.some((field) => field.toLowerCase().includes(query))
      );
    })
    .sort((left, right) => {
      if (sortMode === "value") {
        const leftValue = typeof left.value === "number" ? Math.abs(left.value) : -1;
        const rightValue = typeof right.value === "number" ? Math.abs(right.value) : -1;
        return rightValue - leftValue || left.label.localeCompare(right.label);
      }

      if (sortMode === "source") {
        return (
          left.source.localeCompare(right.source) ||
          left.groupLabel.localeCompare(right.groupLabel) ||
          left.label.localeCompare(right.label)
        );
      }

      return (
        left.groupLabel.localeCompare(right.groupLabel) ||
        left.label.localeCompare(right.label)
      );
    });

  const currentPrice = activeResult ? getNumericFeature(activeResult, "current_price") : null;
  const dayChangePct = activeResult ? getNumericFeature(activeResult, "day_change_pct") : null;
  const companyName = activeResult?.feature_snapshot.company_name?.trim() || null;
  const quoteCurrency = activeResult
    ? getDisplayCurrency(activeResult.ticker, activeResult.feature_snapshot.quote_currency)
    : null;
  const sentimentScore = activeResult ? getNumericFeature(activeResult, "gemini_sentiment_avg") : null;
  const headlineCount = activeResult ? getNumericFeature(activeResult, "headline_count") : null;
  const vixLevel = activeResult ? getNumericFeature(activeResult, "vix_level") : null;
  const spyChangePct = activeResult ? getNumericFeature(activeResult, "spy_change_pct") : null;
  const cacheStatus = activeResult?.feature_snapshot.cache_status ?? null;
  const cacheTone = getCacheTone(cacheStatus);
  const volumeVsAvg10d = activeResult
    ? getNumericFeature(activeResult, "volume_vs_avg_10d")
    : null;
  const priceVs50d = activeResult
    ? getNumericFeature(activeResult, "price_vs_50d_avg")
    : null;
  const topDriverFeatures = activeResult
    ? (activeResult.model_quality?.top_features ?? [])
      .map((row) => getFeature(activeResult, row.feature))
      .filter((feature): feature is FeatureRow => Boolean(feature))
    : [];
  const newsTone = getNewsTone(sentimentScore, headlineCount);
  const hasFredMacro =
    !!activeResult &&
    flattenFeatures(activeResult.feature_snapshot.groups).some(
      (feature) => feature.source === "fred" || feature.providers.includes("fred"),
    );
  const normalizeTickerInput = (raw: string, market: Exchange) => {
    const normalized = raw.trim().toUpperCase();
    if (!normalized) return "";
    if (market === "TSX") {
      if (normalized.includes(".")) return normalized;
      return `${normalized}.TO`;
    }
    return normalized;
  };
  const historySourceLabel = (item: HistoryItem) => {
    const cacheStatus = (item.feature_snapshot as { cache_status?: CacheStatus | null })?.cache_status;
    if (cacheStatus?.source_mode === "live_api") return "Live API";
    if (cacheStatus?.source_mode === "bigquery_cache") return "BQ cache";
    return "Pending";
  };
  const historySnapshotAge = (item: HistoryItem) => {
    const cacheStatus = (item.feature_snapshot as { cache_status?: CacheStatus | null })?.cache_status;
    if (cacheStatus?.snapshot_age_minutes !== undefined && cacheStatus?.snapshot_age_minutes !== null) {
      return `${cacheStatus.snapshot_age_minutes.toFixed(1)}m`;
    }
    const baseTs = item.snapshot_timestamp ?? item.predicted_at;
    return baseTs ? formatRelativeTime(baseTs, nowMs) : "Pending";
  };

  function selectTicker(nextTicker: string) {
    setSelectedTicker(nextTicker);
    const matched = results.find((result) => result.ticker === nextTicker) ?? null;
    setSelectedResultKey(matched ? getResultKey(matched) : null);
    setInterviewQuestion("");
    setError(null);
  }

  function selectResult(result: PredictionResult) {
    setSelectedTicker(result.ticker);
    setSelectedResultKey(getResultKey(result));
    setInterviewQuestion("");
    setError(null);
  }

  function addTicker() {
    const normalized = normalizeTickerInput(ticker, exchange);
    if (!normalized || tickers.includes(normalized)) {
      setTicker("");
      return;
    }

    if (!session) {
      setError("Sign in to manage tickers. Guests can run the default market universe only.");
      return;
    }

    if (totalTickerCount >= FREE_TICKER_LIMIT) {
      setError(`Free accounts can track up to ${FREE_TICKER_LIMIT} tickers total.`);
      return;
    }

    setError(null);
    setTickers((prev) => [...prev, normalized]);
    setSelectedTicker((current) => current ?? normalized);
    setTicker("");
  }

  function removeTicker(target: string) {
    setTickers((prev) => prev.filter((tickerItem) => tickerItem !== target));
    setResults((prev) => prev.filter((result) => result.ticker !== target));
    setSelectedTicker((current) => (current === target ? null : current));
  }

  function ensureTickerRunAllowed(actionLabel: string) {
    if (tickers.length > FREE_TICKER_LIMIT) {
      setError(`Free accounts can run up to ${FREE_TICKER_LIMIT} tickers at a time. Remove extra tickers before ${actionLabel}.`);
      return false;
    }

    if (!session && tickers.some((tickerItem) => !isDefaultTicker(tickerItem))) {
      setError("Sign in to run custom tickers. Guests can use the default market universe only.");
      return false;
    }

    return true;
  }

  async function predict() {
    if (tickers.length === 0) {
      return;
    }

    if (!ensureTickerRunAllowed("running prediction")) {
      return;
    }

    setLoading(true);
    setError(null);
    Object.values(swarmAbortControllers.current).forEach((controller) => controller.abort());
    swarmAbortControllers.current = {};

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instances: tickers.map((tickerItem) => ({ ticker: tickerItem })),
        }),
      });

      const data = (await res.json()) as {
        predictions?: PredictionResult[];
        error?: unknown;
        limit_reached?: boolean;
      };

      if (!res.ok) {
        throw new Error(getErrorMessage(data.error) || `HTTP ${res.status}`);
      }

      const predictions = sanitizePredictionResults(data.predictions);
      setResults(predictions);
      setSelectedResultKey(predictions[0] ? getResultKey(predictions[0]) : null);
      setSelectedTicker(predictions[0]?.ticker ?? null);
      if (analysisMode === "quick") {
        setSwarmByTicker({});
      }
      if (session?.user?.email) {
        setTimeout(() => fetchHistory(), 1500);
        fetchUsage();
      }
      setSearch("");
      setSourceFilter("all");
      setGroupFilter("all");
      setSortMode("group");
      setShowAllInputs(false);
      if (analysisMode === "deep" && predictions.length > 0) {
        void startSwarmAnalysis(predictions[0]);
      }
    } catch (err: unknown) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(13,148,136,0.18),transparent_32%),radial-gradient(circle_at_top_right,rgba(245,158,11,0.16),transparent_28%),linear-gradient(180deg,#f8f3e8_0%,#eff3ef_48%,#e9eff4_100%)] text-slate-950">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
        >
          <div className="flex flex-wrap items-center justify-between gap-6">
            <div className="flex flex-wrap items-center gap-3 text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
              <span className="rounded-full border border-white/70 bg-white/70 px-3 py-1">
                Finnhub primary
              </span>
              <span className="rounded-full border border-white/70 bg-white/70 px-3 py-1">
                Yahoo fallback
              </span>
              <span className="rounded-full border border-white/70 bg-white/70 px-3 py-1">
                BigQuery audit trail
              </span>
            </div>

            <div className="flex items-center gap-4">
              {session ? (
                <div className="flex items-center gap-3">
                  <div className="flex flex-col items-end">
                    <span className="text-xs font-semibold text-slate-900">{session.user?.name}</span>
                    <span className="text-[10px] text-slate-500">{session.user?.email}</span>
                  </div>
                  {session.user?.image ? (
                    <Image
                      src={session.user.image}
                      alt=""
                      width={32}
                      height={32}
                      className="h-8 w-8 rounded-full border border-white"
                    />
                  ) : (
                    <div className="flex h-8 w-8 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-400">
                      <User className="h-4 w-4" />
                    </div>
                  )}
                  <button
                    onClick={() => signOut()}
                    className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-900 text-white transition hover:bg-slate-800"
                    title="Sign out"
                  >
                    <LogOut className="h-3.5 w-3.5" />
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => signIn("google")}
                  className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-semibold uppercase tracking-[0.12em] text-slate-600 transition hover:bg-slate-50"
                >
                  <LogIn className="h-3.5 w-3.5" />
                  Sign in
                </button>
              )}
            </div>
          </div>

          <div className="mt-8 grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
            <div className="rounded-[36px] border border-white/70 bg-white/75 p-6 shadow-[0_28px_90px_-52px_rgba(15,23,42,0.5)] backdrop-blur">
              <div className="flex items-start justify-between gap-6">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-500">
                    Market Intelligence Terminal
                  </p>
                  <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-5xl">
                    Institutional-grade signals from live market, news, and macro data.
                  </h1>
                  <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-600 sm:text-base">
                    Your default market universe loads automatically. Sign in to manage up to 3
                    tickers, then review the signal board and model drivers.
                  </p>
                </div>
                <div className="hidden rounded-[28px] border border-slate-200/80 bg-white/90 p-4 lg:block">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    <Database className="h-4 w-4 text-slate-500" />
                    Active frame
                  </div>
                  <div className="mt-3 text-3xl font-semibold tracking-tight text-slate-950">
                    {activeResult?.feature_snapshot?.selected_feature_count ?? activeResult?.model_quality?.used_feature_count ?? "Pending"}
                  </div>
                  <p className="mt-2 max-w-[14rem] text-sm leading-6 text-slate-600">
                    {activeResult?.feature_snapshot?.feature_count
                      ? `out of ${activeResult.feature_snapshot.feature_count} features selected for training`
                      : activeResult?.model_quality?.used_feature_count
                        ? `out of ${activeResult.model_quality.used_feature_count} features selected for training`
                        : "Auto-scoring default universe. Features align with gold training contract."}
                  </p>
                </div>
              </div>

              <div className="mt-8 grid gap-4 xl:grid-cols-[1fr_auto_auto_auto]">
                <label className="rounded-[28px] border border-slate-200/80 bg-white px-4 py-4 shadow-inner shadow-slate-100">
                  <span className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Ticker
                  </span>
                  <div className="mt-2 flex items-center gap-3">
                    <BarChart3 className="h-5 w-5 text-slate-400" />
                    <input
                      type="text"
                      value={ticker}
                      onChange={(event) => setTicker(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          addTicker();
                        }
                      }}
                      placeholder={exchange === "TSX" ? "RY" : "AAPL"}
                      className="w-full bg-transparent text-lg font-semibold uppercase tracking-[0.14em] text-slate-950 outline-none placeholder:text-slate-300"
                    />
                  </div>
                </label>

                <div className="rounded-[28px] border border-slate-200/80 bg-white px-4 py-4 shadow-inner shadow-slate-100">
                  <span className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Market
                  </span>
                  <div className="mt-2 flex items-center gap-3">
                    <Filter className="h-5 w-5 text-slate-400" />
                    <select
                      value={exchange}
                      onChange={(event) => setExchange(event.target.value as Exchange)}
                      className="w-full bg-transparent text-sm font-semibold uppercase tracking-[0.18em] text-slate-900 outline-none"
                    >
                      <option value="NYSE">NYSE</option>
                      <option value="TSX">TSX</option>
                    </select>
                  </div>
                </div>

                <button
                  onClick={addTicker}
                  disabled={!ticker.trim()}
                  className="inline-flex items-center justify-center gap-2 rounded-[28px] bg-slate-900 px-5 py-4 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Plus className="h-4 w-4" />
                  Add
                </button>

                <button
                  onClick={predict}
                  disabled={tickers.length === 0 || loading || wouldExceedDailyTickerLimit}
                  className="inline-flex items-center justify-center gap-2 rounded-[28px] bg-[linear-gradient(135deg,#0f766e,#0b3b5a)] px-5 py-4 text-sm font-semibold text-white transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Refreshing data
                    </>
                  ) : wouldExceedDailyTickerLimit ? (
                    <>
                      <AlertCircle className="h-4 w-4" />
                      Daily limit reached
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      {results.length > 0 ? "Refresh data" : "Run prediction"}
                    </>
                  )}
                </button>
              </div>

              <div className="mt-3 text-xs text-slate-500">
                {!session
                  ? "Sign in to manage tickers. Guests can run the default market universe."
                  : `Free plan: ${totalTickerCount} / ${FREE_TICKER_LIMIT} tickers selected.`}
              </div>

              <div className="mt-4 inline-flex rounded-full border border-slate-200 bg-white/90 p-1">
                <button
                  onClick={() => setAnalysisMode("quick")}
                  className={clsx(
                    "rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] transition",
                    analysisMode === "quick"
                      ? "bg-slate-900 text-white"
                      : "text-slate-600 hover:bg-slate-100",
                  )}
                >
                  Quick Predict
                </button>
                <button
                  onClick={() => setAnalysisMode("deep")}
                  className={clsx(
                    "rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] transition",
                    analysisMode === "deep"
                      ? "bg-[linear-gradient(135deg,#0f766e,#0b3b5a)] text-white"
                      : "text-slate-600 hover:bg-slate-100",
                  )}
                >
                  Deep Analysis
                </button>
              </div>

              {!session ? (
                <div className="mt-4 rounded-[22px] border border-violet-200/70 bg-[linear-gradient(135deg,rgba(139,92,246,0.08),rgba(79,70,229,0.06))] px-5 py-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-violet-100">
                      <LogIn className="h-5 w-5 text-violet-600" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-slate-900">Sign in to manage your ticker list</p>
                      <p className="text-xs text-slate-500">Guests can run the default market universe. Sign in to manage up to 3 tickers, save history, and keep usage tied to your account.</p>
                    </div>
                    <button
                      onClick={() => signIn("google")}
                      className="ml-auto rounded-full bg-[linear-gradient(135deg,#7c3aed,#4f46e5)] px-5 py-2.5 text-xs font-semibold text-white transition hover:brightness-110"
                    >
                      Sign in with Google
                    </button>
                  </div>
                </div>
              ) : (
                <div className="mt-4 rounded-[22px] border border-slate-200/70 bg-white/80 px-5 py-4">
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      <Gauge className="h-3.5 w-3.5" />
                      Daily usage
                    </div>
                    <div className="text-xs font-semibold text-slate-600">
                      {usageCount} / {usageLimit} tickers refreshed today
                    </div>
                  </div>
                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
                    <motion.div
                      className={clsx(
                        "h-full rounded-full transition-all duration-500",
                        usageCount >= usageLimit
                          ? "bg-rose-500"
                          : usageCount >= usageLimit - 1
                            ? "bg-amber-500"
                            : "bg-emerald-500",
                      )}
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(100, (usageCount / usageLimit) * 100)}%` }}
                      transition={{ duration: 0.6, ease: "easeOut" }}
                    />
                  </div>
                  {usedTickersToday.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {usedTickersToday.map((usedTicker) => (
                        <span
                          key={usedTicker}
                          className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-emerald-700"
                        >
                          {usedTicker}
                        </span>
                      ))}
                    </div>
                  )}
                  {wouldExceedDailyTickerLimit && (
                    <div className="mt-3 flex items-center gap-3 rounded-[18px] border border-rose-200 bg-rose-50 px-4 py-3">
                      <AlertCircle className="h-4 w-4 shrink-0 text-rose-600" />
                      <p className="text-xs text-rose-800">
                        Free accounts can refresh up to {usageLimit} unique tickers per day. This selection would take today&apos;s total to {projectedUsageCount}.
                      </p>
                    </div>
                  )}
                  {usageCount >= usageLimit && (
                    <div className="mt-3 flex items-center gap-3 rounded-[18px] border border-amber-200 bg-amber-50 px-4 py-3">
                      <AlertCircle className="h-4 w-4 shrink-0 text-amber-600" />
                      <p className="text-xs text-amber-800">
                        You&apos;ve used all {usageLimit} free tickers today. <span className="font-semibold">Upgrade to Pro</span> for unlimited access.
                      </p>
                    </div>
                  )}
                </div>
              )}

              <div className="mt-3 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                {analysisMode === "quick"
                  ? "Quick Predict returns the ML ensemble result only."
                  : "Deep Analysis reuses the latest refreshed snapshot, then streams a four-agent review for the active ticker."}
              </div>

              <div className="mt-5 flex flex-wrap gap-2">
                {tickers.length === 0 ? (
                  <div className="rounded-full border border-dashed border-slate-300 px-3 py-1.5 text-sm text-slate-500">
                    Default universe ready
                  </div>
                ) : (
                  tickers.map((tickerItem) => (
                    <button
                      key={tickerItem}
                      onClick={() => selectTicker(tickerItem)}
                      className={clsx(
                        "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition",
                        tickerItem === selectedTicker
                          ? "border-slate-900 bg-slate-900 text-white"
                          : "border-slate-200 bg-white/80 text-slate-700 hover:bg-white",
                      )}
                    >
                      {tickerItem}
                      <span
                        onClick={(event) => {
                          event.stopPropagation();
                          removeTicker(tickerItem);
                        }}
                        className={clsx(
                          "rounded-full p-0.5",
                          tickerItem === selectedTicker ? "bg-white/15" : "bg-slate-100",
                        )}
                      >
                        <X className="h-3.5 w-3.5" />
                      </span>
                    </button>
                  ))
                )}
              </div>
            </div>

            <div className="rounded-[36px] border border-white/70 bg-[linear-gradient(150deg,rgba(255,255,255,0.92),rgba(255,255,255,0.72))] p-6 shadow-[0_28px_90px_-52px_rgba(15,23,42,0.5)] backdrop-blur">
              {activeResult ? (
                <>
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
                        Active ticker
                      </p>
                      <div className="mt-2 flex items-center gap-3">
                        <div className="relative flex h-3 w-3">
                          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
                          <span className="relative inline-flex h-3 w-3 rounded-full bg-emerald-500" />
                        </div>
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-slate-600">
                              {activeResult.ticker}
                            </span>
                            {quoteCurrency && (
                              <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">
                                {quoteCurrency}
                              </span>
                            )}
                          </div>
                          <h2 className="mt-3 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">
                            {companyName ? `${companyName} (${activeResult.ticker})` : activeResult.ticker}
                          </h2>
                        </div>
                      </div>
                    </div>
                    <div className="w-full flex flex-wrap items-center gap-2 sm:w-auto sm:justify-end">
                      <div
                        className={clsx(
                          "rounded-full bg-gradient-to-r px-4 py-2 text-sm font-semibold",
                          getPredictionClasses(activeResult.prediction),
                        )}
                      >
                        {activeResult.prediction}
                      </div>
                      <div
                        className={clsx(
                          "rounded-full border px-4 py-2 text-sm font-semibold",
                          newsTone.chipClasses,
                        )}
                      >
                        {newsTone.label}
                      </div>
                      <div
                        className={clsx(
                          "rounded-full border px-4 py-2 text-sm font-semibold",
                          cacheTone.chipClasses,
                        )}
                      >
                        {cacheTone.label}
                      </div>
                      <div className="flex flex-wrap items-center gap-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                        <span className="rounded-full border border-slate-200 bg-white px-2 py-1">
                          source {cacheStatus?.source_mode === "live_api" ? "Live API" : cacheStatus?.source_mode === "bigquery_cache" ? "BQ cache" : "Pending"}
                        </span>
                        <span className="rounded-full border border-slate-200 bg-white px-2 py-1">
                          age {cacheStatus?.snapshot_age_minutes !== undefined && cacheStatus?.snapshot_age_minutes !== null
                            ? `${cacheStatus.snapshot_age_minutes.toFixed(1)}m`
                            : formatRelativeTime(activeResult.as_of, nowMs)}
                          {cacheStatus?.fresh_ttl_minutes !== undefined && cacheStatus?.fresh_ttl_minutes !== null
                            ? ` / ttl ${cacheStatus.fresh_ttl_minutes.toFixed(0)}m`
                            : ""}
                        </span>
                        <span className="rounded-full border border-slate-200 bg-white px-2 py-1">
                          request {activeResult.request_id}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 rounded-[30px] border border-slate-200/80 bg-slate-950 p-5 text-white">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-white/60">
                        <Gauge className="h-4 w-4" />
                        Up probability
                      </div>
                      <span className="text-sm text-white/70">
                        {activeResult.prediction === "Up" ? "above 50% = Up" : "below 50% = Down"}
                      </span>
                    </div>
                    <div className="mt-5">
                      <div className="flex items-end justify-between">
                        <div className="text-5xl font-semibold tracking-tight">
                          {(activeResult.prob_up * 100).toFixed(1)}%
                        </div>
                        <div className={clsx("text-base font-semibold", getPredictionAccent(activeResult.prediction))}>
                          {activeResult.prediction === "Up" ? (
                            <span className="inline-flex items-center gap-2">
                              <TrendingUp className="h-4 w-4" />
                              predicted up
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-2">
                              <TrendingDown className="h-4 w-4" />
                              predicted down
                            </span>
                          )}
                        </div>
                      </div>
                      <p className="mt-3 text-sm leading-6 text-white/70">
                        This shows how likely the model thinks this ticker will move up. It is
                        a probability score, not the model&apos;s historical accuracy.
                      </p>
                      <div className="mt-5 h-3 overflow-hidden rounded-full bg-white/10">
                        <div
                          className={clsx(
                            "h-full rounded-full bg-gradient-to-r",
                            activeResult.prediction === "Up"
                              ? "from-emerald-400 to-cyan-300"
                              : "from-orange-400 to-rose-400",
                          )}
                          style={{ width: `${Math.max(8, activeResult.prob_up * 100)}%` }}
                        />
                      </div>
                      <div className="mt-2 flex items-center justify-between text-xs uppercase tracking-[0.18em] text-white/50">
                        <span>Down</span>
                        <span>Balanced</span>
                        <span>Up</span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-5 grid gap-3 sm:grid-cols-2">
                    <div className="rounded-[24px] border border-slate-200/70 bg-white/90 p-4">
                      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                        <Clock3 className="h-4 w-4 text-slate-500" />
                        Freshness
                      </div>
                      <p className="mt-3 text-lg font-semibold text-slate-950">
                        snapshot {formatRelativeTime(activeResult.as_of, nowMs)}
                      </p>
                      <p className="mt-1 text-sm text-slate-600">
                        {formatClock(activeResult.as_of)}
                      </p>
                    </div>

                    <div className="rounded-[24px] border border-slate-200/70 bg-white/90 p-4">
                      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                        <Activity className="h-4 w-4 text-slate-500" />
                        Prediction event
                      </div>
                      <p className="mt-3 text-lg font-semibold text-slate-950">
                        scored {formatRelativeTime(activeResult.predicted_at, nowMs)}
                      </p>
                      <p className="mt-1 text-sm text-slate-600">
                        request {activeResult.request_id}
                      </p>
                    </div>
                  </div>
                </>
              ) : (
                <div className="flex h-full min-h-[22rem] flex-col justify-between">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
                      Waiting for a live request
                    </p>
                    <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-950">
                      The right panel turns into a live signal board once a ticker is scored.
                    </h2>
                    <p className="mt-4 text-sm leading-7 text-slate-600">
                      You will see prediction strength, snapshot age, provider mix, model
                      lineage, the news set used for inference, and a searchable feature grid.
                    </p>
                  </div>

                  <div className="mt-8 grid gap-3 sm:grid-cols-2">
                    <MetricCard
                      icon={<Gauge className="h-4 w-4 text-slate-500" />}
                      label="Signal board"
                      value="Live"
                      detail="Request-time snapshot rendered beside the input queue."
                    />
                    <MetricCard
                      icon={<Brain className="h-4 w-4 text-slate-500" />}
                      label="Transparency"
                      value="Full frame"
                      detail="Grouped model inputs, per-headline Gemini sentiment, and live-vs-cache lineage."
                    />
                  </div>
                </div>
              )}
            </div>
          </div>


          {error && (
            <div className="mt-6 flex items-start gap-3 rounded-[26px] border border-rose-200 bg-rose-50/95 px-4 py-3 text-sm text-rose-700 shadow-sm">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {session && (
            <div className="mt-6 rounded-[36px] border border-white/70 bg-white/75 p-6 shadow-[0_28px_90px_-52px_rgba(15,23,42,0.3)] backdrop-blur">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                    <History className="h-4 w-4 text-slate-500" />
                    Your prediction history
                  </div>
                  <p className="mt-2 text-sm text-slate-600">
                    Showing the latest saved row for each ticker linked to your account.
                  </p>
                </div>
                {loadingHistory && <Loader2 className="h-4 w-4 animate-spin text-slate-400" />}
              </div>

              <div className="mt-6 overflow-hidden rounded-[24px] border border-slate-200/60 bg-white/50">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-slate-50/80 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-400">
                      <tr>
                        <th className="px-6 py-4">Ticker</th>
                        <th className="px-6 py-4">Prediction</th>
                        <th className="px-6 py-4">Deep Analysis</th>
                        <th className="px-6 py-4">Prob</th>
                        <th className="px-6 py-4">Price</th>
                        <th className="px-6 py-4">News</th>
                        <th className="px-6 py-4">Source</th>
                        <th className="px-6 py-4 text-right">Date</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100/60">
                      {history.length > 0 ? (
                        history.map((item, idx) => (
                          <tr
                            key={`${buildHistoryKey(item)}-${idx}`}
                            className={clsx(
                              "group hover:bg-slate-50/40",
                              archiveHistoryKey === buildHistoryKey(item) && "bg-cyan-50/50",
                            )}
                          >
                            <td className="px-6 py-4 font-semibold text-slate-900">
                              {item.ticker}
                            </td>
                            <td className="px-6 py-4">
                              <span className={clsx(
                                "rounded-full px-3 py-1 text-[11px] font-semibold",
                                item.prediction === "Up" ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700"
                              )}>
                                {item.prediction}
                              </span>
                            </td>
                            <td className="px-6 py-4">
                              {item.swarm_deliberation_id ? (
                                <div className="flex flex-col items-start gap-2">
                                  <button
                                    onClick={() => void openArchivedSwarm(item)}
                                    className="rounded-full bg-[linear-gradient(135deg,#0f766e,#0b3b5a)] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-white transition hover:brightness-105"
                                  >
                                    {archiveHistoryKey === buildHistoryKey(item) ? "Replay loaded" : "Open replay"}
                                  </button>
                                  <span className="text-xs text-slate-500">
                                    {item.swarm_final_direction ?? "Ready"}
                                    {item.swarm_final_confidence !== null && item.swarm_final_confidence !== undefined
                                      ? ` · ${(item.swarm_final_confidence * 100).toFixed(1)}%`
                                      : ""}
                                  </span>
                                </div>
                              ) : (
                                <span className="text-xs text-slate-400">Not saved</span>
                              )}
                            </td>
                            <td className="px-6 py-4 tabular-nums text-slate-600">
                              {(item.prob_up * 100).toFixed(1)}%
                            </td>
                            <td className="px-6 py-4 tabular-nums text-slate-600">
                              {formatQuotedPrice(item.current_price)}
                            </td>
                            <td className="px-6 py-4 text-slate-500">
                              {item.headline_count} headlines
                            </td>
                            <td className="px-6 py-4 text-slate-500">
                              <div className="flex flex-col">
                                <span className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                                  {historySourceLabel(item)}
                                </span>
                                <span className="text-xs text-slate-400">
                                  {historySnapshotAge(item)}
                                </span>
                              </div>
                            </td>
                            <td className="px-6 py-4 text-right tabular-nums text-slate-400">
                              {formatClock(item.predicted_at)}
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={8} className="px-6 py-12 text-center text-slate-400">
                            {loadingHistory ? "Loading your history..." : "History will appear after your first prediction."}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {(archiveLoading || archiveError || archivedSwarm || selectedArchiveItem) && (
                <div className="mt-6">
                  {archiveLoading && (
                    <div className="mb-4 flex items-center gap-2 rounded-[20px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Loading archived deep analysis…
                    </div>
                  )}
                  {archiveError && (
                    <div className="mb-4 flex items-center gap-2 rounded-[20px] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                      <AlertCircle className="h-4 w-4" />
                      {archiveError}
                    </div>
                  )}
                  {archivedSwarm && selectedArchiveItem && (
                    <SwarmAnalysisPanel
                      ticker={selectedArchiveItem.ticker}
                      swarm={archivedSwarm}
                      variant="archive"
                      archivedAt={selectedArchiveItem.swarm_created_at ?? selectedArchiveItem.predicted_at}
                      nowMs={nowMs}
                    />
                  )}
                </div>
              )}
            </div>
          )}

          {results.length > 0 && activeResult && (
            <div className="mt-6">
              <div className="flex flex-wrap gap-2">
                {results.map((result) => (
                  <button
                    key={`${result.request_id}:${result.ticker}`}
                    onClick={() => selectResult(result)}
                    className={clsx(
                      "inline-flex items-center whitespace-nowrap rounded-full border px-4 py-2 text-sm transition",
                      getResultKey(result) === getResultKey(activeResult)
                        ? "border-slate-900 bg-slate-900 text-white"
                        : "border-white/70 bg-white/80 text-slate-700 hover:bg-white",
                    )}
                  >
                    <span className="font-semibold">{result.ticker}</span>
                    <span className="ml-2 text-xs uppercase tracking-[0.16em] opacity-70">
                      {result.prediction}
                    </span>
                  </button>
                ))}
              </div>

              <AnimatePresence mode="wait">
                <motion.div
                  key={`${activeResult.request_id}:${activeResult.ticker}`}
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -18 }}
                  transition={{ duration: 0.28 }}
                  className="mt-6 grid gap-6 xl:grid-cols-[1.2fr_0.8fr]"
                >
                  <div className="space-y-6">
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <MetricCard
                        icon={<BarChart3 className="h-4 w-4 text-slate-500" />}
                        label={quoteCurrency ? `Live price ${quoteCurrency}` : "Live price"}
                        value={formatQuotedPrice(currentPrice, quoteCurrency)}
                        detail={
                          quoteCurrency
                            ? `Currency ${quoteCurrency}. Day move ${formatSignedPercent(dayChangePct)}`
                            : `Day move ${formatSignedPercent(dayChangePct)}`
                        }
                        accent={
                          dayChangePct !== null
                            ? dayChangePct >= 0
                              ? "text-emerald-700"
                              : "text-rose-700"
                            : "text-slate-950"
                        }
                      />
                      <MetricCard
                        icon={
                          spyChangePct !== null && spyChangePct >= 0 ? (
                            <ArrowUpRight className="h-4 w-4 text-emerald-600" />
                          ) : (
                            <ArrowDownRight className="h-4 w-4 text-rose-600" />
                          )
                        }
                        label="Market context"
                        value={formatSignedPercent(spyChangePct)}
                        detail={
                          vixLevel !== null
                            ? `VIX ${formatFeatureValue({ name: "vix_level", value: vixLevel })}`
                            : "VIX —"
                        }
                        accent={
                          spyChangePct !== null && spyChangePct >= 0
                            ? "text-emerald-700"
                            : "text-rose-700"
                        }
                      />
                      <MetricCard
                        icon={<BarChart3 className="h-4 w-4 text-slate-500" />}
                        label="Volume vs 10D"
                        value={
                          volumeVsAvg10d !== null
                            ? formatFeatureValue({ name: "volume_vs_avg_10d", value: volumeVsAvg10d })
                            : "—"
                        }
                        detail={
                          priceVs50d !== null
                            ? `Price vs 50D ${formatSignedPercent(priceVs50d)}`
                            : "50D comparison unavailable"
                        }
                        accent={
                          volumeVsAvg10d !== null && volumeVsAvg10d >= 1
                            ? "text-emerald-700"
                            : "text-slate-950"
                        }
                      />
                      <MetricCard
                        icon={<Clock3 className="h-4 w-4 text-slate-500" />}
                        label="Data mode"
                        value={cacheTone.label}
                        detail={
                          cacheStatus?.snapshot_age_minutes !== undefined && cacheStatus?.snapshot_age_minutes !== null
                            ? `${cacheTone.summary} Age ${cacheStatus.snapshot_age_minutes.toFixed(1)}m${cacheStatus?.fresh_ttl_minutes !== undefined && cacheStatus?.fresh_ttl_minutes !== null ? ` · TTL ${cacheStatus.fresh_ttl_minutes.toFixed(0)}m` : ""}.`
                            : cacheTone.summary
                        }
                        accent={
                          cacheStatus?.source_mode === "live_api"
                            ? "text-teal-700"
                            : cacheStatus?.stale_served
                              ? "text-amber-800"
                              : "text-indigo-700"
                        }
                      />
                    </div>

                    <TopDriverCards driverFeatures={topDriverFeatures} />

                    {analysisMode === "deep" && (
                    <div ref={activeSwarmPanelRef}>
                      <SwarmAnalysisPanel
                        ticker={activeResult.ticker}
                        swarm={activeSwarm}
                        onRetry={() => void startSwarmAnalysis(activeResult)}
                        onAsk={() => void askSwarmAgent()}
                        interviewAgent={interviewAgent}
                        setInterviewAgent={setInterviewAgent}
                        interviewQuestion={interviewQuestion}
                        setInterviewQuestion={setInterviewQuestion}
                        interviewLoading={interviewLoading}
                        nowMs={nowMs}
                      />
                    </div>
                    )}

                    <div className="rounded-[32px] border border-white/70 bg-white/80 p-5 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.45)] backdrop-blur">
                      <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                        <BarChart3 className="h-4 w-4 text-slate-500" />
                        Price history
                      </div>
                      <div className="mt-4">
                        {chartLoading ? (
                          <div className="flex h-72 items-center justify-center gap-2 text-sm text-slate-400">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Loading chart…
                          </div>
                        ) : chartPoints.length > 0 ? (
                          <PriceChart
                            points={chartPoints}
                            ticker={activeResult.ticker}
                            currency={chartCurrency}
                            prediction={activeResult.prediction}
                          />
                        ) : (
                          <div className="flex h-72 items-center justify-center text-sm text-slate-400">
                            Chart unavailable
                          </div>
                        )}
                      </div>
                    </div>

                    <NewsRail result={activeResult} nowMs={nowMs} tone={newsTone} />

                    <div className="rounded-[32px] border border-white/70 bg-white/80 p-5 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.45)] backdrop-blur">
                      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                        <div>
                          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                            <Database className="h-4 w-4 text-slate-500" />
                            Technical input frame
                          </div>
                          <p className="mt-2 text-sm leading-6 text-slate-600">
                            Expand this only when you want the raw model input frame. The
                            `tfidf_*` headline-vector fields are internal dimensions, not natural-language explanations.
                          </p>
                        </div>
                        <button
                          onClick={() => setShowAllInputs((current) => !current)}
                          className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800"
                        >
                          {showAllInputs ? "Hide raw inputs" : `Show all ${activeResult.feature_snapshot.feature_count} inputs`}
                        </button>
                      </div>
                    </div>

                    {showAllInputs && (
                      <FeatureWorkbench
                        features={filteredFeatures}
                        search={search}
                        setSearch={setSearch}
                        sourceFilter={sourceFilter}
                        setSourceFilter={setSourceFilter}
                        sortMode={sortMode}
                        setSortMode={setSortMode}
                        groupFilter={groupFilter}
                        setGroupFilter={setGroupFilter}
                        groups={activeResult.feature_snapshot.groups}
                      />
                    )}
                  </div>

                  <div className="space-y-6 xl:sticky xl:top-6 xl:self-start">
                    <div className="rounded-[28px] border border-slate-200/70 bg-white/85 p-5 shadow-[0_22px_70px_-44px_rgba(15,23,42,0.45)] backdrop-blur">
                      <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                        <Gauge className="h-4 w-4 text-slate-500" />
                        Technical model report
                      </div>
                      <p className="mt-3 text-sm leading-6 text-slate-600">
                        Reliability metrics, fold breakdown, and learned feature ranking are now
                        on a separate technical page sourced from BigQuery `model_metadata`.
                      </p>
                      <Link
                        href="/model-reliability"
                        className="mt-4 inline-flex items-center gap-2 rounded-full bg-[linear-gradient(135deg,#155e75,#1d4ed8)] px-4 py-2 text-sm font-semibold text-white transition hover:brightness-105"
                      >
                        Open technical report
                        <ArrowUpRight className="h-4 w-4" />
                      </Link>
                    </div>

                    <div className="rounded-[28px] border border-slate-200/70 bg-white/85 p-5 shadow-[0_22px_70px_-44px_rgba(15,23,42,0.45)] backdrop-blur">
                      <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                        <Sparkles className="h-4 w-4 text-slate-500" />
                        Data transparency
                      </div>
                      <div className="mt-4 grid gap-3">
                        <div className="rounded-[22px] bg-slate-100 px-4 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                            snapshot source
                          </p>
                          <p className="mt-2 text-2xl font-semibold text-slate-950">
                            {cacheTone.label}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">{cacheTone.summary}</p>
                        </div>
                        <div className="rounded-[22px] bg-slate-100 px-4 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                            freshness + quota
                          </p>
                          <p className="mt-2 text-2xl font-semibold text-slate-950">
                            {cacheStatus?.snapshot_age_minutes !== undefined &&
                              cacheStatus?.snapshot_age_minutes !== null
                              ? `${cacheStatus.snapshot_age_minutes.toFixed(1)}m old`
                              : "Freshness pending"}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">
                            {cacheStatus
                              ? cacheStatus.daily_live_fetch_count !== null &&
                                cacheStatus.daily_live_fetch_count !== undefined &&
                                cacheStatus.daily_live_fetch_limit !== null &&
                                cacheStatus.daily_live_fetch_limit !== undefined
                                ? `${cacheStatus.daily_live_fetch_count} of ${cacheStatus.daily_live_fetch_limit} live fetches used today for this ticker`
                                : "Live quota telemetry pending"
                              : "Live-cache telemetry pending"}
                          </p>
                        </div>
                        <div className="rounded-[22px] bg-slate-100 px-4 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                            feature frame
                          </p>
                          <p className="mt-2 text-2xl font-semibold text-slate-950">
                            {activeResult.feature_snapshot.feature_count}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">
                            {activeResult.feature_snapshot.selected_feature_count ?? activeResult.model_quality?.used_feature_count ?? "—"} selected by the active model
                          </p>
                        </div>
                        <div className="rounded-[22px] bg-slate-100 px-4 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                            provider mix
                          </p>
                          <p className="mt-2 text-2xl font-semibold text-slate-950">
                            {activeResult.feature_snapshot.provider_summary
                              ? `F${activeResult.feature_snapshot.provider_summary.finnhub ?? 0} · Y${activeResult.feature_snapshot.provider_summary.yahoo ?? 0} · D${activeResult.feature_snapshot.provider_summary.derived ?? 0}`
                              : "Pending"}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">
                            Finnhub, Yahoo, and derived signals per snapshot
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-[28px] border border-slate-200/70 bg-white/85 p-5 shadow-[0_22px_70px_-44px_rgba(15,23,42,0.45)] backdrop-blur">
                      <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                        <Activity className="h-4 w-4 text-slate-500" />
                        Request telemetry
                      </div>
                      <div className="mt-4 space-y-4">
                        <div className="rounded-[22px] bg-slate-100 px-4 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                            Snapshot timestamp
                          </p>
                          <p className="mt-2 text-sm font-semibold text-slate-950">
                            {formatClock(activeResult.as_of)}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">
                            {formatRelativeTime(activeResult.as_of, nowMs)}
                          </p>
                        </div>
                        <div className="rounded-[22px] bg-slate-100 px-4 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                            Prediction timestamp
                          </p>
                          <p className="mt-2 text-sm font-semibold text-slate-950">
                            {formatClock(activeResult.predicted_at)}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">
                            {formatRelativeTime(activeResult.predicted_at, nowMs)}
                          </p>
                        </div>
                        <div className="rounded-[22px] bg-slate-100 px-4 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                            Cache lineage
                          </p>
                          <p className="mt-2 text-sm font-semibold text-slate-950">
                            {cacheStatus?.origin_snapshot_id ?? "Current live snapshot"}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">
                            {cacheStatus?.origin_request_id
                              ? `Derived from request ${cacheStatus.origin_request_id}`
                              : `Current request ${activeResult.request_id}`}
                          </p>
                        </div>
                      </div>
                    </div>

                  </div>
                </motion.div>
              </AnimatePresence>
            </div>
          )}

          {/* ── User Feedback Panel ── */}
          {session && activeResult && (
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.15 }}
              className="mt-10"
            >
              <div className="relative overflow-hidden rounded-[36px] border border-violet-300/30 bg-[radial-gradient(circle_at_top_left,rgba(139,92,246,0.18),transparent_38%),radial-gradient(circle_at_bottom_right,rgba(59,130,246,0.14),transparent_32%),linear-gradient(155deg,rgba(255,255,255,0.92),rgba(245,243,255,0.88),rgba(240,245,255,0.92))] p-6 shadow-[0_30px_100px_-50px_rgba(109,40,217,0.35)] backdrop-blur">
                <div className="pointer-events-none absolute inset-x-6 top-0 h-px bg-gradient-to-r from-transparent via-violet-400/50 to-transparent" />

                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.24em] text-violet-600">
                      <MessageSquare className="h-4 w-4" />
                      Your market intel
                    </div>
                    <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">
                      Share your read on {activeResult.ticker}
                    </h2>
                    <p className="mt-2 max-w-2xl text-sm leading-7 text-slate-600">
                      Help us improve by telling us whether you agree with the model, what you think the
                      direction will be, and how confident you are. Your feedback trains smarter signals.
                    </p>
                  </div>
                  {feedbackSuccess && (
                    <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-semibold text-emerald-700">
                      <CheckCircle className="h-4 w-4" />
                      Submitted
                    </div>
                  )}
                </div>

                <div className="mt-6 grid gap-6 lg:grid-cols-3">
                  {/* Direction */}
                  <div className="rounded-[28px] border border-violet-200/60 bg-white/80 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.7)]">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      <TrendingUp className="h-3.5 w-3.5" />
                      Your prediction
                    </div>
                    <p className="mt-2 text-sm text-slate-600">
                      Model says <span className={clsx("font-semibold", activeResult.prediction === "Up" ? "text-emerald-700" : "text-rose-700")}>{activeResult.prediction}</span>.
                      Do you agree?
                    </p>
                    <div className="mt-4 flex gap-3">
                      <button
                        onClick={() => setFeedbackDirection("Up")}
                        className={clsx(
                          "flex flex-1 items-center justify-center gap-2 rounded-[20px] border-2 px-4 py-3.5 text-sm font-semibold transition-all duration-200",
                          feedbackDirection === "Up"
                            ? "border-emerald-400 bg-emerald-50 text-emerald-700 shadow-[0_8px_24px_-10px_rgba(16,185,129,0.4)]"
                            : "border-slate-200 bg-white text-slate-600 hover:border-emerald-300 hover:bg-emerald-50/50",
                        )}
                      >
                        <ThumbsUp className="h-4 w-4" />
                        Up
                      </button>
                      <button
                        onClick={() => setFeedbackDirection("Down")}
                        className={clsx(
                          "flex flex-1 items-center justify-center gap-2 rounded-[20px] border-2 px-4 py-3.5 text-sm font-semibold transition-all duration-200",
                          feedbackDirection === "Down"
                            ? "border-rose-400 bg-rose-50 text-rose-700 shadow-[0_8px_24px_-10px_rgba(244,63,94,0.4)]"
                            : "border-slate-200 bg-white text-slate-600 hover:border-rose-300 hover:bg-rose-50/50",
                        )}
                      >
                        <ThumbsDown className="h-4 w-4" />
                        Down
                      </button>
                    </div>
                  </div>

                  {/* Sentiment */}
                  <div className="rounded-[28px] border border-violet-200/60 bg-white/80 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.7)]">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      <Sparkles className="h-3.5 w-3.5" />
                      Market sentiment
                    </div>
                    <p className="mt-2 text-sm text-slate-600">
                      How do you read the overall market right now?
                    </p>
                    <div className="mt-4 flex gap-2">
                      {(["bullish", "neutral", "bearish"] as const).map((s) => (
                        <button
                          key={s}
                          onClick={() => setFeedbackSentiment(s)}
                          className={clsx(
                            "flex-1 rounded-[18px] border-2 px-3 py-3 text-sm font-semibold capitalize transition-all duration-200",
                            feedbackSentiment === s
                              ? s === "bullish"
                                ? "border-emerald-400 bg-emerald-50 text-emerald-700 shadow-[0_6px_20px_-8px_rgba(16,185,129,0.4)]"
                                : s === "bearish"
                                  ? "border-rose-400 bg-rose-50 text-rose-700 shadow-[0_6px_20px_-8px_rgba(244,63,94,0.4)]"
                                  : "border-amber-400 bg-amber-50 text-amber-700 shadow-[0_6px_20px_-8px_rgba(245,158,11,0.4)]"
                              : "border-slate-200 bg-white text-slate-500 hover:border-slate-300 hover:bg-slate-50",
                          )}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Confidence */}
                  <div className="rounded-[28px] border border-violet-200/60 bg-white/80 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.7)]">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      <Star className="h-3.5 w-3.5" />
                      Confidence level
                    </div>
                    <p className="mt-2 text-sm text-slate-600">
                      How sure are you about your call? (1 = guess, 5 = very confident)
                    </p>
                    <div className="mt-4 flex gap-2">
                      {[1, 2, 3, 4, 5].map((level) => (
                        <button
                          key={level}
                          onClick={() => setFeedbackConfidence(level)}
                          className={clsx(
                            "flex h-12 w-12 items-center justify-center rounded-2xl border-2 text-sm font-bold transition-all duration-200",
                            feedbackConfidence === level
                              ? "border-violet-400 bg-violet-50 text-violet-700 shadow-[0_6px_20px_-8px_rgba(139,92,246,0.5)] scale-110"
                              : "border-slate-200 bg-white text-slate-500 hover:border-violet-300 hover:bg-violet-50/50",
                          )}
                        >
                          {level}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Comment + Submit */}
                <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_auto]">
                  <div className="rounded-[24px] border border-violet-200/50 bg-white/80 p-4">
                    <label className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      <MessageSquare className="h-3.5 w-3.5" />
                      Optional comment
                    </label>
                    <textarea
                      value={feedbackComment}
                      onChange={(event) => setFeedbackComment(event.target.value)}
                      placeholder="Any specific reasoning, news you've seen, or trade thesis…"
                      rows={2}
                      maxLength={2000}
                      className="mt-2 w-full resize-none bg-transparent text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400"
                    />
                  </div>
                  <button
                    disabled={feedbackSubmitting || (!feedbackDirection && !feedbackSentiment)}
                    onClick={async () => {
                      if (!activeResult || !session?.user?.email) return;
                      setFeedbackSubmitting(true);
                      setFeedbackError(null);
                      setFeedbackSuccess(false);
                      try {
                        const res = await fetch("/api/feedback", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            ticker: activeResult.ticker,
                            request_id: activeResult.request_id,
                            feedback_type: feedbackDirection ? "prediction_rating" : "market_sentiment",
                            model_prediction: activeResult.prediction,
                            user_prediction: feedbackDirection,
                            confidence_rating: feedbackConfidence,
                            sentiment_label: feedbackSentiment,
                            comment: feedbackComment.trim() || null,
                            predicted_at: activeResult.predicted_at,
                            model_confidence: activeResult.confidence,
                          }),
                        });
                        if (!res.ok) {
                          const data = await res.json().catch(() => ({ error: "Submission failed." }));
                          throw new Error((data as { error?: string }).error || "Submission failed.");
                        }
                        setFeedbackSuccess(true);
                        setFeedbackDirection(null);
                        setFeedbackSentiment(null);
                        setFeedbackConfidence(3);
                        setFeedbackComment("");
                        setTimeout(() => setFeedbackSuccess(false), 5000);
                      } catch (err: unknown) {
                        setFeedbackError(err instanceof Error ? err.message : "Submission failed.");
                      } finally {
                        setFeedbackSubmitting(false);
                      }
                    }}
                    className={clsx(
                      "flex items-center justify-center gap-2 self-end rounded-[22px] px-8 py-4 text-sm font-semibold transition-all duration-200",
                      feedbackSubmitting || (!feedbackDirection && !feedbackSentiment)
                        ? "cursor-not-allowed border-2 border-slate-200 bg-slate-100 text-slate-400"
                        : "border-2 border-violet-400/60 bg-[linear-gradient(135deg,rgba(139,92,246,0.92),rgba(79,70,229,0.95))] text-white shadow-[0_12px_36px_-12px_rgba(139,92,246,0.6)] hover:shadow-[0_16px_48px_-12px_rgba(139,92,246,0.7)] hover:brightness-110",
                    )}
                  >
                    {feedbackSubmitting ? (
                      <><Loader2 className="h-4 w-4 animate-spin" /> Submitting…</>
                    ) : (
                      <><Send className="h-4 w-4" /> Submit feedback</>
                    )}
                  </button>
                </div>

                {feedbackError && (
                  <div className="mt-4 rounded-[20px] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                    <AlertCircle className="mr-2 inline-block h-4 w-4" />
                    {feedbackError}
                  </div>
                )}
              </div>
            </motion.div>
          )}

          <div className="mt-10 grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
            <div className="rounded-[36px] border border-white/70 bg-white/80 p-6 shadow-[0_24px_70px_-50px_rgba(15,23,42,0.35)] backdrop-blur">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
                <Shield className="h-4 w-4 text-slate-500" />
                Trust & transparency
              </div>
              <h2 className="mt-3 text-2xl font-semibold tracking-tight text-slate-950">
                Clear inputs, visible lineage, repeatable results.
              </h2>
              <p className="mt-3 text-sm leading-7 text-slate-600">
                Built for financial workflows: you see source mix, freshness, and cache lineage on every run.
              </p>
              <div className="mt-5 grid gap-3 sm:grid-cols-2">
                <div className="rounded-[22px] border border-slate-200/70 bg-white/90 p-4">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    <Newspaper className="h-4 w-4 text-slate-500" />
                    Data sources
                  </div>
                  <p className="mt-2 text-sm text-slate-600">
                    Three sources: Finnhub live, Yahoo fallback, BigQuery audit. Every snapshot is stamped.
                  </p>
                </div>
                <div className="rounded-[22px] border border-slate-200/70 bg-white/90 p-4">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    <Clock3 className="h-4 w-4 text-slate-500" />
                    Freshness
                  </div>
                  <p className="mt-2 text-sm text-slate-600">
                    Every prediction includes snapshot age, cache status, and TTL window.
                  </p>
                </div>
                <div className="rounded-[22px] border border-slate-200/70 bg-white/90 p-4">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    <Database className="h-4 w-4 text-slate-500" />
                    Audit trail
                  </div>
                  <p className="mt-2 text-sm text-slate-600">
                    BigQuery logs every request with timestamped feature snapshots for review.
                  </p>
                </div>
                <div className="rounded-[22px] border border-slate-200/70 bg-white/90 p-4">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    <Brain className="h-4 w-4 text-slate-500" />
                    Model context
                  </div>
                  <p className="mt-2 text-sm text-slate-600">
                    Feature groups, top drivers, and technical report links are all exposed.
                  </p>
                </div>
              </div>
              <div className="mt-5 rounded-[20px] border border-emerald-200/70 bg-emerald-50/70 px-4 py-3 text-sm text-emerald-900">
                <span className="font-semibold">Receipts built-in:</span>{" "}
                {activeResult
                  ? `showing request ${activeResult.request_id} from ${formatClock(activeResult.predicted_at)}`
                  : "run a prediction to stamp a request id and snapshot time right here"}
              </div>
              <p className="mt-4 text-xs uppercase tracking-[0.18em] text-slate-400">
                We store tickers, timestamps, and model features. We never store account passwords.
              </p>
            </div>

            <div className="rounded-[36px] border border-white/70 bg-white/75 p-6 shadow-[0_24px_70px_-50px_rgba(15,23,42,0.35)] backdrop-blur">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
                <Activity className="h-4 w-4 text-slate-500" />
                Live readiness
              </div>
              <div className="mt-4 grid gap-3">
                <div className="rounded-[22px] bg-slate-50 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Snapshot cache
                  </p>
                  <p className="mt-2 text-sm text-slate-600">
                    Fresh snapshots reuse cached frames to minimize API drift and reduce noise.
                  </p>
                </div>
                <div className="rounded-[22px] bg-slate-50 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    News provenance
                  </p>
                  <p className="mt-2 text-sm text-slate-600">
                    Headlines show provider, timestamp, and sentiment scoring time per item.
                  </p>
                </div>
                <div className="rounded-[22px] bg-slate-50 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Model lineage
                  </p>
                  <p className="mt-2 text-sm text-slate-600">
                    Each prediction is tagged with the model artifact URI and training snapshot.
                  </p>
                </div>
                <div className="rounded-[22px] bg-slate-50 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    FRED macro
                  </p>
                  <p className="mt-2 text-sm font-semibold text-slate-900">
                    {activeResult ? (hasFredMacro ? "Active" : "Offline") : "Awaiting run"}
                  </p>
                  <p className="mt-1 text-sm text-slate-600">
                    {activeResult
                      ? "Macro indicators sourced from FRED when available."
                      : "Run a prediction to confirm macro feed status."}
                  </p>
                </div>
                <div className="rounded-[22px] bg-slate-50 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Provider mix
                  </p>
                  <p className="mt-2 text-sm text-slate-600">
                    Finnhub, Yahoo, and derived signals are tracked per snapshot for consistency.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
