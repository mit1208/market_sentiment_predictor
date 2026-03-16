"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useSession, signIn, signOut } from "next-auth/react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Brain,
  Clock3,
  Database,
  Filter,
  Gauge,
  History,
  Loader2,
  LogIn,
  LogOut,
  Newspaper,
  Plus,
  Search,
  Sparkles,
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

type SourceFilter = "all" | "finnhub" | "yahoo" | "derived";
type SortMode = "group" | "value" | "source";

interface HistoryItem {
  request_id: string;
  ticker: string;
  prediction: string;
  prob_up: number;
  confidence: number;
  current_price: number | null;
  headline_count: number | null;
  predicted_at: string;
  feature_snapshot: Record<string, unknown>;
  model_artifact_uri?: string | null;
}

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
    return "Unknown time";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown time";
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
    return "Unknown time";
  }

  const date = new Date(value);
  const targetMs = date.getTime();
  if (Number.isNaN(targetMs)) {
    return "Unknown time";
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
      scoreLabel: score === null ? "n/a" : score.toFixed(2),
      chipClasses: "border-emerald-200 bg-emerald-50 text-emerald-700",
      cardClasses: "border-emerald-200/80 bg-emerald-50/35 hover:border-emerald-300 hover:bg-emerald-50/55",
      scoreClasses: "text-emerald-700",
    };
  }

  if ((score !== null && score <= -0.05) || label === "negative") {
    return {
      label: "Negative",
      scoreLabel: score === null ? "n/a" : score.toFixed(2),
      chipClasses: "border-rose-200 bg-rose-50 text-rose-700",
      cardClasses: "border-rose-200/80 bg-rose-50/35 hover:border-rose-300 hover:bg-rose-50/55",
      scoreClasses: "text-rose-700",
    };
  }

  return {
    label: "Neutral",
    scoreLabel: score === null ? "n/a" : score.toFixed(2),
    chipClasses: "border-amber-200 bg-amber-50 text-amber-800",
    cardClasses: "border-amber-200/80 bg-amber-50/35 hover:border-amber-300 hover:bg-amber-50/55",
    scoreClasses: "text-amber-800",
  };
}

function getCacheTone(cacheStatus: CacheStatus | null | undefined) {
  if (!cacheStatus) {
    return {
      label: "Mode n/a",
      chipClasses: "border-slate-200 bg-slate-100 text-slate-700",
      summary: "No cache metadata was returned for this snapshot.",
    };
  }

  if (cacheStatus.source_mode === "live_api") {
    return {
      label: "Live API",
      chipClasses: "border-teal-200 bg-teal-50 text-teal-700",
      summary:
        "This prediction refreshed live market and news inputs directly from upstream providers.",
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

function getAverageNewsAgeHours(items: NewsItem[], nowMs: number): number | null {
  const ageHours = items
    .map((item) => {
      if (!item.published_at) {
        return null;
      }
      const publishedAtMs = new Date(item.published_at).getTime();
      if (Number.isNaN(publishedAtMs)) {
        return null;
      }
      return Math.max(0, (nowMs - publishedAtMs) / (1000 * 60 * 60));
    })
    .filter((value): value is number => value !== null);

  if (ageHours.length === 0) {
    return null;
  }

  return ageHours.reduce((sum, value) => sum + value, 0) / ageHours.length;
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
            <p className="mt-2 text-xs text-slate-500">{feature.raw_fields.join(", ") || "n/a"}</p>
          </div>
        ))}
      </div>
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
                {feature.providers.join(" • ") || "n/a"}
              </p>
              <p>
                <span className="font-semibold uppercase tracking-[0.16em] text-slate-400">
                  raw fields
                </span>{" "}
                {feature.raw_fields.join(", ") || "n/a"}
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
  const H = 220;
  const padL = 62;
  const padR = 18;
  const padT = 16;
  const padB = 36;
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

  const xTickCount = Math.min(6, points.length);
  const xTickIndices = Array.from({ length: xTickCount }, (_, i) =>
    Math.round((i / (xTickCount - 1)) * (points.length - 1)),
  );

  const lastClose = points[points.length - 1].close;
  const firstClose = points[0].close;
  const totalReturn = ((lastClose - firstClose) / firstClose) * 100;

  const formatPrice = (v: number) =>
    currency === "USD" || !currency ? `$${v.toFixed(2)}` : `${v.toFixed(2)} ${currency}`;

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
              {val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val.toFixed(1)}
            </text>
          </g>
        ))}

        {xTickIndices.map((idx) => (
          <text
            key={idx}
            x={xOf(idx)}
            y={padT + innerH + 20}
            textAnchor="middle"
            fontSize="10"
            fill="#94a3b8"
            fontFamily="ui-sans-serif, sans-serif"
          >
            {points[idx].date.slice(5)}
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
  const [tickers, setTickers] = useState<string[]>([]);
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
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

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 15000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (session?.user?.email) {
      fetchHistory();
    } else {
      setHistory([]);
    }
  }, [session]);

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
      .catch(() => {})
      .finally(() => setChartLoading(false));
  }, [selectedTicker]);

  const fetchHistory = async () => {
    setLoadingHistory(true);
    try {
      const res = await fetch("/api/history");
      if (res.ok) {
        const data = await res.json();
        setHistory(data.history || []);
      }
    } catch (err) {
      console.error("Failed to fetch history:", err);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    if (results.length === 0) {
      setSelectedTicker(null);
      return;
    }

    setSelectedTicker((current) => {
      if (current && results.some((result) => result.ticker === current)) {
        return current;
      }
      return results[0]?.ticker ?? null;
    });
  }, [results]);

  const activeResult =
    results.find((result) => result.ticker === selectedTicker) ?? results[0] ?? null;

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
  const avgNewsAgeHours = activeResult
    ? getAverageNewsAgeHours(getDedupedNewsItems(activeResult.feature_snapshot.news_items), nowMs)
    : null;
  const volumeVsAvg10d = activeResult
    ? getNumericFeature(activeResult, "volume_vs_avg_10d")
    : null;
  const priceVs50d = activeResult
    ? getNumericFeature(activeResult, "price_vs_50d_avg")
    : null;
  const uniqueNewsCount = activeResult
    ? getDedupedNewsItems(activeResult.feature_snapshot.news_items).length
    : 0;
  const topDriverFeatures = activeResult
    ? (activeResult.model_quality?.top_features ?? [])
        .map((row) => getFeature(activeResult, row.feature))
        .filter((feature): feature is FeatureRow => Boolean(feature))
    : [];
  const newsTone = getNewsTone(sentimentScore, headlineCount);

  function addTicker() {
    const normalized = ticker.trim().toUpperCase();
    if (!normalized || tickers.includes(normalized)) {
      setTicker("");
      return;
    }

    setTickers((prev) => [...prev, normalized]);
    setSelectedTicker((current) => current ?? normalized);
    setTicker("");
  }

  function removeTicker(target: string) {
    setTickers((prev) => prev.filter((tickerItem) => tickerItem !== target));
    setResults((prev) => prev.filter((result) => result.ticker !== target));
    setSelectedTicker((current) => (current === target ? null : current));
  }

  async function predict() {
    if (tickers.length === 0) {
      return;
    }

    setLoading(true);
    setError(null);

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
      };

      if (!res.ok) {
        throw new Error(getErrorMessage(data.error) || `HTTP ${res.status}`);
      }

      setResults(data.predictions ?? []);
      if (session?.user?.email) {
        setTimeout(() => fetchHistory(), 1500);
      }
      setSearch("");
      setSourceFilter("all");
      setGroupFilter("all");
      setSortMode("group");
      setShowAllInputs(false);
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
                    <img src={session.user.image} alt="" className="h-8 w-8 rounded-full border border-white" />
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

          {session && (
            <div className="mt-5 rounded-[36px] border border-white/70 bg-white/75 p-6 shadow-[0_28px_90px_-52px_rgba(15,23,42,0.3)] backdrop-blur">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                    <History className="h-4 w-4 text-slate-500" />
                    Your prediction history
                  </div>
                  <p className="mt-2 text-sm text-slate-600">
                    Showing your last 50 predictions linked to your account.
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
                        <th className="px-6 py-4">Prob</th>
                        <th className="px-6 py-4">Price</th>
                        <th className="px-6 py-4">News</th>
                        <th className="px-6 py-4 text-right">Date</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100/60">
                      {history.length > 0 ? (
                        history.map((item, idx) => (
                          <tr key={`${item.request_id}-${idx}`} className="group hover:bg-slate-50/40">
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
                            <td className="px-6 py-4 tabular-nums text-slate-600">
                              {(item.prob_up * 100).toFixed(1)}%
                            </td>
                            <td className="px-6 py-4 tabular-nums text-slate-600">
                              {formatQuotedPrice(item.current_price)}
                            </td>
                            <td className="px-6 py-4 text-slate-500">
                              {item.headline_count} headlines
                            </td>
                            <td className="px-6 py-4 text-right tabular-nums text-slate-400">
                              {formatClock(item.predicted_at)}
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={6} className="px-6 py-12 text-center text-slate-400">
                            {loadingHistory ? "Loading your history..." : "No prediction history found yet. Run a prediction to get started."}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          <div className="mt-8 grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
            <div className="rounded-[36px] border border-white/70 bg-white/75 p-6 shadow-[0_28px_90px_-52px_rgba(15,23,42,0.5)] backdrop-blur">
              <div className="flex items-start justify-between gap-6">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-500">
                    Live Prediction Desk
                  </p>
                  <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-5xl">
                    Real-time ticker intelligence for live market snapshots.
                  </h1>
                  <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-600 sm:text-base">
                    Queue one or more tickers, request a fresh snapshot, then inspect
                    the prediction, market context, and the signals driving the readout.
                  </p>
                </div>
                <div className="hidden rounded-[28px] border border-slate-200/80 bg-white/90 p-4 lg:block">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    <Database className="h-4 w-4 text-slate-500" />
                    Stored
                  </div>
                  <div className="mt-3 text-3xl font-semibold tracking-tight text-slate-950">
                    50
                  </div>
                  <p className="mt-2 max-w-[14rem] text-sm leading-6 text-slate-600">
                    model features retained with snapshot timestamps
                  </p>
                </div>
              </div>

              <div className="mt-8 grid gap-4 xl:grid-cols-[1fr_auto_auto]">
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
                      placeholder="AAPL"
                      className="w-full bg-transparent text-lg font-semibold uppercase tracking-[0.14em] text-slate-950 outline-none placeholder:text-slate-300"
                    />
                  </div>
                </label>

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
                  disabled={tickers.length === 0 || loading}
                  className="inline-flex items-center justify-center gap-2 rounded-[28px] bg-[linear-gradient(135deg,#0f766e,#0b3b5a)] px-5 py-4 text-sm font-semibold text-white transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Refreshing live frame
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      {results.length > 0 ? "Refresh live signal" : "Run live prediction"}
                    </>
                  )}
                </button>
              </div>

              <div className="mt-5 flex flex-wrap gap-2">
                {tickers.length === 0 ? (
                  <div className="rounded-full border border-dashed border-slate-300 px-3 py-1.5 text-sm text-slate-500">
                    No tickers queued yet
                  </div>
                ) : (
                  tickers.map((tickerItem) => (
                    <button
                      key={tickerItem}
                      onClick={() => setSelectedTicker(tickerItem)}
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
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
                        Active ticker
                      </p>
                      <div className="mt-2 flex items-center gap-3">
                        <div className="relative flex h-3 w-3">
                          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
                          <span className="relative inline-flex h-3 w-3 rounded-full bg-emerald-500" />
                        </div>
                        <div>
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
                          <h2 className="mt-3 text-3xl font-semibold tracking-tight text-slate-950">
                            {companyName ?? activeResult.ticker}
                          </h2>
                          {companyName && (
                            <p className="mt-1 text-sm text-slate-500">
                              Full company name for {activeResult.ticker}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center justify-end gap-2">
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

          {results.length > 0 && activeResult && (
            <div className="mt-6">
              <div className="flex flex-wrap gap-2">
                {results.map((result) => (
                  <button
                    key={result.request_id}
                    onClick={() => setSelectedTicker(result.ticker)}
                    className={clsx(
                      "rounded-full border px-4 py-2 text-sm transition",
                      result.ticker === activeResult.ticker
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
                  key={activeResult.request_id}
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
                            : "VIX n/a"
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
                            : "n/a"
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
                          cacheStatus
                            ? `${cacheTone.summary} Age ${cacheStatus.snapshot_age_minutes ?? "n/a"}m. Remaining live budget ${cacheStatus.daily_live_fetch_remaining ?? "n/a"}/${cacheStatus.daily_live_fetch_limit ?? "n/a"}.`
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
                      <MetricCard
                        icon={<Clock3 className="h-4 w-4 text-slate-500" />}
                        label="News input"
                        value={
                          headlineCount !== null
                            ? `${headlineCount.toLocaleString()} raw`
                            : "n/a"
                        }
                        detail={
                          avgNewsAgeHours !== null
                            ? `${uniqueNewsCount} unique shown, avg age ${avgNewsAgeHours.toFixed(1)}h`
                            : `${uniqueNewsCount} unique shown`
                        }
                      />
                    </div>

                    <TopDriverCards driverFeatures={topDriverFeatures} />

                    <div className="rounded-[32px] border border-white/70 bg-white/80 p-5 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.45)] backdrop-blur">
                      <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                        <BarChart3 className="h-4 w-4 text-slate-500" />
                        Price history
                      </div>
                      <div className="mt-4">
                        {chartLoading ? (
                          <div className="flex h-48 items-center justify-center gap-2 text-sm text-slate-400">
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
                          <div className="flex h-48 items-center justify-center text-sm text-slate-400">
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
                              : "n/a"}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">
                            {cacheStatus
                              ? `${cacheStatus.daily_live_fetch_count ?? "n/a"} of ${cacheStatus.daily_live_fetch_limit ?? "n/a"} live fetches used today for this ticker`
                              : "No live-cache telemetry available"}
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
                            {activeResult.feature_snapshot.selected_feature_count ?? activeResult.model_quality?.used_feature_count ?? "n/a"} selected by the active model
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
        </motion.div>
      </div>
    </div>
  );
}
