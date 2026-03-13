import Link from "next/link";
import { ArrowLeft, Database, Gauge } from "lucide-react";

import {
  fetchModelReliabilitySnapshot,
  type ModelReliabilitySnapshot,
} from "@/lib/model-reliability";

export const dynamic = "force-dynamic";

function formatPercent(value: number | null): string {
  if (value === null) {
    return "n/a";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function formatFloat(value: number | null, digits = 4): string {
  if (value === null) {
    return "n/a";
  }

  return value.toFixed(digits);
}

function MetricTile({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="rounded-[28px] border border-slate-200/80 bg-white/90 p-5 shadow-[0_18px_60px_-40px_rgba(15,23,42,0.35)]">
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
        {label}
      </p>
      <p className="mt-3 text-3xl font-semibold tracking-tight text-slate-950">{value}</p>
      <p className="mt-2 text-sm leading-6 text-slate-600">{detail}</p>
    </div>
  );
}

function ReliabilityView({ snapshot }: { snapshot: ModelReliabilitySnapshot }) {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(251,191,36,0.18),transparent_34%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_100%)] text-slate-950">
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="rounded-[36px] border border-white/70 bg-white/80 p-6 shadow-[0_28px_90px_-52px_rgba(15,23,42,0.4)] backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
                <Database className="h-4 w-4" />
                Technical Model Reliability
              </div>
              <h1 className="mt-3 text-4xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-5xl">
                Baseline reliability from BigQuery model metadata.
              </h1>
              <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-600 sm:text-base">
                This page is sourced directly from BigQuery `model_metadata`. It is meant for
                technical review, not the business-facing prediction workflow.
              </p>
            </div>
            <Link
              href="/"
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:text-slate-950"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to dashboard
            </Link>
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
            <span className="rounded-full border border-white/70 bg-white/70 px-3 py-1">
              {snapshot.deploymentStatus}
            </span>
            <span className="rounded-full border border-white/70 bg-white/70 px-3 py-1">
              {snapshot.modelType}
            </span>
            <span className="rounded-full border border-white/70 bg-white/70 px-3 py-1">
              run {snapshot.trainingDate}
            </span>
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricTile
            label="CV Accuracy"
            value={formatPercent(snapshot.cvAccuracyMean)}
            detail={`std ${formatFloat(snapshot.cvAccuracyStd)}`}
          />
          <MetricTile
            label="CV AUC"
            value={formatFloat(snapshot.cvAucMean)}
            detail={`std ${formatFloat(snapshot.cvAucStd)}`}
          />
          <MetricTile
            label="Holdout Accuracy"
            value={formatPercent(snapshot.holdoutAccuracy)}
            detail={`${snapshot.holdoutStartDate} to ${snapshot.holdoutEndDate}`}
          />
          <MetricTile
            label="Holdout AUC"
            value={formatFloat(snapshot.holdoutAuc)}
            detail={`${snapshot.trainingSamples.toLocaleString()} training rows`}
          />
        </div>

        <div className="mt-6 grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
          <div className="space-y-6">
            <div className="rounded-[32px] border border-white/70 bg-white/80 p-6 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.4)] backdrop-blur">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                <Gauge className="h-4 w-4 text-slate-500" />
                Baseline summary
              </div>
              <div className="mt-4 grid gap-4 sm:grid-cols-2">
                <div className="rounded-[24px] bg-slate-100 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    training window
                  </p>
                  <p className="mt-2 text-sm font-semibold text-slate-950">
                    {snapshot.trainingStartDate} to {snapshot.trainingEndDate}
                  </p>
                </div>
                <div className="rounded-[24px] bg-slate-100 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    promotion gate
                  </p>
                  <p className="mt-2 text-sm font-semibold text-slate-950">
                    {snapshot.promotionGateMetric} {">"} {formatFloat(snapshot.promotionGateValue)}
                  </p>
                </div>
                <div className="rounded-[24px] bg-slate-100 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    features
                  </p>
                  <p className="mt-2 text-sm font-semibold text-slate-950">
                    {snapshot.featureCount} total
                  </p>
                </div>
                <div className="rounded-[24px] bg-slate-100 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    validation strategy
                  </p>
                  <p className="mt-2 text-sm font-semibold text-slate-950">
                    {snapshot.validationStrategy}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    {snapshot.nSplits !== null ? `${snapshot.nSplits} folds` : "n/a"}
                  </p>
                </div>
              </div>
              {snapshot.notes ? (
                <div className="mt-4 rounded-[24px] bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-900">
                  {snapshot.notes}
                </div>
              ) : null}
            </div>

            <div className="rounded-[32px] border border-white/70 bg-white/80 p-6 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.4)] backdrop-blur">
              <div className="flex items-center justify-between gap-3">
                <div className="text-sm font-semibold text-slate-900">Fold breakdown</div>
                <div className="text-xs uppercase tracking-[0.18em] text-slate-400">
                  pre-holdout validation
                </div>
              </div>
              <div className="mt-4 overflow-x-auto">
                <table className="min-w-full text-left text-sm text-slate-700">
                  <thead className="text-[11px] uppercase tracking-[0.16em] text-slate-400">
                    <tr>
                      <th className="pb-3 pr-4 font-semibold">Fold</th>
                      <th className="pb-3 pr-4 font-semibold">Validation window</th>
                      <th className="pb-3 pr-4 font-semibold">Accuracy</th>
                      <th className="pb-3 font-semibold">AUC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {snapshot.folds.map((fold) => (
                      <tr key={fold.fold} className="border-t border-slate-200/80">
                        <td className="py-3 pr-4 font-semibold text-slate-950">{fold.fold}</td>
                        <td className="py-3 pr-4">
                          {fold.validation_start_date} to {fold.validation_end_date}
                        </td>
                        <td className="py-3 pr-4">{formatPercent(fold.accuracy)}</td>
                        <td className="py-3">{formatFloat(fold.auc)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-[32px] border border-white/70 bg-white/80 p-6 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.4)] backdrop-blur">
              <div className="text-sm font-semibold text-slate-900">Top learned features</div>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                Highest-importance features for the latest promoted model version.
              </p>
              <div className="mt-4 space-y-3">
                {snapshot.topFeatures.map((feature, index) => (
                  <div
                    key={feature.feature}
                    className="flex items-center justify-between rounded-[22px] bg-slate-100 px-4 py-3"
                  >
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                        rank {index + 1}
                      </p>
                      <p className="mt-1 text-sm font-semibold text-slate-950">{feature.feature}</p>
                    </div>
                    <div className="text-sm font-semibold text-slate-700">
                      {feature.importance.toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[32px] border border-white/70 bg-white/80 p-6 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.4)] backdrop-blur">
              <div className="text-sm font-semibold text-slate-900">Holdout detail</div>
              <div className="mt-4 grid gap-3">
                <div className="rounded-[22px] bg-slate-100 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    precision
                  </p>
                  <p className="mt-2 text-xl font-semibold text-slate-950">
                    {formatFloat(snapshot.holdoutPrecision)}
                  </p>
                </div>
                <div className="rounded-[22px] bg-slate-100 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    recall
                  </p>
                  <p className="mt-2 text-xl font-semibold text-slate-950">
                    {formatFloat(snapshot.holdoutRecall)}
                  </p>
                </div>
                <div className="rounded-[22px] bg-slate-100 px-4 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                    f1
                  </p>
                  <p className="mt-2 text-xl font-semibold text-slate-950">
                    {formatFloat(snapshot.holdoutF1)}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ErrorView({ message }: { message: string }) {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(251,191,36,0.18),transparent_34%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_100%)] text-slate-950">
      <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="rounded-[36px] border border-white/70 bg-white/80 p-6 shadow-[0_28px_90px_-52px_rgba(15,23,42,0.4)] backdrop-blur">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
                Technical Model Reliability
              </p>
              <h1 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-slate-950">
                BigQuery model metadata is unavailable.
              </h1>
              <p className="mt-4 text-sm leading-7 text-slate-600">{message}</p>
            </div>
            <Link
              href="/"
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:text-slate-950"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to dashboard
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default async function ModelReliabilityPage() {
  let snapshot: ModelReliabilitySnapshot | null = null;
  let message: string | null = null;

  try {
    snapshot = await fetchModelReliabilitySnapshot();
  } catch (error) {
    message = error instanceof Error ? error.message : "Unknown model metadata error.";
  }

  if (!snapshot) {
    return <ErrorView message={message ?? "Unknown model metadata error."} />;
  }

  return <ReliabilityView snapshot={snapshot} />;
}
