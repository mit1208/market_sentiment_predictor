# Model Performance Report

## Baseline Run

This file records the current promoted baseline model stored in BigQuery `model_metadata`.

- Run timestamp: `2026-03-13 20:27:24 UTC`
- Model version: `2026-03-13T20-27-24.653025+00-00`
- Deployment status: `PROMOTED`
- Promotion gate: `cv_accuracy_mean > 0.50`
- Promotion gate value: `0.5277`
- Training window: `2020-01-02` to `2026-03-12`
- Holdout window: `2024-12-11` to `2026-03-11`
- Training samples: `7,775`
- Holdout samples: `1,555`
- Feature count: `50`
- Validation strategy: date-based `TimeSeriesSplit` on the pre-holdout window with `5` folds

## Metrics

### Cross-validation Summary

| Metric | Mean | Std Dev |
|--------|------|---------|
| Accuracy | 0.5277 | 0.0279 |
| AUC | 0.5036 | 0.0185 |

### Chronological Holdout

| Metric | Value |
|--------|-------|
| Accuracy | 0.5209 |
| AUC | 0.5336 |
| Precision | 0.5230 |
| Recall | 0.7825 |
| F1 | 0.6269 |

### Fold-by-fold Validation

| Fold | Validation Window | Accuracy | AUC |
|------|-------------------|----------|-----|
| 1 | 2020-10-29 to 2021-08-25 | 0.5314 | 0.5134 |
| 2 | 2021-08-26 to 2022-06-22 | 0.5237 | 0.5261 |
| 3 | 2022-06-23 to 2023-04-19 | 0.4773 | 0.4733 |
| 4 | 2023-04-20 to 2024-02-14 | 0.5488 | 0.5123 |
| 5 | 2024-02-15 to 2024-12-10 | 0.5575 | 0.4929 |

## Top Feature Importance

Top 10 features from the promoted baseline:

1. `day_range_position` — `2270.0`
2. `gap_pct` — `2087.0`
3. `volume_vs_avg_10d` — `2006.0`
4. `intraday_range_pct` — `1980.0`
5. `volume_vs_avg_3m` — `1899.0`
6. `price_vs_50d_avg` — `1865.0`
7. `close_vs_open_pct` — `1718.0`
8. `volume` — `1528.0`
9. `avg_volume_10d` — `1527.0`
10. `price_vs_52w_high` — `1475.0`

## Baseline Takeaways

- The promoted baseline clears the deployment gate, but only narrowly: `0.5277` vs the `0.50` threshold.
- The signal is entirely dominated by live price and volume snapshot features.
- All market-context features, sentiment features, and TF-IDF headline features had zero importance in this promoted baseline.
- The holdout AUC (`0.5336`) is slightly better than the CV AUC (`0.5036`), but both remain close to random and should be treated as a starting point rather than a strong production model.
