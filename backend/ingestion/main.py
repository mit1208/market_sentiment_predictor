import os
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import pandas as pd

try:
    from backend.config import get_csv_env, get_env, get_int_env, qualify_table_name
except ImportError:
    # Cloud Functions deploys only the ingestion source directory, so provide
    # local fallbacks when the shared backend package is not bundled.
    def get_env(*names: str, default: str = "") -> str:
        for name in names:
            value = os.getenv(name)
            if value is None:
                continue
            normalized = value.strip()
            if normalized:
                return normalized
        return default

    def get_required_env(*names: str) -> str:
        value = get_env(*names)
        if value:
            return value
        readable_names = ", ".join(names)
        raise RuntimeError(f"Missing required configuration. Set one of: {readable_names}.")

    def get_int_env(*names: str, default: int | None = None) -> int | None:
        value = get_env(*names)
        if not value:
            return default
        return int(value)

    def get_csv_env(*names: str) -> list[str]:
        value = get_env(*names)
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def qualify_table_name(
        table_name: str,
        *,
        project_id: str | None = None,
        dataset_id: str | None = None,
    ) -> str:
        resolved_project_id = project_id or get_required_env("GCP_PROJECT_ID")
        resolved_dataset_id = dataset_id or get_required_env("BQ_DATASET", "BQ_DATASET_ID")
        return f"{resolved_project_id}.{resolved_dataset_id}.{table_name}"

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("GCP_REGION")
DATASET_ID = get_env("BQ_DATASET_ID", "BQ_DATASET")
FINNHUB_KEY = get_env("FINNHUB_KEY", "FINNHUB_API_KEY")
FRED_API_KEY = get_env("FRED_API_KEY")
TRAINING_JOB_NAME = get_env("TRAINING_JOB_NAME")
TICKERS = get_csv_env("INGESTION_TICKERS", "TRAINING_TICKERS")
MARKET_BOOTSTRAP_START = get_env("MARKET_BOOTSTRAP_START", "BOOTSTRAP_MARKET_START")
NEWS_BOOTSTRAP_DAYS = get_int_env("NEWS_BOOTSTRAP_DAYS", "BOOTSTRAP_NEWS_DAYS")
CONTEXT_TICKERS = ["SPY", "^VIX"]
CONTEXT_HISTORY_START_TOLERANCE_DAYS = 7
FRED_SERIES = {
    "T10Y2Y": "yield_curve_spread",
    "FEDFUNDS": "fed_funds_rate",
    "BAMLH0A0HYM2": "credit_spread",
    "DTWEXBGS": "dxy_level",
}


def _table_id(table_name: str) -> str:
    return qualify_table_name(
        table_name,
        project_id=PROJECT_ID or None,
        dataset_id=DATASET_ID or None,
    )


def _training_ticker_table() -> str:
    return _table_id("training_tickers")


def _resolve_tickers() -> list[str]:
    if TICKERS:
        return [ticker.upper() for ticker in TICKERS]

    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=PROJECT_ID or None)
        rows = client.query(
            f"""
            SELECT ticker
            FROM `{_training_ticker_table()}`
            WHERE is_active IS TRUE
            ORDER BY sector, ticker
            """
        ).result()
    except Exception as exc:
        logger.warning("Failed to read training tickers from %s: %s", _training_ticker_table(), exc)
        return []

    return [str(row.ticker).upper() for row in rows if getattr(row, "ticker", None)]


def _market_universe_tickers(tickers: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for ticker in [*tickers, *CONTEXT_TICKERS]:
        normalized = str(ticker).upper().strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def table_has_rows(table_name):
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT_ID or None)
    table_id = _table_id(table_name)
    query = f"SELECT COUNT(1) AS row_count FROM `{table_id}`"
    rows = list(client.query(query).result())
    return bool(rows and rows[0].row_count)


def _resolve_context_backfill_start(training_tickers: list[str]):
    import pandas as pd
    from google.cloud import bigquery

    if MARKET_BOOTSTRAP_START:
        return datetime.strptime(MARKET_BOOTSTRAP_START, "%Y-%m-%d")

    client = bigquery.Client(project=PROJECT_ID or None)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("tickers", "STRING", training_tickers)]
    )
    rows = list(
        client.query(
            f"""
            SELECT MIN(date) AS min_date
            FROM `{_table_id("raw_market_data")}`
            WHERE ticker IN UNNEST(@tickers)
            """,
            job_config=job_config,
        ).result()
    )
    min_date = rows[0].min_date if rows else None
    if min_date is None:
        return None
    return pd.Timestamp(min_date).to_pydatetime()


def _context_history_requires_backfill(required_start):
    import pandas as pd
    from google.cloud import bigquery

    if required_start is None:
        return False

    client = bigquery.Client(project=PROJECT_ID or None)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("tickers", "STRING", CONTEXT_TICKERS)]
    )
    rows = list(
        client.query(
            f"""
            SELECT ticker, MIN(date) AS min_date, MAX(date) AS max_date, COUNT(DISTINCT date) AS date_count
            FROM `{_table_id("raw_market_data")}`
            WHERE ticker IN UNNEST(@tickers)
            GROUP BY ticker
            """,
            job_config=job_config,
        ).result()
    )
    coverage = {str(row.ticker).upper(): row for row in rows}
    required_start_ts = pd.Timestamp(required_start).normalize()
    tolerated_start = required_start_ts + pd.Timedelta(days=CONTEXT_HISTORY_START_TOLERANCE_DAYS)

    for ticker in CONTEXT_TICKERS:
        row = coverage.get(ticker)
        if row is None or getattr(row, "date_count", 0) == 0:
            return True
        min_date = pd.Timestamp(row.min_date).normalize() if getattr(row, "min_date", None) is not None else None
        if min_date is None or min_date > tolerated_start:
            return True
    return False


def fetch_market_data_df(tickers, *, bootstrap=False, start_date_override=None):
    """Fetches market data and returns a cleaned DataFrame."""
    import yfinance as yf

    all_data = []
    end_date = datetime.now()
    if bootstrap and start_date_override is None and not MARKET_BOOTSTRAP_START:
        raise RuntimeError("MARKET_BOOTSTRAP_START is required when bootstrapping market data.")
    if start_date_override is not None:
        start_date = pd.Timestamp(start_date_override).to_pydatetime()
    elif bootstrap:
        start_date = datetime.strptime(MARKET_BOOTSTRAP_START, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=10)

    for ticker in tickers:
        logger.info(f"Downloading market data for {ticker} (bootstrap={bootstrap})...")
        try:
            data = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
            )
            if not data.empty:
                # Reset index to get the date column
                df = data.reset_index()
                # yfinance returns MultiIndex columns in some versions, flatten if necessary
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                # Rename columns to match BQ schema
                df.columns = [col.lower() for col in df.columns]
                df["ticker"] = ticker
                # Ensure date is format BQ likes
                df["date"] = pd.to_datetime(df["date"]).dt.date

                # Filter to required columns
                df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
                all_data.append(df)
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")

    return pd.concat(all_data) if all_data else pd.DataFrame()


def fetch_news_data_df(tickers, *, bootstrap=False):
    """Fetches news data and returns a cleaned DataFrame."""
    import pandas as pd
    import requests
    import yfinance as yf

    all_articles = []
    end_date = datetime.now()
    if bootstrap and NEWS_BOOTSTRAP_DAYS is None:
        raise RuntimeError("NEWS_BOOTSTRAP_DAYS is required when bootstrapping news data.")
    start_date = end_date - timedelta(days=NEWS_BOOTSTRAP_DAYS if bootstrap else 7)

    for ticker in tickers:
        logger.info(f"Fetching Finnhub/Yahoo news for {ticker} (bootstrap={bootstrap})...")
        seen = set()

        if FINNHUB_KEY:
            try:
                query = urlencode(
                    {
                        "symbol": ticker,
                        "from": start_date.strftime("%Y-%m-%d"),
                        "to": end_date.strftime("%Y-%m-%d"),
                        "token": FINNHUB_KEY,
                    }
                )
                response = requests.get(
                    f"https://finnhub.io/api/v1/company-news?{query}",
                    timeout=15,
                )
                response.raise_for_status()
                payload = response.json()
                articles = payload if isinstance(payload, list) else []
                for art in articles:
                    headline = str(art.get("headline") or "").strip()
                    url = str(art.get("url") or "").strip()
                    if not headline or (headline.lower(), url) in seen:
                        continue
                    seen.add((headline.lower(), url))
                    published_at = art.get("datetime")
                    published_at = None if published_at is None else pd.to_datetime(int(published_at), unit="s", utc=True)
                    all_articles.append(
                        {
                            "date": published_at.date() if published_at is not None else end_date.date(),
                            "timestamp": published_at,
                            "ticker": ticker,
                            "headline": headline,
                            "description": art.get("summary"),
                            "source": art.get("source") or "Finnhub",
                            "url": url,
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to fetch Finnhub news for {ticker}: {e}")

        try:
            for art in (yf.Ticker(ticker).news or []):
                content = art.get("content") if isinstance(art, dict) else {}
                headline = art.get("title") or (content.get("title") if isinstance(content, dict) else None)
                headline = str(headline or "").strip()
                if not headline:
                    continue
                url = art.get("link")
                if url is None and isinstance(content, dict):
                    canonical = content.get("canonicalUrl")
                    if isinstance(canonical, dict):
                        url = canonical.get("url")
                url = str(url or "").strip()
                if (headline.lower(), url) in seen:
                    continue
                seen.add((headline.lower(), url))

                published_at = art.get("providerPublishTime")
                if published_at is None and isinstance(content, dict):
                    published_at = content.get("pubDate")
                published_at = pd.to_datetime(published_at, errors="coerce", utc=True)

                all_articles.append(
                    {
                        "date": (published_at.date() if not pd.isna(published_at) else end_date.date()),
                        "timestamp": (None if pd.isna(published_at) else published_at),
                        "ticker": ticker,
                        "headline": headline,
                        "description": None,
                        "source": art.get("publisher") or "Yahoo",
                        "url": url,
                    }
                )
        except Exception as e:
            logger.error(f"Failed to fetch Yahoo news for {ticker}: {e}")

    if all_articles:
        df = pd.DataFrame(all_articles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df
    return pd.DataFrame()


def fetch_macro_data_df(*, bootstrap=False):
    """Fetches raw macro time series from FRED and returns a cleaned DataFrame."""
    import pandas as pd
    from fredapi import Fred

    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is required for bronze macro ingestion.")

    end_date = datetime.now()
    if bootstrap and not MARKET_BOOTSTRAP_START:
        raise RuntimeError("MARKET_BOOTSTRAP_START is required when bootstrapping macro data.")
    start_date = (
        datetime.strptime(MARKET_BOOTSTRAP_START, "%Y-%m-%d")
        if bootstrap
        else end_date - timedelta(days=45)
    )

    fred = Fred(api_key=FRED_API_KEY)
    rows: list[dict] = []
    fetched_at = datetime.now(timezone.utc)
    for series_id in FRED_SERIES:
        logger.info("Fetching FRED macro series %s (bootstrap=%s)...", series_id, bootstrap)
        try:
            series = fred.get_series(
                series_id,
                observation_start=start_date.date(),
                observation_end=end_date.date(),
            )
        except Exception as exc:
            logger.error("Failed to fetch FRED series %s: %s", series_id, exc)
            continue

        for observation_date, value in series.items():
            observation_ts = pd.to_datetime(observation_date, errors="coerce")
            numeric_value = pd.to_numeric(value, errors="coerce")
            if pd.isna(observation_ts) or pd.isna(numeric_value):
                continue
            rows.append(
                {
                    "observation_date": observation_ts.date(),
                    "series_id": series_id,
                    "value": float(numeric_value),
                    "source": "FRED",
                    "fetched_at": fetched_at,
                }
            )

    return pd.DataFrame(rows)


def load_with_merge(df, table_name, unique_keys):
    """Loads a DataFrame into BigQuery and merges with existing data to prevent duplicates."""
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT_ID or None)
    if df.empty:
        logger.info(f"No data to load for {table_name}")
        return

    staging_table_id = _table_id(f"stg_{table_name}")
    target_table_id = _table_id(table_name)

    # 1. Load to staging table (overwrite)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df, staging_table_id, job_config=job_config).result()

    # 2. Construct MERGE statement
    columns = [field.name for field in client.get_table(target_table_id).schema]

    # Calculate partition filter for MERGE if a date-like column exists.
    date_filter = ""
    if "date" in df.columns:
        min_date = df["date"].min()
        max_date = df["date"].max()
        date_filter = f" AND T.date >= '{min_date}' AND T.date <= '{max_date}'"
    elif "observation_date" in df.columns:
        min_date = df["observation_date"].min()
        max_date = df["observation_date"].max()
        date_filter = (
            f" AND T.observation_date >= '{min_date}'"
            f" AND T.observation_date <= '{max_date}'"
        )

    on_clause = " AND ".join([f"T.{k} = S.{k}" for k in unique_keys])
    insert_cols = ", ".join(columns)
    field_types = {field.name: field.field_type.upper() for field in client.get_table(target_table_id).schema}
    values_cols = ", ".join(
        [
            f"TIMESTAMP(S.{column})" if field_types.get(column) == "TIMESTAMP" else f"S.{column}"
            for column in columns
        ]
    )

    merge_sql = f"""
    MERGE `{target_table_id}` T
    USING `{staging_table_id}` S
    ON {on_clause}{date_filter}
    WHEN NOT MATCHED THEN
      INSERT ({insert_cols})
      VALUES ({values_cols})
    """

    logger.info(f"Executing MERGE for {target_table_id}...")
    client.query(merge_sql).result()

    # 3. Clean up staging
    client.delete_table(staging_table_id, not_found_ok=True)
    logger.info(f"Load and Merge complete for {table_name}")


def trigger_training_job():
    """Launch the Cloud Run training job and return the operation name."""
    from google.api_core.client_options import ClientOptions
    from google.cloud import run_v2

    job_name = get_env("TRAINING_JOB_NAME", default=TRAINING_JOB_NAME)
    if not job_name:
        raise RuntimeError("TRAINING_JOB_NAME is required to trigger the training job.")
    if not PROJECT_ID or not REGION:
        raise RuntimeError("GCP_PROJECT_ID and GCP_REGION must be set for ingestion.")

    client = run_v2.JobsClient(
        client_options=ClientOptions(api_endpoint=f"{REGION}-run.googleapis.com")
    )
    request = run_v2.RunJobRequest(
        name=f"projects/{PROJECT_ID}/locations/{REGION}/jobs/{job_name}"
    )

    operation = client.run_job(request=request)
    operation_name = operation.operation.name
    logger.info(f"Triggered Cloud Run Job operation: {operation_name}")
    return operation_name


def run_ingestion(request=None):
    """Main entry point for the ingestion Cloud Function (2nd Gen HTTP)."""
    logger.info("Starting automated data ingestion...")
    training_tickers = _resolve_tickers()
    if not training_tickers:
        raise RuntimeError(
            "Set INGESTION_TICKERS/TRAINING_TICKERS or populate the training_tickers BigQuery table."
        )
    market_tickers = _market_universe_tickers(training_tickers)
    bootstrap_market = not table_has_rows("raw_market_data")
    bootstrap_news = not table_has_rows("raw_news_data")
    bootstrap_macro = not table_has_rows("raw_macro_data")

    market_df = fetch_market_data_df(market_tickers, bootstrap=bootstrap_market)
    if not market_df.empty:
        load_with_merge(market_df, "raw_market_data", ["date", "ticker"])

    if not bootstrap_market:
        context_backfill_start = _resolve_context_backfill_start(training_tickers)
        if _context_history_requires_backfill(context_backfill_start):
            logger.info(
                "Context ticker history is incomplete in bronze. Backfilling %s from %s.",
                CONTEXT_TICKERS,
                context_backfill_start.date() if context_backfill_start is not None else "configured bootstrap start",
            )
            context_market_df = fetch_market_data_df(
                CONTEXT_TICKERS,
                bootstrap=True,
                start_date_override=context_backfill_start,
            )
            if not context_market_df.empty:
                load_with_merge(context_market_df, "raw_market_data", ["date", "ticker"])
                market_df = (
                    pd.concat([market_df, context_market_df], ignore_index=True)
                    .drop_duplicates(subset=["date", "ticker"], keep="last")
                    .reset_index(drop=True)
                )

    news_df = fetch_news_data_df(training_tickers, bootstrap=bootstrap_news)
    if not news_df.empty:
        load_with_merge(news_df, "raw_news_data", ["url", "ticker"])

    macro_df = fetch_macro_data_df(bootstrap=bootstrap_macro)
    if not macro_df.empty:
        load_with_merge(macro_df, "raw_macro_data", ["observation_date", "series_id"])

    response = {
        "status": "success",
        "market_rows_loaded": int(len(market_df)),
        "news_rows_loaded": int(len(news_df)),
        "macro_rows_loaded": int(len(macro_df)),
        "training_tickers": training_tickers,
        "market_universe_tickers": market_tickers,
    }
    logger.info("Ingestion finished successfully: %s", response)
    return response, 200


if __name__ == "__main__":
    # For local testing
    from dotenv import load_dotenv

    load_dotenv()
    if not (os.getenv("FINNHUB_KEY") or os.getenv("FINNHUB_API_KEY")):
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("FINNHUB_KEY"):
                        os.environ["FINNHUB_KEY"] = line.split("=")[1].strip()
        except:
            pass
    run_ingestion()
