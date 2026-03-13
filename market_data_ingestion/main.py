import os
import logging
from datetime import datetime, timedelta
from urllib.parse import urlencode

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (These should ideally be environment variables in Cloud Function)
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "llm-testing-489113")
REGION = os.getenv("GCP_REGION", "us-central1")
DATASET_ID = os.getenv("BQ_DATASET_ID", "market_sentiment_predictor")
FINNHUB_KEY = os.getenv("FINNHUB_KEY") or os.getenv("FINNHUB_API_KEY")
TRAINING_JOB_NAME = os.getenv("TRAINING_JOB_NAME", "market-sentiment-train")
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
MARKET_BOOTSTRAP_START = os.getenv("MARKET_BOOTSTRAP_START", "2020-01-01")
NEWS_BOOTSTRAP_DAYS = int(os.getenv("NEWS_BOOTSTRAP_DAYS", "30"))


def table_has_rows(table_name):
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
    query = f"SELECT COUNT(1) AS row_count FROM `{table_id}`"
    rows = list(client.query(query).result())
    return bool(rows and rows[0].row_count)


def fetch_market_data_df(tickers, *, bootstrap=False):
    """Fetches market data and returns a cleaned DataFrame."""
    import pandas as pd
    import yfinance as yf

    all_data = []
    end_date = datetime.now()
    start_date = (
        datetime.strptime(MARKET_BOOTSTRAP_START, "%Y-%m-%d")
        if bootstrap
        else end_date - timedelta(days=10)
    )

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


def load_with_merge(df, table_name, unique_keys):
    """Loads a DataFrame into BigQuery and merges with existing data to prevent duplicates."""
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT_ID)
    if df.empty:
        logger.info(f"No data to load for {table_name}")
        return

    staging_table_id = f"{PROJECT_ID}.{DATASET_ID}.stg_{table_name}"
    target_table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    # 1. Load to staging table (overwrite)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df, staging_table_id, job_config=job_config).result()

    # 2. Construct MERGE statement
    columns = [field.name for field in client.get_table(target_table_id).schema]

    # Calculate date range for partition elimination if date is a column
    date_filter = ""
    if "date" in df.columns:
        min_date = df["date"].min()
        max_date = df["date"].max()
        date_filter = f" AND T.date >= '{min_date}' AND T.date <= '{max_date}'"

    on_clause = " AND ".join([f"T.{k} = S.{k}" for k in unique_keys])
    insert_cols = ", ".join(columns)
    values_cols = ", ".join([f"S.{c}" for c in columns])

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

    job_name = os.environ.get("TRAINING_JOB_NAME", TRAINING_JOB_NAME)
    if not job_name:
        logger.error("TRAINING_JOB_NAME not set — cannot trigger Cloud Run job")
        return None

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
    bootstrap_market = not table_has_rows("raw_market_data")
    bootstrap_news = not table_has_rows("raw_news_data")

    # Fetch Market Data
    market_df = fetch_market_data_df(TICKERS, bootstrap=bootstrap_market)
    if not market_df.empty:
        load_with_merge(market_df, "raw_market_data", ["date", "ticker"])

    # Fetch News Data
    news_df = fetch_news_data_df(TICKERS, bootstrap=bootstrap_news)
    if not news_df.empty:
        # Use URL and Ticker as unique key for news
        load_with_merge(news_df, "raw_news_data", ["url", "ticker"])

    # Trigger the asynchronous Cloud Run training job.
    execution_name = trigger_training_job()

    logger.info(f"Ingestion finished successfully. Training execution: {execution_name}")
    return {"status": "success", "training_execution": execution_name}, 200


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
