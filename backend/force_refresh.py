from google.cloud import bigquery

PROJECT_ID = "mkt-sent-friend-260313-225751"
DATASET_ID = "market_sentiment_predictor"
TICKER = "O"

bq = bigquery.Client(project=PROJECT_ID)

query = f"""
    DELETE FROM `{PROJECT_ID}.{DATASET_ID}.live_feature_snapshots`
    WHERE ticker = '{TICKER}'
"""

print(f"Deleting cached data for {TICKER}...")
job = bq.query(query)
job.result()
print("Done. The next prediction request will fetch fresh data.")
