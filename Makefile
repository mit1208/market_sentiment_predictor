ifneq (,$(wildcard .env))
include .env
export FINNHUB_KEY
endif

PROJECT_ID ?= llm-testing-489113
REGION ?= us-central1
BQ_DATASET_ID ?= market_sentiment_predictor
REPO ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/market-sentiment
GCS_BUCKET ?= llm-testing-489113-model-artifacts-01783489
PIPELINE_SERVICE_ACCOUNT ?= market-sentiment-sa@$(PROJECT_ID).iam.gserviceaccount.com

PIPELINE_IMAGE ?= $(REPO)/pipeline:latest
ENDPOINT_IMAGE ?= $(REPO)/endpoint:latest

TRAIN_JOB ?= market-sentiment-train
BATCH_JOB ?= market-sentiment-batch-infer
PREDICT_SERVICE ?= market-sentiment-predict
INGEST_FUNCTION ?= market-data-ingestion
PREDICT_HEALTH_URL ?= $$(gcloud run services describe $(PREDICT_SERVICE) --region=$(REGION) --format='value(status.url)')/health

.PHONY: auth build-pipeline build-endpoint ensure-train-job update-pipeline-jobs update-endpoint-service deploy-pipeline deploy-endpoint deploy-function ingest train verify-endpoint test-live-predict rollout migrate

auth:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev --quiet

build-pipeline:
	docker buildx build --platform linux/amd64 -f pipeline/Dockerfile -t $(PIPELINE_IMAGE) --push .

build-endpoint:
	docker buildx build --platform linux/amd64 -f endpoint/Dockerfile -t $(ENDPOINT_IMAGE) --push .

ensure-train-job:
	@if gcloud run jobs describe $(TRAIN_JOB) --region=$(REGION) >/dev/null 2>&1; then \
		echo "Cloud Run job $(TRAIN_JOB) already exists."; \
	else \
		echo "Creating Cloud Run job $(TRAIN_JOB)..."; \
		gcloud run jobs create $(TRAIN_JOB) \
			--image $(PIPELINE_IMAGE) \
			--region=$(REGION) \
			--command python \
			--args=-m,pipeline.job_runner \
			--memory 4Gi \
			--cpu 1 \
			--max-retries 0 \
			--task-timeout 3600 \
			--service-account $(PIPELINE_SERVICE_ACCOUNT) \
			--set-env-vars GCP_PROJECT_ID=$(PROJECT_ID),GCP_REGION=$(REGION),BQ_DATASET=$(BQ_DATASET_ID),GCS_BUCKET=$(GCS_BUCKET),MIN_ACCURACY=0.50,MODEL_NAME=market_sentiment_lgbm,FINNHUB_KEY=$(FINNHUB_KEY); \
	fi

update-pipeline-jobs:
	$(MAKE) ensure-train-job
	gcloud run jobs update $(TRAIN_JOB) --image $(PIPELINE_IMAGE) --region=$(REGION) --update-env-vars FINNHUB_KEY=$(FINNHUB_KEY)
	gcloud run jobs update $(BATCH_JOB) --image $(PIPELINE_IMAGE) --region=$(REGION) --update-env-vars FINNHUB_KEY=$(FINNHUB_KEY)

deploy-pipeline: auth build-pipeline update-pipeline-jobs

train:
	gcloud run jobs execute $(TRAIN_JOB) --region=$(REGION) --wait

update-endpoint-service:
	gcloud run services update $(PREDICT_SERVICE) --image $(ENDPOINT_IMAGE) --region=$(REGION) --update-env-vars FINNHUB_KEY=$(FINNHUB_KEY)

deploy-endpoint: auth build-endpoint update-endpoint-service

deploy-function:
	gcloud functions deploy $(INGEST_FUNCTION) \
		--gen2 \
		--region=$(REGION) \
		--runtime=python311 \
		--source=market_data_ingestion \
		--entry-point=run_ingestion \
		--trigger-http \
		--allow-unauthenticated \
		--service-account=$(PIPELINE_SERVICE_ACCOUNT) \
		--memory=1Gi \
		--timeout=540s \
		--set-env-vars GCP_PROJECT_ID=$(PROJECT_ID),GCP_REGION=$(REGION),BQ_DATASET_ID=$(BQ_DATASET_ID),TRAINING_JOB_NAME=$(TRAIN_JOB),FINNHUB_KEY=$(FINNHUB_KEY)

ingest:
	gcloud functions call $(INGEST_FUNCTION) --region=$(REGION) --gen2

verify-endpoint:
	curl -sSf $(PREDICT_HEALTH_URL)

TICKER ?= AAPL

test-live-predict:
	python3 scripts/test_live_predict.py --ticker $(TICKER)

MIGRATE ?= true
DBMATE_FLAGS ?= --no-dump-schema

# up: Apply all pending BigQuery migrations using dbmate
.PHONY: up
up:
	@echo "Running BigQuery migrations with dbmate..."
	@DATABASE_URL="bigquery://$(PROJECT_ID)/$(BQ_DATASET_ID)" dbmate --migrations-dir db/migrations $(DBMATE_FLAGS) up

# down: Rollback the most recent BigQuery migration using dbmate
.PHONY: down
down:
	@echo "Rolling back BigQuery migrations with dbmate..."
	@DATABASE_URL="bigquery://$(PROJECT_ID)/$(BQ_DATASET_ID)" dbmate --migrations-dir db/migrations rollback

# migrate: Alias for 'up'
migrate: up

# rollback: Alias for 'down'
rollback: down

# rollout: Deploy the full pipeline including database migrations and data refresh
rollout: migrate deploy-function ingest deploy-pipeline train deploy-endpoint verify-endpoint
