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

.PHONY: auth build-pipeline build-endpoint ensure-train-job update-pipeline-jobs update-endpoint-service deploy-pipeline deploy-endpoint deploy-function ingest train verify-endpoint test test-ml test-all test-live-predict scan-secrets rollout migrate

SECRET_GREP_PATTERN := AIza[0-9A-Za-z_-]{35}|sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|ya29\.[A-Za-z0-9._-]+

auth:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev --quiet

build-pipeline:
	docker buildx build --platform linux/amd64 -f backend/ml/Dockerfile -t $(PIPELINE_IMAGE) --push .

build-endpoint:
	docker buildx build --platform linux/amd64 -f backend/api/Dockerfile -t $(ENDPOINT_IMAGE) --push .

ensure-train-job:
	@if gcloud run jobs describe $(TRAIN_JOB) --region=$(REGION) >/dev/null 2>&1; then \
		echo "Cloud Run job $(TRAIN_JOB) already exists."; \
	else \
		echo "Creating Cloud Run job $(TRAIN_JOB)..."; \
		gcloud run jobs create $(TRAIN_JOB) \
			--image $(PIPELINE_IMAGE) \
			--region=$(REGION) \
			--command python \
			--args=-m,backend.ml.training.job_runner \
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
		--source=backend/ingestion \
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

TEST_PATHS ?= backend/tests
ML_TEST_PATHS ?= backend/ml/testing

test:
	PYTHONPATH=. pytest $(TEST_PATHS)

test-ml:
	PYTHONPATH=. pytest $(ML_TEST_PATHS)

test-all:
	PYTHONPATH=. pytest $(ML_TEST_PATHS) $(TEST_PATHS)

scan-secrets:
	@tracked_sensitive="$$(git ls-files '.env' '.env.local' '.env.*.local' 'frontend/.env.local' '*.pem' '*.key' '*.p12' '*.crt' '*.cer' '*.mobileprovision' '*.tfvars' '*.tfvars.json' '*credentials*.json' '*service-account*.json')"; \
	if [ -n "$$tracked_sensitive" ]; then \
		echo "Tracked sensitive files detected:"; \
		printf '%s\n' "$$tracked_sensitive"; \
		exit 1; \
	fi; \
	potential_matches="$$(git grep -lE '$(SECRET_GREP_PATTERN)' -- . ':(exclude)infra/terraform/.terraform.lock.hcl' || true)"; \
	if [ -n "$$potential_matches" ]; then \
		echo "Potential secret-like content detected in tracked files:"; \
		printf '%s\n' "$$potential_matches"; \
		exit 1; \
	fi; \
	echo "No tracked secret files or common secret patterns detected."

TICKER ?= AAPL

test-live-predict:
	@url="$(PREDICT_API_URL)"; \
	if [ -z "$$url" ]; then \
		echo "PREDICT_API_URL is required" >&2; \
		exit 2; \
	fi; \
	case "$$url" in \
		*/predict) predict_url="$$url" ;; \
		*) predict_url="$${url%/}/predict" ;; \
	esac; \
	curl -sS -X POST "$$predict_url" \
		-H "Content-Type: application/json" \
		-d "{\"instances\":[{\"ticker\":\"$(TICKER)\"}]}"

MIGRATE ?= true
DBMATE_FLAGS ?= --no-dump-schema

# up: Apply all pending BigQuery migrations using dbmate
.PHONY: up
up:
	@echo "Running BigQuery migrations with dbmate..."
	@DATABASE_URL="bigquery://$(PROJECT_ID)/$(BQ_DATASET_ID)" dbmate --migrations-dir backend/db/migrations $(DBMATE_FLAGS) up

# down: Rollback the most recent BigQuery migration using dbmate
.PHONY: down
down:
	@echo "Rolling back BigQuery migrations with dbmate..."
	@DATABASE_URL="bigquery://$(PROJECT_ID)/$(BQ_DATASET_ID)" dbmate --migrations-dir backend/db/migrations rollback

# migrate: Alias for 'up'
migrate: up

# rollback: Alias for 'down'
rollback: down

# rollout: Deploy the full pipeline including database migrations and data refresh
rollout: migrate deploy-function ingest deploy-pipeline train deploy-endpoint verify-endpoint
