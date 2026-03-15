ifneq (,$(wildcard .env))
include .env
export FINNHUB_KEY
export FRED_API_KEY
endif

TF_DIR ?= infra/terraform
BOOTSTRAP_TF_DIR ?= infra/terraform/bootstrap

TF_PROJECT_ID := $(shell terraform -chdir=$(TF_DIR) output -raw project_id 2>/dev/null)
TF_REGION := $(shell terraform -chdir=$(TF_DIR) output -raw region 2>/dev/null)
TF_DATASET_ID := $(shell terraform -chdir=$(TF_DIR) output -raw dataset_id 2>/dev/null)
TF_REPO := $(shell terraform -chdir=$(TF_DIR) output -raw artifact_registry_repo 2>/dev/null)
TF_GCS_BUCKET := $(shell terraform -chdir=$(TF_DIR) output -raw model_artifacts_bucket 2>/dev/null)

PROJECT_ID ?= $(if $(TF_PROJECT_ID),$(TF_PROJECT_ID),mkt-sent-friend-260313-225751)
REGION ?= $(if $(TF_REGION),$(TF_REGION),us-central1)
BQ_DATASET_ID ?= $(if $(TF_DATASET_ID),$(TF_DATASET_ID),market_sentiment_predictor)
REPO ?= $(if $(TF_REPO),$(TF_REPO),$(REGION)-docker.pkg.dev/$(PROJECT_ID)/market-sentiment)
GCS_BUCKET ?= $(if $(TF_GCS_BUCKET),$(TF_GCS_BUCKET),mkt-sent-friend-260313-225751-model-artifacts-7cace324)
PIPELINE_SERVICE_ACCOUNT ?= market-sentiment-sa@$(PROJECT_ID).iam.gserviceaccount.com
TRAINING_TICKERS ?=
BOOTSTRAP_MARKET_START ?=
BOOTSTRAP_NEWS_DAYS ?=

PIPELINE_IMAGE ?= $(REPO)/pipeline:latest
ENDPOINT_IMAGE ?= $(REPO)/endpoint:latest

TRAIN_JOB ?= market-sentiment-train
TRAIN_PREP_JOB ?= market-sentiment-prepare-training-data
FEATURE_RANK_SUBMIT_JOB ?= market-sentiment-feature-rank-submit
FEATURE_RANK_FINALIZE_JOB ?= market-sentiment-feature-rank-finalize
FEATURE_RANK_STATUS_JOB ?= market-sentiment-feature-rank-status
BATCH_JOB ?= market-sentiment-batch-infer
PREDICT_SERVICE ?= market-sentiment-predict
INGEST_FUNCTION ?= market-data-ingestion
PREDICT_HEALTH_URL ?= $$(gcloud run services describe $(PREDICT_SERVICE) --project=$(PROJECT_ID) --region=$(REGION) --format='value(status.url)')/health
GCLOUD_ENV_DELIM := ^@^
TRAINING_TICKERS_ENV := $(if $(TRAINING_TICKERS),@TRAINING_TICKERS=$(TRAINING_TICKERS),)
TRAINING_TICKERS_REMOVE_ARGS := $(if $(TRAINING_TICKERS),,--remove-env-vars TRAINING_TICKERS)
FEATURE_RANK_NOTIFICATION_WEBHOOK_ENV := $(if $(FEATURE_RANK_NOTIFICATION_WEBHOOK),@FEATURE_RANK_NOTIFICATION_WEBHOOK=$(FEATURE_RANK_NOTIFICATION_WEBHOOK),)
FEATURE_RANK_NOTIFICATION_REMOVE_ARGS := $(if $(FEATURE_RANK_NOTIFICATION_WEBHOOK),,--remove-env-vars FEATURE_RANK_NOTIFICATION_WEBHOOK)
FEATURE_RANK_RUN_ID_EXEC_ARG := $(if $(FEATURE_RANK_RUN_ID),--update-env-vars FEATURE_RANK_RUN_ID=$(FEATURE_RANK_RUN_ID),)
SMOKE_AUTO_ML_SAMPLE_ROWS ?= 1000
SMOKE_AUTO_ML_MIN_ROWS ?= 1000
SMOKE_AUTO_ML_BUDGET_MILLI_NODE_HOURS ?= 1000
SMOKE_AUTO_ML_MAX_SELECTED_FEATURES ?= 10
TRAINING_ENV_VARS := GCP_PROJECT_ID=$(PROJECT_ID)@GCP_REGION=$(REGION)@BQ_DATASET=$(BQ_DATASET_ID)@GCS_BUCKET=$(GCS_BUCKET)@MIN_ACCURACY=0.52@MODEL_NAME=market_sentiment_lgbm$(TRAINING_TICKERS_ENV)$(FEATURE_RANK_NOTIFICATION_WEBHOOK_ENV)
FEATURE_RANK_SUBMIT_ENV_VARS := GCP_PROJECT_ID=$(PROJECT_ID)@GCP_REGION=$(REGION)@BQ_DATASET=$(BQ_DATASET_ID)@GCS_BUCKET=$(GCS_BUCKET)@MODEL_NAME=market_sentiment_lgbm$(TRAINING_TICKERS_ENV)
FEATURE_RANK_FINALIZE_ENV_VARS := $(FEATURE_RANK_SUBMIT_ENV_VARS)$(FEATURE_RANK_NOTIFICATION_WEBHOOK_ENV)
FEATURE_RANK_STATUS_ENV_VARS := $(FEATURE_RANK_SUBMIT_ENV_VARS)
BATCH_ENV_VARS := GCP_PROJECT_ID=$(PROJECT_ID)@GCP_REGION=$(REGION)@BQ_DATASET=$(BQ_DATASET_ID)@GCS_BUCKET=$(GCS_BUCKET)@MODEL_NAME=market_sentiment_lgbm@FINNHUB_KEY=$(FINNHUB_KEY)@FRED_API_KEY=$(FRED_API_KEY)
INGESTION_ENV_VARS := GCP_PROJECT_ID=$(PROJECT_ID)@GCP_REGION=$(REGION)@BQ_DATASET_ID=$(BQ_DATASET_ID)@FINNHUB_KEY=$(FINNHUB_KEY)@FRED_API_KEY=$(FRED_API_KEY)$(TRAINING_TICKERS_ENV)@BOOTSTRAP_MARKET_START=$(BOOTSTRAP_MARKET_START)@BOOTSTRAP_NEWS_DAYS=$(BOOTSTRAP_NEWS_DAYS)

.PHONY: auth bootstrap-init bootstrap-plan bootstrap-apply infra-init infra-plan infra-apply build-pipeline build-endpoint ensure-train-job ensure-train-prep-job ensure-feature-rank-submit-job ensure-feature-rank-finalize-job ensure-feature-rank-status-job update-pipeline-jobs update-endpoint-service deploy-pipeline deploy-endpoint deploy-function ingest submit-feature-ranking submit-feature-ranking-smoke check-feature-ranking finalize-feature-ranking select-features prepare-training-data train verify-endpoint test test-ml test-all test-live-predict scan-secrets rollout migrate

SECRET_GREP_PATTERN := AIza[0-9A-Za-z_-]{35}|sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|ya29\.[A-Za-z0-9._-]+

auth:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev --quiet

bootstrap-init:
	terraform -chdir=$(BOOTSTRAP_TF_DIR) init

bootstrap-plan:
	terraform -chdir=$(BOOTSTRAP_TF_DIR) plan

bootstrap-apply:
	terraform -chdir=$(BOOTSTRAP_TF_DIR) apply

infra-init:
	terraform -chdir=$(TF_DIR) init

infra-plan:
	terraform -chdir=$(TF_DIR) plan

infra-apply:
	terraform -chdir=$(TF_DIR) apply

build-pipeline:
	docker buildx build --platform linux/amd64 -f backend/ml/Dockerfile -t $(PIPELINE_IMAGE) --push .

build-endpoint:
	docker buildx build --platform linux/amd64 -f backend/api/Dockerfile -t $(ENDPOINT_IMAGE) --push .

ensure-train-job:
	@if gcloud run jobs describe $(TRAIN_JOB) --project=$(PROJECT_ID) --region=$(REGION) >/dev/null 2>&1; then \
		echo "Cloud Run job $(TRAIN_JOB) already exists."; \
	else \
		echo "Creating Cloud Run job $(TRAIN_JOB)..."; \
		gcloud run jobs create $(TRAIN_JOB) \
			--project=$(PROJECT_ID) \
			--image $(PIPELINE_IMAGE) \
			--region=$(REGION) \
			--command python \
			--args=-m,backend.ml.training.job_runner \
			--memory 4Gi \
			--cpu 1 \
			--max-retries 0 \
			--task-timeout 3600 \
			--service-account $(PIPELINE_SERVICE_ACCOUNT) \
			--set-env-vars "$(GCLOUD_ENV_DELIM)$(TRAINING_ENV_VARS)"; \
	fi

ensure-train-prep-job:
	@if gcloud run jobs describe $(TRAIN_PREP_JOB) --project=$(PROJECT_ID) --region=$(REGION) >/dev/null 2>&1; then \
		echo "Cloud Run job $(TRAIN_PREP_JOB) already exists."; \
	else \
		echo "Creating Cloud Run job $(TRAIN_PREP_JOB)..."; \
		gcloud run jobs create $(TRAIN_PREP_JOB) \
			--project=$(PROJECT_ID) \
			--image $(PIPELINE_IMAGE) \
			--region=$(REGION) \
			--command python \
			--args=-m,backend.ml.training.prepare_training_data_job \
			--memory 4Gi \
			--cpu 1 \
			--max-retries 0 \
			--task-timeout 3600 \
			--service-account $(PIPELINE_SERVICE_ACCOUNT) \
			--set-env-vars "$(GCLOUD_ENV_DELIM)$(TRAINING_ENV_VARS)"; \
	fi

ensure-feature-rank-submit-job:
	@if gcloud run jobs describe $(FEATURE_RANK_SUBMIT_JOB) --project=$(PROJECT_ID) --region=$(REGION) >/dev/null 2>&1; then \
		echo "Cloud Run job $(FEATURE_RANK_SUBMIT_JOB) already exists."; \
	else \
		echo "Creating Cloud Run job $(FEATURE_RANK_SUBMIT_JOB)..."; \
		gcloud run jobs create $(FEATURE_RANK_SUBMIT_JOB) \
			--project=$(PROJECT_ID) \
			--image $(PIPELINE_IMAGE) \
			--region=$(REGION) \
			--command python \
			--args=-m,backend.ml.training.feature_ranking_submit_job \
			--memory 4Gi \
			--cpu 1 \
			--max-retries 0 \
			--task-timeout 3600 \
			--service-account $(PIPELINE_SERVICE_ACCOUNT) \
			--set-env-vars "$(GCLOUD_ENV_DELIM)$(FEATURE_RANK_SUBMIT_ENV_VARS)"; \
	fi

ensure-feature-rank-finalize-job:
	@if gcloud run jobs describe $(FEATURE_RANK_FINALIZE_JOB) --project=$(PROJECT_ID) --region=$(REGION) >/dev/null 2>&1; then \
		echo "Cloud Run job $(FEATURE_RANK_FINALIZE_JOB) already exists."; \
	else \
		echo "Creating Cloud Run job $(FEATURE_RANK_FINALIZE_JOB)..."; \
		gcloud run jobs create $(FEATURE_RANK_FINALIZE_JOB) \
			--project=$(PROJECT_ID) \
			--image $(PIPELINE_IMAGE) \
			--region=$(REGION) \
			--command python \
			--args=-m,backend.ml.training.feature_ranking_finalize_job \
			--memory 4Gi \
			--cpu 1 \
			--max-retries 0 \
			--task-timeout 3600 \
			--service-account $(PIPELINE_SERVICE_ACCOUNT) \
			--set-env-vars "$(GCLOUD_ENV_DELIM)$(FEATURE_RANK_FINALIZE_ENV_VARS)"; \
	fi

ensure-feature-rank-status-job:
	@if gcloud run jobs describe $(FEATURE_RANK_STATUS_JOB) --project=$(PROJECT_ID) --region=$(REGION) >/dev/null 2>&1; then \
		echo "Cloud Run job $(FEATURE_RANK_STATUS_JOB) already exists."; \
	else \
		echo "Creating Cloud Run job $(FEATURE_RANK_STATUS_JOB)..."; \
		gcloud run jobs create $(FEATURE_RANK_STATUS_JOB) \
			--project=$(PROJECT_ID) \
			--image $(PIPELINE_IMAGE) \
			--region=$(REGION) \
			--command python \
			--args=-m,backend.ml.training.feature_ranking_status_job \
			--memory 1Gi \
			--cpu 1 \
			--max-retries 0 \
			--task-timeout 600 \
			--service-account $(PIPELINE_SERVICE_ACCOUNT) \
			--set-env-vars "$(GCLOUD_ENV_DELIM)$(FEATURE_RANK_STATUS_ENV_VARS)"; \
	fi

update-pipeline-jobs:
	$(MAKE) ensure-train-job
	$(MAKE) ensure-train-prep-job
	$(MAKE) ensure-feature-rank-submit-job
	$(MAKE) ensure-feature-rank-finalize-job
	$(MAKE) ensure-feature-rank-status-job
	gcloud run jobs update $(TRAIN_JOB) --project=$(PROJECT_ID) --image $(PIPELINE_IMAGE) --region=$(REGION) $(TRAINING_TICKERS_REMOVE_ARGS) --update-env-vars "$(GCLOUD_ENV_DELIM)$(TRAINING_ENV_VARS)"
	gcloud run jobs update $(TRAIN_PREP_JOB) --project=$(PROJECT_ID) --image $(PIPELINE_IMAGE) --region=$(REGION) $(TRAINING_TICKERS_REMOVE_ARGS) --update-env-vars "$(GCLOUD_ENV_DELIM)$(TRAINING_ENV_VARS)"
	gcloud run jobs update $(FEATURE_RANK_SUBMIT_JOB) --project=$(PROJECT_ID) --image $(PIPELINE_IMAGE) --region=$(REGION) $(TRAINING_TICKERS_REMOVE_ARGS) --update-env-vars "$(GCLOUD_ENV_DELIM)$(FEATURE_RANK_SUBMIT_ENV_VARS)"
	gcloud run jobs update $(FEATURE_RANK_FINALIZE_JOB) --project=$(PROJECT_ID) --image $(PIPELINE_IMAGE) --region=$(REGION) $(TRAINING_TICKERS_REMOVE_ARGS) $(FEATURE_RANK_NOTIFICATION_REMOVE_ARGS) --update-env-vars "$(GCLOUD_ENV_DELIM)$(FEATURE_RANK_FINALIZE_ENV_VARS)"
	gcloud run jobs update $(FEATURE_RANK_STATUS_JOB) --project=$(PROJECT_ID) --image $(PIPELINE_IMAGE) --region=$(REGION) $(TRAINING_TICKERS_REMOVE_ARGS) --update-env-vars "$(GCLOUD_ENV_DELIM)$(FEATURE_RANK_STATUS_ENV_VARS)"
	gcloud run jobs update $(BATCH_JOB) --project=$(PROJECT_ID) --image $(PIPELINE_IMAGE) --region=$(REGION) --update-env-vars "$(GCLOUD_ENV_DELIM)$(BATCH_ENV_VARS)"

deploy-pipeline: auth build-pipeline update-pipeline-jobs

submit-feature-ranking:
	gcloud run jobs execute $(FEATURE_RANK_SUBMIT_JOB) --project=$(PROJECT_ID) --region=$(REGION) --wait

submit-feature-ranking-smoke:
	gcloud run jobs execute $(FEATURE_RANK_SUBMIT_JOB) --project=$(PROJECT_ID) --region=$(REGION) --wait --update-env-vars AUTO_ML_REFRESH=true,AUTO_ML_SAMPLE_ROWS=$(SMOKE_AUTO_ML_SAMPLE_ROWS),AUTO_ML_MIN_ROWS=$(SMOKE_AUTO_ML_MIN_ROWS),AUTO_ML_BUDGET_MILLI_NODE_HOURS=$(SMOKE_AUTO_ML_BUDGET_MILLI_NODE_HOURS),AUTO_ML_MAX_SELECTED_FEATURES=$(SMOKE_AUTO_ML_MAX_SELECTED_FEATURES)

check-feature-ranking:
	gcloud run jobs execute $(FEATURE_RANK_STATUS_JOB) --project=$(PROJECT_ID) --region=$(REGION) --wait $(FEATURE_RANK_RUN_ID_EXEC_ARG)

finalize-feature-ranking:
	gcloud run jobs execute $(FEATURE_RANK_FINALIZE_JOB) --project=$(PROJECT_ID) --region=$(REGION) --wait

select-features: submit-feature-ranking

prepare-training-data:
	gcloud run jobs execute $(TRAIN_PREP_JOB) --project=$(PROJECT_ID) --region=$(REGION) --wait

train:
	gcloud run jobs execute $(TRAIN_JOB) --project=$(PROJECT_ID) --region=$(REGION) --wait

update-endpoint-service:
	gcloud run services update $(PREDICT_SERVICE) --project=$(PROJECT_ID) --image $(ENDPOINT_IMAGE) --region=$(REGION) --update-env-vars FINNHUB_KEY=$(FINNHUB_KEY),FRED_API_KEY=$(FRED_API_KEY)

deploy-endpoint: auth build-endpoint update-endpoint-service

deploy-function:
	gcloud functions deploy $(INGEST_FUNCTION) \
		--project=$(PROJECT_ID) \
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
		--set-env-vars "$(GCLOUD_ENV_DELIM)$(INGESTION_ENV_VARS)"

ingest:
	gcloud functions call $(INGEST_FUNCTION) --project=$(PROJECT_ID) --region=$(REGION) --gen2

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

# rollout: Deploy the pipeline, ingest bronze data, and submit feature ranking. Training remains manual after review.
rollout: migrate deploy-pipeline deploy-function ingest submit-feature-ranking deploy-endpoint verify-endpoint
