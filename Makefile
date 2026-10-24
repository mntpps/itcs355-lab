# ITCS355 Lab 1
# `make reproduce` is the one command a grader runs. Keep it working.

SHELL := /bin/bash
IMAGE ?= itcs355-lab1
TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
PLATFORM ?= linux/amd64
SEED ?= 20260101

# cloud.env supplies CONTAINER_REGISTRY and friends to the Lab 3/4 targets. Optional, so
# a grader without one can still run `make reproduce`.
-include cloud.env

.PHONY: help setup cloud-check data test portability-audit train image image-push reproduce verify clean teardown \
        train-remote tune compare reload-check serve serve-image loadtest deploy smoke drift inject-drift pipeline cost swap-check llm-eval llm-gate \
        scan-secrets test-unit test-contract test-behaviour test-integration lab4-setup reference-profile \
        threshold-study monitor-up monitor-down staging-target replay replay-drift drift-live drift-modes cost-report

help:
	@grep -hE "^[a-zA-Z0-9_-]+:.*?## .*$$" $(firstword $(MAKEFILE_LIST)) | awk -F":.*?## " "{printf \"  %-20s %s\\n\", \$$1, \$$2}"

setup: ## Install dependencies and print environment status
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "environment ok"

cloud-check: ## Resolve the eight capability slots
	python scripts/cloud_check.py

data: ## Generate the default dataset (deterministic)
	python scripts/make_dataset.py --seed $(SEED)

test: ## Run data contract and split property tests
	pytest -q tests/

portability-audit: ## Fail if provider strings leak into src/
	python scripts/portability_audit.py

train: ## Train locally, outside the container
	python -m src.train --seed $(SEED) --metrics-out reports/metrics.json

image: ## Build the training image for linux/amd64
	docker buildx build --platform $(PLATFORM) -t $(IMAGE):$(TAG) --load .

image-push: image ## Push to CONTAINER_REGISTRY via your adapter
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	print(get_adapter(config.load()).push_image(\"$(IMAGE):$(TAG)\"))"

reproduce: data image ## THE ONE COMMAND. Grader runs this.
	docker run --rm \
	  -v "$$PWD/data:/app/data:ro" \
	  -v "$$PWD/reports:/app/reports" \
	  -e MLFLOW_TRACKING_URI=sqlite:////app/reports/mlflow.db \
	  $(IMAGE):$(TAG) --seed $(SEED) --metrics-out /app/reports/metrics.json

verify: ## Check the produced metric against the README claim
	python scripts/verify_metric.py

scan-secrets: ## Scan Git history for credential-shaped values
	python scripts/scan_secrets.py

LAB ?= 4
teardown: ## Delete resources tagged course=itcs355 lab=LAB (default 4), and stop the drift schedule
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	cfg=config.load(); print(get_adapter(cfg).teardown(cfg.tags($(LAB))))"
	@# The schedule outlives every resource it watches unless it is switched off too.
	-gh workflow disable drift.yml && echo "drift schedule disabled"
	-$(STACK) down 2>/dev/null

cost-report: ## Confirm teardown, then list anything still tagged for LAB
	python scripts/teardown_verify.py --lab $(LAB)
	-gh workflow list --all | grep -i drift

clean: ## Remove local artifacts
	rm -rf mlruns mlartifacts mlflow.db reports/metrics.json .pytest_cache

# --- Lab 2 -------------------------------------------------------------------
train-remote: ## Submit and wait for Azure ML training
	PYTHONPATH=. python scripts/train_remote.py

tune: ## Budgeted hyperparameter study (>=12 trials)
	python -m src.tune --trials 12 --budget-thb 150

compare: ## Rank runs by metric and by cost per point
	python scripts/compare_runs.py --experiment itcs355-lab2

reload-check: ## Load the registered model by version and score rows
	python scripts/reload_check.py --name $(MODEL_REGISTRY_NAME) --version $(VERSION)

# --- Lab 3 -------------------------------------------------------------------
serve: ## Run the inference service locally on :8080
	python scripts/export_model.py --out reports/model.joblib
	MODEL_PATH=reports/model.joblib MODEL_VERSION=local uvicorn service.app:app --port 8080

serve-image: ## Build the serving image
	docker buildx build --platform $(PLATFORM) -f service/Dockerfile.serve -t itcs355-serve:$(TAG) --load .

# Lab 3 deployed by hand from a laptop; Lab 4's CD does the same thing from CI.
CONTAINER_APP ?= itcs355-predict
INSTANCE      ?= 0.25:0.5Gi
MODEL_VERSION ?= 2
SERVE_IMAGE   ?= $(strip $(CONTAINER_REGISTRY)):$(shell git rev-parse HEAD 2>/dev/null)

deploy: ## Deploy SERVE_IMAGE to the Azure Container App
	MODEL_VERSION=$(MODEL_VERSION) python -c "from cloudlayer.factory import get_adapter; from src import config; \
	print(get_adapter(config.load()).deploy('$(SERVE_IMAGE)', '$(CONTAINER_APP)', '$(INSTANCE)'))"

smoke: ## Three known payloads against the deployed endpoint
	python scripts/smoke_test.py --app $(CONTAINER_APP)

loadtest: ## Load test at three concurrency levels
	@for vus in 1 10 50; do \
	  echo "=== $$vus VUs ==="; \
	  k6 run -e TARGET=$(TARGET) -e VUS=$$vus loadtest/k6.js || true; \
	done

# --- Lab 4 -------------------------------------------------------------------
TARGET_URL ?= http://localhost:8080
STACK = docker compose -f monitoring/stack/docker-compose.yml

test-unit: ## Unit tests: code only, no data needed for most
	pytest -q tests/test_unit.py tests/test_service.py

test-contract: ## Data contract tests: assertions about the data
	pytest -q tests/test_data.py

test-behaviour: ## Model behaviour tests: what the model does, as served
	pytest -q tests/test_model_behaviour.py

test-integration: ## Integration test against a running service at TARGET_URL
	SERVICE_URL=$(TARGET_URL) pytest -q tests/test_integration.py

lab4-setup: ## One-time Azure OIDC identity, roles, GitHub secrets/vars (needs az, gh)
	bash scripts/azure_lab4_setup.sh

reference-profile: ## Rebuild service/reference_profile.json from the training split
	python scripts/build_reference_profile.py

threshold-study: ## Re-derive the drift thresholds from the no-drift noise floor (~2 min)
	python scripts/drift_threshold_study.py

monitor-up: ## Local service + Prometheus + Grafana (http://localhost:3000)
	python scripts/export_model.py --out reports/model.joblib
	BUILD_SHA=$(TAG) $(STACK) up -d --build
	@echo "dashboard: http://localhost:3000/d/itcs355-service   prometheus: http://localhost:9090"

monitor-down: ## Stop the monitoring stack (stops scraping staging, letting it scale to zero)
	$(STACK) down

staging-target: ## Point the dashboard's `staging` job at the Container App
	@fqdn=$$(az containerapp show -n $(CONTAINER_APP) -g $(strip $(PROJECT_ID)) \
	  --query properties.configuration.ingress.fqdn -o tsv) && \
	printf '[{"targets": ["%s:443"], "labels": {"env": "staging"}}]\n' "$$fqdn" \
	  > monitoring/stack/targets/staging.local.json && echo "staging target: $$fqdn"

replay: ## Normal fleet traffic to TARGET_URL (fills the drift window)
	python scripts/replay_traffic.py --url $(TARGET_URL) --source data/raw/sensors.csv --count 1500

inject-drift: ## Shift a feature's distribution on purpose
	python scripts/inject_drift.py --feature temp_c --mode shift --magnitude 6

replay-drift: inject-drift ## Replay the shifted data to TARGET_URL and stamp the injection time
	python scripts/replay_traffic.py --url $(TARGET_URL) --source data/current.csv --count 1500 \
	  --mark-injection --label "temp_c shift +6C -> $(TARGET_URL)"

drift: ## Score drift of data/current.csv against the training reference
	python -m monitoring.drift --current data/current.csv

drift-live: ## Score the live rolling window at TARGET_URL
	python -m monitoring.drift --current-url $(TARGET_URL)

drift-modes: ## Run the detector against all three injection modes (Drill 4 asks)
	@for m in "shift 6" "scale 1.5" "mix 5"; do set -- $$m; \
	  echo "=== --mode $$1 --magnitude $$2"; \
	  python scripts/inject_drift.py --feature temp_c --mode $$1 --magnitude $$2 >/dev/null; \
	  python -m monitoring.drift --current data/current.csv | head -8; done

# --- Lab 5 -------------------------------------------------------------------
pipeline: ## Compile pipeline/pipeline.yaml for your provider
	python -c "from cloudlayer.pipelines import compile_for; from src import config; \
	compile_for(config.load().provider)"

llm-eval: ## Run the LLM golden set against recorded responses (offline, free)
	python scripts/llm_eval.py --out reports/llm_eval-baseline.json

llm-gate: ## Prove the gate fails on a degraded set — expected to exit non-zero
	python scripts/llm_eval.py --out reports/llm_eval-baseline.json >/dev/null
	python scripts/llm_eval.py --responses evals/fixtures/triage-regressed.jsonl \
	  --out reports/llm_eval.json --baseline reports/llm_eval-baseline.json

cost: ## Build the cost report
	python scripts/cost_report.py --estimate $(EST) --actual $(ACT) --rps $(RPS) --instance $(INSTANCE)

swap-check: ## Prove the portability seam against a second provider
	python scripts/portability_swap_check.py --second-provider $(SECOND)
