# ── Forecast Studio — Makefile ─────────────────────────────────────────────
#
#   make local-up       — full stack locally
#   make local-down     — stop
#   make test-backend   — backend tests (uv)
#   make deploy-dev     — deploy to Azure dev (requires az + ACR)
#

.PHONY: help local-up local-down local-logs local-tools local-platform migrate \
        build-backend build-frontend test-backend test-frontend lint \
        deploy-dev deploy-test deploy-prod infra-dev infra-test infra-prod

COMPOSE := docker compose
ACR_NAME ?= $(shell az acr list --query '[0].name' -o tsv 2>/dev/null)

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ═══════════════════════════════════════════════════════════════════════════
# LOCAL DEVELOPMENT
# ═══════════════════════════════════════════════════════════════════════════

local-up: ## Full stack: app + Postgres/Redis + MLflow + Dagster + Langfuse
	$(COMPOSE) --profile observability up -d
	@echo ""
	@echo "  Frontend:  http://localhost:3000"
	@echo "  Backend:   http://localhost:8000"
	@echo "  API docs:  http://localhost:8000/docs"
	@echo "  MLflow:    http://localhost:5000"
	@echo "  Dagster:   http://localhost:3005"
	@echo "  Langfuse:  http://localhost:3001"
	@echo ""

local-down: ## Stop local stack
	$(COMPOSE) down

local-clean: ## Stop and remove volumes
	$(COMPOSE) down -v

local-logs: ## Follow service logs
	$(COMPOSE) logs -f backend celery-worker frontend langfuse-web langfuse-worker

local-tools: ## PgAdmin + Redis Commander
	$(COMPOSE) --profile tools up -d
	@echo "  PgAdmin:    http://localhost:5050"
	@echo "  Redis CLI:  http://localhost:8081"

local-dataops: ## Ensure MLflow is up (included in default stack; idempotent)
	$(COMPOSE) up -d mlflow
	@echo "  MLflow: http://localhost:5000"

local-platform: ## All default services (same as: docker compose up -d)
	$(COMPOSE) up -d
	@echo ""
	@echo "  Frontend:   http://localhost:3000"
	@echo "  Backend:    http://localhost:8000"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Dagster UI: http://localhost:3005"
	@echo ""

local-restart: ## Restart backend + celery
	$(COMPOSE) restart backend celery-worker

local-rebuild: ## Rebuild images and start
	$(COMPOSE) up -d --build

local-infra: ## Postgres + Redis only
	$(COMPOSE) up -d postgres redis
	@echo "  PostgreSQL: localhost:5432"
	@echo "  Redis:      localhost:6379"

migrate: ## Run Alembic migrations
	$(COMPOSE) run --rm migrate

# ═══════════════════════════════════════════════════════════════════════════
# TESTS & QUALITY
# ═══════════════════════════════════════════════════════════════════════════

test-backend: ## Backend pytest (from workspace)
	cd backend && uv run pytest tests/ -v --tb=short

test-frontend: ## Frontend tests
	cd frontend && npm test

lint: ## Ruff on backend + core library + orchestration
	uv run ruff check backend/app packages/forecaster/forecaster orchestration/fs_orch
	cd frontend && npm run lint

# ═══════════════════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════════════════

build-backend: ## Build backend Docker image
	docker build -f backend/Dockerfile -t forecast-studio-backend:local .

build-frontend: ## Build frontend Docker image
	docker build -f frontend/Dockerfile -t forecast-studio-frontend:local .

build-all: build-backend build-frontend

# ═══════════════════════════════════════════════════════════════════════════
# AZURE DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════════

_check-az:
	@az account show > /dev/null 2>&1 || (echo "❌ Log in: az login" && exit 1)

infra-dev: _check-az ## Deploy infra (Bicep) — DEV
	az group create --name forecaster-dev-rg --location westeurope --tags environment=dev project=forecast-studio
	az deployment group create \
		--resource-group forecaster-dev-rg \
		--template-file infra/main.bicep \
		--parameters environment=dev

infra-test: _check-az ## Deploy infra — TEST
	az group create --name forecaster-test-rg --location westeurope --tags environment=test project=forecast-studio
	az deployment group create \
		--resource-group forecaster-test-rg \
		--template-file infra/main.bicep \
		--parameters environment=test

infra-prod: _check-az ## Deploy infra — PROD
	@echo "⚠️  PRODUCTION infrastructure. Ctrl+C to cancel..."
	@sleep 3
	az group create --name forecaster-prod-rg --location westeurope --tags environment=prod project=forecast-studio
	az deployment group create \
		--resource-group forecaster-prod-rg \
		--template-file infra/main.bicep \
		--parameters environment=prod

_push-images: _check-az
	@[ -n "$(ACR_NAME)" ] || (echo "❌ ACR_NAME not set" && exit 1)
	az acr login --name $(ACR_NAME)
	docker build -f backend/Dockerfile -t $(ACR_NAME).azurecr.io/forecaster-backend:$(TAG) .
	docker build -f frontend/Dockerfile -t $(ACR_NAME).azurecr.io/forecaster-frontend:$(TAG) .
	docker push $(ACR_NAME).azurecr.io/forecaster-backend:$(TAG)
	docker push $(ACR_NAME).azurecr.io/forecaster-frontend:$(TAG)

deploy-dev: TAG=dev-$(shell git rev-parse --short HEAD)
deploy-dev: _push-images ## Deploy app — DEV
	az containerapp update --resource-group forecaster-dev-rg --name forecaster-dev-backend \
		--image $(ACR_NAME).azurecr.io/forecaster-backend:$(TAG)
	az containerapp update --resource-group forecaster-dev-rg --name forecaster-dev-frontend \
		--image $(ACR_NAME).azurecr.io/forecaster-frontend:$(TAG)
	az containerapp exec --resource-group forecaster-dev-rg --name forecaster-dev-backend \
		--command "alembic upgrade head" || true
	@echo "✅ Deployed to DEV"

deploy-test: TAG=test-$(shell git rev-parse --short HEAD)
deploy-test: _push-images ## Deploy app — TEST
	az containerapp update --resource-group forecaster-test-rg --name forecaster-test-backend \
		--image $(ACR_NAME).azurecr.io/forecaster-backend:$(TAG)
	az containerapp update --resource-group forecaster-test-rg --name forecaster-test-frontend \
		--image $(ACR_NAME).azurecr.io/forecaster-frontend:$(TAG)
	az containerapp exec --resource-group forecaster-test-rg --name forecaster-test-backend \
		--command "alembic upgrade head" || true
	@echo "✅ Deployed to TEST"

deploy-prod: TAG=prod-$(shell git rev-parse --short HEAD)
deploy-prod: ## Deploy app — PROD
	@echo "⚠️  PRODUCTION. Type 'yes' to continue:"
	@read -r confirm && [ "$$confirm" = "yes" ] || (echo "Cancelled" && exit 1)
	$(MAKE) _push-images TAG=$(TAG)
	az containerapp update --resource-group forecaster-prod-rg --name forecaster-prod-backend \
		--image $(ACR_NAME).azurecr.io/forecaster-backend:$(TAG)
	az containerapp update --resource-group forecaster-prod-rg --name forecaster-prod-frontend \
		--image $(ACR_NAME).azurecr.io/forecaster-frontend:$(TAG)
	az containerapp exec --resource-group forecaster-prod-rg --name forecaster-prod-backend \
		--command "alembic upgrade head" || true
	@echo "✅ Deployed to PROD"

status-dev: _check-az
	@echo "=== DEV ==="
	@az containerapp list --resource-group forecaster-dev-rg --output table 2>/dev/null || echo "No apps"

status-test: _check-az
	@echo "=== TEST ==="
	@az containerapp list --resource-group forecaster-test-rg --output table 2>/dev/null || echo "No apps"

status-prod: _check-az
	@echo "=== PROD ==="
	@az containerapp list --resource-group forecaster-prod-rg --output table 2>/dev/null || echo "No apps"

status-all: status-dev status-test status-prod
