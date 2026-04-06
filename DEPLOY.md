# Deployment Guide

## Spis treści

- [Local Development](#local-development)
- [Branch Strategy & CI/CD](#branch-strategy--cicd)
- [Azure Setup (pierwszy raz)](#azure-setup-pierwszy-raz)
- [GitHub Secrets Configuration](#github-secrets-configuration)
- [Ręczny deploy z CLI](#ręczny-deploy-z-cli)
- [Troubleshooting](#troubleshooting)

---

## Local Development

### Wymagania

- Docker Desktop (20.10+) z Docker Compose v2
- Node.js 18+ (opcjonalnie, do dev frontendu bez Dockera)
- Python 3.12+ + uv (opcjonalnie, do dev backendu bez Dockera)

### Quick Start

```bash
# 1. Sklonuj repo i skonfiguruj .env
git clone <repo_url> && cd forecast-studio
cp .env.example .env
# Uzupełnij LLM_API_KEY w .env (lub legacy DEEPSEEK_API_KEY — patrz CONFIG.md)

# 2. Uruchom cały stack
make local-up
# lub: docker compose up -d

# 3. Uruchom migracje (pierwsza sesja)
make migrate
```

**Dostępne adresy:**

| Serwis           | URL                          |
|------------------|------------------------------|
| Frontend         | http://localhost:3000         |
| Backend API      | http://localhost:8000         |
| API Docs (Swagger) | http://localhost:8000/docs  |
| PgAdmin*         | http://localhost:5050         |
| Redis Commander* | http://localhost:8081         |

\* Wymaga `--profile tools`: `make local-tools`

### Przydatne komendy

```bash
make local-logs      # Logi (backend + celery + frontend)
make local-restart   # Restart backend + celery po zmianach
make local-rebuild   # Rebuild obrazów + restart
make local-infra     # Tylko postgres + redis (do dev natywnie)
make local-clean     # Zatrzymaj i usuń dane
make lint            # Ruff + ESLint
make test-backend    # pytest
make test-frontend   # npm test
```

### Dev bez Dockera (frontend/backend natywnie)

```bash
# Uruchom tylko infra
make local-infra

# Backend (w osobnym terminalu)
cd backend
uv sync --all-packages --group dev
DATABASE_URL=postgresql+asyncpg://forecaster:forecaster_dev_password@localhost:5432/forecaster \
  cd backend && uv run uvicorn app.main:app --reload --port 8000

# Frontend (w osobnym terminalu)
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

---

## Branch Strategy & CI/CD

### Automatyczny deploy (GitHub Actions)

**Wyłączony** — w repozytorium nie ma workflowu ciągłego wdrożenia na Azure; zakładany jest rozwój lokalny (`docker compose`, `make local-up`).
**PR:** [`.github/workflows/pr-checks.yml`](.github/workflows/pr-checks.yml) (m.in. Ruff, testy backendu).

Wdrożenie na Azure wyłącznie **ręcznie**: [Ręczny deploy z CLI](#ręczny-deploy-z-cli) oraz `make infra-*` / `make deploy-*` w Makefile (wymaga `az` i skonfigurowanego ACR).

### Opcjonalny model branchy (na przyszłość)

Jeśli kiedyś przywrócisz pipeline (np. własny workflow lub zewnętrzny runner), typowy podział to `develop` → DEV, `release/*` → TEST, `main` → PROD.

---

## Azure Setup (pierwszy raz)

### 1. Zaloguj się do Azure

```bash
az login
az account set --subscription "<subscription-name>"
```

### 2. Stwórz Service Principal

```bash
# Stwórz SP z rolą Contributor na subskrypcji
az ad sp create-for-rbac \
  --name "forecaster-github-sp" \
  --role Contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv) \
  --sdk-auth

# Zapisz cały output JSON — to będzie secret AZURE_CREDENTIALS
```

### 3. Deploy infrastruktury (ręcznie, pierwszy raz per środowisko)

```bash
# DEV
make infra-dev

# TEST
make infra-test

# PROD
make infra-prod
```

### 4. ACR — pobierz adres registry

```bash
az acr list --query '[0].{name:name, login:loginServer}' -o table
# Zanotuj ACR_NAME (np. forecasterdevacr)
```

---

## GitHub Secrets Configuration

### Wymagane GitHub Environments

Stwórz w **Settings → Environments**:

| Environment | Protected | Reviewers |
|-------------|-----------|-----------|
| `dev`       | Nie       | —         |
| `test`      | Tak       | Opcjonalnie |
| `prod`      | Tak       | **Wymagane** (min. 1 reviewer) |

### Secrets per Environment

W każdym environment (`dev`, `test`, `prod`) ustaw:

| Secret                | Opis                                          |
|-----------------------|-----------------------------------------------|
| `AZURE_CREDENTIALS`  | JSON output z `az ad sp create-for-rbac`     |
| `ACR_NAME`           | Nazwa Azure Container Registry                |
| `DB_ADMIN_PASSWORD`  | Hasło do PostgreSQL w danym środowisku        |
| `JWT_SECRET`         | Losowy string do podpisywania tokenów JWT     |
| `LLM_API_KEY`        | Klucz API do modelu LLM (OpenAI-compatible)   |
| `DEEPSEEK_API_KEY`   | (Opcjonalnie) legacy — jeśli nie używasz `LLM_API_KEY` |

#### Przykład generowania JWT_SECRET

```bash
openssl rand -hex 32
```

### Zmienne repozytorium (opcjonalne)

W **Settings → Secrets and variables → Actions → Variables**:

| Variable         | Wartość          |
|------------------|------------------|
| `AZURE_LOCATION` | `westeurope`     |

---

## Ręczny deploy z CLI

Jeśli potrzebujesz wdrożyć bez CI/CD:

```bash
# Build i push do ACR + update Container App
make deploy-dev      # DEV
make deploy-test     # TEST
make deploy-prod     # PROD (wymaga potwierdzenia)

# Tylko infrastruktura (Bicep)
make infra-dev
make infra-test
make infra-prod

# Status
make status-all      # Pokaż status wszystkich środowisk
```

---

## Resource Groups i nazewnictwo

| Środowisko | Resource Group          | Prefix zasobów    |
|------------|-------------------------|--------------------|
| DEV        | `forecaster-dev-rg`    | `forecaster-dev-`  |
| TEST       | `forecaster-test-rg`   | `forecaster-test-` |
| PROD       | `forecaster-prod-rg`   | `forecaster-prod-` |

### Zasoby per środowisko

| Zasób                   | DEV                       | TEST                      | PROD                      |
|-------------------------|---------------------------|---------------------------|---------------------------|
| PostgreSQL              | B1ms (1 vCore, 2GB)      | B1ms (1 vCore, 2GB)      | B2s (2 vCore, 4GB)       |
| Redis                   | Basic C0 (250MB)          | Basic C0 (250MB)          | Basic C1 (1GB)            |
| Backend CPU/mem         | 0.25 / 0.5Gi             | 0.5 / 1Gi                | 1.0 / 2Gi                |
| Backend replicas        | 0–1                       | 0–2                       | 1–5                       |
| Storage redundancy      | LRS                       | LRS                       | GRS                       |
| Log retention           | 7 dni                     | 14 dni                    | 90 dni                    |

---

## Troubleshooting

### Lokalne

```bash
# Backend nie startuje
docker compose logs backend

# Problemy z bazą
docker compose exec postgres psql -U forecaster -d forecaster

# Reset bazy
make local-clean && make local-up && make migrate

# Build cache
docker compose build --no-cache
```

### Azure

```bash
# Logi Container App
az containerapp logs show \
  --name forecaster-dev-backend \
  --resource-group forecaster-dev-rg \
  --type console

# Restart Container App
az containerapp revision restart \
  --name forecaster-dev-backend \
  --resource-group forecaster-dev-rg \
  --revision <revision-name>

# Status deploymentu
make status-dev
```

### CI (PR checks)

- **Fail na backendzie** — uruchom lokalnie: `cd backend && uv run ruff check app/ && uv run pytest tests/`
- **Sekrety przy ręcznym `make deploy-*`** — `AZURE_CREDENTIALS`, `ACR_NAME` itd. muszą być dostępne w środowisku / skonfigurowane zgodnie z Makefile
