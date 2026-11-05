# Setup and run guide

Updated 2 November 2026. Implementation and measured verification status are in [COMPLETION_REPORT.md](COMPLETION_REPORT.md).

## Current installation

The Python environment, CPU PyTorch, Chronos-Bolt weights, backend, dashboard and Docker/PostgreSQL stack are installed. Real Chronos forecasting passed on 200 SKUs. The root `.env` is configured for Groq. Live verification passed with `openai/gpt-oss-20b`; see [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for the completed supplied-data evaluation.

Currently running:

- Local dashboard: http://127.0.0.1:5173 (real Chronos, adaptive selection, SQLite).
- Local API documentation: http://127.0.0.1:8000/docs.
- Docker dashboard: http://127.0.0.1:8082 (evaluation container: real Chronos, fixed model strategy, PostgreSQL).
- Docker API: http://127.0.0.1:8001/docs.

Servers must be started again after shutdown. Offline results are explicitly labelled `EmpiricalOffline`; they are not Chronos or LLM research evidence.

## 1. Enable live Groq schema alignment

Open the existing root `.env` in your editor. Preserve its other settings and set:

```dotenv
LLM_API_KEY=your_groq_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
LLM_JSON_SCHEMA=false
LLM_REQUIRED=true
FORECAST_MODE=chronos
FORECAST_STRATEGY=adaptive
```

Do not send the key in chat or enter it in the dashboard. The dashboard's **Middleware API key** field is only for the separate `API_KEY` setting protecting your own backend. `.env` is ignored by Git. The hybrid mapper sends column profiles and three sample values to Groq; use the supplied synthetic data for testing.

From the project root in PowerShell:

```powershell
.venv\Scripts\python.exe check_setup.py --llm --chronos --database --output outputs/live_setup.json
```

The command makes provider calls and fails if a requested integration fails. A check without those flags only reports configuration. Restart the backend after changing `.env`. A key or account quota failure cannot be corrected by the local application.

The LLM maps columns; Chronos forecasts numeric demand locally. Chronos needs no Groq key. Setting `LLM_REQUIRED=false` allows explicitly labelled deterministic schema fallback when the key is absent. Use `true` for live verification.

## 2. Start the local application

In a terminal at the project root:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open http://127.0.0.1:5173. Do not launch a second copy if these ports are already serving the project. Stop your terminal's server with Ctrl+C before restarting it.

1. Upload `data/raw/automotive_transactions.csv` or one of the CSV mutations under `data/`.
2. Inspect the proposed column mapping and correct any ambiguous assignments.
3. Choose the horizon and submit the analysis. Progress is polled from a background job.
4. Inspect the demand-regime scatter, SKU forecast bands, selection scores and replenishment table.
5. Reload a saved run from the sidebar or download its JSON.

Browser click-through verification remains manual because the computer-use tool could not establish Chrome's current URL. Backend upload, jobs, forecasting, storage and retrieval were tested through HTTP.

## 3. Generate thesis artifacts

Offline reproducibility check (no provider requests):

```powershell
.venv\Scripts\python.exe run_all_experiments.py --mode offline --output outputs/offline
.venv\Scripts\python.exe verify_artifacts.py outputs/offline
```

After the live setup check passes, generate all five 300-DPI figures, required CSV/LaTeX tables, adaptive selection tables and audit files:

```powershell
.venv\Scripts\python.exe run_all_experiments.py --mode full --output outputs/full_groq
.venv\Scripts\python.exe verify_artifacts.py outputs/full_groq --require-live
```

Full mode makes live provider calls across the schema benchmark and latency repetitions. It can consume quota. Inspect `schema_raw.csv` and `run_manifest.json` for failures before reporting results. Offline schema scores must not be presented as live LLM performance.

To run real forecasting and adaptive selection without waiting for a provider key:

```powershell
.venv\Scripts\python.exe run_forecasting_experiments.py --mode chronos --output outputs/chronos_validation
```

This has already completed for 200 SKUs. It does not generate live schema scores or the entire five-figure report. The full suite above does.

## 4. Tests and integration checks

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe smoke_system.py --base-url http://127.0.0.1:8000 --skus 200 --output outputs/local_chronos_system.json
cd frontend
npm run build
```

The smoke test uploads synthetic data with an explicit reviewed mapping, so it does not test the LLM. It checks the job, saved run, canonical transactions and selection audit. Its concurrent health requests are a connectivity smoke test, not a production inference-capacity benchmark.

## 5. Docker with PostgreSQL

Start Docker Desktop. To reproduce the verified offline stack alongside the local app:

```powershell
$env:API_PORT='8001'
$env:FORECAST_MODE='offline'
docker compose up --build -d
```

Open http://127.0.0.1:8082. The port avoids an unrelated service already using 8080. Remove PowerShell overrides when switching back to local Chronos:

```powershell
Remove-Item Env:FORECAST_MODE -ErrorAction SilentlyContinue
Remove-Item Env:API_PORT -ErrorAction SilentlyContinue
```

For a Chronos-enabled container, set `WITH_CHRONOS=true` and `FORECAST_MODE=chronos` in the Compose environment and rebuild the API image. This optional image downloads additional dependencies and its own model cache; the verified real-Chronos installation currently runs on the host. Do not assume the offline image contains Chronos.

Compose keeps database/model state in named volumes. `docker compose down` stops the stack; avoid `down -v` unless you intend to delete persisted state.

## 6. Database operations and deployment scope

Migrations run automatically at backend startup. Existing legacy JSON runs remain readable; transaction history cannot be reconstructed for old runs that never stored it.

```powershell
.venv\Scripts\python.exe manage_database.py migrate
.venv\Scripts\python.exe manage_database.py backup --output outputs/backups/manual_backup.db
.venv\Scripts\python.exe manage_database.py retention --days 90
```

Backups require a new output filename. Retention is a dry run unless `--apply` is explicitly supplied. Use PostgreSQL's `pg_dump` for PostgreSQL backups.

Use a single backend process: the bounded queue and model lock are process-local. Queued/running jobs interrupted by restart become failed and require resubmission. For deployment, configure a strong `API_KEY` (at least 32 characters with `DEPLOYMENT_MODE=production`), a non-default database password, HTTPS and appropriate access controls. Multi-tenant identity, distributed scheduling and production throughput qualification are not implemented.

## 7. Reinstalling dependencies on another computer

Python 3.11 and Node.js are required. On this computer, use the existing `.venv`:

```powershell
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe -r requirements-chronos.txt
cd frontend
npm ci
```

On a fresh machine create `.venv` first with `py -3.11 -m venv .venv`. Copy `.env.example` only when `.env` does not already exist. Do not replace an existing credential configuration. The first Chronos load needs internet access; subsequent runs can reuse `.cache/huggingface`.

See README for metric definitions, training/holdout separation, the approximate Chronos mean, recursive horizons, and the limitation of summing marginal quantiles for replenishment. These assumptions must remain visible in the thesis.
