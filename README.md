# Adaptive AI Middleware for Automated Demand Forecasting

A modular BSc research implementation for automotive intermittent and lumpy demand. Includes a six-phase FastAPI service, a React client, reproducible synthetic data, classical intermittent forecasting, optional zero-shot Chronos-Bolt, and automated report artifacts.

For the supplied four-file testing specification, use [EVALUATION_PROTOCOL_GUIDE.md](EVALUATION_PROTOCOL_GUIDE.md). The master runner now writes tables under `outputs/tables/` and PNGs under `outputs/figures/`. It supports the supplied 50-SKU dataset and actual Docker/PostgreSQL measurements, with explicit dataset limitations and separately labelled supplemental lumpy illustrations.

**Start with [SETUP_AND_RUN_GUIDE.md](SETUP_AND_RUN_GUIDE.md)** for the reviewed Windows setup, OpenAI/Groq configuration, real Chronos checks, browser workflow, verification status and troubleshooting. Run `python check_setup.py` for a local configuration report; add `--llm --chronos --database` explicitly to perform live integration checks.

## Run locally

Python 3.11 is recommended. From this directory, on Windows:

```powershell
# Use the existing .venv on this computer; uv installs without requiring pip inside it.
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe -r requirements.txt
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item .env.example .env }
.venv\Scripts\python run_all_experiments.py --mode offline
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The working environment created during implementation can be used directly with `.venv\Scripts\python.exe`. Linux/macOS use `python3` and `.venv/bin/python`.

Open `http://localhost:8000/docs` for the API. Optional client:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Select a CSV/XLSX, inspect the mapping, review ambiguous fields, and run the analysis. The client displays persisted replenishment results. `VITE_API_URL` can point the client to another backend. API keys are held in component memory, not local storage.

## Real-model experiment

```powershell
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe -r requirements-chronos.txt
# Edit .env: set LLM_API_KEY, provider settings below, LLM_REQUIRED=true, FORECAST_MODE=chronos
.venv\Scripts\python check_setup.py --llm --chronos --database
.venv\Scripts\python run_all_experiments.py --mode full --output outputs/full
.venv\Scripts\python verify_artifacts.py outputs/full --require-live
```

| Provider | LLM_BASE_URL | LLM_MODEL | LLM_JSON_SCHEMA |
|---|---|---|---|
| OpenAI | https://api.openai.com/v1 | gpt-4o-mini | true |
| Groq | https://api.groq.com/openai/v1 | openai/gpt-oss-20b | false |

The first Chronos execution downloads `autogluon/chronos-bolt-small`. Full mode requires an API key and fails if Chronos cannot load. It never substitutes an empirical forecast for Chronos. API failures in the schema experiment are retained as failed predictions with errors in `schema_raw.csv`; inspect `schema_failed_calls` before using results in the report. Header-only requests include names and the target schema, but no sample values or dtypes. Hybrid requests add dtype, null ratio, uniqueness ratio and three distinct non-null sample values. These samples are sent to the configured external provider.

Offline mode explicitly forces deterministic rules and an `EmpiricalOffline` distribution, even if an API key exists. It provides runnable integration demonstrations and report layouts, **not measured LLM or Chronos performance**. Numeric cryptic columns may be unidentifiable from a small profile; offline matching abstains rather than using the generator's answer key. Live API temperature zero does not guarantee bitwise reproducibility.

## Modules and six phases

| Phase | Implementation | Behavior |
|---|---|---|
| 1 | `backend/app/services/phase1_ingestion.py` | Bounded UTF-8 CSV/XLSX parsing, duplicate header checks, workbook expansion limit, profiles |
| 2 | `phase2_llm_mapper.py` | Global one-to-one RapidFuzz assignment, header-only LLM, hybrid LLM, strict Pydantic validation, one correction retry |
| 3 | `phase3_normalizer.py` | Canonical validation, duplicate transaction aggregation, common daily grid, sample ADI/CV² classification |
| 4 | `phase4_forecaster.py` | Batched Chronos-Bolt and CrostonClassic, CrostonSBA, TSB, SeasonalNaive; nonnegative ordered quantiles |
| 5 | `phase5_replenish.py` | Safety stock, reorder points, inventory position, lead-time lost-sales simulation |
| 6 | `main.py`, `pipeline.py`, `frontend/` | JSON serialization, REST API, SQLite/PostgreSQL result persistence, React client |

`POST /api/schema/preview` accepts multipart `file`. `POST /api/analyze` accepts `file`, optional `mapping_json` (canonical field → exact source column, null allowed), and `horizon` (1–366 daily periods). Missing required mappings return 422. Forecast coverage is automatically extended to the longest lead time plus one day. `GET /api/runs/{run_id}` retrieves persisted results. `GET /health` reports the selected forecasting mode. All analysis endpoints support `X-API-Key` when `API_KEY` is configured. The database stores JSON snapshots, canonical part metadata and aggregated transactions. `POST /api/jobs` submits background analyses; `GET /api/jobs/{job_id}` reports progress. `GET /api/runs` lists saved runs and `GET /api/runs/{run_id}/transactions` provides paginated canonical history.

The API has a bounded job queue with one forecasting worker, durable job status, upload/grid limits, and API-key protection. Queue overflow returns 429. Interrupted jobs are marked failed after restart and must be resubmitted. Use one Uvicorn worker. Alembic migrations preserve legacy runs and add canonical parts, transactions and jobs. `manage_database.py` provides SQLite backup and opt-in retention. Docker Compose includes PostgreSQL, the API and an Nginx-served React dashboard. TLS, identity/tenant isolation, distributed workers and deployment load testing remain deployment responsibilities; this is not an audited production service.

The default `FORECAST_STRATEGY=adaptive` selects among regime-appropriate foundation/statistical candidates using an inner temporal validation window, then refits on all available history. Each SKU includes the selection scores. `FORECAST_STRATEGY=fixed` uses the configured foundation engine directly. Adaptive results and paired SKU bootstrap intervals are exported separately in tables 4.5 and 4.6; the original model comparison remains unchanged. See [COMPLETION_REPORT.md](COMPLETION_REPORT.md) for verification evidence and remaining limits.

## Dataset and experimental protocol

`generate_synthetic_data.py` produces 200 SKUs, 50 per demand regime, with 730 daily periods starting 2023-01-01. That is 104 complete weeks **plus two days**, not exactly 104 weeks. The optional `7D` transformation retains the last partial bin (105 bins); all reported experiments use daily data. Rows are positive sales; missing SKU-days are structural zeros under an explicit continuous-assortment assumption. For real data, supply known observation windows and distinguish missing reporting/stockouts from true zero demand before making this assumption.

Compound occurrence/rounded-lognormal size distributions, mild annual seasonality, varied lead times and LKR costs create automotive-like demand. Rejection sampling guarantees the intended training and full-history regimes; it uses the full generated series for the latter guarantee, so this is a controlled synthetic benchmark rather than an unbiased population sample. No intended regime is exposed to the schema mappers. The seed, raw-file SHA-256, package versions and protocol are saved. Thirty mutations vary synonyms/jargon, randomly permute opaque source names and column order, resample rows, and add an irrelevant export-batch field. Every ground-truth mapping lives separately in `data/manifest.json` and is used only by scoring.

ADI = number of periods / positive-demand periods. CV² uses **sample variance (`ddof=1`) of positive demand / squared positive mean**. Threshold equality belongs to the high-ADI/high-CV² side. All-zero and one-event histories have explicit `All-zero`/`Insufficient` labels and undefined CV². No arbitrary finite value is inserted for undefined statistics.

Forecast evaluation uses a fixed 584-day training / 146-day holdout split, with no actual holdout values used for forecasts, calibration, scaled-error denominators, regime assignment or opening stock. Forecasts extend beyond the holdout to cover the last inventory protection period. MASE and RMSSE scale against lag-7 training differences; zero denominators produce missing scores and valid-score counts. MAE/RMSE use the expected-demand approximation/point forecast. Pinball uses q90. Results are macro-averaged across SKUs within regime; inventory service is demand-weighted. No percentage error metric dividing by actual demand is used.

Croston uses first-positive size and initial interval equal to its 1-based location, with smoothing alpha=.1. SBA applies `(1-alpha/2)`. TSB uses size alpha=.1 and occurrence beta=.1. SeasonalNaive repeats the final seven daily values. Classical q90 is an additive empirical quantile of up to 100 training-only, expanding-origin one-step residuals, clipped at zero. These are approximate marginal uncertainty estimates, not native intermittent predictive distributions; their multi-step calibration is an experimental limitation.

Chronos-Bolt supplies q10/q50/q80/q90. Its library's second `predict_quantiles` output is actually **the median**, despite the variable name `mean`. This implementation integrates all nine decile forecasts, with constant tails below .1 and above .9, to produce an explicitly labelled approximate mean. Quantiles are clipped nonnegative and sorted. The daily test horizon exceeds Bolt's native 64-step horizon: `limit_prediction_length=False` permits the library's recursive extension. Do not represent this as native direct 146-step inference.

## Inventory interpretation

All policies start with the same `ceil(training mean × (lead time + 1))` stock and no outstanding orders. Every day: receive due orders, satisfy demand up to on-hand stock, record lost sales and closing stock, then order up to the fixed-origin forecast target after subtracting on-hand plus outstanding stock. Orders arrive exactly L days later. There are no backorders, order costs, pack sizes, capacity limits, supplier failures or demand censoring. Holding means average end-of-day on-hand units; value is multiplied by synthetic unit cost.

Policies compare sums of q50, sums of q90 and SBA point forecasts over L+1 days. **A sum of marginal q90 values is not generally a q90 of cumulative demand.** It is a comonotonic planning approximation, not a guaranteed 90% cycle service level or fill rate. The same caution applies to summed medians. A calibrated joint lead-time demand distribution would be needed to claim a target service guarantee. Achieved fill rate is actual units served / units demanded; zero-total-demand service is undefined. Dynamic API safety stock is `max(0, sum(q90 over L) - sum(approximate mean over L))`.

## Report artifacts

The default run exports directly into `outputs/`:

- `table_3_1_dataset_outline.csv` and `.tex`: variables, counts and observation grid.
- `table_4_1_schema_benchmark.csv` and `.tex`: field precision, recall, F1, full-file exact match percentage and mean latency for each tier/method/backend. Wrong mappings count as both a false positive and a missed true pair; abstentions reduce recall.
- `table_4_2_forecasting_metrics_by_regime.csv` and `.tex`: all five models, error scores, q90 pinball, simulated fill rate, SKU and valid-score counts.
- `table_4_3_inventory_service_simulation.csv` and `.tex`: achieved service, stock units/value and model provenance by policy/regime.
- `table_4_4_system_latency_benchmark.csv` and `.tex`: all six phases, 10/100/200 SKUs, median and p95 latency, sampled peak process RSS and incremental RSS.
- `fig_4_1_missing_and_zeros.png`, `fig_4_2_syntetos_boylan_quadrants.png`, `fig_4_3_schema_mapping_f1.png`, `fig_4_4_forecast_comparison_lumpy_sku.png`, `fig_4_5_middleware_latency_breakdown.png`: 300-DPI figures.
- Raw per-file/per-SKU/per-repetition scores, training regime labels and `run_manifest.json` support audit and further statistical analysis.

The fan-chart SKU is the lumpy series with the most positive training periods (ties follow SKU order). Selection uses no test errors; the criterion makes the marginal bands more visible. Verify generated files with `python verify_artifacts.py`.

Latency runs are warm, use three repetitions by default (`--repeats` changes this), and include schema calls when in full mode. Model-load time is recorded separately. RSS is sampled every 2 ms, including native library allocations but excluding GPU VRAM; very short peaks may be missed. Phase 6 measures serialization, not network transport or database commit. Shared process RSS includes already-resident models. These are descriptive measurements, not strong significance evidence. Synthetic results do not establish superiority on real automotive demand.

## Reference interfaces

- [Chronos-Bolt source and API](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py), including its median-return note and extended-horizon implementation.
- [Requested model card](https://huggingface.co/autogluon/chronos-bolt-small).
- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat), used for strict JSON schema; compatible endpoints can use JSON-object mode plus local Pydantic validation.

Dependencies use bounded version ranges. `requirements.lock.txt` records the verified Windows offline environment; `frontend/package-lock.json` locks the client dependencies. `run_manifest.json` records exact installed versions for each experiment; preserve it alongside the tables. Unit and integration tests cover classification, zero-filling, baseline recurrence, metrics, schema retry, deterministic data, inventory conservation, the mocked Chronos tensor contract and authenticated API persistence. Real-model and Docker integration require their respective dependencies/services and must be tested in the target environment.
