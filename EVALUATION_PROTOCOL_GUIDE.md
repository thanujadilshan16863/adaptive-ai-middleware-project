# Running the supplied evaluation specification

The master runner reads the four supplied CSV files without modifying them and exports tables to `outputs/tables/`, figures to `outputs/figures/`, and raw scores/manifests to `outputs/`.

## Dataset findings and explicit deviations

The supplied files contain 36,550 dense daily records: **50 SKUs and 731 days**, from 2024-01-01 through 2025-12-31. Training uses the first 584 days and evaluation the remaining 147 days. Training-only classification finds **13 Smooth and 37 Intermittent** SKUs, with no Erratic or Lumpy SKUs. Missing regimes are reported as zero in the dataset outline, not populated with invented accuracy results.

With the user's approval, figures 4.1 and 4.4 use one clearly labelled supplemental synthetic lumpy SKU from `data/processed/daily_demand.csv`. It is excluded from the supplied-data accuracy tables. Its hash, classification and identity are recorded in the manifest. Other figures use the supplied dataset or measured benchmark results.

The files omit lead time and category. The inventory experiment uses an explicit **14-day lead-time scenario** and `Unknown` category. `unit_price` is only a value proxy; stock volume and fill rate are the primary inventory outcomes. No acquisition-cost or currency claim is made.

The 10/100/200-SKU stress workloads contain 7,310 / 73,100 / 146,200 rows. The two larger workloads use deterministic copies of source series with new identifiers for scaling only. These copies never enter forecast accuracy scoring.

## Live model configuration

The configured `llama-3.1-8b-instant` model returned Groq `model_not_found`. With user approval the experiment uses **`openai/gpt-oss-20b` through Groq**, recorded exactly in the output provenance. This is a change from the specification's named Llama/GPT examples; results must not be described as Llama-3 results. The provider key stays in `.env` and is never written to reports.

## Full execution

From the project root in PowerShell, with Docker Desktop running:

```powershell
$env:API_PORT='8001'
docker compose -f docker-compose.yml -f docker-compose.evaluation.yml up --build -d api
.venv\Scripts\python.exe check_setup.py --llm
.venv\Scripts\python.exe run_all_experiments.py --mode full --dataset-dir 'C:\Users\thanu\Downloads\research_dateset' --latency-backend docker --api-url http://127.0.0.1:8001 --synthetic-lumpy-illustration --output outputs
.venv\Scripts\python.exe verify_artifacts.py outputs --require-live --require-docker
```

The Docker override enables CPU Chronos, fixed foundation forecasts and mandatory LLM mapping, and reuses the local Hugging Face cache. Local `.env` adaptive settings remain separate. The benchmark persists warm-up and measured analyses in PostgreSQL. Three repetitions are the default; use `--repeats` to change this. Running the command makes provider requests and may consume quota.

## Measurement definitions

- Schema precision, recall, F1 and exact match score **part_id, transaction_date and quantity_sold only**. Each tier has one supplied file. Exact match is therefore 0% or 100% for a given method/tier, not a robust estimate across many exports. Other mapped fields remain visible in raw JSON predictions.
- Ground truth is derived from unique, exact full-column equality to the independent master file. No mapper sees that answer key. If files no longer match, supply `schema_ground_truth.json` keyed by export filename with canonical-to-source mappings; ambiguous matching fails rather than guessing.
- MAE/RMSE use point forecasts; MASE/RMSSE use lag-7 training scaling. Zero scaling denominators are undefined. Mean pinball loss and weighted quantile loss are separate columns. Weighted q90 loss is `2 * sum(pinball_q90) / sum(abs(actual))`, pooled across SKUs within regime, and undefined for zero total demand. This follows the normalized quantile-loss convention in [AutoGluon documentation](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html).
- Docker timings are actual synchronous FastAPI HTTP requests against PostgreSQL. Four measured groups are parsing, schema mapping, insertion plus commit (including record preparation), and Chronos plus reorder math. In the application, persistence occurs after inference; group numbering is a report grouping, not a changed execution order. Normalization, serialization and HTTP wall time are recorded separately.
- Peak RAM is the sampled sum of the API and PostgreSQL containers' `docker stats` memory, expressed in MiB. It excludes Docker Desktop's VM overhead and other containers. The shared workload peak is repeated in each phase row; do not sum it across phases or describe it as phase-local memory. Sampling can miss short spikes. Raw sample times are retained.
- The initial 10-SKU warm-up is excluded from measured repetitions. Three repetitions and one file per schema tier support descriptive comparisons, not broad significance claims.

The runner retains TSB and adaptive-selection outputs in addition to the models requested by this specification. Tables 4.5/4.6 are supplemental. Summing marginal quantiles remains a replenishment approximation, not a guaranteed 90% service level.

## Offline development check

```powershell
.venv\Scripts\python.exe run_all_experiments.py --mode offline --dataset-dir 'C:\Users\thanu\Downloads\research_dateset' --latency-backend process --synthetic-lumpy-illustration --output outputs/specification_dry_run --repeats 1
```

This does not satisfy the live Docker experiment. It labels its forecast substitutes and process-only measurements and is useful for validating file ingestion, scoring and plots.
