# Implementation and verification report

**Historical implementation snapshot.** Live Groq and the supplied-data Docker evaluation have since passed; see [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md). The browser-interaction limitation below still applies.

Verified locally on 2 November 2026. The six-phase research application is implemented. Live Groq verification and browser interaction verification remain incomplete; this is not a certification of production readiness.

## Completed implementation

- CSV/XLSX ingestion, schema preview and manual mapping review.
- Fuzzy, header-only LLM and hybrid-profile LLM mappers; validated mappings, one semantic correction, bounded transport retry and explicit offline provenance.
- Canonical transaction storage, daily zero-filling and ADI/CV-squared classification.
- Real Chronos-Bolt, CrostonClassic, SBA, TSB and SeasonalNaive; adaptive candidate selection using an inner training validation window and audited scores.
- Quantile replenishment, safety stock and lost-sales inventory experiments.
- Background analysis jobs, progress polling, persisted runs, transaction retrieval, Alembic migrations, SQLite backup and opt-in retention.
- React mapping review, demand-regime scatter, forecast fan chart, selection details, inventory filtering and saved-run loading.
- Original report tables and five 300-DPI figures, plus adaptive selection and paired bootstrap tables.
- Docker API, Nginx frontend and PostgreSQL deployment.

## Observed verification

| Check | Result | Evidence |
|---|---|---|
| Automated unit/API tests | 20 passed | `python -m pytest -q` |
| React production compilation | Passed | `frontend/dist/` |
| Real Chronos model integration | Passed, including 28/175-step forecasts | `outputs/check_chronos.json` |
| Real 200-SKU forecast and adaptive benchmark | Completed | `outputs/chronos_validation/` |
| Real 200-SKU HTTP analysis | Passed, 79,957 input rows, about 47.4 seconds | `outputs/local_chronos_system.json` |
| Adaptive audit in saved HTTP result | 200 SKU audits; Chronos/SBA/TSB/SeasonalNaive selected | Same HTTP evidence |
| Docker/PostgreSQL HTTP analysis | Passed, 4 parts and 1,577 canonical transactions | `outputs/docker_postgres_system.json` |
| Persistence across API restart | Saved PostgreSQL run returned HTTP 200 | Run `e9737d8f-d0cf-445f-baed-45ac90c137f6` |
| Complete offline experiment run | Completed in about 74 seconds | `outputs/completed_offline/` |
| Required artifact validation | Five CSV/LaTeX tables and five 300-DPI PNGs passed | `verify_artifacts.py outputs/completed_offline` |
| Concurrent health smoke checks | 40 requests, concurrency 8, zero errors on each tested stack | HTTP smoke JSON files |

The health concurrency test does not measure inference throughput. HTTP smoke tests supply an explicit reviewed mapping and therefore do not test the external LLM. Offline artifacts do not measure live LLM or Chronos performance. The real forecasting folder contains real Chronos results, but no live schema benchmark.

## Remaining verification and user action

1. Add your Groq key to `LLM_API_KEY` in the existing root `.env`, set `LLM_REQUIRED=true`, and restart the backend. The last safe configuration check reported no key. Do not share it in chat.
2. Run `check_setup.py --llm --chronos --database`, followed by the full benchmark and `verify_artifacts.py --require-live` commands in the setup guide. This produces the remaining live schema/latency evidence and the combined live report.
3. Open the dashboard and manually check upload, mapping review, chart interaction and saved-run selection. Computer Use ended because it could not verify Chrome's current URL with sufficient confidence, even after user authorization. No browser click-through pass is claimed.

## Scope and research limitations

The application supports a trusted single-workspace deployment. It does not provide multi-tenant identity, a distributed job queue, or audited production security/throughput. Interrupted queued jobs are marked failed and need resubmission. Configure HTTPS and credentials for deployment.

Research conclusions remain limited by synthetic data, one fixed outer holdout, approximate baseline uncertainty and recursive Chronos horizons beyond 64 steps. Summed marginal q90 forecasts do not guarantee a 90% inventory service level. Adaptive bootstrap intervals describe this dataset; they do not establish external validity. See README for the complete experimental definitions.

Follow [SETUP_AND_RUN_GUIDE.md](SETUP_AND_RUN_GUIDE.md) to run the system and complete the remaining live checks.
