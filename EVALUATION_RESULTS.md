# Completed evaluation results

Executed 2 November 2026 using `C:\Users\thanu\Downloads\research_dateset`. The full run completed in **233.4 seconds**, excluding the initial Docker build. All **24 automated tests passed**. Artifact verification passed with both `--require-live` and `--require-docker`.

## Deliverables

- [Required and supplemental CSV/LaTeX tables](outputs/tables/)
- [Five 300-DPI PNG figures](outputs/figures/)
- [Run manifest and methodological assumptions](outputs/run_manifest.json)
- [Docker timing and memory audit](outputs/docker_stress_audit.json)
- [Schema predictions and request counts](outputs/schema_raw.csv)
- [Reproduction commands and protocol](EVALUATION_PROTOCOL_GUIDE.md)

Use the nested `tables/` and `figures/` folders for this evaluation. Older files directly under `outputs/` are historical artifacts from earlier experiments.

## Schema alignment

The live provider was **Groq**, using **openai/gpt-oss-20b** with user approval after the previously configured Llama model returned `model_not_found`. All six live mapper requests completed without provider errors.

| Method | Tier 1 F1 | Tier 2 F1 | Tier 3 F1 |
|---|---:|---:|---:|
| Fuzzy | 1.00 | 1.00 | 0.50 |
| Header-only LLM | 1.00 | 0.80 | 0.50 |
| Hybrid-profile LLM | 1.00 | 1.00 | 1.00 |

Only the three required canonical fields are scored. There is **one file per tier**; these outcomes are descriptive, not evidence of general superiority across ERP systems.

## Container scaling

Three measured warm repetitions per workload were sent through FastAPI to real Chronos and PostgreSQL. A separate warm-up was excluded.

| SKUs | Dense input rows | Median HTTP wall time | Sampled peak API + DB RAM |
|---:|---:|---:|---:|
| 10 | 7,310 | 1,202.56 ms | 458.72 MiB |
| 100 | 73,100 | 5,063.83 ms | 567.55 MiB |
| 200 | 146,200 | 9,855.43 ms | 705.10 MiB |

The larger workloads replicate source series solely for scaling. RAM excludes Docker Desktop VM overhead, and sampling can miss short peaks. The four-stage latency chart excludes separately reported normalization, serialization and HTTP overhead; its stacked total is not the complete HTTP wall time.

## Inventory example

On the supplied Intermittent SKUs, median-based replenishment achieved **66.69%** fill rate with **8.13** average on-hand units; q90 replenishment achieved **97.18%** with **73.37** units; SBA achieved **76.85%** with **12.00** units. This shows the observed service/stock trade-off under the specified simulation assumptions, not a guaranteed service level or universal advantage.

## Dataset limits that must remain in the thesis

The files contain **50 SKUs over 731 days**, classified from training history as **13 Smooth and 37 Intermittent**. They contain no Erratic or Lumpy training series, so accuracy tables cannot establish performance for those regimes. With user approval, the lumpy time-series and fan-chart illustrations use a clearly labelled supplemental synthetic SKU, excluded from accuracy tables.

Lead time is absent and was set to a **14-day scenario**. `unit_price` is a value proxy, not verified acquisition cost. The holdout is 147 days, beyond Chronos-Bolt's native 64-step horizon, using its recursive extension. Weighted quantile loss and mean pinball loss are reported separately. Summed marginal quantiles are a replenishment approximation.

The Docker API remains in the fixed-Chronos evaluation configuration on port 8001. The local application remains separately configured for adaptive selection. Restart an already-running local backend to load the new `.env` model/key settings. No new browser interaction test was performed in this evaluation.
