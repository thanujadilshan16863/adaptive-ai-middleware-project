# Adaptive AI Middleware Framework for Automated Demand Forecasting
**Student:** R. A. N. M. T. D. Nawarathne (`KIC-BSCCSAI-25.1P-050` | Coventry ID: `17167851`)  
**Institution:** Kandy Innovation Centre, National Institute of Business Management (NIBM)  
**Target Domain:** Automotive Spare Parts Inventory & Intermittent Demand Forecasting  

---

## 1. System Overview & Architectural Blueprint

This specification defines the complete end-to-end engineering and experimental benchmarking architecture for the **Adaptive AI Middleware Framework**. It is designed to satisfy both the **6-phase middleware architecture** promised in the research proposal and the **Data Science / Experimental Results requirements** of the NIBM BSc Final Report Template (Chapter 03 Methodology & Chapter 04 Results: Data Preprocessing, Exploratory Data Analysis, and Model Comparison).

### 1.1 High-Level 6-Phase Pipeline Architecture

```
[Raw Messy CSV / Excel ERP Export]
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Automated Data Ingestion & Structural Validation (FastAPI)     │
│ - Checks file corruption, encoding, minimum historical length (>=12m)   │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: LLM-Powered Semantic Schema Alignment (Llama-3 via Groq/Ollama)│
│ - Extracts Headers + Inferred Pandas Dtypes + 3 Sample Rows + Null %    │
│ - Enforces Pydantic JSON output with 1-step self-correction loop        │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Data Transformation, Zero-Filling & Regime Stratification      │
│ - Maps to Canonical PostgreSQL Schema & aggregates to daily/weekly grid │
│ - Fills missing transaction dates with explicit 0s                      │
│ - Computes Syntetos-Boylan ADI & CV² -> Smooth, Erratic, Interm., Lumpy │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Zero-Shot AI Inference Engine (Chronos-Bolt vs. Baselines)     │
│ - Zero-shot probabilistic forecasting via `autogluon/chronos-bolt-small`│
│ - Generates quantile distribution: q_0.10, q_0.50 (median), q_0.90      │
│ - Runs parallel benchmark against Croston, SBA, TSB, and Seasonal Naive │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: Business Logic & Quantile-Driven Replenishment Engine          │
│ - Avoids median zero-collapse on lumpy SKUs by using upper quantiles    │
│ - Calculates Dynamic Safety Stock, Reorder Point (ROP), & Order Qty     │
│ - Flags Stockout Risk / Overstock Status                                │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: Frontend Delivery & Automated Report Artifact Generator        │
│ - Serves JSON via FastAPI to React.js Web Dashboard                     │
│ - Saves high-res EDA charts (PNG) and LaTeX/CSV tables for Ch. 4 Report │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Complete Project Directory Structure

```text
adaptive-ai-middleware/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                          # Raw automotive spare parts CSVs
│   ├── mutated_schemas/              # 30 mutated CSVs (Tier 1, 2, 3) for Exp 1
│   └── processed/                    # Canonical zero-filled time-series parquet/CSVs
├── outputs/                          # Auto-generated artifacts for NIBM Final Report Ch. 4
│   ├── figures/
│   │   ├── fig_4_1_missing_and_zeros.png
│   │   ├── fig_4_2_syntetos_boylan_quadrants.png
│   │   ├── fig_4_3_schema_mapping_f1.png
│   │   ├── fig_4_4_forecast_comparison_lumpy_sku.png
│   │   └── fig_4_5_middleware_latency_breakdown.png
│   └── tables/
│       ├── table_3_1_dataset_outline.csv
│       ├── table_4_1_schema_benchmark.csv
│       ├── table_4_2_forecasting_metrics_by_regime.csv
│       ├── table_4_3_inventory_service_simulation.csv
│       └── table_4_4_system_latency_benchmark.csv
├── backend/
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py                   # FastAPI entrypoint & CORS config
│   │   ├── config.py                 # Env settings (DB URL, LLM API keys, Model IDs)
│   │   ├── database.py               # SQLAlchemy / PostgreSQL connection
│   │   ├── models.py                 # PostgreSQL ORM tables
│   │   ├── schemas.py                # Pydantic validation schemas
│   │   └── services/
│   │       ├── phase1_ingestion.py   # File validation & profiling
│   │       ├── phase2_llm_mapper.py  # Llama-3 hybrid schema alignment + retry
│   │       ├── phase3_normalizer.py  # Zero-filling & ADI/CV² classification
│   │       ├── phase4_forecaster.py  # Chronos-Bolt + Croston/SBA/TSB engine
│   │       └── phase5_replenish.py   # Quantile safety stock & ROP calculator
│   └── benchmarks/
│       ├── generate_synthetic_data.py# Builds realistic automotive spare parts dataset
│       ├── run_all_experiments.py    # Master script to generate all Ch. 4 tables/plots
│       ├── exp1_schema_matching.py   # Evaluates Header-Only vs Hybrid LLM across 30 files
│       ├── exp2_forecasting_eval.py  # Evaluates Chronos-Bolt vs SBA/Croston/TSB
│       └── exp3_system_latency.py    # Benchmarks phase-by-phase latency & memory
└── frontend/
    ├── Dockerfile
    ├── package.json
    └── src/
        ├── App.jsx                   # Single-page React.js Inventory Decision Dashboard
        └── components/
            ├── FileUploadMapper.jsx  # Shows live LLM JSON schema mapping
            ├── RegimeMatrixChart.jsx # Displays ADI vs CV² SKU classification
            └── ReorderAlertsTable.jsx# Displays Chronos-Bolt forecasts & ROP alerts
```

---

## 3. Canonical Database Schema & Dataset Outline (For Report Table 3.1)

Regardless of how messy the user's uploaded Excel/CSV file is, **Phase 2 (LLM Schema Mapper)** maps the input columns into the following **Canonical Schema** stored in PostgreSQL. Use this exact structure for **Table 3.1: Outline of the Data** in your Final Report:

| Canonical Field Name | Data Type | Mandatory? | Example Automotive ERP Source Headers | Description |
| :--- | :--- | :--- | :--- | :--- |
| `part_id` | `VARCHAR(64)` | Yes | `OEM_Num`, `Part_SKU`, `Item_Code`, `CD_01` | Unique identifier for the automotive spare part SKU |
| `part_name` | `VARCHAR(255)` | No | `Description`, `Part_Title`, `Nomenclature` | Human-readable name (e.g., *Hilux Brake Caliper Assy*) |
| `category` | `VARCHAR(100)` | No | `Part_Group`, `Dept`, `Veh_System` | Subsystem classification (Engine, Suspension, Body, Electrical) |
| `transaction_date` | `DATE` | Yes | `Inv_Date`, `Sale_Dt`, `Trx_Timestamp`, `DT_A` | Date of sale or inventory movement (`YYYY-MM-DD`) |
| `quantity_sold` | `FLOAT` | Yes | `Qty_Out`, `Issued_Qty`, `Sales_Vol`, `VAL_1` | Number of units demanded/sold on `transaction_date` |
| `current_stock` | `FLOAT` | No* | `SOH`, `Qty_Left`, `Bal_Stock`, `On_Hand` | Current available physical inventory on hand (*default 0 if missing) |
| `lead_time_days` | `INTEGER` | No* | `Supplier_LT`, `Lead_Days`, `L_Time` | Replenishment lead time in days (*default 14 days if missing) |
| `unit_cost` | `FLOAT` | No* | `Cost_LKR`, `Unit_Price`, `W_Cost` | Unit acquisition cost for inventory holding calculation |

---

## 4. Detailed Phase-by-Phase Technical & Mathematical Implementation

### Phase 1: Automated Data Ingestion & Validation (`phase1_ingestion.py`)
1. Accepts `.csv` or `.xlsx` via FastAPI `UploadFile`.
2. Checks file integrity, strips empty trailing rows/columns, and verifies that the dataset contains at least 3 columns (minimum required: an ID, a date, and a quantity) and spans at least 52 weeks (or 365 days) of historical horizon.
3. Extracts a **Lightweight Metadata Profile** for each column $c_i$:
   * Raw header string
   * Inferred Pandas dtype (`object`, `int64`, `float64`, `datetime64`)
   * Null ratio ($\text{Nulls} / N$)
   * Ubiquity ratio ($\text{Unique Values} / N$)
   * 3 non-null anonymized sample values

### Phase 2: LLM-Powered Semantic Schema Alignment (`phase2_llm_mapper.py`)
1. **Why Header-Only Fails:** If an ERP exports columns named `COL_A`, `COL_B`, `COL_C`, a header-only prompt fails. By passing the **Metadata Profile + 3 Sample Rows**, the LLM sees that `COL_A` contains `"TOY-04465-0K090"`, `COL_B` contains `"2025-03-14"`, and `COL_C` contains `"2"`, allowing it to infer the mapping with near-100% accuracy.
2. **Pydantic Structured Output Contract:**
   ```python
   from pydantic import BaseModel, Field
   from typing import Optional, Dict

   class SchemaMappingResult(BaseModel):
       part_id: str = Field(..., description="Source column mapped to part_id")
       transaction_date: str = Field(..., description="Source column mapped to transaction_date")
       quantity_sold: str = Field(..., description="Source column mapped to quantity_sold")
       part_name: Optional[str] = None
       category: Optional[str] = None
       current_stock: Optional[str] = None
       lead_time_days: Optional[str] = None
       unit_cost: Optional[str] = None
       confidence_score: float = Field(..., ge=0.0, le=1.0)
       reasoning: str
   ```
3. **Self-Correction Loop:** If the LLM returns a column name that does not exist in `df.columns` or fails Pydantic validation, the middleware catches the `ValidationError`, appends the exact error traceback to the prompt, and performs 1 automatic retry.

### Phase 3: Data Transformation, Zero-Filling & Demand Regime Stratification (`phase3_normalizer.py`)
1. **Temporal Grid Zero-Filling:**
   Spare parts transaction logs only record days when a sale occurs. For each `part_id`, generate a complete continuous date range $[t_{\min}, t_{\max}]$ (either Daily or Weekly frequency $f$), merge the aggregated sales, and fill missing dates with `0.0`.
2. **Syntetos–Boylan Demand Regime Classification ($ADI$ and $CV^2$):**
   For each SKU over the training window of $N$ total periods with $N_{>0}$ non-zero demand periods:
   $$ADI = \frac{N}{N_{>0}}$$
   Let $\mu_{>0}$ and $\sigma_{>0}$ be the sample mean and standard deviation of the **non-zero** demand values. Then:
   $$CV^2 = \left(\frac{\sigma_{>0}}{\mu_{>0}}\right)^2$$
   Classify each SKU using the standard Syntetos & Boylan (2005) thresholds ($ADI_{\text{crit}} = 1.32$, $CV^2_{\text{crit}} = 0.49$):
   * **Smooth:** $ADI < 1.32$ and $CV^2 < 0.49$
   * **Erratic:** $ADI < 1.32$ and $CV^2 \ge 0.49$
   * **Intermittent:** $ADI \ge 1.32$ and $CV^2 < 0.49$
   * **Lumpy:** $ADI \ge 1.32$ and $CV^2 \ge 0.49$

### Phase 4: Zero-Shot AI Inference Engine (`phase4_forecaster.py`)
1. **Primary Foundation Model:** `autogluon/chronos-bolt-small` (or `chronos-bolt-base`) loaded via `chronos-forecasting`.
   * Unlike the original Chronos-T5 which tokenized each time point individually, `Chronos-Bolt` uses continuous patch embeddings and direct multi-step quantile regression, running up to 250x faster on CPU/Apple Silicon M1.
   * For a forecast horizon $H$, `Chronos-Bolt` outputs a tensor of quantiles: $q \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$.
2. **Statistical Intermittent Baselines (via `statsforecast`):**
   * `CrostonClassic`: Separate exponential smoothing of non-zero demand size and inter-demand intervals.
   * `CrostonSBA` (Syntetos–Boylan Approximation): Applies bias correction factor $(1 - \frac{\alpha}{2})$ to Croston's estimate.
   * `TSB` (Teunter–Syntetos–Babai): Updates demand probability every period, preventing positive bias when parts become obsolete.
   * `SeasonalNaive`: Baseline benchmark.

### Phase 5: Business Logic & Quantile-Driven Replenishment (`phase5_replenish.py`)
1. **Solving the Median Zero-Collapse Problem:**
   In intermittent series where $>80\%$ of periods have 0 demand, the median forecast ($\hat{y}_{t+h}^{(0.50)}$) is often `0.0`. Using the median leads to zero replenishment and stockouts.
2. **Quantile-Driven Lead-Time Demand ($D_L$):**
   Given a target cycle service level $\alpha_{\text{service}} = 0.90$ and lead time $L$, compute the cumulative probabilistic demand over lead time using the upper quantile ($q = 0.90$) and mean estimate:
   $$\hat{D}_{L}^{(0.90)} = \sum_{h=1}^{L} \hat{y}_{t+h}^{(0.90)}, \quad \hat{\mu}_L = \sum_{h=1}^{L} \hat{y}_{\text{mean}, t+h}$$
   * **Dynamic Safety Stock ($SS$):**
     $$SS = \max\left(0, \, \hat{D}_{L}^{(0.90)} - \hat{\mu}_L\right)$$
   * **Reorder Point ($ROP$):**
     $$ROP = \lceil \hat{D}_{L}^{(0.90)} \rceil$$
   * **Recommended Order Quantity ($Q_{\text{order}}$):**
     $$Q_{\text{order}} = \max\left(0, \, ROP + \lceil \hat{\mu}_{\text{review}} \rceil - \text{CurrentStock}\right)$$
   * **Alert Status:**
     * `CRITICAL_STOCKOUT_RISK` if $\text{CurrentStock} \le SS$
     * `REORDER_NOW` if $SS < \text{CurrentStock} \le ROP$
     * `OPTIMAL_STOCK` if $ROP < \text{CurrentStock} \le 2 \times ROP$
     * `OVERSTOCK_WARNING` if $\text{CurrentStock} > 2 \times ROP$

---

## 5. Evaluation Metrics & Experimental Protocol (For Chapter 04 Results)

The `run_all_experiments.py` script must compute and export the following exact metrics so you can paste them directly into Chapter 04 of the NIBM Final Report:

### 5.1 Experiment 1: LLM Schema Alignment Metrics
Across 30 mutated automotive CSV schemas (10 Tier-1 Lexical Synonyms, 10 Tier-2 Automotive Jargon, 10 Tier-3 Cryptic ERP Codes):
* **Field-Level Precision, Recall, and F1-Score**
* **Exact Schema Match Rate (%):** Percentage of files where 100% of required canonical columns are mapped correctly.
* **Comparison:** `Fuzzy String Matching (Levenshtein)` vs. `Llama-3-8B (Header-Only)` vs. `Llama-3-8B (Hybrid Header + 3 Sample Rows + Pydantic Retry)`.

### 5.2 Experiment 2: Forecasting Accuracy & Inventory Service Metrics
Evaluated across all 4 Syntetos–Boylan quadrants (*Smooth, Erratic, Intermittent, Lumpy*) using an 80/20 temporal split:
1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{H}\sum_{t=n+1}^{n+H} |y_t - \hat{y}_t|$$
2. **Root Mean Square Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{H}\sum_{t=n+1}^{n+H} (y_t - \hat{y}_t)^2}$$
3. **Mean Absolute Scaled Error (MASE) & Root Mean Squared Scaled Error (RMSSE):**
   *(Scale-free metrics recommended by Hyndman & the M5 Competition for intermittent data where MAPE divides by zero)*:
   $$\text{MASE} = \frac{\frac{1}{H}\sum_{t=n+1}^{n+H} |y_t - \hat{y}_t|}{\frac{1}{n-1}\sum_{t=2}^{n} |y_t - y_{t-1}|}, \quad \text{RMSSE} = \sqrt{\frac{\frac{1}{H}\sum_{t=n+1}^{n+H} (y_t - \hat{y}_t)^2}{\frac{1}{n-1}\sum_{t=2}^{n} (y_t - y_{t-1})^2}}$$
4. **Weighted Quantile Loss (Pinball Loss at $q=0.5, 0.9$):**
   $$\mathcal{L}_q(y, \hat{y}^{(q)}) = \frac{1}{H}\sum_{t=n+1}^{n+H} \max\left(q(y_t - \hat{y}_t^{(q)}), \, (q - 1)(y_t - \hat{y}_t^{(q)})\right)$$
5. **Simulated Achieved Fill Rate (%):**
   $$\text{Fill Rate} = \left(1 - \frac{\sum \text{Unmet Demand Units}}{\sum \text{Total Demand Units}}\right) \times 100$$

### 5.3 Experiment 3: Middleware Latency & Resource Benchmark
Wall-clock latency (in milliseconds) and peak memory consumption (MB) measured across 3 batch sizes (`10 SKUs`, `100 SKUs`, `500 SKUs`) broken down by:
* Phase 1 (Validation) + Phase 2 (LLM Schema Mapping) + Phase 3 (Normalization & DB Write) + Phase 4 & 5 (Chronos-Bolt Inference & Reorder Calculation).

---

## 6. Dependencies (`requirements.txt`)

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
sqlalchemy==2.0.35
psycopg2-binary==2.9.9
pydantic==2.9.2
pandas==2.2.3
numpy==1.26.4
scipy==1.14.1
torch>=2.2.0
chronos-forecasting>=1.3.0
statsforecast==1.7.8
openai>=1.50.0
groq>=0.11.0
rapidfuzz==3.10.0
matplotlib==3.9.2
seaborn==0.13.2
openpyxl==3.1.5
python-multipart==0.0.12
psutil==6.0.0
```

---

## 7. Master Prompts to Give to Your GPT / Coding AI Model

Below are **two copy-paste prompts**.  
* **Prompt A** instructs an AI coding assistant (ChatGPT / Claude / Cursor) to generate the **complete, runnable Python backend + automated Chapter 4 benchmark suite** so you immediately get all CSV tables and PNG charts for your NIBM Final Report.  
* **Prompt B** is the **exact system prompt used inside Phase 2 of your middleware** to instruct Llama 3 / GPT to perform zero-shot hybrid schema mapping.

---

### PROMPT A: Master Prompt for GPT / Coding Assistant to Build the System & Generate All Report Artifacts

> **Copy and paste everything inside the block below into your GPT / Coding AI:**

```text
You are a Senior AI Systems Architect and Quantitative Supply Chain Researcher. I am building my BSc Honours final research project titled "Adaptive AI Middleware Framework for Automated Demand Forecasting" focused on automotive spare parts with intermittent and lumpy demand.

I need you to write a complete, modular, production-grade Python implementation that builds both:
1. The 6-Phase FastAPI Middleware Backend (`Phase 1: Ingestion`, `Phase 2: LLM Hybrid Schema Alignment`, `Phase 3: Zero-Filling & Syntetos-Boylan ADI/CV² Classification`, `Phase 4: Zero-Shot Chronos-Bolt + Statistical Baselines Engine`, `Phase 5: Quantile-Driven Replenishment & Safety Stock`, `Phase 6: REST API & React JSON Delivery`).
2. An self-contained Automated Experimental Benchmark Suite (`run_all_experiments.py`) that executes all experiments and automatically exports publication-quality PNG figures (300 DPI) and CSV/LaTeX tables required for Chapter 3 and Chapter 4 of my thesis report.

STRICT TECHNICAL REQUIREMENTS:
1. Dataset Generator (`generate_synthetic_data.py`):
   - Create a realistic 24-month (104 weeks / 730 days) automotive spare parts dataset with 200 SKUs spanning all four Syntetos-Boylan demand regimes:
     * Smooth (ADI < 1.32, CV^2 < 0.49) - e.g., Oil Filters, Brake Pads
     * Erratic (ADI < 1.32, CV^2 >= 0.49) - e.g., Fleet Spark Plugs, Coolant
     * Intermittent (ADI >= 1.32, CV^2 < 0.49) - e.g., Water Pumps, Alternators
     * Lumpy (ADI >= 1.32, CV^2 >= 0.49) - e.g., Collision Body Panels, Overhaul Kits
   - Also generate 30 mutated CSV files across 3 difficulty tiers:
     * Tier 1 (10 files): Lexical synonyms (`OEM_Part_No`, `Sale_Dt`, `Qty_Sold`)
     * Tier 2 (10 files): Automotive ERP jargon (`Chas_SKU`, `GRN_Iss_Qty`, `SOH_Bal`, `L_Time`)
     * Tier 3 (10 files): Cryptic ERP exports (`COL_01`, `DT_FLD_A`, `NUM_VAL_1`) where headers are opaque and require sample row inspection.

2. Phase 2 LLM Schema Mapper (`phase2_llm_mapper.py`):
   - Implement three mappers for ablation benchmarking:
     (a) `FuzzyHeuristicMapper` using RapidFuzz string similarity on headers only.
     (b) `HeaderOnlyLLMMapper` passing only column names to an OpenAI/Groq-compatible LLM endpoint (supporting `llama-3.1-8b-instant` or `gpt-4o-mini` with a deterministic fallback if no API key is set).
     (c) `HybridProfileLLMMapper` (Proposed Method) passing column names + Pandas dtype + null ratio + 3 sample values per column, constrained by a strict Pydantic schema (`part_id`, `transaction_date`, `quantity_sold`, `part_name`, `category`, `current_stock`, `lead_time_days`, `unit_cost`) with a 1-step self-correction retry loop.

3. Phase 3 & 4 Forecasting Engine (`phase4_forecaster.py`):
   - Implement zero-filling across the temporal grid and compute exact `ADI` and `CV^2` per SKU.
   - Integrate `autogluon/chronos-bolt-small` via `chronos-forecasting` (extracting quantiles q=0.1, 0.5, 0.8, 0.9 and mean forecast).
   - Integrate classical intermittent baselines: `CrostonClassic`, `CrostonSBA` (Syntetos-Boylan Approximation), `TSB` (Teunter-Syntetos-Babai), and `SeasonalNaive`.
   - Never use MAPE (due to division by zero on intermittent series). Compute `MAE`, `RMSE`, `MASE`, `RMSSE`, `Pinball Loss (q=0.9)`, and `Simulated Inventory Fill Rate (%)` using an 80/20 temporal train-test split.

4. Automated Report Artifact Generator (`run_all_experiments.py`):
   - Must automatically generate and save to `./outputs/`:
     * `table_3_1_dataset_outline.csv`: Summary of dataset variables and regime counts.
     * `table_4_1_schema_benchmark.csv`: Precision, Recall, F1, Exact Match %, and Latency across Tier 1, Tier 2, and Tier 3 for Fuzzy vs Header-Only LLM vs Hybrid LLM.
     * `table_4_2_forecasting_metrics_by_regime.csv`: MAE, RMSE, MASE, RMSSE, and Pinball Loss comparing Chronos-Bolt vs Croston vs SBA vs TSB across Smooth, Erratic, Intermittent, and Lumpy regimes.
     * `table_4_3_inventory_service_simulation.csv`: Achieved Fill Rate (%) vs Average Stock Holding comparing Median Point Forecast (q=0.5) vs Quantile-Driven Replenishment (q=0.9) vs SBA.
     * `table_4_4_system_latency_benchmark.csv`: Phase-by-phase execution time (ms) and peak RAM (MB) for 10, 100, and 200 SKUs.
     * 5 high-resolution Matplotlib/Seaborn figures (`fig_4_1` to `fig_4_5`) including the Syntetos-Boylan log-log scatter quadrant chart and actual vs predicted quantile fan chart for a lumpy SKU.

Please provide the complete, self-contained code for each file without placeholders or `TODO` blocks so I can run it immediately.
```

---

### PROMPT B: Internal System Prompt for Phase 2 (`phase2_llm_mapper.py`)

> **Use this exact prompt inside your Python code when calling the Llama 3 / GPT API to map uploaded inventory spreadsheets:**

```text
SYSTEM PROMPT:
You are an expert Automotive Supply Chain Data Engineer and Semantic Schema Matcher.
Your task is to map messy, heterogeneous column headers from an uploaded automotive spare parts inventory file into our Canonical Database Schema.

CANONICAL TARGET SCHEMA:
1. "part_id" (REQUIRED): Unique identifier, SKU, OEM number, or part code (e.g., strings like "TOY-12345", "SKU_001").
2. "transaction_date" (REQUIRED): Date or timestamp of the sale, dispatch, or inventory movement (e.g., ISO dates, DD/MM/YYYY).
3. "quantity_sold" (REQUIRED): Numeric quantity demanded, sold, issued, or dispatched in that transaction (non-negative numbers, often sparse/small integers).
4. "part_name" (OPTIONAL): Text description or title of the vehicle spare part (e.g., "Front Brake Pad Set").
5. "category" (OPTIONAL): Vehicle subsystem or part category (e.g., "Suspension", "Engine", "Filters").
6. "current_stock" (OPTIONAL): Available stock on hand (SOH) or remaining balance quantity.
7. "lead_time_days" (OPTIONAL): Supplier replenishment lead time in days (typically integers between 1 and 90).
8. "unit_cost" (OPTIONAL): Monetary cost or price per unit (typically larger floats/decimals than quantity_sold).

INSTRUCTIONS:
- Analyze BOTH the raw column header name AND the provided data profile (data type, uniqueness ratio, and 3 sample values).
- If a column header is cryptic or abbreviated (e.g., "CD_01", "VAL_A", "GRN_QTY"), rely heavily on the sample values and data types to deduce its semantic role.
- Never invent a source column name; every mapped value must exactly match one of the strings in `source_columns`, or be `null` if no matching column exists in the uploaded file.
- Return ONLY a valid JSON object matching this exact schema:
{
  "part_id": "<exact_source_col_name>",
  "transaction_date": "<exact_source_col_name>",
  "quantity_sold": "<exact_source_col_name>",
  "part_name": "<exact_source_col_name or null>",
  "category": "<exact_source_col_name or null>",
  "current_stock": "<exact_source_col_name or null>",
  "lead_time_days": "<exact_source_col_name or null>",
  "unit_cost": "<exact_source_col_name or null>",
  "confidence_score": 0.95,
  "reasoning": "<1-sentence explanation of how ambiguous columns were resolved>"
}
```