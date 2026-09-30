# 🛡️ ClearLoss | Enterprise Revenue Assurance & Forensic Audit Intelligence Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Live Production App:** [https://clearloss-audit-engine.streamlit.app](https://clearloss-audit-engine.streamlit.app)  
> **API Source Data:** [USAspending.gov Live REST API](https://api.usaspending.gov)

---

## Executive Overview

Mid-market enterprises, public sector agencies, and healthcare organizations suffer significant revenue leakage annually due to non-compliant billing, duplicate invoicing, vendor price dispersion, and intentional threshold evasions (split-invoicing). Traditional rule-based transactional audits miss non-linear anomalies and scale poorly across high-volume ledgers.

**ClearLoss** is an enterprise-grade forensic analytics engine that ingests real-time federal transaction streams via the **USAspending.gov REST API**. It fuses parametric statistical tests (Benford’s Law logarithmic distribution analysis) with unsupervised machine learning (Isolation Forests) and sliding-window heuristic detection to flag high-risk transactions, estimate total leakage exposure ($), and automate C-suite audit reporting.

---

## Technical Architecture & Pipeline Flow

 ┌──────────────────────────────────────────────┐
              │   USAspending.gov Live REST API Endpoint     │
              └──────────────────────┬───────────────────────┘
                                     │ JSON Stream / Async ETL
                                     ▼
              ┌──────────────────────────────────────────────┐
              │       Ingestion & Data Cleansing Module      │
              │   (Pandas / Requests / Type Standardization) │
              └──────────────────────┬───────────────────────┘
                                     │ Cleaned Transaction Dataframe
                                     ▼
     ┌───────────────────────────────┴───────────────────────────────┐
     │              Parallel Forensic Analytics Core                 │
     ├───────────────────────────────┬───────────────────────────────┤
     │                               │                               │
     ▼                               ▼                               ▼
┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐│  Benford's Law   │           │ Isolation Forest │           │  Split-Invoice   ││  Logarithmic     │           │ Unsupervised ML  │           │ Sliding-Window   ││  Digit Analysis  │           │ Anomaly Engine   │           │ Time Heuristics  │└────────┬─────────┘           └────────┬─────────┘           └────────┬─────────┘│                               │                               │└───────────────────────────────┼───────────────────────────────┘│ Enriched Metrics & Risk Scores▼┌──────────────────────────────────────────────┐│     Streamlit Executive Intelligence UI      │├──────────────────────────────────────────────┤│ • C-Suite KPI Metric Cards                   ││ • Interactive Plotly Anomaly Matrices        ││ • Granular Line-Item Risk Table              ││ • Exportable CSV/PDF Audit Summaries         │└──────────────────────────────────────────────┘
---

## Core Analytical Capabilities

### 1. Benford’s Law Logarithmic Digit Analysis
Applies a Chi-Square ($\chi^2$) goodness-of-fit test against first-digit distributions across transaction amounts. Significant deviations from the standard logarithmic curve trigger systemic warning flags indicating artificial data manipulation or uniform split pricing.

### 2. Unsupervised Machine Learning (Isolation Forests)
Fits an Isolation Forest model on log-transformed transaction amounts and vendor-specific $Z$-scores ($>3\sigma$ variance) to detect multi-dimensional outliers without relying on pre-labeled historical fraud datasets.

### 3. Sliding-Window Split-Invoice Detection
Utilizes time-series grouping with dynamic sliding windows (e.g., 48-hour rolling limits) to identify vendors issuing multiple sub-threshold invoices (e.g., $\$9,800 + \$9,950$) designed to evade managerial authorization limits (e.g., $\$10,000$).

---

## Technical Stack

* **Language:** Python 3.10+
* **Data Pipelines & ETL:** Pandas, NumPy, Requests
* **Machine Learning & Applied Statistics:** Scikit-Learn (IsolationForest), SciPy (stats.chisquare)
* **Data Visualization:** Plotly Express, Plotly Graph Objects
* **Frontend Web Application:** Streamlit Engine
* **Deployment & CI/CD:** Streamlit Cloud / Render GitHub Integration

---

## Project Directory Structure

```text
clearloss-audit-engine/
│
├── app.py                      # Main Streamlit Executive Dashboard UI
├── ingestion.py                # USAspending API extraction & ETL pipeline
├── forensic_engine.py          # Benford's Law & Isolation Forest models
├── split_invoice_detector.py   # Sliding-window threshold evasion engine
├── requirements.txt            # Operational dependencies
├── .gitignore                  # Production exclusion rules
└── README.md                   # Enterprise technical documentation
Local Setup & InstallationBash# 1. Clone Repository
git clone [https://github.com/yourusername/clearloss-audit-engine.git](https://github.com/bamideleadedeji/clearloss-audit-engine.git)
cd clearloss-audit-engine

# 2. Create and Activate Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Operational Dependencies
pip install -r requirements.txt

# 4. Launch Application Locally
streamlit run app.py
