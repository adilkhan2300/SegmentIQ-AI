# SegmentIQ AI — LLM-Powered Customer Segmentation & Strategic Marketing Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-21%2F21%20Passing-brightgreen.svg)]()

> **An end-to-end Machine Learning & GenAI customer intelligence platform that bridges unsupervised behavioral clustering with actionable marketing execution, retention modeling, and financial scenario simulation.**

---

## 📌 Executive Summary

Modern enterprises often segment customers using static demographic attributes (age, gender, geographical location). While demographically intuitive, empirical evidence demonstrates that demographics explain very little variance in customer lifetime value (LTV), margin generation, or purchasing cadence.

**SegmentIQ AI** transforms raw multi-source customer records (demographics, completed transactions, and multi-channel digital interactions) into actionable, profit-maximizing behavioral segments using unsupervised machine learning, statistical hypothesis testing, and Large Language Model (LLM) persona synthesis.

---

## 🚀 Key Capabilities & Modules

1. **Executive Dashboard (`Page 1`)**
   - High-level KPIs: Active Customer Count, Total Net Revenue, Average Order Value (AOV), Repeat Customer Rate, Average CLV.
   - Revenue Pareto distribution and interactive segment breakdown visualizations.

2. **Data Ingestion & Integrity Engine (`Page 2`)**
   - File upload for Customers, Transactions, Interactions, and Experiments.
   - Built-in 1,200+ customer synthetic demo generator with 6 behavioral archetypes (`seed=42`).
   - Schema validation, data cleaning, deduplication, and anomaly reporting.

3. **Multi-Dimensional Feature Engineering (`Page 3`)**
   - RFM metrics (Recency, Frequency, Monetary Value) with **strict data leakage prevention** (only completed orders contribute to revenue; refunds and cancellations tracked separately).
   - Digital engagement indicators (clicks, pageviews, open rates, search activity).
   - Shannon entropy for category diversity and inter-purchase interval velocity.

4. **Unsupervised ML Clustering & PCA (`Page 4`)**
   - **Algorithms**: K-Means, Agglomerative Hierarchical Clustering, and DBSCAN.
   - **Evaluation**: Elbow Method (Inertia), Silhouette Analysis, Davies-Bouldin Index, and Calinski-Harabasz Score.
   - 2D & 3D Principal Component Analysis (PCA) projections for high-dimensional cluster visualization.

5. **Segment Profiling & Radar Analysis (`Page 5`)**
   - Min-Max normalized radar spider charts across 6 core behavioral dimensions.
   - Product category affinity heatmaps and demographic distribution breakdowns.

6. **LLM-Powered Customer Personas (`Page 6`)**
   - Rich persona generation powered by OpenAI GPT-4o (with an **automatic, robust rule-based fallback** when no API key is supplied).
   - Clear labeling of behavioral evidence vs. hypothesized needs (`Hypothesis:` prefix).

7. **Strategic Marketing Recommendations (`Page 7`)**
   - Differentiated campaigns tailored per segment (Lifecycle Stage, Channel Strategy, Offer Architecture, Timing).
   - **8-Point Marketing Justification Framework** linking machine learning data points directly to business execution.

8. **Customer Value & Retention Intelligence (`Page 8`)**
   - Transparent rule-based retention risk scores (0–100) and risk tiers (High / Medium / Low).
   - Supervised ML models (Random Forest & Logistic Regression) for churn risk and repeat purchase propensity with ROC-AUC, precision, recall, and feature importances.

9. **Campaign Experimentation & A/B Testing (`Page 9`)**
   - Two-proportion Z-test and Welch's t-test for conversion and revenue differences.
   - 95% Confidence Intervals, statistical significance indicators, and sample size power calculator.
   - Hypothetical campaign ROI planning simulator.

10. **Demographic vs. Behavioral Segmentation Comparison (`Page 10`)**
    - Side-by-side empirical contrast between Demographic Cohorts (Age, Location) and Behavioral Segments (RFM, ML Clusters).
    - **One-Way ANOVA F-tests** demonstrating that behavioral clustering captures significantly more variance in revenue and order frequency than demographic proxies.

11. **Interactive Financial Scenario Simulator (`Page 11`)**
    - Dynamic what-if modeling: retention lift, repeat order acceleration, discount optimization, and browser conversion.
    - Financial waterfall comparing baseline actuals against projected gross revenue, contribution margin, and marketing ROI.

12. **Automated AI Insights & Anomaly Engine (`Page 12`)**
    - Surfaces high-impact business discoveries (Pareto revenue concentration, everyday micro-shoppers, dormancy risks, discount cannibalization).
    - Includes supporting metrics, business interpretations, recommended actions, and limitations.

13. **Customer Drill-Down & Segment Lookup (`Page 13`)**
    - Searchable individual customer records with cluster assignments, RFM metrics, engagement velocity, and personalized next-best actions.

14. **Management Reporting & Multi-Format Export (`Page 14`)**
    - Multi-tab formatted Excel workbook export (`Customer_Assignments`, `Segment_Profiles`, `Personas`, `Strategies`, `Metrics`).
    - C-suite executive PDF report generated programmatically via ReportLab with typography, KPI summaries, and strategy briefs.
    - SQLite database persistence for segmentation run auditability.

---

## 🛠️ Tech Stack & Architecture

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/) (Light Theme SaaS styling via custom CSS & `.streamlit/config.toml`)
- **Data Manipulation & Mathematics**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Statsmodels](https://www.statsmodels.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (KMeans, AgglomerativeClustering, DBSCAN, PCA, RandomForestClassifier, LogisticRegression, StandardScaler)
- **Data Visualization**: [Plotly](https://plotly.com/python/) (Interactive Scatter, 3D PCA, Radar Charts, Distribution Histograms)
- **Generative AI / LLM**: [OpenAI API](https://platform.openai.com/) (GPT-4o) with intelligent rule-based heuristic fallbacks
- **Export & Storage**: [ReportLab](https://www.reportlab.com/) (Executive PDF), [OpenPyXL](https://openpyxl.readthedocs.io/) (Excel), [SQLite3](https://docs.python.org/3/library/sqlite3.html) (Persistence)
- **Quality Assurance**: [Pytest](https://docs.pytest.org/) (21 unit & integration tests)

---

## 📂 Project Directory Structure

```text
├── .streamlit/
│   └── config.toml               # Light theme configuration and server settings
├── data/
│   ├── customers.csv             # 1,200 synthetic customer records
│   ├── transactions.csv          # 8,700+ transactions
│   ├── interactions.csv          # 23,000+ digital interaction events
│   ├── campaign_experiments.csv  # 1,500 A/B testing campaign rows
│   └── segmentiq.db              # SQLite persistence database
├── modules/
│   ├── __init__.py
│   ├── data_generator.py         # Synthetic multi-archetype generator (seed=42)
│   ├── data_loader.py            # CSV caching & data ingestion
│   ├── data_validation.py        # Schema checking, cleaning, and deduplication
│   ├── feature_engineering.py    # RFM, engagement, leakage prevention
│   ├── clustering.py             # K-Means, Agglomerative, DBSCAN, PCA, metrics
│   ├── segment_profiling.py      # Radar charts, product category affinity
│   ├── llm_personas.py           # OpenAI personas + heuristic fallbacks
│   ├── marketing_recommendations.py # 8-point justification framework
│   ├── retention_analysis.py     # Rule-based scoring + RF/Logistic classifiers
│   ├── experiment_analysis.py    # A/B testing Z-test, CI, sample size calculator
│   ├── demographic_comparison.py # ANOVA F-statistic comparison engine
│   ├── scenario_simulator.py     # Financial what-if projections & ROI
│   ├── insights_engine.py        # Automated pattern and Pareto discovery
│   ├── database.py               # SQLite run tracking and persistence
│   └── reporting.py              # Excel workbook & ReportLab PDF generator
├── tests/
│   ├── __init__.py
│   ├── test_validation.py        # Schema and cleaning tests
│   ├── test_feature_engineering.py # RFM calculation and leakage tests
│   ├── test_clustering.py        # KMeans, silhouette, and candidate k tests
│   ├── test_personas.py          # Persona fallback tests
│   ├── test_retention.py         # Churn risk scoring & ML model tests
│   ├── test_experiments.py       # A/B testing & sample size tests
│   ├── test_simulator_and_insights.py # Financial simulator & ANOVA tests
│   └── test_export_and_db.py     # Database persistence & report export tests
├── utils/
│   ├── __init__.py
│   └── ui_helpers.py             # Light-mode CSS, KPI cards, Plotly layout
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git ignore file
├── app.py                        # Main 14-page Streamlit application
├── requirements.txt              # Production Python dependencies
└── README.md                     # Comprehensive documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/adilkhan2300/SegmentIQ-AI.git
cd SegmentIQ-AI
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure OpenAI API Key
Copy `.env.example` to `.env` and insert your OpenAI key if you wish to use live LLM persona generation:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
> *Note: If no API key is provided, SegmentIQ AI automatically utilizes its built-in rule-based persona synthesis engine.*

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` (or the port indicated in your terminal).

---

## 🧪 Automated Testing

SegmentIQ AI includes a comprehensive test suite with 21 unit and integration tests across data validation, feature engineering, clustering algorithms, churn prediction, experimentation, and database persistence:

```bash
python -m pytest tests/ -v
```

**Test Output:**
```text
============================= 21 passed in 6.81s ==============================
```

---

## 📊 Analytical Methodology

### 1. Data Leakage Prevention
- **Net Completed Revenue**: Only transactions with `status == "Completed"` contribute to Recency, Frequency, and Monetary (RFM) aggregations.
- **Refund Separation**: Refunded transactions are factored separately into the `return_refund_rate` dimension to avoid contaminating core spending signals.

### 2. Feature Standardization
- High-skew monetary metrics are optionally log-transformed ($\ln(1 + x)$).
- All clustering features are scaled using `StandardScaler` to ensure zero mean and unit variance ($\mu = 0, \sigma = 1$).

### 3. Cluster Quality Metrics
- **Silhouette Coefficient**: Measures intra-cluster cohesion versus nearest-cluster separation ($s \in [-1, 1]$).
- **Davies-Bouldin Index**: Ratio of within-cluster scatter to between-cluster separation (lower is better).
- **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion (higher is better).

### 4. Empirical Demographic Contrast (ANOVA)
To empirically prove that behavioral clustering outperforms traditional demographic segmentation:
$$\text{ANOVA } F = \frac{\text{Between-Group Variability}}{\text{Within-Group Variability}}$$
The system computes $F$-statistics on Monetary Spend and Order Frequency across Age Cohorts, Location Regions, and ML Behavioral Clusters. Results demonstrate statistically superior variance explanation from ML behavioral clusters.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
