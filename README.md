# RoadSight Rwanda
### Road Accident Risk Analysis · Group D Big Data Project 2026

> Exploring when, where, and why road accidents happen in Rwanda — and what the data says about preventing them.

---

## What This Project Does

RoadSight Rwanda analyses 3,007 road accident records across all five Rwandan provinces (2022–2025), uncovers temporal and geographic risk patterns, trains a severity classification model, and surfaces the findings through an interactive Plotly dashboard.

The core question we set out to answer: **can available data predict how serious a road accident will be — and what conditions are most dangerous?**

---

## At a Glance

| | |
|---|---|
| **Records** | 3,007 accidents after cleaning |
| **Coverage** | 5 provinces · 2022–2025 |
| **Severity classes** | Damage only · Minor injury · Serious/grievous · Fatal |
| **Fatal rate** | 6.9 % (206 accidents) |
| **Best model** | Random Forest · macro F1 = 0.32 |
| **Dashboard** | Self-contained HTML · ~172 KB · no server needed |
| **Dataset** | Synthetic/simulated — clearly labeled, not real police data |

---

## Project Structure

```
roadsight_rwanda/
│
├── data/
│   ├── raw/                        # Original synthetic dataset
│   │   └── rwanda_road_crashes_SYNTHETIC.csv
│   └── processed/                  # Cleaned, feature-engineered dataset
│       └── accidents_clean.csv
│
├── notebooks/
│   └── analysis.ipynb              # Main notebook — cleaning → EDA → model → dashboard
│
├── scripts/
│   ├── clean_data.py               # Data cleaning pipeline
│   ├── generate_eda.py             # EDA chart generation (11 figures)
│   ├── train_model.py              # Model training + evaluation
│   └── build_dashboard.py          # Plotly dashboard builder
│
├── outputs/
│   ├── figures/                    # 14 PNG charts
│   └── models/                     # model_results.json (pkl files excluded from git)
│
├── dashboard/
│   └── accident_risk_dashboard.html   # ← Open this in any browser
│
├── report/
│   ├── project_report.md           # Full written report
│   └── cleaning_log.md             # Step-by-step data cleaning record
│
├── requirements.txt
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A terminal

### 1 · Clone the repository

```bash
git clone git@github.com:mug1sha/RoadSight-Rwanda.git
cd RoadSight-Rwanda
```

### 2 · Create and activate the virtual environment

```bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Register the Jupyter kernel

This makes the venv available as a kernel inside JupyterLab.

```bash
python -m ipykernel install --user --name roadsight-rwanda --display-name "Python (RoadSight Rwanda)"
```

### 5 · Run the pipeline

Each script is standalone. Run them in this order if starting fresh:

```bash
# Clean the raw dataset → produces data/processed/accidents_clean.csv
python scripts/clean_data.py

# Generate all EDA charts → outputs/figures/
python scripts/generate_eda.py

# Train models + save evaluation artifacts → outputs/models/
python scripts/train_model.py

# Build the interactive dashboard → dashboard/accident_risk_dashboard.html
python scripts/build_dashboard.py
```

### 6 · Open the dashboard

No server needed. Just open the file in any browser:

```bash
# macOS
open dashboard/accident_risk_dashboard.html

# Linux
xdg-open dashboard/accident_risk_dashboard.html

# Windows
start dashboard/accident_risk_dashboard.html
```

### 7 · Launch the notebook

```bash
jupyter lab
```

Then open `notebooks/analysis.ipynb`. Select **"Python (RoadSight Rwanda)"** as the kernel — it will run the full analysis end to end.

---

## Key Findings

**Where** — Kigali City accounts for 63.1% of all accidents, driven by higher traffic density. Accidents are spread across all five provinces, but the city's sheer volume means most high-severity hotspots sit within it.

**When** — The riskiest hour is 17:00, as the evening rush concentrates traffic and fatigue simultaneously. The 07:00–09:00 morning window is a secondary peak. Dark-night accidents carry a disproportionately high fatal share.

**Who and what** — Motorcycles are involved in more accidents than any other vehicle type. Trucks have the highest fatal rate per incident. Drivers aged 18–34 appear most frequently.

**Risk factors** — Speeding is the strongest predictor of severity: suspected speeding raises mean severity score by ~31% above baseline. Alcohol involvement pushes a further ~20% above baseline. These two factors together represent the most actionable targets for enforcement.

**Model** — Random Forest achieved a macro F1 of 0.32 across four severity classes, outperforming XGBoost (0.26). The result is honest: synthetic data with heavy class imbalance (Fatal = 6.9%) limits prediction, but the model reliably identifies high-risk condition combinations and confirms which features matter most.

---

## Dashboard Sections

| Section | What you see |
|---|---|
| **KPI banner** | Total accidents, fatal count, fatal rate, alcohol %, speeding % |
| **Severity overview** | Donut chart + stacked bar by time of day |
| **Temporal heatmap** | Day-of-week × hour grid showing when accidents cluster |
| **Monthly trend** | Fatal accidents per month, 2022–2025 |
| **Geographic map** | Scatter map of accident locations (Plotly, carto-positron basemap) |
| **By province** | Grouped bar — severity breakdown per province |
| **Risk factors** | Mean severity score per risk factor vs baseline |
| **Vehicle fatal rate** | Fatal % by vehicle type |
| **Model performance** | F1-score per severity class, RF vs XGBoost |

---

## Dataset Notice

The dataset (`rwanda_road_crashes_SYNTHETIC.csv`) is **synthetic and simulated**. Every row carries the label *"SYNTHETIC - simulated, not real police data"* in the original `data_source` column. It was generated for educational purposes.

Findings are illustrative. They should not be used to inform real road-safety policy without validation against official Rwanda National Police or RURA records.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Core language |
| pandas · numpy | Data manipulation |
| matplotlib · seaborn | Static EDA charts |
| scikit-learn | Random Forest, preprocessing, evaluation |
| XGBoost | Gradient boosting classifier |
| imbalanced-learn | SMOTE oversampling |
| Plotly | Interactive dashboard |
| JupyterLab | Notebook environment |

---

## Report and Cleaning Log

The full written report is at [`report/project_report.md`](report/project_report.md). It covers problem definition, data cleaning, EDA findings, model results, limitations, ethical considerations, and practical recommendations.

Every cleaning decision is documented step by step in [`report/cleaning_log.md`](report/cleaning_log.md).

---

## Academic Context

**Course:** Big Data Analytics · 2026  
**Group:** D — Road Accident Risk Analysis in Rwanda  
**Thematic area:** Public Safety

> Models produced here are educational decision-support exercises, not substitutes for real road-safety judgment or official data.

---

*RoadSight Rwanda · Group D · Big Data Project 2026*
