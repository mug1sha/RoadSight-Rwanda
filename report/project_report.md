# RoadSight Rwanda
## Road Accident Risk Analysis — Group D Big Data Project Report 2026

**Thematic area:** Public Safety  
**Course:** Big Data Analytics  
**Submission week:** Week 14

---

## Table of Contents
1. [Problem Statement & Objectives](#1-problem-statement--objectives)
2. [Dataset Description](#2-dataset-description)
3. [Data Cleaning Summary](#3-data-cleaning-summary)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Machine Learning Model](#5-machine-learning-model)
6. [Dashboard](#6-dashboard)
7. [Findings & Interpretation](#7-findings--interpretation)
8. [Limitations](#8-limitations)
9. [Ethical Considerations](#9-ethical-considerations)
10. [Practical Recommendations](#10-practical-recommendations)
11. [References & Data Source](#11-references--data-source)

---

## 1. Problem Statement & Objectives

### Problem Statement
Road traffic accidents are a leading cause of preventable death and injury globally. In Rwanda, rapid urbanisation and rising motorisation have increased road safety risks, particularly in Kigali City. Studying historical incident patterns can help identify when and where serious accidents are more likely to occur, enabling targeted preventive interventions.

### Research Question
> What patterns are associated with road accidents in Rwanda, and can available data predict accident severity or risk?

### Specific Objectives
1. Investigate accident patterns by location, time of day, day of week, and road/environmental conditions.
2. Clean and prepare a road-safety dataset, documenting all changes in a cleaning log.
3. Develop and evaluate a severity classification model using Random Forest and XGBoost.
4. Build an interactive Plotly dashboard to communicate risk patterns and prevention insights.

---

## 2. Dataset Description

| Attribute | Detail |
|---|---|
| **File** | `rwanda_road_crashes_SYNTHETIC.csv` |
| **Nature** | Synthetic / simulated — clearly labeled in `data_source` column |
| **Records** | 3,030 rows (3,007 after cleaning) |
| **Columns** | 27 raw → 35 after feature engineering |
| **Coverage** | Rwanda, 5 provinces, 2022–2025 |
| **Target variable** | `severity` (4 classes) |
| **Source note** | Generated for educational purposes. Not real Rwanda National Police or RURA data. |

### Columns of note

| Column | Description |
|---|---|
| `crash_datetime` | Timestamp of the accident |
| `province`, `district`, `sector` | Administrative location |
| `latitude`, `longitude` | GPS coordinates |
| `road_class`, `road_surface`, `road_condition` | Road context |
| `weather`, `lighting` | Environmental conditions |
| `vehicle_primary` | Main vehicle type involved |
| `alcohol_suspected`, `speeding_suspected` | Behavioural risk flags |
| `driver_age`, `driver_gender` | Driver demographics |
| `severity` | **Target:** Damage only / Minor injury / Serious/grievous / Fatal |

---

## 3. Data Cleaning Summary

Full cleaning log: `report/cleaning_log.md`

| Step | Action | Records/Values Affected |
|---|---|---|
| Remove metadata column | Dropped `data_source` (constant) | — |
| Parse datetime | Converted `crash_datetime` to datetime type | 3,030 rows |
| Feature extraction | Added `crash_year`, `crash_month`, `crash_hour`, `crash_dow`, `is_weekend`, `time_of_day` | New columns |
| Case normalisation | Standardised `vehicle_primary` and `other_party` to Title Case | ~77 rows with case variants |
| Implausible ages | Replaced driver age < 15 or > 100 with NaN | 5 values |
| Missing target | Dropped rows where `severity` is null | 23 rows removed |
| Missing categoricals | Filled with `'Unknown'`: `road_condition` (241), `alcohol_suspected` (214), `junction` (91), `weather` (151), `lighting` (119), `driver_gender` (58), `helmet_worn` (118) | 992 values imputed |
| Missing driver age | Imputed with median age (35) | 276 values |
| Age group feature | Added `age_group` bands: ≤24, 25–34, 35–44, 45–59, 60+ | New column |
| Severity encoding | Added `severity_code` (0–3 ordinal) | New column |

**Final dataset: 3,007 rows × 35 columns**

---

## 4. Exploratory Data Analysis

### 4.1 Severity Distribution & Class Balance

| Severity | Count | Share |
|---|---|---|
| Damage only | 1,350 | 44.9% |
| Minor injury | 1,051 | 35.0% |
| Serious/grievous | 400 | 13.3% |
| Fatal | 206 | 6.8% |

The dataset is imbalanced: fatal accidents represent only 6.8% of records, while damage-only accidents account for 44.9%. This imbalance was addressed using SMOTE oversampling on the training set.

### 4.2 Temporal Patterns

- **Peak hours:** Accident frequency rises sharply between 07:00–09:00 (morning rush) and again at 17:00–20:00 (evening rush). The 18:00–19:00 window has the highest single-hour count.
- **Day of week:** Monday through Friday each record roughly similar volumes. Friday slightly peaks. The combined weekend total is lower, but weekend crash-to-fatal ratios are elevated.
- **Monthly trends:** No single month dominates; there is a slight increase in accidents from March to May (long rainy season in Rwanda) and again in November.

### 4.3 Geographic Patterns

| Province | Accidents | Share |
|---|---|---|
| Kigali City | 1,910 | 63.5% |
| Southern | 365 | 12.1% |
| Eastern | 283 | 9.4% |
| Western | 249 | 8.3% |
| Northern | 223 | 7.4% |

Kigali City dominates, consistent with higher traffic density. Fatal accident proportions are broadly similar across provinces, suggesting accidents everywhere carry comparable severity risk.

### 4.4 Vehicle Types

Motorcycles (32.2%) and cars (31.3%) are the most involved vehicle types, reflecting their prevalence in Rwanda's traffic. Trucks have the highest fatal rate per incident — when a truck is the primary vehicle, the likelihood of a fatal outcome is higher than average.

### 4.5 Key Risk Factor Findings

| Factor | Observation |
|---|---|
| **Alcohol suspected** | Mean severity score 0.89 vs 0.74 overall — alcohol-involved crashes trend more severe |
| **Speeding suspected** | Mean severity score 0.96 vs 0.65 for non-speeding — strongest behavioural predictor |
| **Dark night** | Higher proportion of fatal and serious outcomes than daylight |
| **Wet road** | Modest increase in severe outcomes vs dry conditions |
| **No helmet (motorcycle)** | Associated with higher serious/fatal share among motorcycle crashes |

### 4.6 Correlation Analysis

`severity_code` has weak linear correlations with individual numeric features (highest: `persons_killed` r≈0.46 — expected as a consequence, not a predictor). This confirms that severity is driven by combinations of conditions, not any single numeric variable, and motivates an ensemble ML approach.

---

## 5. Machine Learning Model

### 5.1 Approach

**Task:** Multi-class classification — predict `severity` (4 classes).  
**Features used:** 21 features covering time, location, road/weather conditions, vehicle type, and driver characteristics.  
**Class imbalance handling:** SMOTE applied to the training set only (test set left as-is for honest evaluation).  
**Train/test split:** 80/20 stratified.

### 5.2 Models Trained

| Model | Configuration |
|---|---|
| **Random Forest** | 200 trees, max depth 15, `n_jobs=-1` |
| **XGBoost** | 300 estimators, depth 6, learning rate 0.1, `mlogloss` metric |

### 5.3 Results

#### Random Forest (selected model — better performance)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Damage only | 0.51 | 0.54 | 0.52 | 270 |
| Minor injury | 0.38 | 0.37 | 0.38 | 211 |
| Serious/grievous | 0.25 | 0.23 | 0.24 | 80 |
| Fatal | 0.15 | 0.15 | 0.15 | 41 |
| **Macro avg** | **0.32** | **0.32** | **0.32** | 602 |
| Weighted avg | 0.41 | 0.41 | 0.41 | 602 |

**Accuracy: 41.0%** | **Macro F1: 0.32**

#### XGBoost

| Metric | Value |
|---|---|
| Accuracy | 37.0% |
| Macro F1 | 0.26 |

Random Forest outperforms XGBoost on this dataset.

### 5.4 Top Predictive Features (Random Forest importance)

1. `driver_age`
2. `crash_hour`
3. `speed_limit_kmh`
4. `crash_month`
5. `vehicles_involved`

Time and driver characteristics dominate feature importance, with behavioural flags (`alcohol_suspected`, `speeding_suspected`) contributing meaningful signal.

### 5.5 Model Interpretation

The macro F1 of 0.32 is modest. This is expected because:
- **Synthetic data** may lack real-world conditional structure that makes severe accidents genuinely predictable from features.
- **Four-class imbalanced classification** is inherently harder than binary; the Fatal class (41 test samples) has insufficient support for reliable learning.
- The model performs best on the dominant class (Damage only, F1=0.52) and worst on Fatal (F1=0.15), reflecting the class imbalance challenge.

Despite limited predictive accuracy, the model confirms that time-of-day, driver age, and speed context are the strongest available signals — findings that are actionable for prevention policy.

---

## 6. Dashboard

**File:** `dashboard/accident_risk_dashboard.html`  
**Technology:** Plotly (Python), self-contained HTML (~170 KB)

### Dashboard Sections

| Section | Content |
|---|---|
| KPI banner | Total accidents, fatal count, fatal rate, alcohol %, speeding % |
| Severity Overview | Donut chart + stacked bar by time of day |
| Temporal Patterns | Day × Hour heatmap, monthly fatal trend line |
| Geographic | Interactive scatter map (Plotly scatter_map, carto-positron basemap) + severity by province |
| Risk Factors | Mean severity score by factor + fatal rate by vehicle type |
| Model Performance | F1-score per class, both models compared |

The dashboard includes a data notice warning that the dataset is synthetic and findings are illustrative only.

---

## 7. Findings & Interpretation

1. **Most accidents cause property damage only** (44.9%), but 1 in 15 is fatal — a rate that warrants serious preventive attention.
2. **Evening rush hours are the highest-risk window.** Fatigue, rush, and poor lighting converge at 17:00–20:00.
3. **Kigali City carries the majority of accident burden** but provinces are not risk-free; rural road conditions may amplify severity when accidents do occur.
4. **Speeding is the strongest behavioural predictor of severity** — suspected speeding raises mean severity score by ~31% above baseline.
5. **Motorcycles dominate by count; trucks dominate by severity.** Combined, they represent the two most important vehicle types for intervention.
6. **A 4-class severity classifier achieves macro F1 of 0.32** — useful as a screening tool and to identify high-risk conditions, but not sufficient for high-stakes automated decisions.

---

## 8. Limitations

1. **Synthetic dataset:** The data was algorithmically generated and is not drawn from real Rwanda National Police records. Patterns embedded in the data reflect simulation assumptions, not necessarily actual Rwandan accident dynamics.
2. **Missing data:** Seven columns had 4–9% missing values, imputed as 'Unknown'. This creates an artificial category that may confound some analyses.
3. **Class imbalance:** Even after SMOTE, the Fatal class remains the hardest to classify (F1=0.15). Real-world deployment would need far more fatal accident examples.
4. **No causal inference:** Correlations identified in EDA do not establish causation. Alcohol and speeding flags are reported by officers and may be inconsistently recorded.
5. **Static model:** The model is trained on 2022–2025 synthetic data. Road conditions, traffic patterns, and vehicle mix change over time; a deployed model would require periodic retraining.

---

## 9. Ethical Considerations

- **No real personal data was used.** The dataset is synthetic with no names, national IDs, real GPS traces, or individually identifiable records.
- **No privacy risk** exists in sharing or publishing this dataset or analysis.
- **Model limitations must be disclosed** before any real-world application. This model is an educational exercise; it should not be used by police or transport authorities without validation on real, official records.
- **Bias awareness:** Synthetic data may under- or over-represent certain provinces, vehicle types, or demographic groups. Any finding should be validated against census or official traffic counts before informing policy.
- **Fairness:** The model treats all provinces equally, but interventions derived from it should account for socioeconomic differences in road infrastructure and vehicle access.

---

## 10. Practical Recommendations

| Priority | Recommendation | Evidence Base |
|---|---|---|
| High | Increase traffic enforcement during 17:00–20:00, especially in Kigali City | Highest accident count window |
| High | Mandatory breathalyser checkpoints on main roads | Alcohol raises mean severity by ~20% |
| High | Speed camera deployment on roads with high fatal counts | Speeding is strongest severity predictor |
| Medium | Targeted street-lighting improvement on high-incident roads | Dark night associated with more fatal outcomes |
| Medium | Helmet enforcement at key motorcycle junctions | No-helmet rides significantly elevate serious injury rate |
| Medium | Road safety education for 18–34 age bracket | Highest driver count in accidents |
| Low | Collect and publish structured official crash records | Current reliance on synthetic data prevents validated policy |
| Low | Wet-weather signage and speed reductions | Wet road conditions correlate with moderate severity increase |

---

## 11. References & Data Source

| Item | Detail |
|---|---|
| **Dataset** | `rwanda_road_crashes_SYNTHETIC.csv` — synthetic data generated for educational use |
| **Dataset labeling** | Every row labeled: *"SYNTHETIC - simulated, not real police data"* in `data_source` column |
| **Approval status** | Dataset clearly labeled as synthetic alternative; no real data was required |
| **scikit-learn** | Pedregosa et al., 2011. Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830. |
| **XGBoost** | Chen & Guestrin, 2016. XGBoost: A Scalable Tree Boosting System. KDD '16. |
| **imbalanced-learn** | Lemaître et al., 2017. Imbalanced-learn: A Python Toolbox. JMLR 18(17), 1–5. |
| **Plotly** | Plotly Technologies Inc., 2015. Collaborative data science. Montreal, QC. |

---

*RoadSight Rwanda | Report prepared by Group D | Big Data Project 2026*
