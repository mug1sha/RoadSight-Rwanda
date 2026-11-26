"""
Data Cleaning Script — RoadSight Rwanda
Group D | Big Data Project
Cleans raw synthetic dataset and writes:
  - data/processed/accidents_clean.csv
  - report/cleaning_log.md
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw" / "rwanda_road_crashes_SYNTHETIC.csv"
OUT_CSV = Path(__file__).parent.parent / "data" / "processed" / "accidents_clean.csv"
LOG_PATH = Path(__file__).parent.parent / "report" / "cleaning_log.md"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

log_entries = []

def log(msg):
    print(msg)
    log_entries.append(msg)


# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW)
log(f"## Cleaning Log — Rwanda Road Accident Dataset")
log(f"\n### Source\nFile: `{RAW.name}`  \nData is **synthetic/simulated** (not real police records). "
    f"Clearly labeled in `data_source` column.\n")
log(f"### 1. Load\nInitial shape: **{df.shape[0]} rows × {df.shape[1]} columns**")

original_rows = len(df)

# ── 2. Drop the data_source column (constant metadata) ───────────────────────
df.drop(columns=["data_source"], inplace=True)
log("\n### 2. Removed Columns\n- Dropped `data_source`: constant metadata column, not useful for analysis.")

# ── 3. Parse datetime and extract time features ───────────────────────────────
df["crash_datetime"] = pd.to_datetime(df["crash_datetime"])
df["crash_year"]  = df["crash_datetime"].dt.year
df["crash_month"] = df["crash_datetime"].dt.month
df["crash_hour"]  = df["crash_datetime"].dt.hour
df["crash_dow"]   = df["crash_datetime"].dt.dayofweek   # 0=Mon … 6=Sun
df["crash_dow_name"] = df["crash_datetime"].dt.day_name()
df["is_weekend"]  = df["crash_dow"].isin([5, 6]).astype(int)
df["time_of_day"] = pd.cut(
    df["crash_hour"],
    bins=[-1, 5, 11, 17, 20, 23],
    labels=["Night", "Morning", "Afternoon", "Evening", "Late evening"]
)
log("\n### 3. Datetime Parsing\n"
    "- Parsed `crash_datetime` to datetime type.\n"
    "- Extracted `crash_year`, `crash_month`, `crash_hour`, `crash_dow`, `crash_dow_name`.\n"
    "- Added `is_weekend` binary flag.\n"
    "- Added `time_of_day` categorical bin (Night/Morning/Afternoon/Evening/Late evening).")

# ── 4. Normalise case-inconsistent categoricals ───────────────────────────────
for col in ["vehicle_primary", "other_party"]:
    df[col] = df[col].str.strip().str.title()
log("\n### 4. Case Normalisation\n"
    "- `vehicle_primary` and `other_party`: standardised to Title Case "
    "(e.g. 'CAR' → 'Car', 'motorcycle' → 'Motorcycle').")

# ── 5. Fix implausible driver_age values ─────────────────────────────────────
bad_age_mask = (df["driver_age"] < 15) | (df["driver_age"] > 100)
n_bad = bad_age_mask.sum()
df.loc[bad_age_mask, "driver_age"] = np.nan
log(f"\n### 5. Driver Age\n"
    f"- Replaced **{n_bad}** implausible values (age < 15 or > 100) with NaN.")

# ── 6. Drop rows missing the target variable (severity) ──────────────────────
n_missing_target = df["severity"].isna().sum()
df.dropna(subset=["severity"], inplace=True)
log(f"\n### 6. Missing Target\n"
    f"- Dropped **{n_missing_target}** rows with missing `severity` (target variable).")

# ── 7. Impute remaining missing categoricals with 'Unknown' ──────────────────
cat_cols = ["road_condition", "junction", "weather", "lighting",
            "driver_gender", "alcohol_suspected", "helmet_worn"]
imputed_counts = {}
for col in cat_cols:
    n = df[col].isna().sum()
    if n > 0:
        df[col] = df[col].fillna("Unknown")
        imputed_counts[col] = n
log("\n### 7. Missing Categoricals → 'Unknown'\n" +
    "\n".join(f"- `{c}`: {v} values filled with 'Unknown'" for c, v in imputed_counts.items()))

# ── 8. Impute driver_age with median ─────────────────────────────────────────
n_age_null = df["driver_age"].isna().sum()
age_median = df["driver_age"].median()
df["driver_age"] = df["driver_age"].fillna(age_median)
df["driver_age"] = df["driver_age"].astype(int)
log(f"\n### 8. Missing driver_age\n"
    f"- Imputed **{n_age_null}** missing values with median age ({age_median:.0f}).")

# ── 9. Add age group feature ─────────────────────────────────────────────────
df["age_group"] = pd.cut(
    df["driver_age"],
    bins=[0, 24, 34, 44, 59, 100],
    labels=["≤24", "25–34", "35–44", "45–59", "60+"]
)
log("\n### 9. Engineered Features\n"
    "- Added `age_group`: bands ≤24, 25–34, 35–44, 45–59, 60+.")

# ── 10. Encode severity as ordered numeric ────────────────────────────────────
severity_order = {"Damage only": 0, "Minor injury": 1, "Serious/grievous": 2, "Fatal": 3}
df["severity_code"] = df["severity"].map(severity_order)
log("\n### 10. Severity Encoding\n"
    "- Added `severity_code`: Damage only=0, Minor injury=1, Serious/grievous=2, Fatal=3.")

# ── 11. Final state ───────────────────────────────────────────────────────────
removed = original_rows - len(df)
log(f"\n### Summary\n"
    f"| Metric | Value |\n|---|---|\n"
    f"| Original rows | {original_rows} |\n"
    f"| Rows removed | {removed} |\n"
    f"| Final rows | {len(df)} |\n"
    f"| Final columns | {len(df.columns)} |")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"\nCleaned dataset saved to: {OUT_CSV}")

LOG_PATH.write_text("\n".join(log_entries))
print(f"Cleaning log saved to: {LOG_PATH}")
