## Cleaning Log — Rwanda Road Accident Dataset

### Source
File: `rwanda_road_crashes_SYNTHETIC.csv`  
Data is **synthetic/simulated** (not real police records). Clearly labeled in `data_source` column.

### 1. Load
Initial shape: **3030 rows × 27 columns**

### 2. Removed Columns
- Dropped `data_source`: constant metadata column, not useful for analysis.

### 3. Datetime Parsing
- Parsed `crash_datetime` to datetime type.
- Extracted `crash_year`, `crash_month`, `crash_hour`, `crash_dow`, `crash_dow_name`.
- Added `is_weekend` binary flag.
- Added `time_of_day` categorical bin (Night/Morning/Afternoon/Evening/Late evening).

### 4. Case Normalisation
- `vehicle_primary` and `other_party`: standardised to Title Case (e.g. 'CAR' → 'Car', 'motorcycle' → 'Motorcycle').

### 5. Driver Age
- Replaced **5** implausible values (age < 15 or > 100) with NaN.

### 6. Missing Target
- Dropped **23** rows with missing `severity` (target variable).

### 7. Missing Categoricals → 'Unknown'
- `road_condition`: 241 values filled with 'Unknown'
- `junction`: 91 values filled with 'Unknown'
- `weather`: 151 values filled with 'Unknown'
- `lighting`: 119 values filled with 'Unknown'
- `driver_gender`: 58 values filled with 'Unknown'
- `alcohol_suspected`: 214 values filled with 'Unknown'
- `helmet_worn`: 118 values filled with 'Unknown'

### 8. Missing driver_age
- Imputed **276** missing values with median age (35).

### 9. Engineered Features
- Added `age_group`: bands ≤24, 25–34, 35–44, 45–59, 60+.

### 10. Severity Encoding
- Added `severity_code`: Damage only=0, Minor injury=1, Serious/grievous=2, Fatal=3.

### Summary
| Metric | Value |
|---|---|
| Original rows | 3030 |
| Rows removed | 23 |
| Final rows | 3007 |
| Final columns | 35 |