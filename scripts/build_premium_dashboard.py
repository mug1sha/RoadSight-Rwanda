"""
RoadSight Rwanda — Premium Analytics Dashboard (Tailwind CSS rebuild)
Generates dashboard/accident_risk_dashboard.html
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE    = Path(__file__).parent.parent
CSV     = BASE / "data" / "processed" / "accidents_clean.csv"
MR_PATH = BASE / "outputs" / "models" / "model_results.json"
OUT     = BASE / "dashboard" / "accident_risk_dashboard.html"

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df["crash_datetime"] = pd.to_datetime(df["crash_datetime"])

SEV   = ["Damage only", "Minor injury", "Serious/grievous", "Fatal"]
SEV_S = ["Damage only", "Minor injury", "Serious/grievous", "Fatal"]   # alias
PROVS = ["Kigali City", "Southern", "Eastern", "Western", "Northern"]
TOP_DIST = df["district"].value_counts().head(12).index.tolist()
TOD   = ["Morning", "Afternoon", "Evening", "Late evening", "Night"]
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DOW   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
YEARS = sorted(df["crash_year"].unique().tolist())

with open(MR_PATH) as f:
    model_results = json.load(f)

# ── Global KPIs ───────────────────────────────────────────────────────────────
total      = len(df)
fatal_n    = int((df["severity"] == "Fatal").sum())
serious_n  = int((df["severity"] == "Serious/grievous").sum())
minor_n    = int((df["severity"] == "Minor injury").sum())
damage_n   = int((df["severity"] == "Damage only").sum())
fatal_rate = round(fatal_n / total * 100, 1)
speeding_n = int((df["speeding_suspected"] == "Yes").sum())
alcohol_n  = int((df["alcohol_suspected"] == "Yes").sum())
rf_f1      = round(model_results["Random Forest"]["macro_f1"], 3)
rf_acc     = round(model_results["Random Forest"]["report"]["accuracy"] * 100, 1)

# ── Aggregate tables for every chart (pre-grouped, indexed by filter keys) ───
# Structure: { year|"all": { province|"all": { nested data } } }
# This lets JS filter by year + province + severity client-side without raw rows.

def agg_severity_counts(subset):
    return {s: int((subset["severity"] == s).sum()) for s in SEV}

def agg_yearly(subset):
    out = {}
    for yr in YEARS:
        s = subset[subset["crash_year"] == yr]
        out[str(yr)] = agg_severity_counts(s)
    return out

def agg_tod(subset):
    out = {}
    for t in TOD:
        s = subset[subset["time_of_day"] == t]
        out[t] = agg_severity_counts(s)
    return out

def agg_monthly(subset):
    out = {}
    for i, m in enumerate(MONTHS, 1):
        s = subset[subset["crash_month"] == i]
        out[m] = agg_severity_counts(s)
    return out

def agg_heatmap(subset):
    # 7 × 24 matrix  [dow_index][hour]
    mat = [[0]*24 for _ in range(7)]
    dow_map = {d: i for i, d in enumerate(DOW)}
    for _, row in subset.iterrows():
        di = dow_map.get(row["crash_dow_name"], None)
        if di is not None:
            mat[di][int(row["crash_hour"])] += 1
    return mat

def agg_province(subset):
    out = {}
    for p in PROVS:
        s = subset[subset["province"] == p]
        out[p] = agg_severity_counts(s)
    return out

def agg_district(subset):
    out = {}
    for d in TOP_DIST:
        s = subset[subset["district"] == d]
        out[d] = {**agg_severity_counts(s), "total": len(s)}
    return out

def agg_vehicle(subset):
    vehs = ["Motorcycle","Car","Minibus/Taxi","Truck","Bus","Bicycle","Other"]
    out = {}
    for v in vehs:
        s = subset[subset["vehicle_primary"] == v]
        out[v] = agg_severity_counts(s)
    return out

def agg_factors(subset):
    baseline = round(float(subset["severity_code"].mean()), 3) if len(subset) else 0
    result = {"baseline": baseline, "items": []}
    for col, vals in [
        ("weather",            ["Clear","Rain","Fog"]),
        ("road_condition",     ["Dry","Wet"]),
        ("lighting",           ["Daylight","Street-lit night","Dark night","Dawn/Dusk"]),
        ("alcohol_suspected",  ["No","Yes"]),
        ("speeding_suspected", ["No","Yes"]),
    ]:
        for val in vals:
            s = subset[subset[col] == val]
            if len(s) >= 10:
                mean = round(float(s["severity_code"].mean()), 3)
                result["items"].append({"label": f"{col.replace('_',' ').title()}: {val}",
                                        "value": mean, "n": len(s)})
    result["items"].sort(key=lambda x: x["value"])
    return result

# Build a compact aggregated dataset keyed by (year, province) combinations
# "all" means no filter on that dimension
print("Building aggregate tables…")
AGG = {}

filter_combos = [("all", "all")]
for yr in YEARS:
    filter_combos.append((str(yr), "all"))
for pv in PROVS:
    filter_combos.append(("all", pv))
for yr in YEARS:
    for pv in PROVS:
        filter_combos.append((str(yr), pv))

for (yr, pv) in filter_combos:
    sub = df.copy()
    if yr != "all":
        sub = sub[sub["crash_year"] == int(yr)]
    if pv != "all":
        sub = sub[sub["province"] == pv]

    key = f"{yr}|{pv}"
    # Heatmap is expensive; skip year-province combos to keep file size reasonable
    include_heatmap = (pv == "all")

    AGG[key] = {
        "kpi":      {
            "total":   len(sub),
            "fatal":   int((sub["severity"] == "Fatal").sum()),
            "serious": int((sub["severity"] == "Serious/grievous").sum()),
            "damage":  int((sub["severity"] == "Damage only").sum()),
            "minor":   int((sub["severity"] == "Minor injury").sum()),
        },
        "donut":    agg_severity_counts(sub),
        "yearly":   agg_yearly(sub),
        "tod":      agg_tod(sub),
        "monthly":  agg_monthly(sub),
        "province": agg_province(sub),
        "district": agg_district(sub),
        "vehicle":  agg_vehicle(sub),
        "factors":  agg_factors(sub),
        "heatmap":  agg_heatmap(sub) if include_heatmap else None,
    }
    print(f"  {key}: {len(sub)} rows")

# ── Model performance data ────────────────────────────────────────────────────
MODEL_DATA = {}
for model in ["Random Forest", "XGBoost"]:
    MODEL_DATA[model] = {}
    for cls in SEV:
        r = model_results[model]["report"][cls]
        MODEL_DATA[model][cls] = {
            "precision": round(r["precision"], 3),
            "recall":    round(r["recall"],    3),
            "f1":        round(r["f1-score"],  3),
            "support":   int(r["support"]),
        }
    MODEL_DATA[model]["macro_f1"] = model_results[model]["macro_f1"]
    MODEL_DATA[model]["accuracy"] = round(
        model_results[model]["report"]["accuracy"] * 100, 1)

# ── District ranking (global, for right-column panel) ────────────────────────
DISTRICT_RANK = []
for d in TOP_DIST[:8]:
    s = df[df["district"] == d]
    DISTRICT_RANK.append({
        "name":  d,
        "total": len(s),
        "fatal": int((s["severity"] == "Fatal").sum()),
    })

agg_json   = json.dumps(AGG,          separators=(",", ":"))
model_json = json.dumps(MODEL_DATA,   separators=(",", ":"))
rank_json  = json.dumps(DISTRICT_RANK,separators=(",", ":"))
kpi_json   = json.dumps({
    "total": total, "fatal": fatal_n, "serious": serious_n,
    "minor": minor_n, "damage": damage_n,
    "fatal_rate": fatal_rate, "rf_f1": rf_f1, "rf_acc": rf_acc,
}, separators=(",", ":"))
meta_json  = json.dumps({
    "years": [str(y) for y in YEARS],
    "provinces": PROVS,
    "severities": SEV,
    "tod": TOD,
    "months": MONTHS,
    "dow": DOW,
    "top_districts": TOP_DIST,
}, separators=(",", ":"))

print(f"AGG size: {len(agg_json)//1024} KB")
print("Aggregation complete.")


# ════════════════════════════════════════════════════════════════════════════
# HTML PARTS
# ════════════════════════════════════════════════════════════════════════════

HEAD = """\
<!DOCTYPE html>
<html lang="en" class="light" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RoadSight Rwanda — Road Accident Risk Analysis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com?plugins=forms"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
/* Tailwind config — extend with our design tokens */
tailwind.config = {
  darkMode: 'class',
  theme: {
    fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui'] },
    extend: {
      colors: {
        green: {
          50:  '#E6F2EB',
          100: '#C5E3D0',
          200: '#8DC8A9',
          400: '#19B875',
          500: '#0B8F5A',
          700: '#075B3E',
          900: '#03221A',
        },
        surface: {
          light: '#FFFFFF',
          light2: '#F5F4F4',
          dark:  '#19231D',
          dark2: '#202D25',
        },
        bg: {
          light: '#E8E6E6',
          dark:  '#101512',
        },
      },
      boxShadow: {
        card:      '0 1px 4px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.06)',
        'card-dark':'0 1px 4px rgba(0,0,0,0.3),  0 4px 16px rgba(0,0,0,0.25)',
        'card-hover':'0 4px 12px rgba(0,0,0,0.12), 0 8px 32px rgba(0,0,0,0.09)',
      },
      borderRadius: { card: '12px', btn: '8px' },
      transitionProperty: {
        theme: 'background-color, border-color, color, fill, stroke',
      },
    },
  },
};
</script>
<style>
/* ── CSS custom properties for Plotly & progressive overrides ─── */
:root {
  --chart-c0:#19B875; --chart-c1:#0B8F5A;
  --chart-c2:#075B3E; --chart-c3:#C0392B;
  --chart-c4:#E67E22; --chart-c5:#D35400; --chart-c6:#8B0000;
  --grid:rgba(0,0,0,0.07); --tick:#777777;
  --title-c:#1a1a1a; --legend-bg:rgba(255,255,255,0.92);
  --hm-lo:#E6F2EB; --hm-mid:#19B875; --hm-hi:#075B3E; --hm-peak:#030f07;
  --anno:#555555;
}
.dark {
  --chart-c0:#35C786; --chart-c1:#19B875;
  --chart-c2:#0B8F5A; --chart-c3:#E74C3C;
  --chart-c4:#F39C12; --chart-c5:#D35400; --chart-c6:#8B0000;
  --grid:rgba(255,255,255,0.07); --tick:#A0ADA4;
  --title-c:#E8F0EB; --legend-bg:rgba(25,35,29,0.95);
  --hm-lo:#101512; --hm-mid:#0B8F5A; --hm-hi:#19B875; --hm-peak:#E8F0EB;
  --anno:#A0ADA4;
}

/* scrollbar */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(0,0,0,0.15); border-radius:4px; }
.dark ::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); }

/* sidebar transition on mobile */
#sidebar { transition: transform 0.28s cubic-bezier(0.4,0,0.2,1); }

/* progress bar animation */
.bar-fill { transition: width 0.9s cubic-bezier(0.4,0,0.2,1); }

/* KPI counter */
.kpi-val { font-variant-numeric: tabular-nums; }

/* nav active indicator */
.nav-active {
  background: #E6F2EB;
  color: #075B3E;
  font-weight: 600;
}
.dark .nav-active {
  background: rgba(53,199,134,0.14);
  color: #35C786;
}

/* smooth theme */
*, *::before, *::after {
  transition: background-color 0.2s, border-color 0.2s, color 0.15s;
}
/* but NOT transforms or Plotly canvases */
svg, canvas, .js-plotly-plot * { transition: none !important; }

/* chart container */
.chart-pane { background: transparent; width: 100%; }

/* filter select arrow fix on dark */
.dark select option { background: #19231D; color: #F4F7F4; }
</style>
</head>
"""

# ── Sidebar ───────────────────────────────────────────────────────────────────
SIDEBAR = """\
<aside id="sidebar"
  class="fixed top-0 left-0 h-full w-60 z-50 flex flex-col
         bg-white dark:bg-surface-dark
         border-r border-black/[0.07] dark:border-white/[0.07]
         -translate-x-full lg:translate-x-0 overflow-y-auto overflow-x-hidden">

  <!-- Brand -->
  <div class="flex items-center gap-3 px-5 py-5 border-b border-black/[0.07] dark:border-white/[0.07] flex-shrink-0">
    <div class="w-9 h-9 rounded-lg bg-green-700 dark:bg-green-400 flex items-center justify-center flex-shrink-0">
      <svg class="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M3 12l9-9 9 9M5 10v10h5v-5h4v5h5V10"/>
      </svg>
    </div>
    <div>
      <div class="text-sm font-bold text-gray-900 dark:text-green-50 leading-tight">RoadSight</div>
      <div class="text-[10px] text-gray-500 dark:text-green-200/60">Rwanda Analytics</div>
    </div>
  </div>

  <!-- Nav -->
  <nav class="flex-1 px-3 py-4 space-y-5">

    <div>
      <p class="text-[10px] font-semibold uppercase tracking-widest text-gray-400 dark:text-green-200/40 px-2 mb-1.5">Main Menu</p>
      <button onclick="showSection('overview',this)"   class="nav-item nav-active w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-gray-700 dark:text-green-100 hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
        <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
        Overview
      </button>
      <button onclick="showSection('reports',this)"    class="nav-item w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-gray-700 dark:text-green-100 hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
        <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
        Accident Reports
      </button>
      <button onclick="showSection('risk',this)"       class="nav-item w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-gray-700 dark:text-green-100 hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
        <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
        Risk Analysis
      </button>
      <button onclick="showSection('geo',this)"        class="nav-item w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-gray-700 dark:text-green-100 hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
        <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0zM15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
        Geographic Analysis
      </button>
    </div>

    <div>
      <p class="text-[10px] font-semibold uppercase tracking-widest text-gray-400 dark:text-green-200/40 px-2 mb-1.5">Analytics</p>
      <button onclick="showSection('trends',this)"     class="nav-item w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-gray-700 dark:text-green-100 hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
        <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        Accident Trends
      </button>
      <button onclick="showSection('severity',this)"   class="nav-item w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-gray-700 dark:text-green-100 hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
        <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
        Severity Analysis
      </button>
      <button onclick="showSection('factors',this)"    class="nav-item w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-gray-700 dark:text-green-100 hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
        <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 010 14.14M4.93 4.93a10 10 0 000 14.14"/></svg>
        Contributing Factors
      </button>
      <button onclick="showSection('model',this)"      class="nav-item w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-gray-700 dark:text-green-100 hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
        <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><path d="M12 2a10 10 0 100 20A10 10 0 0012 2zM2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>
        ML Predictions
      </button>
    </div>

    <div>
      <p class="text-[10px] font-semibold uppercase tracking-widest text-gray-400 dark:text-green-200/40 px-2 mb-1.5">System</p>
      <button onclick="showSection('about',this)"      class="nav-item w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-gray-700 dark:text-green-100 hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
        <svg class="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4m0-4h.01"/></svg>
        About Project
      </button>
    </div>

  </nav>

  <!-- Footer -->
  <div class="px-4 py-3 border-t border-black/[0.07] dark:border-white/[0.07] flex-shrink-0">
    <p class="text-[11px] text-gray-400 dark:text-green-200/40 leading-relaxed">
      Group D · Big Data 2026
    </p>
    <span class="inline-block mt-1 text-[9px] font-semibold uppercase tracking-wide
                 bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400
                 px-2 py-0.5 rounded-full">
      ⚠ Synthetic dataset
    </span>
  </div>
</aside>

<!-- Mobile backdrop -->
<div id="backdrop" onclick="closeSidebar()"
     class="hidden fixed inset-0 bg-black/50 z-40 lg:hidden"></div>
"""

# ── Header ────────────────────────────────────────────────────────────────────
HEADER = """\
<header class="sticky top-0 z-30 h-16 flex items-center gap-3 px-5
               bg-white dark:bg-surface-dark
               border-b border-black/[0.07] dark:border-white/[0.07]">

  <!-- Hamburger (mobile) -->
  <button onclick="toggleSidebar()" class="lg:hidden p-1.5 rounded-lg
          text-gray-500 hover:bg-green-50 dark:hover:bg-green-400/10"
          aria-label="Open menu">
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <line x1="3" y1="6"  x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/>
      <line x1="3" y1="18" x2="21" y2="18"/>
    </svg>
  </button>

  <div class="flex-1 min-w-0">
    <h1 id="page-title" class="text-[15px] font-700 font-bold text-gray-900 dark:text-green-50 leading-tight truncate">
      Road Safety Overview
    </h1>
    <p class="text-[11px] text-gray-400 dark:text-green-200/50 leading-none mt-0.5">
      Rwanda road accident risk analysis · 2022–2025
    </p>
  </div>

  <div class="flex items-center gap-2 flex-shrink-0">
    <div id="date-badge"
         class="hidden sm:block text-[11px] text-gray-400 dark:text-green-200/50
                bg-gray-100 dark:bg-surface-dark2 border border-black/[0.07] dark:border-white/[0.07]
                px-3 py-1 rounded-full whitespace-nowrap">
    </div>

    <!-- Refresh -->
    <button onclick="refreshCharts()" title="Refresh"
            class="w-9 h-9 flex items-center justify-center rounded-lg
                   bg-gray-100 dark:bg-surface-dark2 border border-black/[0.07] dark:border-white/[0.07]
                   text-gray-500 dark:text-green-200/60
                   hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
      <svg id="refresh-icon" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="23 4 23 10 17 10"/>
        <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/>
      </svg>
    </button>

    <!-- Theme toggle -->
    <button onclick="toggleTheme()" title="Toggle theme"
            class="w-9 h-9 flex items-center justify-center rounded-lg
                   bg-gray-100 dark:bg-surface-dark2 border border-black/[0.07] dark:border-white/[0.07]
                   text-gray-500 dark:text-green-200/60
                   hover:bg-green-50 dark:hover:bg-green-400/10 hover:text-green-700 dark:hover:text-green-400">
      <svg id="icon-moon" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>
      <svg id="icon-sun"  class="w-4 h-4 hidden" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
    </button>
  </div>
</header>
"""

# ── KPI cards ─────────────────────────────────────────────────────────────────
def kpi_card(icon_path, ring_cls, label, val_id, default_val, sub_html):
    return f"""\
<div class="bg-white dark:bg-surface-dark rounded-card p-4 shadow-card flex items-start gap-3
            hover:shadow-card-hover hover:-translate-y-0.5 transition-all duration-200">
  <div class="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 {ring_cls}">
    <svg class="w-4.5 h-4.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="{icon_path}"/></svg>
  </div>
  <div class="min-w-0">
    <p class="text-[11px] font-semibold uppercase tracking-wide text-gray-400 dark:text-green-200/50">{label}</p>
    <p class="text-[22px] font-bold text-gray-900 dark:text-green-50 kpi-val leading-tight" id="{val_id}">{default_val}</p>
    <p class="text-[11px] text-gray-400 dark:text-green-200/40 mt-0.5">{sub_html}</p>
  </div>
</div>"""

KPI_GRID = f"""\
<div id="kpi-grid" class="grid grid-cols-2 xl:grid-cols-4 gap-3 mb-5">
  {kpi_card("M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z",
            "bg-green-50 dark:bg-green-400/10 text-green-700 dark:text-green-400",
            "Total Accidents","kpi-total",f"{total:,}","2022–2025 · 5 provinces")}
  {kpi_card("M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z",
            "bg-red-50 dark:bg-red-400/10 text-red-600 dark:text-red-400",
            "Fatal Accidents","kpi-fatal",f"{fatal_n}",f'<span class="text-red-500 dark:text-red-400">▲ {fatal_rate}%</span> of all accidents')}
  {kpi_card("M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z",
            "bg-orange-50 dark:bg-orange-400/10 text-orange-600 dark:text-orange-400",
            "Serious Injuries","kpi-serious",f"{serious_n}","13.3% of all accidents")}
  {kpi_card("M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z",
            "bg-blue-50 dark:bg-blue-400/10 text-blue-600 dark:text-blue-400",
            "RF Model F1","kpi-f1",f"{rf_f1}",f"Macro avg · accuracy {rf_acc}%")}
</div>"""

# ── Filter bar ─────────────────────────────────────────────────────────────────
# Single persistent filter bar — rendered ONCE in the layout (outside sections).
# All sections share the same three select elements. No duplicate IDs anywhere.
FILTER_BAR = """\
<div id="filter-bar" class="flex flex-wrap items-center gap-2 mb-4
     bg-white dark:bg-surface-dark rounded-card p-3 shadow-card
     border border-black/[0.05] dark:border-white/[0.05]">
  <span class="text-[11px] font-semibold text-gray-400 dark:text-green-200/40 uppercase tracking-wide mr-1">
    Filter:
  </span>
  <select id="f-year" onchange="applyFilters()"
          class="text-[12px] rounded-btn border border-black/[0.08] dark:border-white/[0.08]
                 bg-gray-50 dark:bg-surface-dark2 text-gray-700 dark:text-green-100
                 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-green-500">
    <option value="">All Years</option>
    <option value="2022">2022</option>
    <option value="2023">2023</option>
    <option value="2024">2024</option>
    <option value="2025">2025</option>
  </select>
  <select id="f-province" onchange="applyFilters()"
          class="text-[12px] rounded-btn border border-black/[0.08] dark:border-white/[0.08]
                 bg-gray-50 dark:bg-surface-dark2 text-gray-700 dark:text-green-100
                 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-green-500">
    <option value="">All Provinces</option>
    <option value="Kigali City">Kigali City</option>
    <option value="Southern">Southern</option>
    <option value="Eastern">Eastern</option>
    <option value="Western">Western</option>
    <option value="Northern">Northern</option>
  </select>
  <select id="f-severity" onchange="applyFilters()"
          class="text-[12px] rounded-btn border border-black/[0.08] dark:border-white/[0.08]
                 bg-gray-50 dark:bg-surface-dark2 text-gray-700 dark:text-green-100
                 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-green-500">
    <option value="">All Severities</option>
    <option value="Damage only">Damage only</option>
    <option value="Minor injury">Minor injury</option>
    <option value="Serious/grievous">Serious/grievous</option>
    <option value="Fatal">Fatal</option>
  </select>
  <button onclick="clearFilters()"
          class="text-[12px] px-3 py-1.5 rounded-btn
                 bg-gray-100 dark:bg-surface-dark2
                 border border-black/[0.08] dark:border-white/[0.08]
                 text-gray-600 dark:text-green-200/60
                 hover:bg-green-50 dark:hover:bg-green-400/10
                 hover:text-green-700 dark:hover:text-green-400">
    Clear
  </button>
  <span id="filter-status" class="ml-auto text-[11px] text-gray-400 dark:text-green-200/40 hidden sm:inline"></span>
  <span id="filter-active-badge"
        class="hidden text-[10px] font-semibold uppercase tracking-wide
               bg-green-100 dark:bg-green-400/15 text-green-700 dark:text-green-400
               px-2 py-0.5 rounded-full ml-1">
    Filtered
  </span>
</div>"""

# ── Card helper ───────────────────────────────────────────────────────────────
def card(title, subtitle, body_html, extra_cls=""):
    return f"""\
<div class="bg-white dark:bg-surface-dark rounded-card shadow-card
            border border-black/[0.05] dark:border-white/[0.05] overflow-hidden {extra_cls}">
  <div class="flex items-start justify-between px-4 py-3
              border-b border-black/[0.05] dark:border-white/[0.05]">
    <div>
      <p class="text-[13px] font-semibold text-gray-800 dark:text-green-50">{title}</p>
      <p class="text-[11px] text-gray-400 dark:text-green-200/40 mt-0.5">{subtitle}</p>
    </div>
  </div>
  {body_html}
</div>"""

def chart_card(title, subtitle, div_id, height=300, extra_cls=""):
    body = f'<div id="{div_id}" style="height:{height}px" class="chart-pane px-1"></div>'
    return card(title, subtitle, body, extra_cls)

# ── District ranking panel ─────────────────────────────────────────────────────
def district_panel():
    rows = ""
    max_total = DISTRICT_RANK[0]["total"]
    for item in DISTRICT_RANK:
        pct = round(item["total"] / max_total * 100)
        rows += f"""\
<div class="mb-3 last:mb-0">
  <div class="flex justify-between items-baseline mb-1">
    <span class="text-[12px] font-medium text-gray-700 dark:text-green-100">{item["name"]}</span>
    <span class="text-[11px] text-gray-400 dark:text-green-200/40">
      {item["total"]:,} &middot; <span class="text-red-500 dark:text-red-400">{item["fatal"]} fatal</span>
    </span>
  </div>
  <div class="h-1.5 bg-gray-100 dark:bg-surface-dark2 rounded-full overflow-hidden">
    <div class="bar-fill h-full rounded-full bg-green-400 dark:bg-green-400"
         style="width:{pct}%"></div>
  </div>
</div>"""

    body = f'<div class="px-4 py-3">{rows}</div>'
    return card("Highest-Risk Districts",
                "Accident count · fatal incidents", body)

# ── Model metrics table ────────────────────────────────────────────────────────
def model_metrics_panel():
    SEV_LABELS = {"Damage only":"Damage only",
                  "Minor injury":"Minor injury",
                  "Serious/grievous":"Serious/grievous",
                  "Fatal":"Fatal"}
    rows = ""
    for cls in SEV:
        rf  = MODEL_DATA["Random Forest"][cls]
        xgb = MODEL_DATA["XGBoost"][cls]
        bar = int(rf["f1"] * 72)
        rows += f"""\
<tr class="border-b border-black/[0.04] dark:border-white/[0.04] last:border-0
           hover:bg-gray-50 dark:hover:bg-surface-dark2 transition-colors">
  <td class="py-2 pr-3 text-[12px] font-medium text-gray-700 dark:text-green-100 whitespace-nowrap">{cls}</td>
  <td class="py-2 pr-2 text-[12px] text-gray-500 dark:text-green-200/60">{rf["precision"]:.2f}</td>
  <td class="py-2 pr-2 text-[12px] text-gray-500 dark:text-green-200/60">{rf["recall"]:.2f}</td>
  <td class="py-2 pr-3 text-[12px] text-gray-700 dark:text-green-100">
    {rf["f1"]:.2f}
    <span class="inline-block h-1 rounded bg-green-400 dark:bg-green-400 ml-1 align-middle"
          style="width:{bar}px"></span>
  </td>
  <td class="py-2 text-[12px] text-gray-500 dark:text-green-200/60">{xgb["f1"]:.2f}</td>
</tr>"""

    rf_mf1  = MODEL_DATA["Random Forest"]["macro_f1"]
    xgb_mf1 = MODEL_DATA["XGBoost"]["macro_f1"]
    body = f"""\
<div class="px-4 py-3 overflow-x-auto">
  <table class="w-full text-left">
    <thead>
      <tr class="border-b border-black/[0.07] dark:border-white/[0.07]">
        <th class="pb-2 text-[10.5px] font-semibold uppercase tracking-wide text-gray-400 dark:text-green-200/40 pr-3">Class</th>
        <th class="pb-2 text-[10.5px] font-semibold uppercase tracking-wide text-gray-400 dark:text-green-200/40 pr-2">RF P</th>
        <th class="pb-2 text-[10.5px] font-semibold uppercase tracking-wide text-gray-400 dark:text-green-200/40 pr-2">RF R</th>
        <th class="pb-2 text-[10.5px] font-semibold uppercase tracking-wide text-gray-400 dark:text-green-200/40 pr-3">RF F1</th>
        <th class="pb-2 text-[10.5px] font-semibold uppercase tracking-wide text-gray-400 dark:text-green-200/40">XGB F1</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <p class="mt-3 text-[10.5px] text-gray-400 dark:text-green-200/40 leading-relaxed">
    RF macro F1 = <strong class="text-green-700 dark:text-green-400">{rf_mf1:.3f}</strong> &nbsp;|&nbsp;
    XGB macro F1 = <strong class="text-orange-500">{xgb_mf1:.3f}</strong><br>
    ⚠ Class imbalance: 44.9% Damage only vs 6.9% Fatal.
    Low Fatal F1 (RF 0.146) is expected without resampling.
  </p>
</div>"""
    return card("Model Metrics — Per Class",
                "Random Forest vs XGBoost", body)

# ── Insight chips ──────────────────────────────────────────────────────────────
INSIGHT_ROW = """\
<div class="grid grid-cols-2 xl:grid-cols-4 gap-3 mb-5">
  <div class="bg-white dark:bg-surface-dark rounded-card p-3 shadow-card
              border border-black/[0.05] dark:border-white/[0.05]">
    <p class="text-[12px] font-semibold text-gray-800 dark:text-green-50">63.1% — Kigali City</p>
    <p class="text-[11px] text-gray-400 dark:text-green-200/40 mt-1 leading-relaxed">
      1,897 of 3,007 accidents occur in the capital due to traffic density.
    </p>
  </div>
  <div class="bg-white dark:bg-surface-dark rounded-card p-3 shadow-card
              border border-black/[0.05] dark:border-white/[0.05]">
    <p class="text-[12px] font-semibold text-gray-800 dark:text-green-50">Peak: Afternoon</p>
    <p class="text-[11px] text-gray-400 dark:text-green-200/40 mt-1 leading-relaxed">
      1,060 accidents (35.3%) in the afternoon. Evening rush adds 539 more.
    </p>
  </div>
  <div class="bg-white dark:bg-surface-dark rounded-card p-3 shadow-card
              border border-black/[0.05] dark:border-white/[0.05]">
    <p class="text-[12px] font-semibold text-gray-800 dark:text-green-50">Speeding &amp; alcohol</p>
    <p class="text-[11px] text-gray-400 dark:text-green-200/40 mt-1 leading-relaxed">
      18% of crashes involve speeding; 5.9% alcohol. Both raise mean severity.
    </p>
  </div>
  <div class="bg-white dark:bg-surface-dark rounded-card p-3 shadow-card
              border border-black/[0.05] dark:border-white/[0.05]">
    <p class="text-[12px] font-semibold text-gray-800 dark:text-green-50">Motorcycles dominate</p>
    <p class="text-[11px] text-gray-400 dark:text-green-200/40 mt-1 leading-relaxed">
      982 accidents (32.7%) involve motorcycles — Rwanda's most common vehicle.
    </p>
  </div>
</div>"""

# ── Notice bar ─────────────────────────────────────────────────────────────────
NOTICE = """\
<div class="flex items-start gap-2 bg-amber-50 dark:bg-amber-900/20
            border border-amber-200 dark:border-amber-700/40
            rounded-card px-4 py-2.5 mb-5 text-[11.5px]
            text-amber-700 dark:text-amber-400">
  <svg class="w-4 h-4 mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
  </svg>
  <span>
    <strong>Synthetic dataset —</strong> not real police records.
    Findings are illustrative only. Do not use for policy decisions
    without validation against official Rwanda National Police or RURA data.
  </span>
</div>"""

# ── Section content builders ───────────────────────────────────────────────────

def section_overview():
    left = f"""\
<div class="flex flex-col gap-4">
  {chart_card("Accident Trends by Year","Grouped by severity class","chart-yearly",280)}
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    {chart_card("Time of Day","Stacked by severity","chart-tod",280)}
    {chart_card("Monthly Pattern","All years combined","chart-monthly",280)}
  </div>
  {chart_card("Hour \u00d7 Day of Week Heatmap","Accident frequency intensity","chart-heatmap",250)}
</div>"""

    right = f"""\
<div class="flex flex-col gap-4">
  {chart_card("Severity Distribution","All accidents","chart-donut",300)}
  {district_panel()}
</div>"""

    return f"""\
<div id="sec-overview" class="section-panel">
  {NOTICE}
  {KPI_GRID}
  {INSIGHT_ROW}
  <div class="grid grid-cols-1 xl:grid-cols-[1fr_300px] gap-4">
    {left}
    {right}
  </div>
</div>"""


def section_reports():
    return f"""\
<div id="sec-reports" class="section-panel hidden">
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
    {chart_card("Accident Trends by Year","Grouped by severity","chart-yearly-r",280)}
    {chart_card("Monthly Pattern","All years combined","chart-monthly-r",280)}
  </div>
  <div class="mb-4">
    {chart_card("Top Districts — Stacked by Severity","Accident count per district","chart-districts",380)}
  </div>
  {chart_card("Accidents by Province","Stacked by severity","chart-province",300)}
</div>"""


def section_risk():
    return f"""\
<div id="sec-risk" class="section-panel hidden">
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    {chart_card("Accidents by Vehicle Type","Stacked by severity","chart-vehicle",320)}
    {chart_card("Contributing Factors","Mean severity score (0\u20133)","chart-factors",380)}
  </div>
</div>"""


def section_geo():
    return f"""\
<div id="sec-geo" class="section-panel hidden">
  <div class="grid grid-cols-1 xl:grid-cols-[1fr_300px] gap-4">
    <div class="flex flex-col gap-4">
      {chart_card("Accidents by Province","Stacked by severity","chart-province-g",300)}
      {chart_card("Top Districts","Stacked by severity","chart-districts-g",380)}
    </div>
    <div>
      {district_panel()}
    </div>
  </div>
</div>"""


def section_trends():
    return f"""\
<div id="sec-trends" class="section-panel hidden">
  <div class="flex flex-col gap-4">
    {chart_card("Accident Trends by Year","Grouped by severity class","chart-yearly-t",300)}
    {chart_card("Monthly Pattern","All years combined","chart-monthly-t",300)}
    {chart_card("Hour \u00d7 Day Heatmap","Intensity = accident frequency","chart-heatmap-t",260)}
    {chart_card("Time of Day","Stacked by severity","chart-tod-t",280)}
  </div>
</div>"""


def section_severity():
    return f"""\
<div id="sec-severity" class="section-panel hidden">
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
    {chart_card("Severity Distribution","Donut \u2014 all accidents","chart-donut-s",320)}
    {chart_card("Severity by Province","Stacked bar","chart-province-s",320)}
  </div>
  {chart_card("Severity by Vehicle Type","Stacked bar","chart-vehicle-s",320)}
</div>"""


def section_factors():
    return f"""\
<div id="sec-factors" class="section-panel hidden">
  <div class="flex flex-col gap-4">
    {chart_card("Contributing Factors \u2014 Mean Severity Score",
                "0 = Damage only \u00b7 3 = Fatal \u00b7 dotted line = dataset average",
                "chart-factors-f",440)}
    {chart_card("Accidents by Vehicle Type","Stacked by severity","chart-vehicle-f",320)}
  </div>
</div>"""


def section_model():
    return f"""\
<div id="sec-model" class="section-panel hidden">
  <div class="flex flex-col gap-4">
    {chart_card("Model Performance by Class",
                "Precision / Recall / F1 \u2014 Random Forest vs XGBoost",
                "chart-model",340)}
    {model_metrics_panel()}
  </div>
</div>"""


def section_about():
    return """\
<div id="sec-about" class="section-panel hidden">
  <div class="bg-white dark:bg-surface-dark rounded-card shadow-card
              border border-black/[0.05] dark:border-white/[0.05] p-6 max-w-2xl">
    <h2 class="text-[16px] font-bold text-gray-900 dark:text-green-50 mb-3">About RoadSight Rwanda</h2>
    <p class="text-[13px] text-gray-600 dark:text-green-200/60 leading-relaxed mb-4">
      RoadSight Rwanda is a Big Data analytics project by Group D (2026) that analyses
      road accident risk patterns across Rwanda using machine learning.
    </p>
    <div class="space-y-2 text-[13px] text-gray-600 dark:text-green-200/60">
      <p><strong class="text-gray-800 dark:text-green-100">Dataset:</strong> 3,007 synthetic/simulated accidents, 2022&ndash;2025, 5 provinces, 35 features.</p>
      <p><strong class="text-gray-800 dark:text-green-100">Models:</strong> Random Forest (macro F1 = 0.320) and XGBoost (macro F1 = 0.265).</p>
      <p><strong class="text-gray-800 dark:text-green-100">Stack:</strong> Python &middot; Pandas &middot; scikit-learn &middot; XGBoost &middot; Plotly &middot; Tailwind CSS.</p>
      <p><strong class="text-gray-800 dark:text-green-100">Important:</strong> This dataset is synthetic and was generated for educational purposes.
         Results must not inform real policy without validation against official Rwanda National Police or RURA data.</p>
    </div>
  </div>
</div>"""

# ── All sections ───────────────────────────────────────────────────────────────
ALL_SECTIONS = "\n".join([
    section_overview(),
    section_reports(),
    section_risk(),
    section_geo(),
    section_trends(),
    section_severity(),
    section_factors(),
    section_model(),
    section_about(),
])

FOOTER = """\
<footer class="mt-8 py-5 border-t border-black/[0.07] dark:border-white/[0.07]
               text-center text-[11px] text-gray-400 dark:text-green-200/30 leading-relaxed">
  <strong class="text-green-700 dark:text-green-400 tracking-widest uppercase text-[12px]">
    RoadSight Rwanda
  </strong><br>
  Group D · Big Data Project 2026 · Rwanda<br>
  Python · Plotly 2.27 · scikit-learn · XGBoost · Tailwind CSS<br>
  <span class="opacity-60">Dataset: Synthetic — not real police records</span>
</footer>"""

# ── JavaScript ────────────────────────────────────────────────────────────────
def build_js(agg_j, model_j, rank_j, kpi_j, meta_j):
    return f"""<script>
// ═══════════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════════
var AGG   = {agg_j};
var MODEL = {model_j};
var RANK  = {rank_j};
var GKPI  = {kpi_j};
var META  = {meta_j};

var SEV    = META.severities;
var MONTHS = META.months;
var TOD    = META.tod;
var DOW    = META.dow;

// ═══════════════════════════════════════════════════════════════
// THEME
// ═══════════════════════════════════════════════════════════════
var PALETTES = {{
  light: {{
    colors:  ["#19B875","#0B8F5A","#075B3E","#C0392B","#E67E22","#D35400"],
    donut:   ["#19B875","#0B8F5A","#075B3E","#C0392B"],
    hm_cs:   [[0,"#E6F2EB"],[0.4,"#19B875"],[0.75,"#075B3E"],[1,"#030f07"]],
    grid:    "rgba(0,0,0,0.07)", tick:"#777", title:"#1a1a1a",
    legend:  "rgba(255,255,255,0.92)", anno:"#555",
    factor_lo:"#19B875", factor_mid:"#E67E22", factor_hi:"#C0392B",
  }},
  dark: {{
    colors:  ["#35C786","#19B875","#0B8F5A","#E74C3C","#F39C12","#D35400"],
    donut:   ["#35C786","#19B875","#0B8F5A","#E74C3C"],
    hm_cs:   [[0,"#101512"],[0.4,"#0B8F5A"],[0.75,"#19B875"],[1,"#E8F0EB"]],
    grid:    "rgba(255,255,255,0.07)", tick:"#A0ADA4", title:"#E8F0EB",
    legend:  "rgba(25,35,29,0.95)", anno:"#A0ADA4",
    factor_lo:"#35C786", factor_mid:"#F39C12", factor_hi:"#E74C3C",
  }}
}};

function pal() {{ return PALETTES[isDark() ? "dark" : "light"]; }}
function isDark() {{ return document.documentElement.classList.contains("dark"); }}
var PLOTLY_CFG = {{responsive:true, displayModeBar:false}};
var BG = "rgba(0,0,0,0)";

// ═══════════════════════════════════════════════════════════════
// FILTER STATE
// ═══════════════════════════════════════════════════════════════
var F = {{ year:"all", province:"all", severity:"" }};

function aggKey() {{ return F.year + "|" + F.province; }}
function aggData() {{ return AGG[aggKey()] || AGG["all|all"]; }}

function applyFilters() {{
  F.year     = document.getElementById("f-year").value     || "all";
  F.province = document.getElementById("f-province").value || "all";
  F.severity = document.getElementById("f-severity").value || "";

  // Update active badge
  var active = [F.year!=="all", F.province!=="all", F.severity!==""].filter(Boolean).length;
  var badge = document.getElementById("filter-active-badge");
  var status = document.getElementById("filter-status");
  if(badge) badge.classList.toggle("hidden", active === 0);
  if(status) status.textContent = active ? active+" filter"+(active>1?"s":"")+" active" : "";

  // Update KPIs and district panel from filtered aggregate
  updateKPIs();
  updateDistrictPanel();

  // Re-render only charts that are currently visible (in the active section)
  renderVisibleCharts();
}}

function clearFilters() {{
  F = {{year:"all", province:"all", severity:""}};
  var selects = ["f-year","f-province","f-severity"];
  selects.forEach(function(id) {{
    var el = document.getElementById(id);
    if(el) el.value = "";
  }});
  var badge = document.getElementById("filter-active-badge");
  var status = document.getElementById("filter-status");
  if(badge) badge.classList.add("hidden");
  if(status) status.textContent = "";
  updateKPIs();
  updateDistrictPanel();
  renderVisibleCharts();
}}

// ═══════════════════════════════════════════════════════════════
// KPI update
// ═══════════════════════════════════════════════════════════════
function updateKPIs() {{
  var d = aggData().kpi;
  animCount("kpi-total",   d.total,   false);
  animCount("kpi-fatal",   d.fatal,   false);
  animCount("kpi-serious", d.serious, false);
  // F1 is model-level, doesn't change with filters
}}

function animCount(id, target, isFloat) {{
  var el = document.getElementById(id);
  if(!el) return;
  var start = 0, dur = 900, t0 = null;
  function step(ts) {{
    if(!t0) t0 = ts;
    var p = Math.min((ts-t0)/dur, 1);
    var e = 1-Math.pow(1-p, 3);
    var v = target * e;
    el.textContent = isFloat ? v.toFixed(3) : Math.round(v).toLocaleString();
    if(p<1) requestAnimationFrame(step);
  }}
  requestAnimationFrame(step);
}}

// ═══════════════════════════════════════════════════════════════
// DISTRICT PANEL update
// ═══════════════════════════════════════════════════════════════
function updateDistrictPanel() {{
  var d = aggData().district;
  var panels = document.querySelectorAll(".district-dyn");
  panels.forEach(function(panel) {{
    var items = panel.querySelectorAll(".dist-item");
    var maxTotal = 0;
    items.forEach(function(item) {{
      var name = item.dataset.district;
      var total = (d[name] ? Object.values(d[name]).reduce(function(a,b){{return a+(typeof b==="number"?b:0);}},0) : 0);
      // don't count total key if present
      if(d[name] && d[name].total !== undefined) total = d[name].total || total;
      if(total > maxTotal) maxTotal = total;
      item._filteredTotal = total;
      item._filteredFatal = d[name] ? (d[name]["Fatal"]||0) : 0;
    }});
    items.forEach(function(item) {{
      var t = item._filteredTotal, f = item._filteredFatal;
      var pct = maxTotal > 0 ? Math.round(t/maxTotal*100) : 0;
      var countEl = item.querySelector(".dist-count");
      var fillEl  = item.querySelector(".bar-fill");
      var fatalEl = item.querySelector(".dist-fatal");
      if(countEl) countEl.textContent = t.toLocaleString();
      if(fatalEl) fatalEl.textContent = f + " fatal";
      if(fillEl)  fillEl.style.width  = pct + "%";
    }});
  }});
}}

// ═══════════════════════════════════════════════════════════════
// LAYOUT HELPERS
// ═══════════════════════════════════════════════════════════════
function baseLayout(title, height, margin) {{
  var p = pal();
  margin = margin || {{t:46,b:36,l:52,r:16}};
  return {{
    paper_bgcolor: BG, plot_bgcolor: BG, height: height,
    margin: margin,
    title: {{text:title, font:{{size:13,family:"Inter",color:p.title}}, x:0, xanchor:"left", pad:{{l:4}}}},
    font:  {{family:"Inter", size:11, color:p.tick}},
    legend: {{bgcolor:p.legend, font:{{size:10,color:p.tick}}, borderwidth:0}},
    hovermode: "closest",
  }};
}}

function axisStyle(title) {{
  var p = pal();
  return {{
    title: title||"", title_font:{{size:11,color:p.tick}},
    tickfont:{{size:10,color:p.tick}},
    gridcolor: p.grid, zerolinecolor: p.grid,
    showgrid: true,
  }};
}}

function sevFilter(obj) {{
  // If a severity filter is active, zero out other severities
  if(!F.severity) return obj;
  var out = {{}};
  SEV.forEach(function(s) {{ out[s] = s===F.severity ? obj[s] : 0; }});
  return out;
}}

// ═══════════════════════════════════════════════════════════════
// CHART BUILDERS  — each returns {{data, layout}}
// ═══════════════════════════════════════════════════════════════
var p_colors = function() {{ return pal().colors; }};

function buildDonut(divId) {{
  var d = aggData().donut;
  d = sevFilter(d);
  var vals = SEV.map(function(s){{return d[s]||0;}});
  var total_n = vals.reduce(function(a,b){{return a+b;}},0);
  var p = pal();
  return {{
    data: [{{
      type:"pie", labels:SEV, values:vals, hole:0.62,
      marker:{{colors:p.donut, line:{{color:"#fff",width:2}}}},
      textinfo:"percent", textfont:{{size:11}},
      hovertemplate:"<b>%{{label}}</b><br>%{{value:,}} · %{{percent}}<extra></extra>",
      sort:false,
    }}],
    layout: Object.assign(baseLayout("", 290, {{t:10,b:10,l:10,r:10}}), {{
      showlegend:true,
      legend:{{orientation:"v",x:1.02,y:0.5,font:{{size:10,color:p.tick}},bgcolor:BG}},
      annotations:[{{text:"<b>"+total_n.toLocaleString()+"</b><br><span style='font-size:10px'>Total</span>",
                    x:0.5,y:0.5,font:{{size:13,color:p.title}},showarrow:false}}],
    }}),
  }};
}}

function buildYearly(divId) {{
  var d = aggData().yearly;
  var p = pal();
  var sev = F.severity ? [F.severity] : SEV;
  var traces = sev.map(function(s,i) {{
    var c = F.severity ? p.colors[SEV.indexOf(s)] : p.colors[i];
    return {{
      type:"bar", name:s,
      x:META.years,
      y:META.years.map(function(yr){{ return (d[yr]||{{}})[s]||0; }}),
      marker:{{color:c}},
      hovertemplate:"<b>"+s+"</b><br>%{{x}}: %{{y:,}}<extra></extra>",
    }};
  }});
  return {{
    data:traces,
    layout:Object.assign(baseLayout("Accident Trends",280),{{
      barmode:"group", bargap:0.25, bargroupgap:0.08,
      xaxis:axisStyle("Year"), yaxis:axisStyle("Count"),
    }}),
  }};
}}

function buildTOD(divId) {{
  var d = aggData().tod;
  var p = pal();
  var sev = F.severity ? [F.severity] : SEV;
  var traces = sev.map(function(s,i) {{
    var c = F.severity ? p.colors[SEV.indexOf(s)] : p.colors[i];
    return {{
      type:"bar", name:s,
      x:TOD,
      y:TOD.map(function(t){{ return (d[t]||{{}})[s]||0; }}),
      marker:{{color:c}},
      hovertemplate:"<b>"+s+"</b><br>%{{x}}: %{{y:,}}<extra></extra>",
    }};
  }});
  return {{
    data:traces,
    layout:Object.assign(baseLayout("Time of Day",280),{{
      barmode:"stack", bargap:0.3,
      xaxis:axisStyle(), yaxis:axisStyle("Count"),
    }}),
  }};
}}

function buildMonthly(divId) {{
  var d = aggData().monthly;
  var p = pal();
  var sev = F.severity ? [F.severity] : SEV;
  var traces = sev.map(function(s,i) {{
    var c = F.severity ? p.colors[SEV.indexOf(s)] : p.colors[i];
    return {{
      type:"scatter", mode:"lines+markers", name:s,
      x:MONTHS,
      y:MONTHS.map(function(m){{ return (d[m]||{{}})[s]||0; }}),
      line:{{color:c,width:2.5}}, marker:{{size:5,color:c}},
      hovertemplate:"<b>"+s+"</b><br>%{{x}}: %{{y:,}}<extra></extra>",
    }};
  }});
  return {{
    data:traces,
    layout:Object.assign(baseLayout("Monthly Pattern",280),{{
      xaxis:axisStyle(), yaxis:axisStyle("Count"),
    }}),
  }};
}}

function buildHeatmap(divId) {{
  var key = F.year+"|all";   // heatmap only stored for year|all combos
  var src = AGG[key] || AGG["all|all"];
  var mat = src ? src.heatmap : null;
  if(!mat) mat = AGG["all|all"].heatmap;
  var p = pal();
  return {{
    data:[{{
      type:"heatmap", z:mat,
      x:Array.from({{length:24}},function(_,i){{return String(i).padStart(2,"0")+":00";}}),
      y:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
      colorscale:p.hm_cs, showscale:true,
      hovertemplate:"<b>%{{y}} %{{x}}</b><br>%{{z}} accidents<extra></extra>",
      colorbar:{{len:0.85,thickness:9,tickfont:{{size:9,color:p.tick}}}},
    }}],
    layout:Object.assign(baseLayout("Hour × Day Heatmap",250,{{t:46,b:52,l:52,r:60}}),{{
      xaxis:{{tickfont:{{size:9}},tickangle:-45}},
      yaxis:{{tickfont:{{size:10}}}},
    }}),
  }};
}}

function buildProvince(divId) {{
  var d = aggData().province;
  var p = pal();
  var sev = F.severity ? [F.severity] : SEV;
  var provs = META.provinces;
  var traces = sev.map(function(s,i) {{
    var c = F.severity ? p.colors[SEV.indexOf(s)] : p.colors[i];
    return {{
      type:"bar", name:s,
      x:provs,
      y:provs.map(function(pv){{ return (d[pv]||{{}})[s]||0; }}),
      marker:{{color:c}},
      hovertemplate:"<b>"+s+"</b><br>%{{x}}: %{{y:,}}<extra></extra>",
    }};
  }});
  return {{
    data:traces,
    layout:Object.assign(baseLayout("By Province",300),{{
      barmode:"stack", bargap:0.3,
      xaxis:axisStyle(), yaxis:axisStyle("Count"),
    }}),
  }};
}}

function buildDistricts(divId) {{
  var d = aggData().district;
  var p = pal();
  var dists = META.top_districts.slice(0,12);
  var sev = F.severity ? [F.severity] : SEV;
  // compute totals for sorting
  var totals = dists.map(function(dn){{
    var nd = d[dn]||{{}};
    return sev.reduce(function(a,s){{return a+(nd[s]||0);}},0);
  }});
  var order = dists.map(function(_,i){{return i;}})
                   .sort(function(a,b){{return totals[a]-totals[b];}});
  var sortedDists = order.map(function(i){{return dists[i];}});

  var traces = sev.map(function(s,i) {{
    var c = F.severity ? p.colors[SEV.indexOf(s)] : p.colors[i];
    return {{
      type:"bar", orientation:"h", name:s,
      y:sortedDists,
      x:sortedDists.map(function(dn){{return (d[dn]||{{}})[s]||0;}}),
      marker:{{color:c}},
      hovertemplate:"<b>"+s+"</b><br>%{{y}}: %{{x:,}}<extra></extra>",
    }};
  }});
  return {{
    data:traces,
    layout:Object.assign(baseLayout("Top Districts",380,{{t:46,b:36,l:100,r:16}}),{{
      barmode:"stack",
      xaxis:axisStyle("Count"), yaxis:{{tickfont:{{size:10}}}},
    }}),
  }};
}}

function buildVehicle(divId) {{
  var d = aggData().vehicle;
  var p = pal();
  var vehs = ["Motorcycle","Car","Minibus/Taxi","Truck","Bus","Bicycle","Other"];
  var sev = F.severity ? [F.severity] : SEV;
  var totals = vehs.map(function(v){{
    var nd=d[v]||{{}};
    return sev.reduce(function(a,s){{return a+(nd[s]||0);}},0);
  }});
  var order = vehs.map(function(_,i){{return i;}})
                  .sort(function(a,b){{return totals[a]-totals[b];}});
  var sv = order.map(function(i){{return vehs[i];}});

  var traces = sev.map(function(s,i) {{
    var c = F.severity ? p.colors[SEV.indexOf(s)] : p.colors[i];
    return {{
      type:"bar", orientation:"h", name:s,
      y:sv,
      x:sv.map(function(v){{return (d[v]||{{}})[s]||0;}}),
      marker:{{color:c}},
      hovertemplate:"<b>"+s+"</b><br>%{{y}}: %{{x:,}}<extra></extra>",
    }};
  }});
  return {{
    data:traces,
    layout:Object.assign(baseLayout("By Vehicle Type",320,{{t:46,b:36,l:110,r:16}}),{{
      barmode:"stack",
      xaxis:axisStyle("Count"), yaxis:{{tickfont:{{size:10}}}},
    }}),
  }};
}}

function buildFactors(divId) {{
  var d = aggData().factors;
  var p = pal();
  var items = d.items||[];
  var baseline = d.baseline||0;
  var labels = items.map(function(x){{return x.label;}});
  var values = items.map(function(x){{return x.value;}});
  var colors = values.map(function(v){{
    if(v > baseline*1.15) return p.factor_hi;
    if(v > baseline)      return p.factor_mid;
    return p.factor_lo;
  }});
  return {{
    data:[{{
      type:"bar", orientation:"h",
      y:labels, x:values,
      marker:{{color:colors}},
      text:values.map(function(v){{return v.toFixed(2);}}),
      textposition:"outside",
      textfont:{{size:10,color:p.tick}},
      hovertemplate:"<b>%{{y}}</b><br>Mean severity: %{{x:.3f}}<extra></extra>",
    }}],
    layout:Object.assign(baseLayout("Contributing Factors",440,{{t:50,b:36,l:190,r:70}}),{{
      xaxis:Object.assign(axisStyle("Mean severity score (0–3)"),{{range:[0,1.65]}}),
      yaxis:{{tickfont:{{size:10,color:p.tick}}}},
      shapes:[{{
        type:"line", x0:baseline, x1:baseline, y0:-0.5, y1:labels.length-0.5,
        line:{{color:p.anno, dash:"dot", width:1.5}},
      }}],
      annotations:[{{
        x:baseline, y:labels.length-0.3, xanchor:"left",
        text:"avg "+baseline.toFixed(2), showarrow:false,
        font:{{size:10,color:p.anno}},
      }}],
    }}),
  }};
}}

function buildModel(divId) {{
  var p = pal();
  var RF  = MODEL["Random Forest"];
  var XGB = MODEL["XGBoost"];
  var metrics = [
    {{label:"RF Precision",  vals:SEV.map(function(s){{return RF[s].precision;}}),  color:p.colors[0], dash:"solid"}},
    {{label:"RF Recall",     vals:SEV.map(function(s){{return RF[s].recall;}}),     color:p.colors[1], dash:"dot"}},
    {{label:"RF F1",         vals:SEV.map(function(s){{return RF[s].f1;}}),         color:p.colors[2], dash:"dash"}},
    {{label:"XGB Precision", vals:SEV.map(function(s){{return XGB[s].precision;}}), color:p.colors[3], dash:"solid"}},
    {{label:"XGB Recall",    vals:SEV.map(function(s){{return XGB[s].recall;}}),    color:p.colors[4], dash:"dot"}},
    {{label:"XGB F1",        vals:SEV.map(function(s){{return XGB[s].f1;}}),        color:p.colors[5], dash:"dash"}},
  ];
  var traces = metrics.map(function(m) {{
    return {{
      type:"scatter", mode:"lines+markers", name:m.label,
      x:SEV, y:m.vals,
      line:{{color:m.color, width:2, dash:m.dash}}, marker:{{size:6}},
      hovertemplate:"<b>"+m.label+"</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
    }};
  }});
  return {{
    data:traces,
    layout:Object.assign(baseLayout("Model Performance",340),{{
      xaxis:axisStyle("Severity Class"),
      yaxis:Object.assign(axisStyle("Score"),{{range:[0,0.75]}}),
    }}),
  }};
}}

// ═══════════════════════════════════════════════════════════════
// CHART REGISTRY — maps div IDs to builder functions
// ═══════════════════════════════════════════════════════════════
var BUILDERS = {{
  "chart-donut":       buildDonut,
  "chart-yearly":      buildYearly,
  "chart-tod":         buildTOD,
  "chart-monthly":     buildMonthly,
  "chart-heatmap":     buildHeatmap,
  "chart-districts":   buildDistricts,
  "chart-province":    buildProvince,
  "chart-vehicle":     buildVehicle,
  "chart-factors":     buildFactors,
  "chart-model":       buildModel,
  // duplicate IDs in other sections
  "chart-yearly-r":    buildYearly,
  "chart-monthly-r":   buildMonthly,
  "chart-districts-g": buildDistricts,
  "chart-province-g":  buildProvince,
  "chart-yearly-t":    buildYearly,
  "chart-monthly-t":   buildMonthly,
  "chart-heatmap-t":   buildHeatmap,
  "chart-tod-t":       buildTOD,
  "chart-donut-s":     buildDonut,
  "chart-province-s":  buildProvince,
  "chart-vehicle-s":   buildVehicle,
  "chart-factors-f":   buildFactors,
  "chart-vehicle-f":   buildVehicle,
}};

// ═══════════════════════════════════════════════════════════════
// RENDER
// ═══════════════════════════════════════════════════════════════
function renderChart(id) {{
  var el = document.getElementById(id);
  if(!el || el.offsetParent === null) return;  // skip hidden
  var builder = BUILDERS[id];
  if(!builder) return;
  var fig = builder(id);
  Plotly.react(id, fig.data, fig.layout, PLOTLY_CFG);
}}

function renderAllCharts() {{
  // Re-render every chart whose div exists and is visible in the DOM
  Object.keys(BUILDERS).forEach(renderChart);
}}

function renderVisibleCharts() {{
  // Only render charts inside the currently-visible section panel
  var activeSec = document.querySelector(".section-panel:not(.hidden)");
  if(!activeSec) return;
  Object.keys(BUILDERS).forEach(function(id) {{
    var el = document.getElementById(id);
    if(el && activeSec.contains(el)) renderChart(id);
  }});
}}

// ═══════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════
var SECTION_TITLES = {{
  "overview":  "Road Safety Overview",
  "reports":   "Accident Reports",
  "risk":      "Risk Analysis",
  "geo":       "Geographic Analysis",
  "trends":    "Accident Trends",
  "severity":  "Severity Analysis",
  "factors":   "Contributing Factors",
  "model":     "ML Predictions",
  "about":     "About Project",
}};

var SEC_TO_DIV = {{
  "overview":"sec-overview","reports":"sec-reports","risk":"sec-risk",
  "geo":"sec-geo","trends":"sec-trends","severity":"sec-severity",
  "factors":"sec-factors","model":"sec-model","about":"sec-about",
}};

var currentSection = "overview";

function showSection(name, btn) {{
  // Hide all sections
  document.querySelectorAll(".section-panel").forEach(function(el) {{
    el.classList.add("hidden");
  }});
  // Show target
  var target = document.getElementById(SEC_TO_DIV[name]);
  if(target) target.classList.remove("hidden");

  // Update nav active state
  document.querySelectorAll(".nav-item").forEach(function(el) {{
    el.classList.remove("nav-active");
  }});
  if(btn) btn.classList.add("nav-active");

  // Update page title
  var titleEl = document.getElementById("page-title");
  if(titleEl) titleEl.textContent = SECTION_TITLES[name] || name;

  currentSection = name;

  // Render charts now visible
  setTimeout(renderVisibleCharts, 60);

  // Close mobile sidebar
  if(window.innerWidth < 1024) closeSidebar();
}}

// ═══════════════════════════════════════════════════════════════
// SIDEBAR (mobile)
// ═══════════════════════════════════════════════════════════════
function toggleSidebar() {{
  var sb = document.getElementById("sidebar");
  var bd = document.getElementById("backdrop");
  var open = !sb.classList.contains("-translate-x-full");
  if(open) {{
    sb.classList.add("-translate-x-full");
    bd.classList.add("hidden");
  }} else {{
    sb.classList.remove("-translate-x-full");
    bd.classList.remove("hidden");
  }}
}}
function closeSidebar() {{
  document.getElementById("sidebar").classList.add("-translate-x-full");
  document.getElementById("backdrop").classList.add("hidden");
}}

// ═══════════════════════════════════════════════════════════════
// THEME
// ═══════════════════════════════════════════════════════════════
function toggleTheme() {{
  applyTheme(isDark() ? "light" : "dark");
}}
function applyTheme(t) {{
  var root = document.documentElement;
  if(t === "dark") {{
    root.classList.add("dark");
    root.setAttribute("data-theme","dark");
  }} else {{
    root.classList.remove("dark");
    root.setAttribute("data-theme","light");
  }}
  localStorage.setItem("rs-theme", t);
  document.getElementById("icon-moon").classList.toggle("hidden", t==="dark");
  document.getElementById("icon-sun").classList.toggle("hidden", t==="light");
  renderAllCharts();
}}

// ═══════════════════════════════════════════════════════════════
// REFRESH
// ═══════════════════════════════════════════════════════════════
function refreshCharts() {{
  var icon = document.getElementById("refresh-icon");
  if(icon) {{
    icon.style.transform = "rotate(360deg)";
    icon.style.transition = "transform 0.5s ease";
    setTimeout(function() {{ icon.style.transform=""; icon.style.transition=""; }}, 520);
  }}
  renderAllCharts();
}}

// ═══════════════════════════════════════════════════════════════
// DATE
// ═══════════════════════════════════════════════════════════════
function updateDate() {{
  var el = document.getElementById("date-badge");
  if(el) el.textContent = new Date().toLocaleDateString("en-GB",
    {{weekday:"short",year:"numeric",month:"short",day:"numeric"}});
}}

// ═══════════════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", function() {{
  // Theme (no FOUC — applied before paint via inline script below)
  var saved = localStorage.getItem("rs-theme") || "light";
  applyTheme(saved);

  updateDate();
  setInterval(updateDate, 60000);

  // Initial KPI animation
  animCount("kpi-total",   GKPI.total,   false);
  animCount("kpi-fatal",   GKPI.fatal,   false);
  animCount("kpi-serious", GKPI.serious, false);
  animCount("kpi-f1",      GKPI.rf_f1,  true);

  // Render charts for default section (overview)
  setTimeout(renderVisibleCharts, 100);

  // Resize handler
  window.addEventListener("resize", function() {{
    Object.keys(BUILDERS).forEach(function(id) {{
      var el = document.getElementById(id);
      if(el && el.offsetParent !== null) Plotly.relayout(id, {{autosize:true}});
    }});
  }});
}});
</script>"""


# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

# Anti-FOUC inline script (runs before DOMContentLoaded)
ANTI_FOUC = """\
<script>
(function(){
  var t = localStorage.getItem("rs-theme") || "light";
  if(t === "dark") {
    document.documentElement.classList.add("dark");
    document.documentElement.setAttribute("data-theme","dark");
  }
})();
</script>"""

def build_full_html():
    js = build_js(agg_json, model_json, rank_json, kpi_json, meta_json)
    return "\n".join([
        HEAD,
        ANTI_FOUC,
        "<body class='bg-bg-light dark:bg-bg-dark min-h-screen font-sans'>",
        SIDEBAR,
        # Main wrapper — offset by sidebar on large screens
        "<div class='lg:ml-60 flex flex-col min-h-screen'>",
        HEADER,
        # Page content — filter bar is OUTSIDE sections so it persists across nav
        "<main class='flex-1 p-4 md:p-5'>",
        FILTER_BAR,          # single shared filter bar, unique IDs
        ALL_SECTIONS,
        FOOTER,
        "</main>",
        "</div>",  # lg:ml-60
        js,
        "</body>",
        "</html>",
    ])

print("Assembling HTML…")
OUT.parent.mkdir(parents=True, exist_ok=True)
html = build_full_html()
OUT.write_text(html, encoding="utf-8")
size_kb = OUT.stat().st_size / 1024
print(f"Written: {OUT}")
print(f"Size:    {size_kb:.0f} KB")
