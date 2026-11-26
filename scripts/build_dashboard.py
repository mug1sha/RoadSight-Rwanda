"""
Interactive Plotly Dashboard — RoadSight Rwanda
Group D | Big Data Project

Generates a self-contained HTML dashboard: dashboard/accident_risk_dashboard.html
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = Path(__file__).parent.parent
DATA = BASE / "data" / "processed" / "accidents_clean.csv"
MODELS_DIR = BASE / "outputs" / "models"
DASH_DIR = BASE / "dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["crash_datetime"] = pd.to_datetime(df["crash_datetime"])

SEV_ORDER = ["Damage only", "Minor injury", "Serious/grievous", "Fatal"]
SEV_COLORS = {
    "Damage only": "#4CAF50",
    "Minor injury": "#FF9800",
    "Serious/grievous": "#FF5722",
    "Fatal": "#9C27B0",
}
PLOTLY_TEMPLATE = "plotly_white"

# Load model results
with open(MODELS_DIR / "model_results.json") as f:
    model_results = json.load(f)

figures = []

# ── Title banner ──────────────────────────────────────────────────────────────
# (handled via HTML)

# ── Fig 1: Severity donut ─────────────────────────────────────────────────────
sev_counts = df["severity"].value_counts().reindex(SEV_ORDER)
fig1 = go.Figure(go.Pie(
    labels=sev_counts.index,
    values=sev_counts.values,
    hole=0.45,
    marker_colors=[SEV_COLORS[s] for s in SEV_ORDER],
    textinfo="label+percent",
    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
))
fig1.update_layout(
    title="Accident Severity Distribution (n=3,007)",
    template=PLOTLY_TEMPLATE,
    legend=dict(orientation="h", y=-0.1),
    height=420,
)
figures.append(("severity_donut", fig1))

# ── Fig 2: Hourly heatmap by day-of-week ─────────────────────────────────────
dow_hour = df.groupby(["crash_dow", "crash_hour"]).size().reset_index(name="count")
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_hour["dow_name"] = dow_hour["crash_dow"].map(lambda x: dow_names[x])
pivot = dow_hour.pivot(index="dow_name", columns="crash_hour", values="count").fillna(0)
pivot = pivot.reindex(dow_names)
fig2 = go.Figure(go.Heatmap(
    z=pivot.values,
    x=[str(h) for h in pivot.columns],
    y=pivot.index,
    colorscale="YlOrRd",
    hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Accidents: %{z}<extra></extra>",
    colorbar=dict(title="Count"),
))
fig2.update_layout(
    title="Accident Frequency: Day of Week × Hour of Day",
    xaxis_title="Hour of Day",
    yaxis_title="",
    template=PLOTLY_TEMPLATE,
    height=360,
)
figures.append(("heatmap_dow_hour", fig2))

# ── Fig 3: Severity by province (grouped bar) ────────────────────────────────
prov_sev = df.groupby(["province", "severity"]).size().reset_index(name="count")
fig3 = px.bar(
    prov_sev, x="province", y="count", color="severity",
    color_discrete_map=SEV_COLORS,
    category_orders={"severity": SEV_ORDER},
    barmode="group",
    title="Accident Severity by Province",
    labels={"province": "Province", "count": "Count", "severity": "Severity"},
    template=PLOTLY_TEMPLATE,
    height=420,
)
figures.append(("severity_by_province", fig3))

# ── Fig 4: Accidents by time of day and severity ─────────────────────────────
tod_order = ["Night", "Morning", "Afternoon", "Evening", "Late evening"]
tod_sev = df.groupby(["time_of_day", "severity"]).size().reset_index(name="count")
tod_sev["time_of_day"] = pd.Categorical(tod_sev["time_of_day"], categories=tod_order, ordered=True)
tod_sev = tod_sev.sort_values("time_of_day")
fig4 = px.bar(
    tod_sev, x="time_of_day", y="count", color="severity",
    color_discrete_map=SEV_COLORS,
    category_orders={"severity": SEV_ORDER},
    barmode="stack",
    title="Accident Counts by Time of Day",
    labels={"time_of_day": "Time of Day", "count": "Count", "severity": "Severity"},
    template=PLOTLY_TEMPLATE,
    height=400,
)
figures.append(("time_of_day", fig4))

# ── Fig 5: Geographic scatter map ────────────────────────────────────────────
# Sample to keep HTML manageable; fatal always included
fatal_df = df[df["severity"] == "Fatal"]
other_df = df[df["severity"] != "Fatal"].sample(n=min(800, len(df)), random_state=42)
map_df = pd.concat([fatal_df, other_df]).drop_duplicates()

fig5 = px.scatter_map(
    map_df,
    lat="latitude", lon="longitude",
    color="severity",
    color_discrete_map=SEV_COLORS,
    category_orders={"severity": SEV_ORDER},
    hover_data={"district": True, "vehicle_primary": True, "weather": True,
                "severity": True, "latitude": False, "longitude": False},
    zoom=9,
    center={"lat": -1.95, "lon": 30.06},
    title="Accident Locations (Kigali focus — fatal always shown)",
    map_style="carto-positron",
    height=520,
    opacity=0.7,
    size_max=10,
)
fig5.update_layout(template=PLOTLY_TEMPLATE, legend=dict(orientation="h", y=-0.05))
figures.append(("accident_map", fig5))

# ── Fig 6: Risk factors bar chart ────────────────────────────────────────────
risk_data = {
    "Alcohol suspected (Yes)": df[df["alcohol_suspected"] == "Yes"]["severity_code"].mean(),
    "Speeding suspected (Yes)": df[df["speeding_suspected"] == "Yes"]["severity_code"].mean(),
    "Dark night lighting": df[df["lighting"] == "Dark night"]["severity_code"].mean(),
    "Wet road": df[df["road_condition"] == "Wet"]["severity_code"].mean(),
    "Foggy weather": df[df["weather"] == "Fog"]["severity_code"].mean(),
    "Motorcycle involved": df[df["vehicle_primary"] == "Motorcycle"]["severity_code"].mean(),
    "No helmet (motorcycle)": df[(df["vehicle_primary"] == "Motorcycle") & (df["helmet_worn"] == "No")]["severity_code"].mean(),
    "Baseline (all accidents)": df["severity_code"].mean(),
}
risk_df = pd.DataFrame(list(risk_data.items()), columns=["Factor", "Mean Severity Score"])
risk_df = risk_df.sort_values("Mean Severity Score", ascending=True)
bar_colors = ["#c0392b" if v > risk_data["Baseline (all accidents)"] else "#2980b9"
              for v in risk_df["Mean Severity Score"]]
fig6 = go.Figure(go.Bar(
    x=risk_df["Mean Severity Score"],
    y=risk_df["Factor"],
    orientation="h",
    marker_color=bar_colors,
    hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    text=[f"{v:.3f}" for v in risk_df["Mean Severity Score"]],
    textposition="outside",
))
fig6.add_vline(x=risk_data["Baseline (all accidents)"], line_dash="dash",
               line_color="gray", annotation_text="Baseline")
fig6.update_layout(
    title="Mean Severity Score by Risk Factor  (0=Damage only … 3=Fatal)",
    xaxis_title="Mean Severity Score",
    yaxis_title="",
    template=PLOTLY_TEMPLATE,
    height=420,
    xaxis=dict(range=[0, 2.0]),
)
figures.append(("risk_factors", fig6))

# ── Fig 7: Vehicle type fatal rate ────────────────────────────────────────────
top_veh = df["vehicle_primary"].value_counts().head(7).index
veh_fatal = (
    df[df["vehicle_primary"].isin(top_veh)]
    .groupby("vehicle_primary")
    .apply(lambda g: (g["severity"] == "Fatal").mean() * 100)
    .reset_index(name="fatal_pct")
    .sort_values("fatal_pct", ascending=True)
)
fig7 = px.bar(
    veh_fatal, x="fatal_pct", y="vehicle_primary", orientation="h",
    title="Fatal Accident Rate by Vehicle Type (%)",
    labels={"fatal_pct": "Fatal Rate (%)", "vehicle_primary": ""},
    color="fatal_pct", color_continuous_scale="Reds",
    template=PLOTLY_TEMPLATE, height=380,
    text=veh_fatal["fatal_pct"].map("{:.1f}%".format),
)
fig7.update_traces(textposition="outside")
fig7.update_layout(coloraxis_showscale=False)
figures.append(("vehicle_fatal_rate", fig7))

# ── Fig 8: Monthly trend ──────────────────────────────────────────────────────
monthly = df.groupby(["crash_year", "crash_month", "severity"]).size().reset_index(name="count")
monthly["date"] = pd.to_datetime(
    monthly["crash_year"].astype(str) + "-" + monthly["crash_month"].astype(str).str.zfill(2) + "-01"
)
fatal_monthly = monthly[monthly["severity"] == "Fatal"].sort_values("date")
fig8 = px.line(
    fatal_monthly, x="date", y="count", markers=True,
    title="Fatal Accidents per Month Over Time",
    labels={"date": "Date", "count": "Fatal Accidents"},
    template=PLOTLY_TEMPLATE, height=360,
    color_discrete_sequence=["#9C27B0"],
)
fig8.update_traces(line_width=2.5)
figures.append(("monthly_trend", fig8))

# ── Fig 9: Model performance ─────────────────────────────────────────────────
models_comp = []
for model_name, res in model_results.items():
    report = res["report"]
    for cls in SEV_ORDER:
        if cls in report:
            models_comp.append({
                "Model": model_name,
                "Class": cls,
                "Precision": round(report[cls]["precision"], 3),
                "Recall": round(report[cls]["recall"], 3),
                "F1-Score": round(report[cls]["f1-score"], 3),
            })
perf_df = pd.DataFrame(models_comp)
fig9 = px.bar(
    perf_df, x="Class", y="F1-Score", color="Model", barmode="group",
    title="Model F1-Score by Severity Class",
    labels={"Class": "Severity Class", "F1-Score": "F1-Score"},
    template=PLOTLY_TEMPLATE, height=400,
    color_discrete_sequence=["#2196F3", "#FF5722"],
    category_orders={"Class": SEV_ORDER},
)
macro_rf  = model_results["Random Forest"]["macro_f1"]
macro_xgb = model_results["XGBoost"]["macro_f1"]
fig9.add_annotation(
    x=1.5, y=0.95, xref="paper", yref="paper",
    text=f"Macro F1 — RF: {macro_rf:.3f} | XGB: {macro_xgb:.3f}",
    showarrow=False, bgcolor="lightyellow", bordercolor="gray", borderwidth=1,
)
figures.append(("model_performance", fig9))

# ── Assemble HTML ─────────────────────────────────────────────────────────────
print("Assembling dashboard HTML...")

html_parts = []

html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rwanda Road Accident Risk Analysis — Group D Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; color: #2c3e50; }
  .header {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1565c0 100%);
    color: white; padding: 32px 40px; text-align: center;
  }
  .header h1 { font-size: 2rem; letter-spacing: 0.5px; margin-bottom: 8px; }
  .header p  { font-size: 1rem; opacity: 0.85; }
  .badge { display: inline-block; background: rgba(255,255,255,0.2);
           border-radius: 20px; padding: 4px 14px; margin: 4px; font-size: 0.85rem; }
  .kpi-row { display: flex; flex-wrap: wrap; gap: 16px;
             padding: 24px 32px 8px; justify-content: center; }
  .kpi-card { background: white; border-radius: 12px; padding: 20px 28px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; min-width: 160px; }
  .kpi-card .value { font-size: 2rem; font-weight: 700; }
  .kpi-card .label { font-size: 0.85rem; color: #666; margin-top: 4px; }
  .section { padding: 16px 32px; }
  .section-title { font-size: 1.1rem; font-weight: 600; color: #1a237e;
                   border-left: 4px solid #1565c0; padding-left: 10px; margin-bottom: 12px; }
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 20px; }
  .chart-card { background: white; border-radius: 12px; padding: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.07); overflow: hidden; }
  .chart-card-full { background: white; border-radius: 12px; padding: 16px;
                     box-shadow: 0 2px 8px rgba(0,0,0,0.07); overflow: hidden; margin-bottom: 20px; }
  .footer { text-align: center; padding: 24px; color: #888; font-size: 0.82rem;
            border-top: 1px solid #e0e0e0; margin-top: 20px; }
  .warning-banner { background: #fff8e1; border-left: 4px solid #ffc107;
                    padding: 10px 20px; margin: 0 32px 16px; border-radius: 6px;
                    font-size: 0.88rem; color: #5d4037; }
</style>
</head>
<body>
""")

# KPI counts
n_total    = len(df)
n_fatal    = (df["severity"] == "Fatal").sum()
n_serious  = (df["severity"] == "Serious/grievous").sum()
fatal_rate = n_fatal / n_total * 100
alc_pct    = (df["alcohol_suspected"] == "Yes").sum() / n_total * 100
spd_pct    = (df["speeding_suspected"] == "Yes").sum() / n_total * 100

html_parts.append(f"""
<div class="header">
  <h1>🚦 Rwanda Road Accident Risk Analysis</h1>
  <p>Interactive risk dashboard — Group D Big Data Project</p>
  <div style="margin-top:12px">
    <span class="badge">📅 2022–2025</span>
    <span class="badge">📍 Rwanda (5 Provinces)</span>
    <span class="badge">⚠️ Synthetic dataset — not real police records</span>
  </div>
</div>

<div class="warning-banner">
  ⚠️ <strong>Data notice:</strong> This dataset is synthetic/simulated, generated for educational purposes.
  Findings are illustrative and should not be used for real policy decisions without validation against official Rwanda National Police or RURA data.
</div>

<div class="kpi-row">
  <div class="kpi-card"><div class="value" style="color:#2196F3">{n_total:,}</div><div class="label">Total Accidents</div></div>
  <div class="kpi-card"><div class="value" style="color:#9C27B0">{n_fatal:,}</div><div class="label">Fatal Accidents</div></div>
  <div class="kpi-card"><div class="value" style="color:#FF5722">{n_serious:,}</div><div class="label">Serious / Grievous</div></div>
  <div class="kpi-card"><div class="value" style="color:#9C27B0">{fatal_rate:.1f}%</div><div class="label">Fatal Rate</div></div>
  <div class="kpi-card"><div class="value" style="color:#FF9800">{alc_pct:.1f}%</div><div class="label">Alcohol Suspected</div></div>
  <div class="kpi-card"><div class="value" style="color:#FF9800">{spd_pct:.1f}%</div><div class="label">Speeding Suspected</div></div>
</div>
""")


def embed(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


html_parts.append('<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n')

# Section 1 — Overview
html_parts.append('<div class="section"><div class="section-title">1 · Severity Overview &amp; Temporal Patterns</div><div class="chart-grid">')
html_parts.append(f'<div class="chart-card">{embed(fig1)}</div>')
html_parts.append(f'<div class="chart-card">{embed(fig4)}</div>')
html_parts.append('</div></div>')

# Section 2 — Temporal heatmap (full width)
html_parts.append('<div class="section"><div class="section-title">2 · When Do Accidents Happen?</div>')
html_parts.append(f'<div class="chart-card-full">{embed(fig2)}</div>')
html_parts.append(f'<div class="chart-card-full">{embed(fig8)}</div>')
html_parts.append('</div>')

# Section 3 — Map (full width)
html_parts.append('<div class="section"><div class="section-title">3 · Where Do Accidents Happen?</div>')
html_parts.append(f'<div class="chart-card-full">{embed(fig5)}</div>')
html_parts.append(f'<div class="chart-card-full">{embed(fig3)}</div>')
html_parts.append('</div>')

# Section 4 — Risk factors
html_parts.append('<div class="section"><div class="section-title">4 · Risk Factors &amp; Vehicle Analysis</div><div class="chart-grid">')
html_parts.append(f'<div class="chart-card">{embed(fig6)}</div>')
html_parts.append(f'<div class="chart-card">{embed(fig7)}</div>')
html_parts.append('</div></div>')

# Section 5 — Model performance
html_parts.append('<div class="section"><div class="section-title">5 · Machine Learning Model Performance</div>')
html_parts.append(f'<div class="chart-card-full">{embed(fig9)}</div>')
html_parts.append('</div>')

html_parts.append(f"""
<div class="footer">
  Group D — Road Accident Risk Analysis in Rwanda | Big Data Project 2026 |
  Dataset: Synthetic (not real police data) | Built with Python, Plotly, scikit-learn, XGBoost
</div>
</body>
</html>
""")

out_path = DASH_DIR / "accident_risk_dashboard.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

size_kb = out_path.stat().st_size / 1024
print(f"Dashboard saved: {out_path}  ({size_kb:.0f} KB)")
