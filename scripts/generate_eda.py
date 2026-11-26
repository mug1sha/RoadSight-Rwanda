"""
EDA Chart Generation — RoadSight Rwanda
Group D | Big Data Project
Saves all EDA figures to outputs/figures/
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
DATA = BASE / "data" / "processed" / "accidents_clean.csv"
FIGS = BASE / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["crash_datetime"] = pd.to_datetime(df["crash_datetime"])

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548"]
SEV_ORDER = ["Damage only", "Minor injury", "Serious/grievous", "Fatal"]
SEV_COLORS = {"Damage only": "#4CAF50", "Minor injury": "#FF9800",
              "Serious/grievous": "#FF5722", "Fatal": "#9C27B0"}

sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13, "axes.titleweight": "bold"})

# ── 1. Severity distribution ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = df["severity"].value_counts().reindex(SEV_ORDER)
colors = [SEV_COLORS[s] for s in SEV_ORDER]
axes[0].bar(SEV_ORDER, counts.values, color=colors, edgecolor="white", linewidth=0.5)
axes[0].set_title("Accident Severity Distribution")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(SEV_ORDER, rotation=15, ha="right")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 10, str(v), ha="center", fontsize=10)

pcts = counts / counts.sum() * 100
axes[1].pie(pcts.values, labels=SEV_ORDER, autopct="%1.1f%%",
            colors=colors, startangle=140, wedgeprops={"edgecolor": "white"})
axes[1].set_title("Severity Share (%)")
plt.tight_layout()
fig.savefig(FIGS / "01_severity_distribution.png", bbox_inches="tight")
plt.close()
print("Saved 01_severity_distribution.png")

# ── 2. Accidents by hour of day ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
hourly = df.groupby(["crash_hour", "severity"]).size().unstack(fill_value=0)
hourly = hourly.reindex(columns=SEV_ORDER, fill_value=0)
hourly.plot(kind="bar", ax=ax, color=colors, stacked=True, edgecolor="white", width=0.85)
ax.set_title("Accidents by Hour of Day")
ax.set_xlabel("Hour (0–23)")
ax.set_ylabel("Count")
ax.set_xticklabels([str(h) for h in range(24)], rotation=0, fontsize=8)
ax.legend(title="Severity", loc="upper left", fontsize=8)
plt.tight_layout()
fig.savefig(FIGS / "02_accidents_by_hour.png", bbox_inches="tight")
plt.close()
print("Saved 02_accidents_by_hour.png")

# ── 3. Accidents by day of week ───────────────────────────────────────────────
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
fig, ax = plt.subplots(figsize=(10, 5))
dow_counts = df["crash_dow_name"].value_counts().reindex(dow_order)
bar_colors = ["#FF5722" if d in ["Saturday", "Sunday"] else "#2196F3" for d in dow_order]
ax.bar(dow_order, dow_counts.values, color=bar_colors, edgecolor="white")
ax.set_title("Accidents by Day of Week  (orange = weekend)")
ax.set_ylabel("Count")
ax.set_xticklabels(dow_order, rotation=20, ha="right")
for i, v in enumerate(dow_counts.values):
    ax.text(i, v + 5, str(v), ha="center", fontsize=9)
plt.tight_layout()
fig.savefig(FIGS / "03_accidents_by_dow.png", bbox_inches="tight")
plt.close()
print("Saved 03_accidents_by_dow.png")

# ── 4. Accidents by month ─────────────────────────────────────────────────────
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(10, 5))
monthly = df["crash_month"].value_counts().sort_index()
ax.plot(range(1, 13), [monthly.get(m, 0) for m in range(1, 13)],
        marker="o", color="#2196F3", linewidth=2.5, markersize=7)
ax.fill_between(range(1, 13), [monthly.get(m, 0) for m in range(1, 13)], alpha=0.15, color="#2196F3")
ax.set_title("Accident Frequency by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Count")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
plt.tight_layout()
fig.savefig(FIGS / "04_accidents_by_month.png", bbox_inches="tight")
plt.close()
print("Saved 04_accidents_by_month.png")

# ── 5. Severity by province ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
prov_sev = df.groupby(["province", "severity"]).size().unstack(fill_value=0)
prov_sev = prov_sev.reindex(columns=SEV_ORDER, fill_value=0)
prov_sev.plot(kind="bar", ax=ax, color=colors, stacked=False, edgecolor="white")
ax.set_title("Accident Severity by Province")
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.set_xticklabels(prov_sev.index, rotation=15, ha="right")
ax.legend(title="Severity", fontsize=8)
plt.tight_layout()
fig.savefig(FIGS / "05_severity_by_province.png", bbox_inches="tight")
plt.close()
print("Saved 05_severity_by_province.png")

# ── 6. Severity by vehicle type ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
top_vehicles = df["vehicle_primary"].value_counts().head(7).index
veh_sev = df[df["vehicle_primary"].isin(top_vehicles)].groupby(
    ["vehicle_primary", "severity"]).size().unstack(fill_value=0)
veh_sev = veh_sev.reindex(columns=SEV_ORDER, fill_value=0)
veh_sev.plot(kind="bar", ax=ax, color=colors, stacked=True, edgecolor="white")
ax.set_title("Accident Severity by Primary Vehicle Type")
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.set_xticklabels(veh_sev.index, rotation=20, ha="right")
ax.legend(title="Severity", fontsize=8)
plt.tight_layout()
fig.savefig(FIGS / "06_severity_by_vehicle.png", bbox_inches="tight")
plt.close()
print("Saved 06_severity_by_vehicle.png")

# ── 7. Driver age distribution by severity ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for sev, color in SEV_COLORS.items():
    subset = df[df["severity"] == sev]["driver_age"].dropna()
    ax.hist(subset, bins=20, alpha=0.55, color=color, label=sev, edgecolor="white")
ax.set_title("Driver Age Distribution by Severity")
ax.set_xlabel("Driver Age")
ax.set_ylabel("Count")
ax.legend(title="Severity", fontsize=8)
plt.tight_layout()
fig.savefig(FIGS / "07_driver_age_severity.png", bbox_inches="tight")
plt.close()
print("Saved 07_driver_age_severity.png")

# ── 8. Weather and road condition vs severity (heatmap) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title in [
    (axes[0], "weather", "Weather"),
    (axes[1], "road_condition", "Road Condition"),
]:
    pivot = df.groupby([col, "severity"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=SEV_ORDER, fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    sns.heatmap(pivot_pct, ax=ax, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "% of row"})
    ax.set_title(f"Severity % by {title}")
    ax.set_xlabel("Severity")
    ax.set_ylabel(col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
fig.savefig(FIGS / "08_weather_road_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved 08_weather_road_heatmap.png")

# ── 9. Alcohol & speeding vs severity ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, title in [
    (axes[0], "alcohol_suspected", "Alcohol Suspected"),
    (axes[1], "speeding_suspected", "Speeding Suspected"),
]:
    ct = df.groupby([col, "severity"]).size().unstack(fill_value=0)
    ct = ct.reindex(columns=SEV_ORDER, fill_value=0)
    ct.plot(kind="bar", ax=ax, color=colors, stacked=True, edgecolor="white")
    ax.set_title(f"Severity by {title}")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ct.index, rotation=0)
    ax.legend(title="Severity", fontsize=7)
plt.tight_layout()
fig.savefig(FIGS / "09_risk_factors.png", bbox_inches="tight")
plt.close()
print("Saved 09_risk_factors.png")

# ── 10. Lighting condition vs severity ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
light_sev = df.groupby(["lighting", "severity"]).size().unstack(fill_value=0)
light_sev = light_sev.reindex(columns=SEV_ORDER, fill_value=0)
light_pct = light_sev.div(light_sev.sum(axis=1), axis=0) * 100
light_pct.plot(kind="barh", ax=ax, color=colors, stacked=True, edgecolor="white")
ax.set_title("Severity Distribution by Lighting Condition (%)")
ax.set_xlabel("Percentage")
ax.set_ylabel("")
ax.legend(title="Severity", fontsize=8, loc="lower right")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
fig.savefig(FIGS / "10_lighting_severity.png", bbox_inches="tight")
plt.close()
print("Saved 10_lighting_severity.png")

# ── 11. Correlation heatmap (numeric features) ───────────────────────────────
num_cols = ["crash_hour", "crash_month", "crash_dow", "is_weekend",
            "speed_limit_kmh", "vehicles_involved", "driver_age",
            "severity_code", "persons_killed", "persons_injured"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, square=True)
ax.set_title("Correlation Matrix — Numeric Features")
plt.tight_layout()
fig.savefig(FIGS / "11_correlation_matrix.png", bbox_inches="tight")
plt.close()
print("Saved 11_correlation_matrix.png")

print("\nAll EDA charts saved to:", FIGS)
