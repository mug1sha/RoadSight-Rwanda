"""
ML Model — RoadSight Rwanda — Accident Severity Classification
Group D | Big Data Project

Trains Random Forest and XGBoost classifiers on the cleaned dataset.
Saves:
  - outputs/models/rf_model.pkl
  - outputs/models/xgb_model.pkl
  - outputs/models/label_encoder.pkl
  - outputs/figures/confusion_matrix_rf.png
  - outputs/figures/confusion_matrix_xgb.png
  - outputs/figures/feature_importance.png
  - outputs/models/model_results.json
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
DATA = BASE / "data" / "processed" / "accidents_clean.csv"
MODELS = BASE / "outputs" / "models"
FIGS = BASE / "outputs" / "figures"
MODELS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"Loaded {len(df)} rows")

# ── Feature selection ─────────────────────────────────────────────────────────
FEATURES = [
    "crash_hour", "crash_month", "crash_dow", "is_weekend",
    "speed_limit_kmh", "vehicles_involved", "driver_age",
    "road_class", "road_surface", "road_condition", "junction",
    "weather", "lighting", "vehicle_primary",
    "alcohol_suspected", "speeding_suspected", "driver_gender",
    "helmet_worn", "province", "time_of_day", "age_group",
]
TARGET = "severity"

df_model = df[FEATURES + [TARGET]].copy()

# ── Encode categoricals ───────────────────────────────────────────────────────
label_encoders = {}
cat_cols = df_model.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]

for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

# Encode target
target_le = LabelEncoder()
# Fix order: Damage only, Minor injury, Serious/grievous, Fatal
class_order = ["Damage only", "Minor injury", "Serious/grievous", "Fatal"]
target_le.fit(class_order)
df_model[TARGET] = target_le.transform(df_model[TARGET])
label_encoders["__target__"] = target_le

with open(MODELS / "label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"Class distribution:\n{pd.Series(target_le.inverse_transform(y)).value_counts()}\n")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── SMOTE on training set only ────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(target_le.inverse_transform(y_train_sm)).value_counts().to_dict()}\n")

class_names = target_le.classes_

results = {}

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    report = classification_report(y_te, y_pred, target_names=class_names, output_dict=True)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    print(f"=== {name} ===")
    print(classification_report(y_te, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=20)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_").replace("/", "_")
    fig.savefig(FIGS / f"confusion_matrix_{fname}.png", dpi=150)
    plt.close()

    results[name] = {
        "macro_f1": round(macro_f1, 4),
        "report": report,
    }
    return model

# ── Random Forest ─────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf = evaluate("Random Forest", rf, X_train_sm, y_train_sm, X_test, y_test)
with open(MODELS / "rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# ── XGBoost ───────────────────────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, n_jobs=-1
)
xgb = evaluate("XGBoost", xgb, X_train_sm, y_train_sm, X_test, y_test)
with open(MODELS / "xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

# ── Feature importance (from RF) ──────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#c0392b" if i >= len(importances) - 5 else "#2980b9" for i in range(len(importances))]
importances.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Mean Decrease in Impurity")
ax.axvline(importances.mean(), color="gray", linestyle="--", alpha=0.6, label="Mean")
ax.legend()
plt.tight_layout()
fig.savefig(FIGS / "feature_importance.png", dpi=150)
plt.close()
print("Feature importance chart saved.")

# ── Save results JSON ─────────────────────────────────────────────────────────
# Make JSON-serialisable
def make_serialisable(obj):
    if isinstance(obj, dict):
        return {k: make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

with open(MODELS / "model_results.json", "w") as f:
    json.dump(make_serialisable(results), f, indent=2)

print("\nAll model artifacts saved.")
print(f"  RF macro-F1:  {results['Random Forest']['macro_f1']}")
print(f"  XGB macro-F1: {results['XGBoost']['macro_f1']}")
