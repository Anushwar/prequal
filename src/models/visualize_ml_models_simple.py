#!/usr/bin/env python
"""
Generate visualizations for traditional ML models (LightGBM, CatBoost)
Creates ROC curves and feature importance plots
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.config import IMAGES_ML_DIR, TRAIN_CSV
from src.models.catboost_model import LoanPredictionSystem as CatBoostSystem
from src.models.lightgbm_model import LoanPredictionSystem as LightGBMSystem

warnings.filterwarnings("ignore")

# Ensure output directory exists
IMAGES_ML_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Traditional ML Model Visualizations")
print("=" * 70)
print(f"Output directory: {IMAGES_ML_DIR}\n")

# Load training data
df = pd.read_csv(TRAIN_CSV)
print(f"Loaded training data: {df.shape}\n")


def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.title(f"{model_name} - Top {top_n} Feature Importances", fontweight="bold")
        plt.barh(range(top_n), importances[indices], align="center")
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        plt.savefig(IMAGES_ML_DIR / filename, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {filename}")


def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
    plt.savefig(IMAGES_ML_DIR / filename, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Denied", "Approved"],
        yticklabels=["Denied", "Approved"],
    )
    plt.title(f"{model_name} - Confusion Matrix", fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(IMAGES_ML_DIR / filename, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {filename}")


# 1. LightGBM Visualizations
print("\n[1/2] Generating LightGBM visualizations...")
lgbm_system = LightGBMSystem()
X = lgbm_system.prepare_features(df, fit=True)
y = df["loan_status"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_scaled = lgbm_system.scaler.fit_transform(X_train)
X_val_scaled = lgbm_system.scaler.transform(X_val)

smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

from lightgbm import LGBMClassifier
lgbm_system.model = LGBMClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    random_state=42,
    verbose=-1,
)
lgbm_system.model.fit(X_train_scaled, y_train)
y_pred = lgbm_system.model.predict(X_val_scaled)
y_pred_proba = lgbm_system.model.predict_proba(X_val_scaled)[:, 1]

plot_feature_importance(lgbm_system.model, lgbm_system.feature_columns, "LightGBM")
plot_roc_curve(y_val, y_pred_proba, "LightGBM")
plot_confusion_matrix(y_val, y_pred, "LightGBM")

# 2. CatBoost Visualizations
print("\n[2/2] Generating CatBoost visualizations...")
catboost_system = CatBoostSystem()
X = catboost_system.prepare_features(df, fit=True)
y = df["loan_status"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_scaled = catboost_system.scaler.fit_transform(X_train)
X_val_scaled = catboost_system.scaler.transform(X_val)

smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

from catboost import CatBoostClassifier
catboost_system.model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=False,
)
catboost_system.model.fit(X_train_scaled, y_train, verbose=False)
y_pred = catboost_system.model.predict(X_val_scaled)
y_pred_proba = catboost_system.model.predict_proba(X_val_scaled)[:, 1]

plot_feature_importance(catboost_system.model, catboost_system.feature_columns, "CatBoost")
plot_roc_curve(y_val, y_pred_proba, "CatBoost")
plot_confusion_matrix(y_val, y_pred, "CatBoost")

print("\n" + "=" * 70)
print("All traditional ML visualizations generated successfully!")
print(f"Total plots saved: 6")
print(f"Location: {IMAGES_ML_DIR}")
print("=" * 70)
