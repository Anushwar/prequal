#!/usr/bin/env python
"""
Export all EDA visualizations to artifacts/images/eda/
Modified from src/eda.py to save plots instead of showing them
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    IMAGES_EDA_DIR,
    TEST_CSV,
    TEST_PROCESSED_CSV,
    TRAIN_CSV,
    TRAIN_PROCESSED_CSV,
)

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Ensure output directory exists
IMAGES_EDA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EDA Visualization Export")
print("=" * 70)
print(f"Output directory: {IMAGES_EDA_DIR}\n")

# Load Data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Training: {train_df.shape}")
print(f"Test: {test_df.shape}\n")

# 1. Target Distribution
fig, ax = plt.subplots(figsize=(8, 5))
vc = train_df["loan_status"].value_counts().reindex([0, 1])
labels = ["Denied (0)", "Approved (1)"]
colors = ["#2ecc71", "#e74c3c"]

bars = ax.bar(labels, vc.values, color=colors)
ax.set_title("Loan Status Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Loan Status")
ax.set_ylabel("Count")

total = vc.sum()
for bar, cnt in zip(bars, vc.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{(cnt/total)*100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
    )

plt.tight_layout()
plt.savefig(IMAGES_EDA_DIR / "target_distribution.png", dpi=100, bbox_inches="tight")
plt.close()
print("✓ Saved: target_distribution.png")

# Prepare features
continuous_cols = (
    train_df.select_dtypes(include=["int64", "float64"])
    .columns.difference(["id", "loan_status"])
    .tolist()
)
categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

# One-hot encoding
train_ohe = pd.get_dummies(train_df[categorical_cols], drop_first=True, dtype="uint8")
test_ohe = pd.get_dummies(test_df[categorical_cols], drop_first=True, dtype="uint8")
test_ohe = test_ohe.reindex(columns=train_ohe.columns, fill_value=0)

train_df[train_ohe.columns] = train_ohe
test_df[train_ohe.columns] = test_ohe
ohe_cols = list(train_ohe.columns)
numerical_cols = [c for c in continuous_cols if c in train_df.columns]

# 2. Numerical Features Distribution
plot_cols = [c for c in numerical_cols if c not in ohe_cols]
n = len(plot_cols)
ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
axes = np.array(axes).reshape(-1)

for idx, col in enumerate(plot_cols):
    axes[idx].hist(train_df[col].dropna(), bins=50, edgecolor="black", alpha=0.7)
    axes[idx].set_title(col, fontsize=11, fontweight="bold")
    axes[idx].set_ylabel("Frequency")

for ax in axes[n:]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(IMAGES_EDA_DIR / "numerical_distributions.png", dpi=100, bbox_inches="tight")
plt.close()
print("✓ Saved: numerical_distributions.png")

# 3. Correlation Matrix (before dropping person_age)
corr_cols = numerical_cols + ["loan_status"]
corr_matrix = train_df[corr_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5
)
plt.title("Correlation Matrix (Before Feature Engineering)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(IMAGES_EDA_DIR / "correlation_matrix_before.png", dpi=100, bbox_inches="tight")
plt.close()
print("✓ Saved: correlation_matrix_before.png")

# Drop person_age due to multicollinearity
TO_DROP = ["person_age"]
for df in (train_df, test_df):
    df.drop(columns=[c for c in TO_DROP if c in df.columns], inplace=True, errors="ignore")

continuous_cols = [c for c in continuous_cols if c not in TO_DROP]
numerical_cols = [c for c in numerical_cols if c not in TO_DROP]

# 4. Correlation Matrix (after dropping person_age)
corr_cols = numerical_cols + ["loan_status"]
corr_matrix = train_df[corr_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5
)
plt.title("Correlation Matrix (After Dropping person_age)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(IMAGES_EDA_DIR / "correlation_matrix_after.png", dpi=100, bbox_inches="tight")
plt.close()
print("✓ Saved: correlation_matrix_after.png")

# 5. Feature Engineering
for df in (train_df, test_df):
    df["interest_to_income_ratio"] = df["loan_int_rate"] / df["person_income"]
    df["debt_to_income_ratio"] = df["loan_amnt"] / df["person_income"]

continuous_cols = [c for c in continuous_cols if c in train_df.columns]

# 6. Feature Transformation
for col in ["person_income", "interest_to_income_ratio", "debt_to_income_ratio"]:
    train_df[f"{col}_log"] = np.log1p(train_df[col])
    test_df[f"{col}_log"] = np.log1p(test_df[col])

for col in ["person_emp_length", "loan_amnt", "cb_person_cred_hist_length"]:
    train_df[f"{col}_sqrt"] = np.sqrt(train_df[col])
    test_df[f"{col}_sqrt"] = np.sqrt(test_df[col])

# 7. Transformation Comparisons
cols_to_check = [
    "person_income",
    "person_income_log",
    "interest_to_income_ratio",
    "interest_to_income_ratio_log",
    "person_emp_length",
    "person_emp_length_sqrt",
]

for idx in range(0, len(cols_to_check), 2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(train_df[cols_to_check[idx]], kde=True, ax=ax[0])
    ax[0].set_title(f"Before: {cols_to_check[idx]}")
    sns.histplot(train_df[cols_to_check[idx + 1]], kde=True, ax=ax[1])
    ax[1].set_title(f"After: {cols_to_check[idx+1]}")
    plt.tight_layout()
    transform_name = cols_to_check[idx].replace("_", " ").title()
    filename = f"transformation_{cols_to_check[idx]}.png"
    plt.savefig(IMAGES_EDA_DIR / filename, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {filename}")

# 8. Numerical Features by Target (Boxplots)
plot_cols = [
    c
    for c in [
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "debt_to_income_ratio",
        "interest_to_income_ratio",
    ]
    if c in train_df.columns
]

n_cols = 3
n_rows = int(np.ceil(len(plot_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = np.array(axes).reshape(-1)

for idx, col in enumerate(plot_cols):
    data = [
        train_df.loc[train_df["loan_status"] == 0, col].dropna(),
        train_df.loc[train_df["loan_status"] == 1, col].dropna(),
    ]
    bp = axes[idx].boxplot(data, labels=["Denied", "Approved"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#e74c3c", "#2ecc71"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[idx].set_title(col, fontsize=11, fontweight="bold")

for ax in axes[len(plot_cols) :]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(IMAGES_EDA_DIR / "boxplots_by_target.png", dpi=100, bbox_inches="tight")
plt.close()
print("✓ Saved: boxplots_by_target.png")

# 9. Categorical Features vs Target
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols):
    ct = pd.crosstab(train_df[col], train_df["loan_status"], normalize="index") * 100
    ct.plot(kind="bar", ax=axes[idx], color=["#e74c3c", "#2ecc71"], alpha=0.8)
    axes[idx].set_title(f"{col} vs Loan Status", fontsize=12, fontweight="bold")
    axes[idx].set_ylabel("Percentage")
    axes[idx].legend(["Denied", "Approved"])
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig(IMAGES_EDA_DIR / "categorical_vs_target.png", dpi=100, bbox_inches="tight")
plt.close()
print("✓ Saved: categorical_vs_target.png")

# 10. PCA Analysis
numeric_df = train_df.select_dtypes(include=[np.number]).drop(
    columns=["loan_status"] + (ohe_cols if "ohe_cols" in globals() else []),
    errors="ignore",
)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

pca = PCA()
pca_data = pca.fit_transform(scaled_data)

explained_var = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(explained_var) + 1),
    explained_var,
    marker="o",
    linestyle="--",
    color="b",
)
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig(IMAGES_EDA_DIR / "pca_variance.png", dpi=100, bbox_inches="tight")
plt.close()
print("✓ Saved: pca_variance.png")

# 11. PCA Scatter Plot
pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(pca_data.shape[1])])
pca_df["loan_status"] = train_df["loan_status"].values

plt.figure(figsize=(8, 6))
plt.scatter(
    pca_df[pca_df["loan_status"] == 0]["PC1"],
    pca_df[pca_df["loan_status"] == 0]["PC2"],
    label="Denied",
    alpha=0.6,
    c="#e74c3c",
)
plt.scatter(
    pca_df[pca_df["loan_status"] == 1]["PC1"],
    pca_df[pca_df["loan_status"] == 1]["PC2"],
    label="Approved",
    alpha=0.6,
    c="#2ecc71",
)
plt.title("PCA Scatter Plot (PC1 vs PC2)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(IMAGES_EDA_DIR / "pca_scatter.png", dpi=100, bbox_inches="tight")
plt.close()
print("✓ Saved: pca_scatter.png")

print("\n" + "=" * 70)
print("All EDA visualizations exported successfully!")
print(f"Total plots saved: 11")
print(f"Location: {IMAGES_EDA_DIR}")
print("=" * 70)

# Save processed data for NN training
print("\n" + "=" * 70)
print("Saving Processed Data for NN Training")
print("=" * 70)

train_df.to_csv(TRAIN_PROCESSED_CSV, index=False)
test_df.to_csv(TEST_PROCESSED_CSV, index=False)

print(f"Saved train_processed.csv: {TRAIN_PROCESSED_CSV}")
print(f"  Shape: {train_df.shape}")
print(f"Saved test_processed.csv: {TEST_PROCESSED_CSV}")
print(f"  Shape: {test_df.shape}")
print("=" * 70)
