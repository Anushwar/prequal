#!/usr/bin/env python

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
    TEST_CSV,
    TEST_PROCESSED_CSV,
    TRAIN_CSV,
    TRAIN_PROCESSED_CSV,
)

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Load Data

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Training: {train_df.shape}")
print(f"Test: {test_df.shape}")
print("Target distribution:")
print(train_df["loan_status"].value_counts(normalize=True))
train_df.head()


# ## 2. Basic Statistics

print("Data Info:")
train_df.info()
print("Missing values:")
print(
    train_df.isnull().sum()[train_df.isnull().sum() > 0]
    if train_df.isnull().sum().any()
    else "None"
)
print("Basic statistics:")
train_df.describe()


# ## 3. Target Distribution


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
plt.show()

ratio = vc.loc[0] / vc.loc[1]
print(f"Class Imbalance Ratio (Denied:Approved) = {ratio:.2f}:1")


# ## 4. Feature Types

continuous_cols = (
    train_df.select_dtypes(include=["int64", "float64"])
    .columns.difference(["id", "loan_status"])
    .tolist()
)

categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

print(f"Numerical features ({len(continuous_cols)}):")
for col in continuous_cols:
    print(f"  - {col}")

print(f"Categorical features ({len(categorical_cols)}):")
for col in categorical_cols:
    print(f"  - {col}: {train_df[col].unique().tolist()}")


# In[6]:


train_ohe = pd.get_dummies(train_df[categorical_cols], drop_first=True, dtype="uint8")
test_ohe = pd.get_dummies(test_df[categorical_cols], drop_first=True, dtype="uint8")
test_ohe = test_ohe.reindex(columns=train_ohe.columns, fill_value=0)

train_df[train_ohe.columns] = train_ohe
test_df[train_ohe.columns] = test_ohe

ohe_cols = list(train_ohe.columns)

numerical_cols = [c for c in continuous_cols if c in train_df.columns]

# when you model later, use:
# model_cols = numerical_cols + ohe_cols   (X = train_df[model_cols])


# In[7]:


print(f"continuous_cols: {continuous_cols}")


# In[8]:


print(f"numerical_cols: {numerical_cols}")


# In[9]:


ohe_cols


# In[10]:


categorical_cols


# In[ ]:


# In[ ]:


# ## 5. Numerical Features Distribution

# In[11]:


plot_cols = [
    c for c in numerical_cols if "ohe_cols" not in globals() or c not in ohe_cols
]

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
plt.show()


# ## 6. Correlation Analysis

# In[12]:


corr_cols = numerical_cols + ["loan_status"]
corr_matrix = train_df[corr_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5
)
plt.title("Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("Correlations with loan_status:")
print(corr_matrix["loan_status"].sort_values(ascending=False))


# In[13]:


TO_DROP = ["person_age"]

for df in (train_df, test_df):
    df.drop(
        columns=[c for c in TO_DROP if c in df.columns], inplace=True, errors="ignore"
    )


def _rm(lst, items):
    return [c for c in lst if c not in set(items)]


if "continuous_cols" in globals():
    continuous_cols = _rm(continuous_cols, TO_DROP)

if "numerical_cols" in globals():
    numerical_cols = _rm(numerical_cols, TO_DROP)

assert "person_age" not in train_df.columns
assert "person_age" not in test_df.columns


# In[ ]:


# In[14]:


corr_cols = numerical_cols + ["loan_status"]
corr_matrix = train_df[corr_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5
)
plt.title("Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("Correlations with loan_status:")
print(corr_matrix["loan_status"].sort_values(ascending=False))


# We drop person_age due to severe multicollinearity with cb_person_cred_hist_length (Pearson r ≈ 0.87), which exceeds common thresholds. Keeping both inflates coefficient variance and muddies interpretation, while credit-history length is the more causally relevant signal for credit risk.

# ##7. Feature Engineering:

# In[15]:


for df in (train_df, test_df):
    df["interest_to_income_ratio"] = df["loan_int_rate"] / df["person_income"]
    df["debt_to_income_ratio"] = df["loan_amnt"] / df["person_income"]
    # df['eff_cost_borrowing'] = df['loan_amnt'] * df['loan_int_rate']
    # df['cred_hist_x_int_rate'] = df['cb_person_cred_hist_length'] * df['loan_int_rate']
    # df['cred_hist_x_income'] = df['cb_person_cred_hist_length'] * df['person_income']


train_df[["debt_to_income_ratio", "interest_to_income_ratio"]].describe()
print("\n" + "=" * 70)
print("SANITY CHECK 1: After Feature Engineering")
print("=" * 70)
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {len(train_df.columns)}")
print(f"Test columns: {len(test_df.columns)}")


# In[16]:


# continuous_cols


# ## Skewness check

# In[17]:


continuous_cols = [c for c in continuous_cols if c in train_df.columns]

skew_vals = train_df[continuous_cols].skew().sort_values(ascending=False)
print(skew_vals)

# In[17.5]: SANITY CHECK BEFORE TRANSFORMATION
print("\n" + "=" * 70)
print("SANITY CHECK 2: Before Feature Transformation")
print("=" * 70)
print(f"continuous_cols ({len(continuous_cols)}): {continuous_cols}")
print(f"\nTrain DataFrame shape: {train_df.shape}")
print(f"Test DataFrame shape: {test_df.shape}")

# ## Feature Transformation

# In[18]:


for col in ["person_income", "interest_to_income_ratio", "debt_to_income_ratio"]:
    #'debt_to_income_ratio', 'eff_cost_borrowing', 'cred_hist_x_income']:
    train_df[f"{col}_log"] = np.log1p(train_df[col])
    test_df[f"{col}_log"] = np.log1p(test_df[col])

for col in ["person_emp_length", "loan_amnt", "cb_person_cred_hist_length"]:
    train_df[f"{col}_sqrt"] = np.sqrt(train_df[col])
    test_df[f"{col}_sqrt"] = np.sqrt(test_df[col])

print("Transformations applied successfully!")

# === SANITY CHECK 3: After transformation ===
print("\n" + "=" * 70)
print("SANITY CHECK 3: After Feature Transformation")
print("=" * 70)
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
all_train_cols = train_df.columns.tolist()
print(f"Total columns in train_df: {len(all_train_cols)}")
print("\nAll train columns:")
for i, col in enumerate(all_train_cols, 1):
    print(f"  {i:2d}. {col}")


# In[19]:


cols_to_check = [
    "person_income",
    "person_income_log",
    "interest_to_income_ratio",
    "interest_to_income_ratio_log",
    "person_emp_length",
    "person_emp_length_sqrt",
]

for i in range(0, len(cols_to_check), 2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(train_df[cols_to_check[i]], kde=True, ax=ax[0])
    ax[0].set_title(f"Before: {cols_to_check[i]}")
    sns.histplot(train_df[cols_to_check[i + 1]], kde=True, ax=ax[1])
    ax[1].set_title(f"After: {cols_to_check[i+1]}")
    plt.tight_layout()
    plt.show()


# ## 8. Numerical Features by Target

# In[20]:


numerical_cols


# In[21]:


continuous_cols


# In[22]:


plotting = [
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "debt_to_income_ratio",
    "interest_to_income_ratio",
    "eff_cost_borrowing",
    "cred_hist_x_int_rate",
    "cred_hist_x_income",
]


plot_cols = [c for c in continuous_cols if c in plotting and c in train_df.columns]

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
plt.show()


# ## 9. Categorical Features vs Target

# In[23]:


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
plt.show()


# In[ ]:


# In[ ]:


# ## 10. PCA

# In[24]:


numeric_df = train_df.select_dtypes(include=[np.number]).drop(
    columns=["loan_status"] + (ohe_cols if "ohe_cols" in globals() else []),
    errors="ignore",
)

assert not set(numeric_df.columns) & set(
    ohe_cols
), "numeric_df still contains OHE dummies."
assert (
    "loan_status" not in numeric_df.columns
), "numeric_df must not include the target."

print("Sanity checks 2/2 passed.")

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
plt.show()


for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"Principal Component {i+1}: {var:.2%} variance explained")


# In[25]:


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
plt.show()


# # PCA Analysis Summary
# The PCA analysis was performed on the standardized numeric features to understand the underlying variance structure of the dataset.
# - The first principal component (PC1) explains about 24.4% of the total variance, followed by PC2 (18.3%) and PC3 (12.9%).
# - Together, the first five components explain approximately 80–85% of the dataset’s total variance.
# - This indicates that most of the information in the dataset can be represented with only a few components, reducing dimensionality without significant information loss.
# - PCA also helps visualize relationships and potential separability between approved and denied loans when plotted on the first two components.
# - Since the project’s goal is loan approval prediction, PCA confirms that the numeric variables carry overlapping information, but 4–5 components are sufficient to retain key patterns if dimensionality reduction is applied in the modeling phase.

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# ## 11. Statistical Significance Tests

# In[26]:


print("Mann-Whitney U Test (Numerical Features):")
print("=" * 70)
for col in numerical_cols:
    g0 = train_df[train_df["loan_status"] == 0][col].dropna()
    g1 = train_df[train_df["loan_status"] == 1][col].dropna()
    stat, p = stats.mannwhitneyu(g0, g1)
    sig = "✓ Significant" if p < 0.05 else "Not significant"
    print(f"{col:30s} p={p:.6f}  {sig}")

print("Chi-Square Test (Categorical Features):")
print("=" * 70)
for col in categorical_cols:
    ct = pd.crosstab(train_df[col], train_df["loan_status"])
    chi2, p, dof, _ = chi2_contingency(ct)
    sig = "✓ Significant" if p < 0.05 else "Not significant"
    print(f"{col:30s} p={p:.6f}  {sig}")


# # Key insights
#
# * **Class Imbalance:**
#    Around **86% of loans are denied** and **14% approved**, indicating strong class imbalance that will require **SMOTE or class weighting** during modeling.
# * **High collinearity detected.** `person_age` is strongly correlated with `cb_person_cred_hist_length` (~0.87). To reduce multicollinearity (and for fairness hygiene), `person_age` was excluded from features used in PCA/modeling.
# * **Categoricals handled via one-hot encoding (OHE).** OHE columns were **kept out of scaling/PCA** (since they’re already 0/1) but are available for downstream models. Train/test OHE columns are aligned to avoid column drift.
# * **Feature engineering adds signal:**
#
#   * Burden/affordability: `debt_to_income_ratio`, `loan_percent_income`.
#   * Cost interactions: `eff_cost_borrowing = loan_amnt × loan_int_rate`, `cred_hist_x_int_rate`, `cred_hist_x_income`.
#   * These new features show heavy right-skew; targeted transforms (e.g., `log1p` for `eff_cost_borrowing` and `cred_hist_x_income`, `sqrt` for `cred_hist_x_int_rate`) help stabilize variance.
# * **Numerical vs. Target Relationships:**
#    Boxplots showed clear separation trends — higher debt ratios and interest rates are associated with **denied loans**, while higher income or longer credit history improves approval likelihood.
#
# * **Categorical Insights:**
#    Certain categorical variables (like `cb_person_default_on_file` and `person_home_ownership`) showed strong patterns against loan status — e.g., renters and those with default history tend to face more denials.
#
# * **Data Quality:**
#    No missing values were detected. Some outliers persist in income and employment length, but after transformation, their effect is reduced.
#
# * **Next Steps:**
#    Proceed with model building using the transformed dataset — apply scaling, handle imbalance, and evaluate model performance using precision-recall metrics due to the skewed target distribution.

# In[ ]:

# === FINAL SANITY CHECK: Ready for Modeling ===
print("\n" + "=" * 70)
print("FINAL SANITY CHECK: Feature Summary for Modeling")
print("=" * 70)
numerical_cols = [
    "person_income_log",  # Log transformed
    "person_emp_length_sqrt",  # Sqrt transformed
    "loan_amnt_sqrt",  # Sqrt transformed
    "loan_int_rate",  # Original (already a rate)
    "loan_percent_income",  # Original (already a ratio)
    "cb_person_cred_hist_length_sqrt",  # Sqrt transformed
    "interest_to_income_ratio_log",  # Engineered + Log transformed
    "debt_to_income_ratio_log",  # Engineered + Log transformed
]
print(f"\nnumerical_cols ({len(numerical_cols)}):")
for col in numerical_cols:
    print(f"  - {col}")

print(f"\nohe_cols ({len(ohe_cols)}):")
for col in ohe_cols:
    print(f"  - {col}")

model_cols = numerical_cols + ohe_cols
print(f"\n{'='*70}")
print(f"TOTAL MODEL INPUT FEATURES: {len(model_cols)}")
print(f"{'='*70}")
print("\nTo use for modeling:")
print(f"  X_train = train_df{model_cols}")
print("  y_train = train_df['loan_status']")

# === Save Processed Data for Training ===
print("\n" + "=" * 70)
print("Saving Processed Data")
print("=" * 70)

# Save the fully processed dataframes
train_df.to_csv(TRAIN_PROCESSED_CSV, index=False)
test_df.to_csv(TEST_PROCESSED_CSV, index=False)

print("Processed data saved:")
print(f"  Train: {TRAIN_PROCESSED_CSV}")
print(f"  Test: {TEST_PROCESSED_CSV}")

print(f"Saved train_processed.csv with shape: {train_df.shape}")
print(f"Saved test_processed.csv with shape: {test_df.shape}")
print(f"{'='*70}\n")
