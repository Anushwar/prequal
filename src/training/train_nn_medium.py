import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax.training import train_state
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from tqdm import trange

from src.config import IMAGES_NN_DIR, NN_PARAMS_PKL, TRAIN_PROCESSED_CSV
from src.models.nn_medium import ANN_128_64_32_16 as LoanApprovalMLP


def save_params(params, filename=None):
    if filename is None:
        filename = NN_PARAMS_PKL
    with open(filename, "wb") as f:
        pickle.dump(params, f)


def train_with_cv(X_train, y_train, X_test, y_test, n_splits=5):
    """Train with k-fold cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nFold {fold+1}/{n_splits}...")

        # Split into train/val for this fold
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        # Train model on this fold
        model = LoanApprovalMLP()
        rng = jax.random.PRNGKey(42)
        params = model.init(rng, jnp.ones((1, X_fold_train.shape[1])))

        tx = optax.adam(learning_rate=1e-3)
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx
        )

        # Training loop
        num_epochs = 30
        batch_size = 32
        num_batches = len(X_fold_train) // batch_size

        neg_ratio = (y_fold_train == 0).mean()
        pos_ratio = (y_fold_train == 1).mean()
        pos_weight = neg_ratio / pos_ratio

        for epoch in range(num_epochs):
            epoch_losses = []
            perm = np.random.permutation(len(X_fold_train))

            for i in range(num_batches):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                batch_X = X_fold_train[idx]
                batch_y = y_fold_train[idx]
                state, loss = train_step(
                    state, (batch_X, batch_y), alpha=1e-4, pos_weight=pos_weight
                )
                epoch_losses.append(loss)

        # Evaluate on validation fold
        val_preds = state.apply_fn(state.params, X_fold_val)
        fold_auc = roc_auc_score(y_fold_val, val_preds)
        cv_scores.append(fold_auc)
        print(f"  Fold AUC: {fold_auc:.4f}")

    print(f"\n{'='*70}")
    print("CV Results:")
    print(f"{'='*70}")
    print(f"  Mean AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
    print(f"{'='*70}\n")
    return cv_scores


def load_and_prepare_data(train_path, use_smote=True):
    train_df = pd.read_csv(train_path)

    # Drop person_age due to multicollinearity
    if "person_age" in train_df.columns:
        train_df.drop(columns=["person_age"], inplace=True)

    # Feature Engineering
    train_df["interest_to_income_ratio"] = train_df["loan_int_rate"] / train_df["person_income"]
    train_df["debt_to_income_ratio"] = train_df["loan_amnt"] / train_df["person_income"]

    # Feature Transformations
    for col in ["person_income", "interest_to_income_ratio", "debt_to_income_ratio"]:
        train_df[f"{col}_log"] = np.log1p(train_df[col])

    for col in ["person_emp_length", "loan_amnt", "cb_person_cred_hist_length"]:
        train_df[f"{col}_sqrt"] = np.sqrt(train_df[col])

    # One-hot encoding for categorical variables
    categorical_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    train_ohe = pd.get_dummies(train_df[categorical_cols], drop_first=True, dtype="uint8")
    train_df[train_ohe.columns] = train_ohe

    numerical_cols = [
        "person_income_log",
        "person_emp_length_sqrt",
        "loan_amnt_sqrt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length_sqrt",
        "interest_to_income_ratio_log",
        "debt_to_income_ratio_log",
    ]

    ohe_cols = list(train_ohe.columns)

    feature_cols = numerical_cols + ohe_cols

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df["loan_status"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler2 = RobustScaler()
    X_train = scaler2.fit_transform(X_train).astype(np.float32)
    X_test = scaler2.transform(X_test).astype(np.float32)

    if use_smote:
        print("\nApplying SMOTE...")
        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"  Resampled to {X_train.shape[0]} samples")

    print(f"\n{'='*70}")
    print("Data Loading Summary")
    print(f"{'='*70}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Feature columns: {feature_cols}")
    print("\nClass distribution in training:")
    print(f"  Denied (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"  Approved (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"{'='*70}\n")

    return X_train, y_train, X_test, y_test


def print_feature_stats(X, name=""):
    print(f"\n{name} Feature Statistics:")
    print(f"  Min: {X.min(axis=0).min():.4f}")
    print(f"  Max: {X.max(axis=0).max():.4f}")
    print(f"  Mean: {X.mean(axis=0).mean():.4f}")
    print(f"  Std: {X.std(axis=0).mean():.4f}")


def weighted_binary_cross_entropy(logits, targets, pos_weight=6.04):
    eps = 1e-8
    predictions = jax.nn.sigmoid(logits)
    predictions = jnp.clip(predictions, eps, 1 - eps)

    pos_loss = -targets * jnp.log(predictions) * pos_weight
    neg_loss = -(1 - targets) * jnp.log(1 - predictions)

    loss = pos_loss + neg_loss
    return jnp.mean(loss)


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    eps = 1e-8
    predictions = jax.nn.sigmoid(logits)
    predictions = jnp.clip(predictions, eps, 1 - eps)

    bce = -targets * jnp.log(predictions) - (1 - targets) * jnp.log(1 - predictions)
    pt = jnp.where(targets == 1, predictions, 1 - predictions)
    focal_weight = jnp.power(1 - pt, gamma)

    alpha_weight = jnp.where(targets == 1, alpha, 1 - alpha)
    loss = alpha_weight * focal_weight * bce

    return jnp.mean(loss)


def ridge_penalty(params, alpha):
    return alpha * sum(
        jnp.sum(jnp.square(w)) for w in jax.tree_util.tree_leaves(params)
    )


@jax.jit
def train_step(state, batch, alpha=1e-4, pos_weight=6.04):
    def loss_fn(params):
        X, y = batch
        logits = state.apply_fn(params, X)
        loss = weighted_binary_cross_entropy(logits, y, pos_weight)
        # loss = focal_loss(logits, y, alpha=focal_alpha, gamma=focal_gamma)
        loss += ridge_penalty(params, alpha)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval_step(params, apply_fn, batch):
    X, y = batch
    logits = apply_fn(params, X)
    return logits


def main():
    train_csv = str(TRAIN_PROCESSED_CSV)

    USE_SMOTE = True

    print("Loading preprocessed data from EDA...")
    X_train, y_train, X_test, y_test = load_and_prepare_data(
        train_csv, use_smote=USE_SMOTE
    )

    # Run cross-validation
    print("\nStarting 5-Fold Cross-Validation...")
    train_with_cv(X_train, y_train, X_test, y_test, n_splits=5)

    # After CV, train final model on full training data for test evaluation
    print("\n" + "=" * 70)
    print("Training Final Model on All Training Data")
    print("=" * 70)

    model = LoanApprovalMLP()
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, jnp.ones((1, X_train.shape[1])))

    tx = optax.adam(learning_rate=1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    num_epochs = 50
    batch_size = 32
    num_batches = len(X_train) // batch_size

    neg_ratio = (y_train == 0).mean()
    pos_ratio = (y_train == 1).mean()
    pos_weight = neg_ratio / pos_ratio

    for epoch in range(num_epochs):
        epoch_losses = []
        perm = np.random.permutation(len(X_train))

        for i in trange(num_batches, desc=f"Epoch {epoch+1}/{num_epochs}"):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            batch_X = X_train[idx]
            batch_y = y_train[idx]
            state, loss = train_step(
                state, (batch_X, batch_y), alpha=1e-4, pos_weight=pos_weight
            )
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Evaluate on test set
    print(f"\n{'='*70}")
    print("Final Evaluation on Test Set")
    print(f"{'='*70}")

    test_preds = state.apply_fn(state.params, X_test)
    test_preds_binary = (test_preds > 0.5).astype(np.float32)

    test_acc = accuracy_score(y_test, test_preds_binary)
    test_precision = precision_score(y_test, test_preds_binary, zero_division=0)
    test_recall = recall_score(y_test, test_preds_binary, zero_division=0)
    test_f1 = f1_score(y_test, test_preds_binary, zero_division=0)
    test_auc = roc_auc_score(y_test, test_preds)

    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1 Score:  {test_f1:.4f}")
    print(f"Test AUC-ROC:   {test_auc:.4f}")
    print(f"{'='*70}\n")

    save_params(state.params, "NN1_params.pkl")


if __name__ == "__main__":
    main()
