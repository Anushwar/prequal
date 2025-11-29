import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from flax.training import train_state
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tqdm import trange

from src.config import IMAGES_DIR, NN_PARAMS_PKL, TRAIN_PROCESSED_CSV
from src.models.nn_large import ANN_64_128_256_128_64_32 as LoanApprovalMLP


def save_params(params, filename=None):
    if filename is None:
        filename = NN_PARAMS_PKL
    with open(filename, "wb") as f:
        pickle.dump(params, f)


def load_and_prepare_data(train_path):
    train_df = pd.read_csv(train_path)

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

    ohe_cols = [
        "person_home_ownership_OTHER",
        "person_home_ownership_OWN",
        "person_home_ownership_RENT",
        "loan_intent_HOMEIMPROVEMENT",
        "loan_intent_MEDICAL",
        "loan_intent_EDUCATION",
        "loan_intent_PERSONAL",
        "loan_intent_VENTURE",
        "loan_grade_B",
        "loan_grade_C",
        "loan_grade_D",
        "loan_grade_E",
        "loan_grade_F",
        "loan_grade_G",
        "cb_person_default_on_file_Y",
    ]

    feature_cols = numerical_cols + ohe_cols

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df["loan_status"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler2 = RobustScaler()
    X_train = scaler2.fit_transform(X_train).astype(np.float32)
    X_test = scaler2.transform(X_test).astype(np.float32)

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

    # === ADD THIS DEBUGGING ===
    print(f"\n{'='*70}")
    print("Data Quality Check")
    print(f"{'='*70}")
    print(f"X_train - NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
    print(f"X_test - NaN: {np.isnan(X_test).sum()}, Inf: {np.isinf(X_test).sum()}")
    print(f"X_train min: {np.nanmin(X_train):.6f}, max: {np.nanmax(X_train):.6f}")
    print(f"X_test min: {np.nanmin(X_test):.6f}, max: {np.nanmax(X_test):.6f}")

    # Check each feature column
    for i, col in enumerate(feature_cols):
        nan_count = np.isnan(X_train[:, i]).sum()
        inf_count = np.isinf(X_train[:, i]).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  PROBLEM in {col}: NaN={nan_count}, Inf={inf_count}")

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

    print("Loading preprocessed data from EDA...")
    X_train, y_train, X_test, y_test = load_and_prepare_data(train_csv)

    print_feature_stats(X_train, "Train")
    print_feature_stats(X_test, "Test")

    model = LoanApprovalMLP()
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, jnp.ones((1, X_train.shape[1])))

    tx = optax.adam(learning_rate=1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    num_epochs = 75
    batch_size = 32
    num_batches = len(X_train) // batch_size

    neg_ratio = (y_train == 0).mean()
    pos_ratio = (y_train == 1).mean()
    pos_weight = neg_ratio / pos_ratio

    print("\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print("  Learning rate: 1e-3")
    print("  L2 regularization: 1e-4")
    print(f"  Positive class weight: {pos_weight:.2f}")
    print(f"  Number of batches per epoch: {num_batches}\n")

    train_losses = []
    test_accuracies = []

    print("Starting training...")
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
        train_losses.append(avg_loss)

        test_preds = eval_step(state.params, state.apply_fn, (X_test, y_test))
        test_preds_binary = (test_preds > 0.5).astype(np.float32)
        test_acc = accuracy_score(y_test, test_preds_binary)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    print(f"\n{'='*70}")
    print("Final Evaluation on Test Set")
    print(f"{'='*70}")

    test_preds = eval_step(state.params, state.apply_fn, (X_test, y_test))
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

    save_params(state.params)
    print(f"Model parameters saved to '{NN_PARAMS_PKL}'")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = IMAGES_DIR / "nn_large_training.png"
    plt.savefig(plot_path)
    print(f"Training plot saved as '{plot_path}'")


if __name__ == "__main__":
    main()
