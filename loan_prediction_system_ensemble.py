"""
Loan Approval Prediction System – Ensemble Version
A complete stacking ensemble integrating LightGBM, XGBoost, and CatBoost
with full EDA-driven feature engineering and 5-Fold ROC-AUC evaluation.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

DATA_DIR = "data/"
MODEL_DIR = "models/"


class LoanPredictionEnsemble:
    """Stacked ensemble model combining LGBM, XGBoost, and CatBoost"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.models = {}
        self.meta_model = None

    def engineer_features(self, df):
        """Feature engineering – ratios, transformations, and stability indicators"""
        df = df.copy()

        # Drop collinear feature
        if "person_age" in df.columns:
            df.drop(columns=["person_age"], inplace=True)

        # Handle missing values
        if "person_emp_length" in df.columns:
            df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)

        # Financial ratios and interaction terms
        df["debt_to_income_ratio"] = df["loan_amnt"] / (df["person_income"] + 1)
        df["interest_to_income_ratio"] = df["loan_int_rate"] / (df["person_income"] + 1)
        df["eff_cost_borrowing"] = df["loan_amnt"] * df["loan_int_rate"]
        df["cred_hist_x_int_rate"] = df["cb_person_cred_hist_length"] * df["loan_int_rate"]
        df["cred_hist_x_income"] = df["cb_person_cred_hist_length"] * df["person_income"]

        # Log transformations
        for col in [
            "person_income", "interest_to_income_ratio",
            "debt_to_income_ratio", "eff_cost_borrowing", "cred_hist_x_income"
        ]:
            df[f"{col}_log"] = np.log1p(df[col])

        # Square root transformations
        for col in [
            "person_emp_length", "loan_amnt",
            "cb_person_cred_hist_length", "cred_hist_x_int_rate"
        ]:
            df[f"{col}_sqrt"] = np.sqrt(df[col])

        # Binary indicators
        df["employment_stable"] = (df["person_emp_length"] >= 5).astype(int)
        df["high_interest"] = (df["loan_int_rate"] > df["loan_int_rate"].median()).astype(int)

        return df

    def encode_categorical(self, df, fit=True):
        """Encode categorical variables using label and one-hot encoding"""
        df = df.copy()
        categorical_cols = [
            "person_home_ownership", "loan_intent",
            "loan_grade", "cb_person_default_on_file"
        ]

        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[f"{col}_encoded"] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    df[f"{col}_encoded"] = self.label_encoders[col].transform(df[col])

        df = pd.get_dummies(
            df,
            columns=["person_home_ownership", "loan_intent", "loan_grade"],
            prefix=["home", "intent", "grade"],
        )
        return df

    def prepare_features(self, df, fit=True):
        """Complete feature preparation pipeline"""
        df = self.engineer_features(df)
        df = self.encode_categorical(df, fit)

        drop_cols = ["id", "cb_person_default_on_file"]
        if "loan_status" in df.columns:
            drop_cols.append("loan_status")

        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        if fit:
            self.feature_columns = X.columns.tolist()
        else:
            for c in self.feature_columns:
                if c not in X.columns:
                    X[c] = 0
            X = X[self.feature_columns]

        return X

    def train(self, train_path=DATA_DIR + "train.csv", use_smote=True):
        """Train stacked ensemble (LGBM + XGBoost + CatBoost)"""
        print("=" * 70)
        print("LOAN PREDICTION SYSTEM - ENSEMBLE TRAINING")
        print("=" * 70)

        # Load data
        df = pd.read_csv(train_path)
        X = self.prepare_features(df, fit=True)
        y = df["loan_status"]

        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Scale numeric features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # Handle class imbalance
        if use_smote:
            sm = SMOTE(random_state=self.random_state, sampling_strategy=0.5)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"After SMOTE: {X_train.shape}")

        # Base models
        base_models = {
            "lgbm": LGBMClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=7,
                subsample=0.8, random_state=self.random_state
            ),
            "xgb": XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=self.random_state
            ),
            "cat": CatBoostClassifier(
                iterations=400, learning_rate=0.05, depth=7,
                verbose=False, random_seed=self.random_state
            )
        }

        # 5-Fold training for base out-of-fold predictions
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((len(X_train), len(base_models)))
        val_preds = np.zeros((len(X_val), len(base_models)))

        for i, (name, model) in enumerate(base_models.items()):
            print(f"Training {name} ...")
            oof = np.zeros(len(X_train))
            for tr, va in skf.split(X_train, y_train):
                model.fit(X_train[tr], y_train.iloc[tr])
                oof[va] = model.predict_proba(X_train[va])[:, 1]
            oof_preds[:, i] = oof
            val_preds[:, i] = model.predict_proba(X_val)[:, 1]
            self.models[name] = model

        # Meta-learner using CatBoost
        self.meta_model = CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=5,
            verbose=False, random_seed=self.random_state
        )
        self.meta_model.fit(oof_preds, y_train)
        meta_val = self.meta_model.predict_proba(val_preds)[:, 1]
        roc_auc = roc_auc_score(y_val, meta_val)

        # Optimize F1 threshold
        thresholds = np.linspace(0.1, 0.9, 80)
        f1s = [f1_score(y_val, (meta_val > t).astype(int)) for t in thresholds]
        best_thr = thresholds[np.argmax(f1s)]
        f1_best = np.max(f1s)

        print("\nValidation Performance:")
        print(classification_report(y_val, (meta_val > best_thr).astype(int),
              target_names=["Denied", "Approved"]))
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Optimal F1 Threshold: {best_thr:.3f}, F1={f1_best:.4f}")

        # Full 5-Fold ROC-AUC evaluation
        print("\nCross-Validation (5-Fold Ensemble ROC-AUC):")
        cv_aucs = []
        skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for fold, (tr_idx, va_idx) in enumerate(skf_outer.split(X_train, y_train), 1):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            fold_preds = np.zeros((len(X_va), len(base_models)))
            for i2, (name, model) in enumerate(base_models.items()):
                model.fit(X_tr, y_tr)
                fold_preds[:, i2] = model.predict_proba(X_va)[:, 1]

            meta_train = np.column_stack([
                m.predict_proba(X_tr)[:, 1] for m in base_models.values()
            ])
            meta_model_fold = CatBoostClassifier(
                iterations=400, learning_rate=0.05, depth=5,
                verbose=False, random_seed=self.random_state
            )
            meta_model_fold.fit(meta_train, y_tr)
            meta_val_pred = meta_model_fold.predict_proba(fold_preds)[:, 1]
            auc = roc_auc_score(y_va, meta_val_pred)
            cv_aucs.append(auc)
            print(f"  Fold {fold} ROC-AUC: {auc:.4f}")

        print(f"Mean 5-Fold Ensemble ROC-AUC: {np.mean(cv_aucs):.4f} (+/- {np.std(cv_aucs)*2:.4f})")

        # Save trained system
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_DIR + "loan_ensemble.pkl", "wb") as f:
            pickle.dump({
                "models": self.models,
                "meta": self.meta_model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns
            }, f)
        print(f"\nSystem saved to {MODEL_DIR}loan_ensemble.pkl")

        return {"roc_auc": roc_auc, "f1": f1_best, "cv_roc_auc": np.mean(cv_aucs)}

    def predict(self, test_path=DATA_DIR + "test.csv", output_path=DATA_DIR + "submission.csv"):
        """Generate ensemble predictions on test data"""
        print("=" * 70)
        print("GENERATING ENSEMBLE PREDICTIONS")
        print("=" * 70)

        df = pd.read_csv(test_path)
        X = self.prepare_features(df, fit=False)
        X = self.scaler.transform(X)

        base_preds = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models.values()])
        final = self.meta_model.predict_proba(base_preds)[:, 1]

        submission = pd.DataFrame({
            "id": df["id"],
            "loan_status": (final > 0.5).astype(int)
        })
        submission.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        return submission


def main():
    """Main execution – train, save, and predict using ensemble"""
    ens = LoanPredictionEnsemble(random_state=42)
    results = ens.train(DATA_DIR + "train.csv", use_smote=True)
    ens.predict(DATA_DIR + "test.csv", DATA_DIR + "submission.csv")
    print(f"\n✓ System training complete! ROC-AUC={results['roc_auc']:.4f}, "
          f"F1={results['f1']:.4f}, CV-ROC-AUC={results['cv_roc_auc']:.4f}")


if __name__ == "__main__":
    main()