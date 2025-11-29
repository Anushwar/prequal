"""
Loan Approval Prediction System – CatBoost Version
A complete, modular pipeline integrating EDA-driven features,
log/sqrt transformations, and optional PCA (95% variance retention)
for enhanced loan approval prediction.
"""

import pickle
import warnings
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


class LoanPredictionSystem:
    """Loan approval prediction system using CatBoost with EDA and PCA"""

    def __init__(self, random_state=42, use_pca=False, pca_var=0.95):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=pca_var, random_state=random_state) if use_pca else None
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.use_pca = use_pca

    def engineer_features(self, df):
        """Feature engineering – creates ratios, transformations, and binary indicators"""
        df = df.copy()

        # Drop highly collinear feature
        if "person_age" in df.columns:
            df.drop(columns=["person_age"], inplace=True)

        # Handle missing values
        if "person_emp_length" in df.columns and df["person_emp_length"].isnull().sum() > 0:
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
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col])

        # Square root transformations
        for col in [
            "person_emp_length", "loan_amnt",
            "cb_person_cred_hist_length", "cred_hist_x_int_rate"
        ]:
            if col in df.columns:
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
                    if col in self.label_encoders:
                        df[f"{col}_encoded"] = self.label_encoders[col].transform(df[col])

        # One-hot encoding for nominal features
        df = pd.get_dummies(
            df,
            columns=["person_home_ownership", "loan_intent", "loan_grade"],
            prefix=["home", "intent", "grade"],
        )
        return df

    def prepare_features(self, df, fit=True):
        """Complete feature preparation pipeline"""
        df = self.engineer_features(df)
        df = self.encode_categorical(df, fit=fit)

        # Drop unnecessary columns
        drop_cols = ["id", "cb_person_default_on_file"]
        if "loan_status" in df.columns:
            drop_cols.append("loan_status")

        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        if fit:
            self.feature_columns = X.columns.tolist()
        else:
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_columns]

        return X

    def train(self, train_path="data/train.csv", use_smote=True):
        """Train the CatBoost model"""
        print("=" * 70)
        print("LOAN PREDICTION SYSTEM - TRAINING (CatBoost + EDA + PCA)")
        print("=" * 70)

        # Load data
        df = pd.read_csv(train_path)
        print(f"Loaded {len(df)} samples")

        # Prepare features
        X = self.prepare_features(df, fit=True)
        y = df["loan_status"]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Scale numeric features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Apply PCA if enabled
        if self.use_pca:
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_val_scaled = self.pca.transform(X_val_scaled)
            print(f"PCA applied → {self.pca.n_components_} components retained (95% variance)")

        # Handle imbalance
        if use_smote:
            sm = SMOTE(random_state=self.random_state, sampling_strategy=0.5)
            X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: {X_train_scaled.shape[0]} samples")

        # Train CatBoost model
        self.model = CatBoostClassifier(
            iterations=900,
            depth=7,
            learning_rate=0.045,
            l2_leaf_reg=2.5,
            random_seed=self.random_state,
            verbose=False,
            eval_metric="AUC",
            auto_class_weights="Balanced",
        )
        self.model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val))

        # Evaluate performance
        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]

        print("\nValidation Performance:")
        print(classification_report(y_val, y_pred, target_names=["Denied", "Approved"]))
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        # 5-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        aucs = []
        for tr_idx, va_idx in skf.split(X_train_scaled, y_train):
            X_t, X_v = X_train_scaled[tr_idx], X_train_scaled[va_idx]
            y_t, y_v = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            self.model.fit(X_t, y_t, verbose=False)
            preds = self.model.predict_proba(X_v)[:, 1]
            aucs.append(roc_auc_score(y_v, preds))
        print(f"5-Fold ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs)*2:.4f})")

        # Feature importance
        print("\nTop 10 Important Features:")
        if self.use_pca:
            feature_names = [f"PC{i+1}" for i in range(len(self.model.get_feature_importance()))]
        else:
            feature_names = self.feature_columns

        feature_imp = (
            pd.DataFrame({
                "feature": feature_names,
                "importance": self.model.get_feature_importance()
            })
            .sort_values("importance", ascending=False)
            .head(10)
        )
        print(feature_imp.to_string(index=False))

        print("\nTraining Complete ✓")
        return {"roc_auc": roc_auc, "cv_scores": aucs, "feature_importance": feature_imp}

    def predict(self, test_path="data/test.csv", output_path="data/submission.csv"):
        """Generate predictions on new test data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Load and prepare test data
        df = pd.read_csv(test_path)
        test_ids = df["id"]
        X_test = self.prepare_features(df, fit=False)
        X_test_scaled = self.scaler.transform(X_test)
        if self.use_pca:
            X_test_scaled = self.pca.transform(X_test_scaled)

        # Predict and save
        predictions = self.model.predict(X_test_scaled)
        submission = pd.DataFrame({"id": test_ids, "loan_status": predictions})
        submission.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        return submission

    def save(self, path="models/loan_catboost.pkl"):
        """Save the trained CatBoost system"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "pca": self.pca,
                "feature_columns": self.feature_columns,
                "label_encoders": self.label_encoders,
                "random_state": self.random_state,
                "use_pca": self.use_pca,
            }, f)
        print(f"System saved to {path}")

    @classmethod
    def load(cls, path="models/loan_catboost.pkl"):
        """Load a saved CatBoost prediction system"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        system = cls(random_state=data["random_state"], use_pca=data["use_pca"])
        system.model = data["model"]
        system.scaler = data["scaler"]
        system.pca = data["pca"]
        system.feature_columns = data["feature_columns"]
        system.label_encoders = data["label_encoders"]
        print(f"System loaded from {path}")
        return system


def main():
    """Main execution – train, save, and generate predictions"""
    system = LoanPredictionSystem(random_state=42, use_pca=True)
    system.train(train_path="data/train.csv", use_smote=True)
    system.save("models/loan_catboost.pkl")
    system.predict(test_path="data/test.csv", output_path="data/submission.csv")
    print("\n✓ System training and prediction complete!")


if __name__ == "__main__":
    main()