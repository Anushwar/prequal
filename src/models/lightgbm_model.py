"""
Loan Approval Prediction System
A complete, modular data mining pipeline for loan prediction

This system can be imported and used for:
- Training new models
- Making predictions
- Evaluating performance
"""

import pickle
import warnings

import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")


class LoanPredictionSystem:
    """Complete loan approval prediction system"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}

    def engineer_features(self, df):
        """Create engineered features"""
        df = df.copy()

        # Handle missing values
        if (
            "person_emp_length" in df.columns
            and df["person_emp_length"].isnull().sum() > 0
        ):
            df["person_emp_length"].fillna(
                df["person_emp_length"].median(), inplace=True
            )

        # Financial ratios
        df["debt_to_income"] = df["loan_amnt"] / (df["person_income"] + 1)
        df["income_per_year_employed"] = df["person_income"] / (
            df["person_emp_length"] + 1
        )
        df["credit_hist_to_age"] = df["cb_person_cred_hist_length"] / df["person_age"]

        # Binary indicators
        df["employment_stable"] = (df["person_emp_length"] >= 5).astype(int)
        df["high_interest"] = (
            df["loan_int_rate"] > df["loan_int_rate"].median()
        ).astype(int)

        # Categorical binning
        df["age_group"] = pd.cut(
            df["person_age"], bins=[0, 25, 35, 45, 100], labels=[0, 1, 2, 3]
        )
        df["age_group"] = df["age_group"].cat.codes

        df["income_bracket"] = pd.cut(
            df["person_income"],
            bins=[0, 30000, 60000, 100000, float("inf")],
            labels=[0, 1, 2, 3],
        )
        df["income_bracket"] = df["income_bracket"].cat.codes

        return df

    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        df = df.copy()

        categorical_cols = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]

        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[f"{col}_encoded"] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df[f"{col}_encoded"] = self.label_encoders[col].transform(
                            df[col]
                        )

        # One-hot encoding
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

        X = df.drop(columns=[col for col in drop_cols if col in df.columns])

        if fit:
            self.feature_columns = X.columns.tolist()
        else:
            # Ensure same columns as training
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_columns]

        return X

    def train(self, train_path="data/train.csv", use_smote=True):
        """Train the model"""
        print("=" * 70)
        print("LOAN PREDICTION SYSTEM - TRAINING")
        print("=" * 70)

        # Load data
        print("\n[1/5] Loading training data...")
        df = pd.read_csv(train_path)
        print(f"  Loaded {len(df)} samples")

        # Prepare features
        print("\n[2/5] Preparing features...")
        X = self.prepare_features(df, fit=True)
        y = df["loan_status"]
        print(f"  Created {X.shape[1]} features")

        # Split data
        print("\n[3/5] Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

        # Scale features
        print("\n[4/5] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Handle imbalance
        if use_smote:
            print("\n[5/5] Handling class imbalance with SMOTE...")
            smote = SMOTE(random_state=self.random_state, sampling_strategy=0.5)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"  Resampled to {X_train_scaled.shape[0]} samples")

        # Train model
        print("\n[6/6] Training LightGBM model...")
        self.model = LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=self.random_state,
            verbose=-1,
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\n" + "=" * 70)
        print("TRAINING RESULTS")
        print("=" * 70)

        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]

        print("\nValidation Performance:")
        print(classification_report(y_val, y_pred, target_names=["Denied", "Approved"]))

        roc_auc = roc_auc_score(y_val, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        # Cross-validation
        print("\nCross-Validation (5-fold):")
        cv_scores = cross_val_score(
            self.model,
            X_train_scaled,
            y_train,
            cv=StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            ),
            scoring="roc_auc",
            n_jobs=-1,
        )
        print(f"  Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Feature importance
        print("\nTop 10 Important Features:")
        feature_imp = (
            pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(10)
        )
        print(feature_imp.to_string(index=False))

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        return {
            "roc_auc": roc_auc,
            "cv_scores": cv_scores,
            "feature_importance": feature_imp,
        }

    def predict(self, test_path="data/test.csv", output_path="data/submission.csv"):
        """Make predictions on test data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print("\n" + "=" * 70)
        print("GENERATING PREDICTIONS")
        print("=" * 70)

        # Load test data
        print(f"\nLoading test data from {test_path}...")
        df = pd.read_csv(test_path)
        test_ids = df["id"]

        # Prepare features
        print("Preparing features...")
        X_test = self.prepare_features(df, fit=False)

        # Scale and predict
        print("Making predictions...")
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)

        # Create submission
        submission = pd.DataFrame({"id": test_ids, "loan_status": predictions})

        submission.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        dist = pd.Series(predictions).value_counts(normalize=True).to_dict()
        print(f"Distribution: {dist}")

        return submission

    def save(self, path="models/loan_system.pkl"):
        """Save the trained system"""
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "feature_columns": self.feature_columns,
                    "label_encoders": self.label_encoders,
                    "random_state": self.random_state,
                },
                f,
            )
        print(f"\nSystem saved to {path}")

    @classmethod
    def load(cls, path="models/loan_system.pkl"):
        """Load a trained system"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        system = cls(random_state=data["random_state"])
        system.model = data["model"]
        system.scaler = data["scaler"]
        system.feature_columns = data["feature_columns"]
        system.label_encoders = data["label_encoders"]

        print(f"System loaded from {path}")
        return system


def main():
    """Main execution"""
    # Create and train system
    system = LoanPredictionSystem(random_state=42)

    # Train
    system.train(train_path="data/train.csv", use_smote=True)

    # Save model
    system.save("models/loan_system.pkl")

    # Make predictions
    system.predict(test_path="data/test.csv", output_path="data/submission.csv")

    print("\n✓ System training and prediction complete!")
    print("\nTo use this system programmatically:")
    print("  from loan_prediction_system import LoanPredictionSystem")
    print("  system = LoanPredictionSystem.load('models/loan_system.pkl')")
    print("  system.predict('data/test.csv', 'output.csv')")


if __name__ == "__main__":
    main()
