# Loan Approval Prediction

ML system for predicting loan approval.

## Tech Stack

- JAX, Optax Linen
- Custom Multi Layer Perceptron, Focal Loss, Weighted Binary Cross Entropy Loss
- Python 3.10
- scikit-learn, LightGBM, imbalanced-learn
- pandas, numpy, matplotlib, seaborn

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

### 2. Train Model

```bash
python loan_prediction_system.py
```

```bash
python loan_prediction_system_catboost.py
```

```bash
python loan_prediction_system_ensemble.py
```

Outputs:
- `models/loan_system.pkl` - Trained model
- Performance metrics printed

### 3. Use Trained Model

```python
from loan_prediction_system import LoanPredictionSystem

system = LoanPredictionSystem.load('models/loan_system.pkl')
system.predict('data/test.csv', 'predictions.csv')
```

## Development

### Code Formatting & Linting

Before making code changes, ensure code quality:

```bash
# Format all Python files
black .

# Lint and auto-fix issues
ruff check . --fix

# Verify formatting
black --check .
ruff check .
```

Or for specific files:
```bash
black <file>.py
ruff check <file>.py --fix
```

## Project Structure

```
prequal/
├── notebooks/
│   └── eda.ipynb              # Exploratory analysis
├── data/
│   ├── train.csv              # Training data
│   └── test.csv               # Test data
├── models/
│   └── loan_system.pkl        # Trained model (generated)
├── loan_prediction_system.py  # Main ML system
├── loan_prediction_system_catboost.py # CatBoost ML system
├── loan_prediction_system_ensemble.py # Ensemble ML system
├── doc.md                     # Project documentation
└── requirements.txt

```

## Performance

<img width="999" height="370" alt="Screenshot 2025-11-02 at 7 30 20 PM" src="https://github.com/user-attachments/assets/c869e629-7262-4864-921a-28a9c99c3cf5" />
## Features

- Automated feature engineering (8+ features)
- Handles class imbalance with SMOTE
- RobustScaler for outliers
- LightGBM classifier with cross-validation
- Modular, reusable pipeline

## License

MIT
