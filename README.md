# Loan Approval Prediction

ML system for predicting loan approval. Project checkpoint for data mining course.

## Tech Stack

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
├── doc.md                     # Project documentation
└── requirements.txt
```

## Performance

- ROC-AUC: 95.0%
- Cross-Validation: 98.3%
- Accuracy: 95.2%
- F1-Score: 81.2%

## Features

- Automated feature engineering (8+ features)
- Handles class imbalance with SMOTE
- RobustScaler for outliers
- LightGBM classifier with cross-validation
- Modular, reusable pipeline

## License

MIT
