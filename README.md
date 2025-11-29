# Loan Approval Prediction

Machine learning system for predicting loan approval using traditional ML (LightGBM, CatBoost, XGBoost) and custom neural networks (JAX/Flax).

## Features

- **Multiple Model Implementations**: LightGBM, CatBoost, Ensemble (stacking), and custom neural networks
- **Advanced Feature Engineering**: Automated creation of financial ratios, log/sqrt transformations, and interaction terms
- **Class Imbalance Handling**: SMOTE oversampling and weighted loss functions
- **Neural Network Architectures**: 3 sizes (small/medium/large) with Leaky ReLU, Focal Loss, and Weighted BCE
- **Robust Preprocessing**: RobustScaler for outlier handling, optional PCA for dimensionality reduction
- **Cross-Validation**: 5-fold stratified cross-validation for reliable performance estimates

## Tech Stack

- **ML**: LightGBM, XGBoost, CatBoost, scikit-learn, imbalanced-learn
- **Deep Learning**: JAX, Flax, Optax
- **Data**: pandas, numpy, scipy
- **Dev Tools**: black, ruff

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Run EDA & Preprocessing:**
```bash
python -m src.eda
```

**Train Traditional ML Models:**
```bash
python -m src.models.lightgbm_model
python -m src.models.catboost_model
python -m src.models.ensemble_model
```

**Train Neural Networks:**
```bash
python -m src.training.train_nn_small
python -m src.training.train_nn_medium
python -m src.training.train_nn_large
```

**Use Trained Model:**
```python
from src.models.lightgbm_model import LoanPredictionSystem

system = LoanPredictionSystem.load('models/lightgbm_system.pkl')
system.predict('data/test.csv', 'predictions.csv')
```

## Development

```bash
# Format code
black src/

# Lint
ruff check src/ --fix
```

## License

MIT
