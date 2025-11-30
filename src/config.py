"""
Configuration and constants for the loan prediction system
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = ARTIFACTS_DIR / "images"
ANALYSIS_DIR = ARTIFACTS_DIR / "analysis"

# Image subdirectories
IMAGES_NN_DIR = IMAGES_DIR / "neural_networks"
IMAGES_ML_DIR = IMAGES_DIR / "traditional_ml"
IMAGES_EDA_DIR = IMAGES_DIR / "eda"

# Data files
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
TRAIN_PROCESSED_CSV = DATA_DIR / "train_processed.csv"  # Preprocessed by EDA
TEST_PROCESSED_CSV = DATA_DIR / "test_processed.csv"  # Preprocessed by EDA
SUBMISSION_CSV = DATA_DIR / "submission.csv"

# Model files
LIGHTGBM_MODEL_PKL = MODELS_DIR / "lightgbm_system.pkl"
CATBOOST_MODEL_PKL = MODELS_DIR / "catboost_system.pkl"
ENSEMBLE_MODEL_PKL = MODELS_DIR / "ensemble_system.pkl"
NN_PARAMS_PKL = MODELS_DIR / "nn_params.pkl"

# Artifact files (moved to analysis subdirectory)
VIF_CSV = ANALYSIS_DIR / "vif.csv"
VIF_AFTER_CSV = ANALYSIS_DIR / "vif_after.csv"
MUTUAL_INFO_CSV = ANALYSIS_DIR / "mutual_info.csv"
MUTUAL_INFO_AFTER_CSV = ANALYSIS_DIR / "mutual_info_after.csv"
HIGH_CORR_05_CSV = ANALYSIS_DIR / "high_corr_0_5_pairs.csv"
HIGH_CORR_05_AFTER_CSV = ANALYSIS_DIR / "high_corr_0_5_pairs_after.csv"
HIGH_CORR_08_CSV = ANALYSIS_DIR / "high_corr_0_8_pairs.csv"
HIGH_CORR_08_AFTER_CSV = ANALYSIS_DIR / "high_corr_0_8_pairs_after.csv"

# Training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
