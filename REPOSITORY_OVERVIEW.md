# Alzheimer's ML Repository - Complete Overview

## Project Summary

This repository contains a complete machine learning pipeline for predicting Alzheimer's disease progression using the OASIS dataset. The project includes:

- **Cross-Sectional Model**: Random Forest classifier trained on baseline features
- **Longitudinal Model**: XGBoost classifier trained on temporal features from multiple visits
- **Feature Engineering**: Extracts baseline, delta, and slope features from longitudinal data
- **Hyperparameter Optimization**: Uses Optuna for automated tuning
- **Comprehensive Evaluation**: Metrics, visualizations, and SHAP analysis

## Repository Structure

```
alzheimers-ml/
│
├── src/                          # Main source code package
│   ├── __init__.py                  # Package initialization (version info)
│   │
│   ├── data/                     # Data handling modules
│   │   ├── __init__.py
│   │   ├── load_data.py             # Data loading functions
│   │   └── preprocessing.py       # Preprocessing pipelines
│   │
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py   # Longitudinal feature extraction
│   │
│   ├── models/                   # ML model definitions
│   │   ├── __init__.py
│   │   ├── random_forest.py         # RF with Optuna optimization
│   │   └── xgboost_model.py         # XGBoost with Optuna optimization
│   │
│   ├── evaluation/               # Evaluation utilities
│   │   ├── __init__.py
│   │   ├── metrics.py               # Evaluation metrics
│   │   └── visualization.py         # Plotting functions
│   │
│   └── utils/                    # Utility functions
│       └── __init__.py
│
├── scripts/                      # Executable training scripts
│   ├── train_cross_sectional.py    # Train RF on cross-sectional data
│   ├── train_longitudinal.py       # Train XGBoost on longitudinal data
│   ├── main.py                      # Full training pipeline
│   ├── predict.py                   # Prediction functions
│   ├── example_usage.py             # Example usage demonstrations
│   └── generate_visualizations.py   # Generate model visualizations
│
├── notebooks/                    # Jupyter notebooks (optional)
│
├── data/                         # Data directory (gitignored)
│   ├── oasis_cross-sectional-5708aa0a98d82080.xlsx
│   └── oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx
│
├── models/                       # Saved models (gitignored)
│   ├── rf_cross_sectional.pkl
│   ├── xgb_longitudinal.pkl
│   ├── preprocessor_cross_sectional.pkl
│   ├── preprocessor_longitudinal.pkl
│   └── label_encoder.pkl
│
├── configs/                      # Configuration files
│   └── config.yaml                  # Centralized configuration
│
├── tests/                        # Unit tests (to be implemented)
│
├── results/                      # Output results (gitignored)
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # Main documentation
├── QUICK_START.md                # Quick start guide
├── STRUCTURE.md                  # Detailed structure documentation
└── REPOSITORY_OVERVIEW.md        # This file
```

## Data Flow Pipeline

### 1. Cross-Sectional Pipeline

```
┌─────────────────┐
│  Raw Excel File │
│ (Cross-Section) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load & Clean   │
│  (load_data.py) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Train-Test      │
│ Split           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ Pipeline        │
│ (preprocessing) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Random Forest   │
│ + Optuna        │
│ (random_forest) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluation &    │
│ Save Model      │
└─────────────────┘
```

### 2. Longitudinal Pipeline

```
┌─────────────────┐
│  Raw Excel File │
│ (Longitudinal)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load & Clean   │
│  (load_data.py) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature         │
│ Engineering     │
│ (feature_eng)   │
│                 │
│ • Baseline      │
│ • Delta         │
│ • Slope         │
│ • CDR_pred      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ Pipeline        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ XGBoost         │
│ + Optuna        │
│ (xgboost_model) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluation &    │
│ Save Model      │
└─────────────────┘
```

### 3. Full Pipeline (Combined)

```
Cross-Sectional Data
        │
        ▼
   RF Model ──────┐
        │         │
        │         │ (CDR predictions)
        │         │
        │         ▼
        │   Longitudinal Data
        │         │
        │         ▼
        │   Feature Engineering
        │         │
        │         ▼
        │   XGBoost Model
        │         │
        └─────────┴──► Final Predictions
```

## Key Files and Their Contents

### Data Loading (`src/data/load_data.py`)

**Functions:**
- `load_cross_sectional_data()`: Loads Excel, drops missing CDR, converts to binary
- `load_longitudinal_data()`: Loads Excel, creates binary columns, handles missing values
- `prepare_train_test_split()`: Creates stratified train-test split

**Key Code:**
```python
df = pd.read_excel(data_path)
df = df.dropna(subset=["CDR"])
df['CDR'] = (df['CDR'] > 0).astype(int)
```

### Preprocessing (`src/data/preprocessing.py`)

**Functions:**
- `create_preprocessing_pipeline()`: Creates ColumnTransformer with separate pipelines for numeric and categorical features
- `encode_labels()`: Encodes target labels

**Key Code:**
```python
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
bin_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("OneHotEncoder", OneHotEncoder(...))
])
preprocess = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("bin", bin_pipeline, bin_cols)
])
```

### Feature Engineering (`src/features/feature_engineering.py`)

**Functions:**
- `extract_longitudinal_features()`: Extracts per-subject features from multiple visits
- `prepare_longitudinal_train_test_split()`: Subject-level train-test split

**Features Extracted:**
- Static: Sex, Hand, Educ, SES
- Baseline: MMSE, nWBV, Age, eTIV, ASF, CDR_pred
- Delta: MMSE_delta, nWBV_delta
- Slope: MMSE_slope, nWBV_slope (per year)
- Time: n_visits, followup_years
- Indicator: has_multiple_visits

### Model Training (`src/models/`)

**Random Forest (`random_forest.py`):**
- Optuna optimization with 75 trials
- Hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
- Uses ROC-AUC as optimization metric

**XGBoost (`xgboost_model.py`):**
- Optuna optimization with 300 trials
- Hyperparameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, regularization
- Uses ROC-AUC as optimization metric

### Evaluation (`src/evaluation/`)

**Metrics (`metrics.py`):**
- `evaluate_model()`: Computes accuracy, F1, ROC-AUC, Brier score, classification report, confusion matrix
- `cross_validate_model()`: Performs cross-validation

**Visualization (`visualization.py`):**
- `plot_calibration_curve()`: Calibration plots
- `plot_roc_curve()`: ROC curves
- `plot_feature_importance()`: Feature importance plots
- `plot_shap_summary()`: SHAP values

### Training Scripts (`scripts/`)

**`train_cross_sectional.py`:**
1. Load data
2. Train-test split
3. Preprocessing
4. Train RF model
5. Evaluate
6. Save models

**`train_longitudinal.py`:**
1. Load longitudinal data
2. Extract features (uses CDR model if available)
3. Preprocessing
4. Train XGBoost model
5. Evaluate
6. Save models

**`main.py`:**
- Runs both scripts sequentially
- Passes RF model to longitudinal training

## Quick Start Guide

### 1. Setup

```bash
# Clone repository
git clone <repository-url>
cd alzheimers-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your data files in the `data/` directory:
- `oasis_cross-sectional-5708aa0a98d82080.xlsx`
- `oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx`

### 3. Train Models

**Option A: Full Pipeline (Recommended)**
```bash
python scripts/main.py
```

**Option B: Individual Models**
```bash
# Train cross-sectional model
python scripts/train_cross_sectional.py

# Train longitudinal model
python scripts/train_longitudinal.py
```

### 4. Use Models

```python
import joblib

# Load models
rf_model = joblib.load("models/rf_cross_sectional.pkl")
xgb_model = joblib.load("models/xgb_longitudinal.pkl")
preprocessor = joblib.load("models/preprocessor_cross_sectional.pkl")

# Make predictions
X_new_trf = preprocessor.transform(X_new)
predictions = rf_model.predict(X_new_trf)
```

## Model Performance

### Cross-Sectional Random Forest
- **ROC-AUC**: ~0.93
- **F1 Score**: ~0.86
- **Accuracy**: ~0.89

### Longitudinal XGBoost
- **ROC-AUC**: ~0.98
- **F1 Score**: ~0.90
- **Accuracy**: ~0.90

## 🔧 Configuration

Edit `configs/config.yaml` to modify:
- Data file paths
- Model hyperparameter search ranges
- Training parameters (test_size, random_state)
- Number of optimization trials

## 📚 Documentation Files

- **README.md**: Main project documentation with installation and usage
- **STRUCTURE.md**: Detailed file-by-file breakdown
- **QUICK_START.md**: Quick reference guide
- **REPOSITORY_OVERVIEW.md**: This file - high-level overview

## 🎓 Key Concepts

1. **Modular Design**: Each component is in its own module for reusability
2. **Pipeline-Based**: Uses sklearn pipelines for preprocessing
3. **Subject-Level Splitting**: Longitudinal data split at subject level, not visit level
4. **Temporal Features**: Extracts slopes and deltas from longitudinal data
5. **Model Stacking**: Cross-sectional model predictions used as features in longitudinal model
6. **Hyperparameter Optimization**: Optuna for automated tuning

## File Size Notes

- Data files (`.xlsx`): Large, gitignored
- Model files (`.pkl`): Large, gitignored
- Source code: Small, version controlled
- Results/plots: Gitignored

## Development

### Adding New Features

1. Add feature extraction logic to `src/features/feature_engineering.py`
2. Update preprocessing if needed in `src/data/preprocessing.py`
3. Modify training scripts if workflow changes

### Adding New Models

1. Create new file in `src/models/`
2. Implement optimization function similar to existing models
3. Add training script in `scripts/`

### Testing

```bash
# Run tests (when implemented)
pytest tests/
```

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning
- **xgboost**: Gradient boosting
- **optuna**: Hyperparameter optimization
- **matplotlib**: Visualization
- **shap**: Model interpretability
- **openpyxl**: Excel file reading
- **joblib**: Model serialization


## 🙏 Acknowledgments

- OASIS dataset providers
- Open-source ML community


