# Repository Structure Documentation

This document provides a detailed breakdown of the repository structure and what goes into each file.

## Directory Structure

```
alzheimers-ml/
├── src/                          # Main source code package
│   ├── __init__.py              # Package initialization
│   ├── data/                    # Data handling modules
│   ├── features/                # Feature engineering
│   ├── models/                  # ML model definitions
│   ├── evaluation/              # Evaluation and metrics
│   └── utils/                   # Utility functions
├── scripts/                      # Executable training scripts
├── notebooks/                    # Jupyter notebooks (optional)
├── data/                        # Data files (gitignored)
├── models/                      # Saved models (gitignored)
├── configs/                     # Configuration files
├── tests/                       # Unit tests
├── results/                     # Output results
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── README.md                    # Main documentation
├── QUICK_START.md               # Quick start guide
├── STRUCTURE.md                 # This file
└── REPOSITORY_OVERVIEW.md       # Repository overview
```

## File-by-File Breakdown

### Root Level Files

#### `README.md`
**Purpose**: Main project documentation  
**Contents**:
- Project overview
- Installation instructions
- Usage examples
- Model performance metrics
- Configuration guide

#### `requirements.txt`
**Purpose**: Python package dependencies  
**Contents**:
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
optuna>=3.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
joblib>=1.3.0
shap>=0.42.0
```

#### `scripts/generate_visualizations.py`
**Purpose**: Generate visualizations from trained models  
**Workflow**:
1. Load trained models
2. Load test data
3. Generate ROC curves
4. Generate calibration curves
5. Generate feature importance plots
6. Generate SHAP plots (if available)

**Usage**:
```bash
# Generate all visualizations
python scripts/generate_visualizations.py

# Generate only cross-sectional visualizations
python scripts/generate_visualizations.py --model cross

# Generate only longitudinal visualizations
python scripts/generate_visualizations.py --model long
```

**Outputs**:
- `results/rf_roc_curve.png`
- `results/rf_calibration_curve.png`
- `results/rf_feature_importance.png`
- `results/rf_shap_summary.png`
- `results/xgb_roc_curve.png`
- `results/xgb_calibration_curve.png`
- `results/xgb_feature_importance.png`
- `results/xgb_shap_summary.png`

#### `.gitignore`
**Purpose**: Exclude files from version control  
**Excludes**:
- Python cache files
- Virtual environments
- Data files (.xlsx, .csv)
- Model files (.pkl, .joblib)
- Jupyter notebooks
- IDE files

---

### `src/` - Source Code Package

#### `src/__init__.py`
**Purpose**: Package initialization  
**Code**:
```python
"""
Alzheimer's Disease Prediction ML Project
"""
__version__ = "1.0.0"
```

---

### `src/data/` - Data Handling

#### `src/data/__init__.py`
**Purpose**: Data module initialization  
**Code**:
```python
"""Data loading and preprocessing modules"""
```

#### `src/data/load_data.py`
**Purpose**: Data loading utilities  
**Key Functions**:

1. **`load_cross_sectional_data(data_path)`**
   - Loads Excel file
   - Drops rows with missing CDR
   - Converts CDR to binary (0/1)
   - Returns: DataFrame

2. **`load_longitudinal_data(data_path)`**
   - Loads longitudinal Excel file
   - Creates Group_binary column
   - Converts CDR to binary
   - Handles missing MR Delay values
   - Converts numeric columns
   - Returns: DataFrame

3. **`prepare_train_test_split(df, target_col, test_size, random_state, drop_cols)`**
   - Splits data into train/test
   - Drops specified columns
   - Returns: X_train, X_test, y_train, y_test

**Example Usage**:
```python
from src.data.load_data import load_cross_sectional_data

df = load_cross_sectional_data("data/oasis_cross-sectional-5708aa0a98d82080.xlsx")
```

#### `src/data/preprocessing.py`
**Purpose**: Data preprocessing pipelines  
**Key Functions**:

1. **`create_preprocessing_pipeline(X_train)`**
   - Identifies numeric and categorical columns
   - Creates separate pipelines for each type
   - Numeric: Median imputation
   - Categorical: Mode imputation + OneHot encoding
   - Returns: Fitted ColumnTransformer, column lists

2. **`create_longitudinal_preprocessing_pipeline(X_train)`**
   - Similar to above but for longitudinal features
   - Handles different column types

3. **`encode_labels(y_train, y_test=None)`**
   - Encodes target labels using LabelEncoder
   - Returns: Encoded labels, encoder

**Example Usage**:
```python
from src.data.preprocessing import create_preprocessing_pipeline

preprocessor, num_cols, bin_cols = create_preprocessing_pipeline(X_train)
X_train_trf = preprocessor.fit_transform(X_train)
```

---

### `src/features/` - Feature Engineering

#### `src/features/__init__.py`
**Purpose**: Features module initialization  
**Code**:
```python
"""Feature engineering modules"""
```

#### `src/features/feature_engineering.py`
**Purpose**: Extract features from longitudinal data  
**Key Functions**:

1. **`extract_longitudinal_features(df, is_training, cdr_predictor, preprocessor, label_encoder)`**
   - Groups data by Subject ID
   - Extracts per-subject features:
     - **Static**: Sex, Hand, Educ, SES
     - **Baseline**: MMSE, nWBV, Age, eTIV, ASF, CDR_pred
     - **Delta**: MMSE_delta, nWBV_delta
     - **Slope**: MMSE_slope, nWBV_slope (per year)
     - **Time**: n_visits, followup_years
     - **Indicator**: has_multiple_visits
     - **Target**: target_group (if training)
   - Returns: DataFrame (one row per subject)

2. **`prepare_longitudinal_train_test_split(ldf, test_size, random_state, ...)`**
   - Splits at subject level (not visit level)
   - Uses stratification
   - Calls extract_longitudinal_features for train/test
   - Returns: X_train, X_test, y_train, y_test

**Example Usage**:
```python
from src.features.feature_engineering import extract_longitudinal_features

df_long = extract_longitudinal_features(ldf, is_training=True)
```

---

### `src/models/` - Model Definitions

#### `src/models/__init__.py`
**Purpose**: Models module initialization  
**Code**:
```python
"""Model training and optimization modules"""
```

#### `src/models/random_forest.py`
**Purpose**: Random Forest with Optuna optimization  
**Key Functions**:

1. **`rf_objective(X_train, y_train, trial)`**
   - Optuna objective function
   - Suggests hyperparameters:
     - n_estimators: 200-800
     - max_depth: 3-20
     - min_samples_split: 2-10
     - min_samples_leaf: 1-5
     - max_features: ["sqrt", "log2"]
   - Uses 5-fold stratified CV
   - Returns: Mean ROC-AUC score

2. **`optimize_random_forest(X_train, y_train, n_trials, random_state)`**
   - Creates Optuna study
   - Optimizes hyperparameters
   - Trains best model
   - Returns: Study object, trained model

**Example Usage**:
```python
from src.models.random_forest import optimize_random_forest

study, best_model = optimize_random_forest(X_train, y_train, n_trials=75)
```

#### `src/models/xgboost_model.py`
**Purpose**: XGBoost with Optuna optimization  
**Key Functions**:

1. **`xgb_objective(X_train, y_train, trial)`**
   - Optuna objective function
   - Suggests hyperparameters:
     - n_estimators: 100-600
     - max_depth: 3-8
     - learning_rate: 0.01-0.3 (log scale)
     - subsample: 0.6-1.0
     - colsample_bytree: 0.6-1.0
     - gamma: 0-5
     - reg_alpha: 0-5
     - reg_lambda: 1-10
     - scale_pos_weight: 1-5
   - Uses 5-fold stratified CV
   - Returns: Mean ROC-AUC score

2. **`optimize_xgboost(X_train, y_train, n_trials, random_state)`**
   - Creates Optuna study
   - Optimizes hyperparameters
   - Trains best model
   - Returns: Study object, trained model

**Example Usage**:
```python
from src.models.xgboost_model import optimize_xgboost

study, best_model = optimize_xgboost(X_train, y_train, n_trials=300)
```

---

### `src/evaluation/` - Evaluation

#### `src/evaluation/__init__.py`
**Purpose**: Evaluation module initialization  
**Code**:
```python
"""Model evaluation and visualization modules"""
```

#### `src/evaluation/metrics.py`
**Purpose**: Evaluation metrics and reporting  
**Key Functions**:

1. **`evaluate_model(model, X_test, y_test, y_pred, y_proba)`**
   - Computes comprehensive metrics:
     - Accuracy
     - F1 Score
     - ROC-AUC
     - Brier Score
     - Classification Report
     - Confusion Matrix
   - Returns: Dictionary of metrics

2. **`print_evaluation_results(metrics)`**
   - Pretty-prints evaluation results
   - Formats output

3. **`cross_validate_model(model, X, y, cv_folds, scoring)`**
   - Performs cross-validation
   - Returns: Mean, std, and all scores

**Example Usage**:
```python
from src.evaluation.metrics import evaluate_model, print_evaluation_results

metrics = evaluate_model(model, X_test, y_test)
print_evaluation_results(metrics)
```

#### `src/evaluation/visualization.py`
**Purpose**: Visualization utilities  
**Key Functions**:

1. **`plot_calibration_curve(y_true, y_proba, model_name, n_bins)`**
   - Plots calibration curve
   - Shows predicted vs observed probabilities

2. **`plot_roc_curve(y_true, y_proba, model_name)`**
   - Plots ROC curve
   - Shows AUC score

3. **`plot_feature_importance(model, feature_names, max_features)`**
   - Plots feature importance
   - Works with XGBoost models

4. **`plot_shap_summary(model, X_test, feature_names, class_idx)`**
   - Generates SHAP summary plot
   - Requires shap library

**Example Usage**:
```python
from src.evaluation.visualization import plot_calibration_curve

plot_calibration_curve(y_test, y_proba, "XGBoost")
plt.show()
```

---

### `src/utils/` - Utilities

#### `src/utils/__init__.py`
**Purpose**: Utility module initialization  
**Code**:
```python
"""Utility functions"""
```

---

### `scripts/` - Training Scripts

#### `scripts/train_cross_sectional.py`
**Purpose**: Train Random Forest on cross-sectional data  
**Workflow**:
1. Load cross-sectional data
2. Prepare train-test split
3. Create preprocessing pipeline
4. Encode labels
5. Optimize and train Random Forest
6. Evaluate model
7. Save model, preprocessor, encoder

**Usage**:
```bash
python scripts/train_cross_sectional.py
```

**Outputs**:
- `models/rf_cross_sectional.pkl`
- `models/preprocessor_cross_sectional.pkl`
- `models/label_encoder.pkl`

#### `scripts/train_longitudinal.py`
**Purpose**: Train XGBoost on longitudinal data  
**Workflow**:
1. Load longitudinal data
2. Extract longitudinal features (uses CDR model if available)
3. Create preprocessing pipeline
4. Optimize and train XGBoost
5. Evaluate model
6. Save model and preprocessor

**Usage**:
```bash
python scripts/train_longitudinal.py
```

**Outputs**:
- `models/xgb_longitudinal.pkl`
- `models/preprocessor_longitudinal.pkl`

#### `scripts/main.py`
**Purpose**: Train both models sequentially  
**Workflow**:
1. Train cross-sectional model
2. Use cross-sectional model to predict CDR for longitudinal features
3. Train longitudinal model
4. Print summary

**Usage**:
```bash
python scripts/main.py
```

#### `scripts/predict.py`
**Purpose**: Make predictions on new data  
**Key Functions**:
- `predict_cross_sectional()`: Predict using Random Forest model
- `predict_longitudinal()`: Predict using XGBoost model
- `predict_from_dict_cross_sectional()`: Convenience function for dictionary input

#### `scripts/example_usage.py`
**Purpose**: Example demonstrations of prediction functions  
**Shows**:
- Single patient prediction
- Multiple patients prediction
- Longitudinal prediction

---

### `configs/` - Configuration

#### `configs/config.yaml`
**Purpose**: Centralized configuration  
**Contents**:
- Data file paths
- Model save paths
- Hyperparameter search settings
- Training parameters
- Evaluation metrics

**Example**:
```yaml
data:
  cross_sectional_path: "data/oasis_cross-sectional-5708aa0a98d82080.xlsx"
  longitudinal_path: "data/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"

models:
  rf:
    n_trials: 75
  xgb:
    n_trials: 300
```

---

### Other Directories

#### `data/`
- Contains input data files
- Gitignored (too large for repo)
- Place your `.xlsx` files here

#### `models/`
- Contains saved model files
- Gitignored
- Created automatically during training

#### `notebooks/`
- Optional Jupyter notebooks
- For exploration and experimentation

#### `tests/`
- Unit tests (to be implemented)
- Use pytest framework

#### `results/`
- Output files (plots, reports)
- Gitignored

---

## Data Flow

### Cross-Sectional Pipeline
```
Raw Data → Load → Preprocess → Train RF → Evaluate → Save Model
```

### Longitudinal Pipeline
```
Raw Data → Load → Feature Engineering → Preprocess → Train XGBoost → Evaluate → Save Model
```

### Full Pipeline
```
Cross-Sectional Data → RF Model → CDR Predictions
                                                    ↓
Longitudinal Data → Feature Engineering (with CDR) → XGBoost Model
```

---

## Key Design Decisions

1. **Modular Structure**: Each component is in its own module for reusability
2. **Separation of Concerns**: Data, features, models, and evaluation are separate
3. **Pipeline-Based**: Uses sklearn pipelines for preprocessing
4. **Hyperparameter Optimization**: Optuna for automated tuning
5. **Subject-Level Splitting**: Longitudinal data split at subject level, not visit level
6. **Feature Engineering**: Extracts temporal features (slopes, deltas) from longitudinal data

---

## Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Place data files** in `data/` directory
3. **Run training**: `python scripts/main.py`
4. **Check results** in console output and saved models

---

## Notes

- All paths are relative to repository root
- Models are saved using joblib
- Random state is set to 42 for reproducibility
- Cross-sectional model is used as a feature in longitudinal model



