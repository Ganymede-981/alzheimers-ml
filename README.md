# Alzheimer's Disease Prediction ML Project

A machine learning project for predicting Alzheimer's disease progression using the OASIS dataset. This repository contains models for both cross-sectional and longitudinal analysis.

## 📁 Repository Structure

```
alzheimers-ml/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── load_data.py          # Data loading utilities
│   │   └── preprocessing.py      # Preprocessing pipelines
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py # Longitudinal feature extraction
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   ├── random_forest.py      # RF with Optuna optimization
│   │   └── xgboost_model.py      # XGBoost with Optuna optimization
│   ├── evaluation/               # Evaluation utilities
│   │   ├── __init__.py
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── visualization.py      # Visualization functions
│   └── utils/                    # Utility functions
│       └── __init__.py
├── scripts/                      # Training and prediction scripts
│   ├── train_cross_sectional.py # Train RF on cross-sectional data
│   ├── train_longitudinal.py    # Train XGBoost on longitudinal data
│   ├── main.py                  # Full training pipeline
│   ├── predict.py               # Prediction functions
│   ├── example_usage.py         # Example usage demonstrations
│   └── generate_visualizations.py   # Generate model visualizations
├── notebooks/                    # Jupyter notebooks (optional)
├── data/                         # Data directory (not in repo)
│   ├── oasis_cross-sectional-5708aa0a98d82080.xlsx
│   └── oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx
├── models/                       # Saved models (not in repo)
├── configs/                      # Configuration files
│   └── config.yaml
├── tests/                        # Unit tests
├── results/                      # Results and outputs
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── QUICK_START.md                # Quick start guide
├── STRUCTURE.md                  # Detailed structure documentation
└── REPOSITORY_OVERVIEW.md        # Repository overview
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd alzheimers-ml
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your data files in the `data/` directory:
   - `oasis_cross-sectional-5708aa0a98d82080.xlsx`
   - `oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx`

## Usage

### Step 1: Training Models

** Important:** You must train the models before making predictions!

#### Option 1: Train Full Pipeline (Recommended)
Trains both cross-sectional and longitudinal models sequentially:
```bash
python scripts/main.py
```

This will:
- Train the Random Forest model on cross-sectional data
- Save the model to `models/rf_cross_sectional.pkl`
- Train the XGBoost model on longitudinal data
- Save the model to `models/xgb_longitudinal.pkl`

#### Option 2: Train Models Separately

**Train Cross-Sectional Model:**
```bash
python scripts/train_cross_sectional.py
```

**Train Longitudinal Model:**
```bash
python scripts/train_longitudinal.py
```

**Note:** The longitudinal model requires the cross-sectional model to be trained first.

### Step 2: Making Predictions on Your Own Data

After training, you can make predictions on new patient data using the prediction scripts.

#### Quick Start - Run Examples

See example usage:
```bash
python scripts/example_usage.py
```

#### Method 1: Single Patient Prediction (Dictionary Input)

Create a Python script or use the interactive Python:

```python
from scripts.predict import predict_from_dict_cross_sectional

# Your patient data
patient = {
    'M/F': 'M',      # Male or Female
    'Hand': 'R',      # Right or Left handed
    'Age': 75,        # Age in years
    'Educ': 16,       # Years of education
    'SES': 2,         # Socioeconomic status (1-5)
    'MMSE': 25,       # Mini-Mental State Examination score (0-30)
    'eTIV': 1500,     # Estimated Total Intracranial Volume
    'nWBV': 0.75,     # Normalized Whole Brain Volume
    'ASF': 1.2        # Atlas Scaling Factor
}

# Get prediction
result = predict_from_dict_cross_sectional(patient)

print(f"Prediction: {result['prediction_label']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Confidence: {result['confidence']}")
```

#### Method 2: Multiple Patients (DataFrame Input)

```python
import pandas as pd
from scripts.predict import predict_cross_sectional

# Your patient data as DataFrame
patients_df = pd.DataFrame([
    {
        'M/F': 'F', 'Hand': 'R', 'Age': 68, 'Educ': 12, 'SES': 3,
        'MMSE': 28, 'eTIV': 1400, 'nWBV': 0.78, 'ASF': 1.15
    },
    {
        'M/F': 'M', 'Hand': 'R', 'Age': 82, 'Educ': 14, 'SES': 2,
        'MMSE': 20, 'eTIV': 1600, 'nWBV': 0.68, 'ASF': 1.3
    }
])

# Get predictions
predictions, probabilities = predict_cross_sectional(patients_df)

# Display results
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = 'Demented' if pred == 1 else 'Non-Demented'
    print(f"Patient {i+1}: {label} (probability: {prob:.2%})")
```

#### Method 3: Using CSV File

```python
import pandas as pd
from scripts.predict import predict_cross_sectional

# Load your CSV file
# Make sure it has these columns: M/F, Hand, Age, Educ, SES, MMSE, eTIV, nWBV, ASF
patients_df = pd.read_csv('your_patients.csv')

# Get predictions
predictions, probabilities = predict_cross_sectional(patients_df)

# Add predictions to DataFrame
patients_df['prediction'] = ['Demented' if p == 1 else 'Non-Demented' for p in predictions]
patients_df['probability'] = probabilities

# Save results
patients_df.to_csv('predictions_results.csv', index=False)
print("Predictions saved to predictions_results.csv")
```

#### Method 4: Longitudinal Predictions

For patients with multiple visits:

```python
import pandas as pd
from scripts.predict import predict_longitudinal

# Your longitudinal data
# Each row is a visit, multiple rows per subject
ldf = pd.DataFrame([
    {
        'Subject ID': 'PATIENT_001',
        'Visit': 1,
        'M/F': 'M', 'Hand': 'R', 'Age': 70, 'EDUC': 16, 'SES': 2,
        'MMSE': 28, 'eTIV': 1500, 'nWBV': 0.75, 'ASF': 1.2,
        'MR Delay': 0  # Days since first visit
    },
    {
        'Subject ID': 'PATIENT_001',
        'Visit': 2,
        'M/F': 'M', 'Hand': 'R', 'Age': 72, 'EDUC': 16, 'SES': 2,
        'MMSE': 25, 'eTIV': 1500, 'nWBV': 0.72, 'ASF': 1.2,
        'MR Delay': 730  # 2 years later
    }
])

# Get predictions
results = predict_longitudinal(ldf)
print(results)
```

### Required Input Columns

**For Cross-Sectional Predictions:**
- `M/F`: Male ('M') or Female ('F')
- `Hand`: Right ('R') or Left ('L') handed
- `Age`: Age in years (numeric)
- `Educ`: Years of education (numeric)
- `SES`: Socioeconomic status, 1-5 (numeric)
- `MMSE`: Mini-Mental State Examination score, 0-30 (numeric)
- `eTIV`: Estimated Total Intracranial Volume (numeric)
- `nWBV`: Normalized Whole Brain Volume, typically 0.6-0.9 (numeric)
- `ASF`: Atlas Scaling Factor (numeric)

**For Longitudinal Predictions:**
- All columns above, plus:
- `Subject ID`: Unique identifier for each subject (string)
- `Visit`: Visit number (numeric)
- `MR Delay`: Days since first visit (numeric)

### Model Workflow

1. **Cross-Sectional Model (Random Forest)**
   - Trained on baseline features
   - Predicts CDR (Clinical Dementia Rating) from single visit
   - Used as feature for longitudinal model

2. **Longitudinal Model (XGBoost)**
   - Trained on engineered features from multiple visits
   - Uses CDR predictions from cross-sectional model
   - Predicts dementia progression (Group_binary)

## Code Structure Details

### Data Loading (`src/data/load_data.py`)

**Key Functions:**
- `load_cross_sectional_data()`: Loads and preprocesses cross-sectional data
- `load_longitudinal_data()`: Loads and preprocesses longitudinal data
- `prepare_train_test_split()`: Creates train-test splits

**Example:**
```python
from src.data.load_data import load_cross_sectional_data

df = load_cross_sectional_data("data/oasis_cross-sectional-5708aa0a98d82080.xlsx")
```

### Preprocessing (`src/data/preprocessing.py`)

**Key Functions:**
- `create_preprocessing_pipeline()`: Creates pipeline for cross-sectional data
- `create_longitudinal_preprocessing_pipeline()`: Creates pipeline for longitudinal data
- `encode_labels()`: Encodes target labels

**Example:**
```python
from src.data.preprocessing import create_preprocessing_pipeline

preprocessor, num_cols, bin_cols = create_preprocessing_pipeline(X_train)
X_train_trf = preprocessor.fit_transform(X_train)
```

### Feature Engineering (`src/features/feature_engineering.py`)

**Key Functions:**
- `extract_longitudinal_features()`: Extracts features per subject from longitudinal data
- `prepare_longitudinal_train_test_split()`: Creates subject-level train-test split

**Features Extracted:**
- Baseline features (MMSE, nWBV, Age, etc.)
- Delta features (change from first to last visit)
- Slope features (rate of change per year)
- Time features (follow-up duration, number of visits)
- CDR prediction from cross-sectional model

**Example:**
```python
from src.features.feature_engineering import extract_longitudinal_features

df_long = extract_longitudinal_features(ldf, is_training=True)
```

### Model Training (`src/models/`)

**Random Forest (`random_forest.py`):**
- Optuna hyperparameter optimization
- Optimizes: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
- Uses ROC-AUC as optimization metric

**XGBoost (`xgboost_model.py`):**
- Optuna hyperparameter optimization
- Optimizes: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, regularization
- Uses ROC-AUC as optimization metric

**Example:**
```python
from src.models.random_forest import optimize_random_forest

study, best_model = optimize_random_forest(X_train, y_train, n_trials=75)
```

### Evaluation (`src/evaluation/`)

**Metrics (`metrics.py`):**
- `evaluate_model()`: Comprehensive model evaluation
- `print_evaluation_results()`: Formatted output
- `cross_validate_model()`: Cross-validation

**Visualization (`visualization.py`):**
- `plot_calibration_curve()`: Calibration plots
- `plot_roc_curve()`: ROC curves
- `plot_feature_importance()`: Feature importance
- `plot_shap_summary()`: SHAP values (requires shap library)

**Example:**
```python
from src.evaluation.metrics import evaluate_model, print_evaluation_results

metrics = evaluate_model(model, X_test, y_test)
print_evaluation_results(metrics)
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

## Configuration

Edit `configs/config.yaml` to modify:
- Data paths
- Model hyperparameters
- Training parameters
- Evaluation metrics

## Visualizations

After training, visualizations are automatically generated and saved to the `results/` directory:

- **ROC Curves**: Show model discrimination ability
- **Calibration Curves**: Show prediction probability calibration
- **Feature Importance**: Shows which features are most important
- **SHAP Plots**: Explains individual predictions (requires `shap` library)

To regenerate visualizations without retraining:
```bash
# Generate all visualizations
python scripts/generate_visualizations.py

# Generate only cross-sectional visualizations
python scripts/generate_visualizations.py --model cross

# Generate only longitudinal visualizations
python scripts/generate_visualizations.py --model long
```

## Testing

Run tests (when implemented):
```bash
pytest tests/
```

## Dependencies

Key libraries:
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning
- `xgboost`: Gradient boosting
- `optuna`: Hyperparameter optimization
- `matplotlib`: Visualization
- `shap`: Model interpretability

See `requirements.txt` for full list.
[Add contributors here]

## 🙏 Acknowledgments

- OASIS dataset providers
- Open-source ML community


