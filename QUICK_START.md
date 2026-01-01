# Quick Start Guide

## How to Run the Code

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data

Place your data files in the `data/` directory:
- `oasis_cross-sectional-5708aa0a98d82080.xlsx`
- `oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx`

### Step 3: Train the Models

**Run this command to train both models:**
```bash
python scripts/main.py
```

This will:
1. Train the Random Forest model (takes ~5-10 minutes)
2. Train the XGBoost model (takes ~10-20 minutes)
3. Save models to `models/` directory

### Step 4: Make Predictions

**Option A: Run Examples**
```bash
python scripts/example_usage.py
```

**Option B: Use in Your Own Code**

#### Example 1: Single Patient (Dictionary)

```python
from scripts.predict import predict_from_dict_cross_sectional

patient = {
    'M/F': 'M',
    'Hand': 'R',
    'Age': 75,
    'Educ': 16,
    'SES': 2,
    'MMSE': 25,
    'eTIV': 1500,
    'nWBV': 0.75,
    'ASF': 1.2
}

result = predict_from_dict_cross_sectional(patient)
print(f"Prediction: {result['prediction_label']}")
print(f"Probability: {result['probability']:.2%}")
```

#### Example 2: Multiple Patients (DataFrame)

```python
import pandas as pd
from scripts.predict import predict_cross_sectional

# Load from CSV or create DataFrame
patients_df = pd.read_csv('your_patients.csv')

# Make predictions
predictions, probabilities = predict_cross_sectional(patients_df)

# Add to DataFrame
patients_df['prediction'] = ['Demented' if p == 1 else 'Non-Demented' 
                             for p in predictions]
patients_df['probability'] = probabilities

# Save results
patients_df.to_csv('results.csv', index=False)
```

## Required Input Format

### For Cross-Sectional Predictions

Your data must have these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| M/F | string | Male or Female | 'M' or 'F' |
| Hand | string | Handedness | 'R' or 'L' |
| Age | numeric | Age in years | 75 |
| Educ | numeric | Years of education | 16 |
| SES | numeric | Socioeconomic status (1-5) | 2 |
| MMSE | numeric | MMSE score (0-30) | 25 |
| eTIV | numeric | Estimated Total Intracranial Volume | 1500 |
| nWBV | numeric | Normalized Whole Brain Volume | 0.75 |
| ASF | numeric | Atlas Scaling Factor | 1.2 |

### For Longitudinal Predictions

All columns above, plus:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| Subject ID | string | Unique subject identifier | 'PATIENT_001' |
| Visit | numeric | Visit number | 1, 2, 3... |
| MR Delay | numeric | Days since first visit | 0, 365, 730... |

## Common Use Cases

### Use Case 1: Predict from Single Visit

```python
from scripts.predict import predict_from_dict_cross_sectional

patient = {
    'M/F': 'F', 'Hand': 'R', 'Age': 68, 'Educ': 12, 'SES': 3,
    'MMSE': 28, 'eTIV': 1400, 'nWBV': 0.78, 'ASF': 1.15
}

result = predict_from_dict_cross_sectional(patient)
```

### Use Case 2: Batch Predictions from CSV

```python
import pandas as pd
from scripts.predict import predict_cross_sectional

df = pd.read_csv('patients.csv')
predictions, probs = predict_cross_sectional(df)
df['prediction'] = predictions
df['probability'] = probs
df.to_csv('results.csv', index=False)
```

### Use Case 3: Longitudinal Analysis

```python
import pandas as pd
from scripts.predict import predict_longitudinal

# Load longitudinal data (multiple visits per patient)
ldf = pd.read_excel('longitudinal_data.xlsx')

# Get predictions
results = predict_longitudinal(ldf)
print(results)
```

## Troubleshooting

### Error: "Model files not found"

**Solution:** Train the models first:
```bash
python scripts/main.py
```

### Error: "Missing required columns"

**Solution:** Make sure your data has all required columns. Check the column names match exactly (case-sensitive).

### Error: "Cannot find data files"

**Solution:** Place your Excel files in the `data/` directory with the exact filenames:
- `data/oasis_cross-sectional-5708aa0a98d82080.xlsx`
- `data/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx`

## Need Help?

1. Check `README.md` for detailed documentation
2. Run `python scripts/example_usage.py` to see examples
3. Check `STRUCTURE.md` for detailed file structure
4. Check `REPOSITORY_OVERVIEW.md` for high-level overview

