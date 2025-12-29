import sys
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.load_data import load_cross_sectional_data
from src.data.preprocessing import create_preprocessing_pipeline
from src.features.feature_engineering import extract_longitudinal_features


def predict_cross_sectional(X_new, model_path="models/rf_cross_sectional.pkl", 
                           preprocessor_path="models/preprocessor_cross_sectional.pkl"):
    # Load model and preprocessor
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files not found.\nError: {e}")
    
    # Convert dict to DataFrame if needed
    if isinstance(X_new, dict):
        X_new = pd.DataFrame([X_new])
    
    # Ensure DataFrame
    if not isinstance(X_new, pd.DataFrame):
        raise ValueError(f"{e}")
    
    # Preprocess
    X_new_trf = preprocessor.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_new_trf)
    probabilities = model.predict_proba(X_new_trf)[:, 1]
    
    return predictions, probabilities


def predict_longitudinal(ldf_new, model_path="models/xgb_longitudinal.pkl",
                        preprocessor_path="models/preprocessor_longitudinal.pkl",
                        rf_model_path="models/rf_cross_sectional.pkl",
                        rf_preprocessor_path="models/preprocessor_cross_sectional.pkl"):
    # Load models
    try:
        xgb_model = joblib.load(model_path)
        long_preprocessor = joblib.load(preprocessor_path)
        rf_model = joblib.load(rf_model_path)
        rf_preprocessor = joblib.load(rf_preprocessor_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}")
    
    # Extract features
    df_long = extract_longitudinal_features(
        ldf_new,
        is_training=False,
        cdr_predictor=rf_model,
        preprocessor=rf_preprocessor,
        label_encoder=None
    )
    
    # Prepare features (drop Subject ID and Hand if present)
    feature_cols = df_long.drop(columns=['Subject ID'], errors='ignore').columns
    X_new = df_long[feature_cols]
    
    # Preprocess
    X_new_trf = long_preprocessor.transform(X_new)
    
    # Make predictions
    predictions = xgb_model.predict(X_new_trf)
    probabilities = xgb_model.predict_proba(X_new_trf)[:, 1]
    
    # Add predictions to DataFrame
    df_long['prediction'] = predictions
    df_long['probability'] = probabilities
    df_long['prediction_label'] = df_long['prediction'].map({0: 'Non-Demented', 1: 'Demented'})
    
    return df_long[['Subject ID', 'prediction', 'probability', 'prediction_label']]


def predict_from_dict_cross_sectional(patient_data):
    pred, prob = predict_cross_sectional(patient_data)
    
    return {
        'prediction': int(pred[0]),
        'prediction_label': 'Demented' if pred[0] == 1 else 'Non-Demented',
        'probability': float(prob[0]),
        'confidence': 'High' if prob[0] > 0.8 or prob[0] < 0.2 else 'Medium' if prob[0] > 0.6 or prob[0] < 0.4 else 'Low'
    }


def main():
    print("Alzheimer's Disease Prediction")
    
    # Example 1: Cross-sectional prediction from dictionary
    print("Example 1: Cross-Sectional Prediction from Dictionary")
    
    example_patient = {
        'M/F': 'M',      # Male or Female
        'Hand': 'R',      # Right or Left handed
        'Age': 75,       # Age in years
        'Educ': 16,      # Years of education
        'SES': 2,        # Socioeconomic status (1-5)
        'MMSE': 25,      # Mini-Mental State Examination score
        'eTIV': 1500,    # Estimated Total Intracranial Volume
        'nWBV': 0.75,    # Normalized Whole Brain Volume
        'ASF': 1.2       # Atlas Scaling Factor
    }
    
    try:
        result = predict_from_dict_cross_sectional(example_patient)
        print(f"Patient Data: {example_patient}")
        print(f"Prediction: {result['prediction_label']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']}")
    except FileNotFoundError as e:
        print(f"{e}")
        print("Please train the model first:")
        print("python scripts/train_cross_sectional.py")
    
    # Example 2: Cross-sectional prediction from DataFrame
    print("Example 2: Cross-Sectional Prediction from DataFrame")
    print("-" * 70)
    
    example_patients_df = pd.DataFrame([
        {'M/F': 'F', 'Hand': 'R', 'Age': 68, 'Educ': 12, 'SES': 3, 
         'MMSE': 28, 'eTIV': 1400, 'nWBV': 0.78, 'ASF': 1.15},
        {'M/F': 'M', 'Hand': 'R', 'Age': 82, 'Educ': 14, 'SES': 2, 
         'MMSE': 20, 'eTIV': 1600, 'nWBV': 0.68, 'ASF': 1.3}
    ])
    
    try:
        predictions, probabilities = predict_cross_sectional(example_patients_df)
        print(f"\nPredictions for {len(example_patients_df)} patients:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            label = 'Demented' if pred == 1 else 'Non-Demented'
            print(f"  Patient {i+1}: {label} (probability: {prob:.4f})")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train the model first:")
        print("  python scripts/train_cross_sectional.py")
    
    print("For longitudinal predictions, use predict_longitudinal() function")


if __name__ == "__main__":
    main()

