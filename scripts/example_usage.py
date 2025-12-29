import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.predict import predict_from_dict_cross_sectional, predict_cross_sectional, predict_longitudinal


def example_cross_sectional_single_patient():
    print("Example 1: Single Patient Prediction (Cross-Sectional)")
    
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
    
    print(f"Age: {patient['Age']} years")
    print(f"Sex: {patient['M/F']}")
    print(f"Education: {patient['Educ']} years")
    print(f"MMSE Score: {patient['MMSE']}/30")
    print(f"Brain Volume (nWBV): {patient['nWBV']}")
    
    try:
        result = predict_from_dict_cross_sectional(patient)
        
        print(f"Prediction: {result['prediction_label']}")
        print(f"Probability: {result['probability']:.2%}")
        print(f"Confidence: {result['confidence']}")
        
        if result['prediction'] == 1:
            print("This patient shows signs of dementia.")
        else:
            print("This patient shows no signs of dementia.")
            
    except FileNotFoundError:
        print("Model not found. Please train the model first:")
        print("python scripts/train_cross_sectional.py")


def example_cross_sectional_multiple_patients():
    print("Example 2: Multiple Patients Prediction (Cross-Sectional)")
    
    patients_df = pd.DataFrame([
        {
            'M/F': 'F', 'Hand': 'R', 'Age': 68, 'Educ': 12, 'SES': 3,
            'MMSE': 28, 'eTIV': 1400, 'nWBV': 0.78, 'ASF': 1.15
        },
        {
            'M/F': 'M', 'Hand': 'R', 'Age': 82, 'Educ': 14, 'SES': 2,
            'MMSE': 20, 'eTIV': 1600, 'nWBV': 0.68, 'ASF': 1.3
        },
        {
            'M/F': 'F', 'Hand': 'R', 'Age': 72, 'Educ': 18, 'SES': 1,
            'MMSE': 30, 'eTIV': 1450, 'nWBV': 0.82, 'ASF': 1.1
        }
    ])
    
    print(f"Processing {len(patients_df)} patients... \n")
    
    try:
        predictions, probabilities = predict_cross_sectional(patients_df)
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            label = 'Demented' if pred == 1 else 'Non-Demented'
            patient_info = patients_df.iloc[i]
            print(f"Patient {i+1}: \n")
            print(f"Age: {patient_info['Age']}, MMSE: {patient_info['MMSE']}, nWBV: {patient_info['nWBV']}")
            print(f"Prediction: {label}")
            print(f"Probability: {prob:.2%}")
            
    except FileNotFoundError:
        print("Model not found. Please train the model first:")
        print("python scripts/train_cross_sectional.py")


def example_longitudinal_prediction():
    print("Example 3: Longitudinal Prediction")
    
    ldf_example = pd.DataFrame([
        {
            'Subject ID': 'SUBJECT_001',
            'Visit': 1,
            'M/F': 'M',
            'Hand': 'R',
            'Age': 70,
            'EDUC': 16,
            'SES': 2,
            'MMSE': 28,
            'eTIV': 1500,
            'nWBV': 0.75,
            'ASF': 1.2,
            'MR Delay': 0,
            'Group': 'Nondemented',
            'CDR': 0
        },
        {
            'Subject ID': 'SUBJECT_001',
            'Visit': 2,
            'M/F': 'M',
            'Hand': 'R',
            'Age': 72,
            'EDUC': 16,
            'SES': 2,
            'MMSE': 25,
            'eTIV': 1500,
            'nWBV': 0.72,
            'ASF': 1.2,
            'MR Delay': 730,
            'Group': 'Nondemented',
            'CDR': 0.5
        }
    ])
    
    print("Longitudinal data for Subject SUBJECT_001: \n")
    print(f"Visit 1: Age 70, MMSE 28, nWBV 0.75")
    print(f"Visit 2: Age 72, MMSE 25, nWBV 0.72")
    
    try:
        results = predict_longitudinal(ldf_example)
        
        print("Prediction Results: \n")
        print(results.to_string(index=False))
        
    except FileNotFoundError:
        print(f"{e}")
        print("python scripts/main.py")
    except Exception as e:
        print(f"{e}")
        print("Make sure you have trained both models first")


def main():
    print("Alzheimer's Disease Prediction - Usage Examples")
    print("This script demonstrates how to use the trained models for predictions.")
    print("Make sure you have trained the models first using:")
    print("python scripts/main.py")
    
    example_cross_sectional_single_patient()
    example_cross_sectional_multiple_patients()
    example_longitudinal_prediction()

if __name__ == "__main__":
    main()

