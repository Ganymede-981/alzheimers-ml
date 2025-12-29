import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_cross_sectional import main as train_cross_sectional
from scripts.train_longitudinal import main as train_longitudinal
import joblib


def main():
    
    print("Training Cross-Sectional Model \n")
    rf_model, preprocessor, label_encoder, rf_metrics = train_cross_sectional()
    
    print("Training Longitudinal Model \n")
    xgb_model, long_preprocessor, feature_names, xgb_metrics = train_longitudinal(
        cdr_model=rf_model,
        cdr_preprocessor=preprocessor,
        cdr_encoder=label_encoder
    )

    print("Summary\n")
    print("Cross-Sectional Random Forest:")
    print(f"ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    print(f"F1 Score: {rf_metrics['f1_score']:.4f}")
    print(f"Accuracy: {rf_metrics['accuracy']:.4f}")
    
    print("Longitudinal XGBoost:")
    print(f"ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
    print(f"F1 Score: {xgb_metrics['f1_score']:.4f}")
    print(f"Accuracy: {xgb_metrics['accuracy']:.4f}")

    print("All models saved successfully!")


if __name__ == "__main__":
    main()


