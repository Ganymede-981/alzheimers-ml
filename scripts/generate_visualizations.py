import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.load_data import load_cross_sectional_data, load_longitudinal_data, prepare_train_test_split
from src.data.preprocessing import create_preprocessing_pipeline, create_longitudinal_preprocessing_pipeline, encode_labels
from src.features.feature_engineering import prepare_longitudinal_train_test_split
from src.evaluation.visualization import (
    plot_roc_curve, 
    plot_calibration_curve, 
    plot_feature_importance,
    plot_shap_summary
)
import matplotlib.pyplot as plt


def generate_cross_sectional_visualizations():
    print("Generating Cross-Sectional Model Visualizations")
    
    # Load model and preprocessor
    try:
        model = joblib.load("models/rf_cross_sectional.pkl")
        preprocessor = joblib.load("models/preprocessor_cross_sectional.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
    except FileNotFoundError as e:
        print(f" {e} \n")
        print("Please train the model first: python scripts/train_cross_sectional.py")
        return
    
    # Load and prepare data
    print("Loading data... \n")
    df = load_cross_sectional_data("data/oasis_cross-sectional-5708aa0a98d82080.xlsx")
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # Preprocess
    X_test_trf = preprocessor.transform(X_test)
    y_test_trf = label_encoder.transform(y_test)
    
    # Generate predictions
    y_proba = model.predict_proba(X_test_trf)[:, 1]
    
    # Get feature names
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    bin_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    feature_names = []
    feature_names.extend(num_cols)
    for col in bin_cols:
        unique_vals = sorted(X_train[col].dropna().unique())
        if len(unique_vals) > 1:
            for val in unique_vals[1:]:
                feature_names.append(f"{col}_{val}")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("Generating ROC Curve... \n")
    plt_roc = plot_roc_curve(y_test_trf, y_proba, "Random Forest (Cross-Sectional)")
    plt_roc.savefig(results_dir / "rf_roc_curve.png", dpi=300, bbox_inches='tight')
    print(f"Saved to results/rf_roc_curve.png")
    plt_roc.close()
    
    print("Generating Calibration Curve... \n")
    plt_cal = plot_calibration_curve(y_test_trf, y_proba, "Random Forest", n_bins=10)
    plt_cal.savefig(results_dir / "rf_calibration_curve.png", dpi=300, bbox_inches='tight')
    print(f"Saved to results/rf_calibration_curve.png")
    plt_cal.close()
    
    print("Generating Feature Importance Plot... \n")
    try:
        n_features = len(model.feature_importances_)
        if len(feature_names) != n_features:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        plt_imp = plot_feature_importance(model, feature_names, max_features=15)
        plt_imp.savefig(results_dir / "rf_feature_importance.png", dpi=300, bbox_inches='tight')
        print(f"Saved to results/rf_feature_importance.png")
        plt_imp.close()
    except Exception as e:
        print(f"{e} \n")
    
    print("Generating SHAP Summary Plot... \n")
    try:
        n_samples = min(100, len(X_test_trf))
        X_test_shap = X_test_trf[:n_samples]
        shap_feature_names = feature_names[:X_test_shap.shape[1]] if len(feature_names) >= X_test_shap.shape[1] else [f"Feature_{i}" for i in range(X_test_shap.shape[1])]
        plt_shap = plot_shap_summary(model, X_test_shap, shap_feature_names, class_idx=1, max_display=15)
        if plt_shap is not None:
            plt_shap.savefig(results_dir / "rf_shap_summary.png", dpi=300, bbox_inches='tight')
            print(f"Saved to results/rf_shap_summary.png")
            plt_shap.close()
    except Exception as e:
        print(f"{e} \n")
    
    print("Visualizations Complete! \n")


def generate_longitudinal_visualizations():
    print("Generating Longitudinal Model Visualizations")
    
    # Load models
    try:
        xgb_model = joblib.load("models/xgb_longitudinal.pkl")
        long_preprocessor = joblib.load("models/preprocessor_longitudinal.pkl")
        rf_model = joblib.load("models/rf_cross_sectional.pkl")
        rf_preprocessor = joblib.load("models/preprocessor_cross_sectional.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
    except FileNotFoundError as e:
        print(f"{e} \n")
        print("Please train the models first: python scripts/main.py")
        return
    
    # Load and prepare data
    print("Loading data... \n")
    ldf = load_longitudinal_data("data/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx")
    
    X_train_long, X_test_long, y_train_long, y_test_long = prepare_longitudinal_train_test_split(
        ldf,
        cdr_predictor=rf_model,
        preprocessor=rf_preprocessor,
        label_encoder=label_encoder
    )
    
    # Preprocess
    X_test_long_trf = long_preprocessor.transform(X_test_long)
    
    # Generate predictions
    y_proba = xgb_model.predict_proba(X_test_long_trf)[:, 1]
    
    # Get feature names
    feature_names = X_train_long.drop(columns=['Hand'], errors='ignore').columns.tolist()
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("Generating ROC Curve... \n")
    plt_roc = plot_roc_curve(y_test_long, y_proba, "XGBoost (Longitudinal)")
    plt_roc.savefig(results_dir / "xgb_roc_curve.png", dpi=300, bbox_inches='tight')
    print(f"Saved to results/xgb_roc_curve.png")
    plt_roc.close()
    
    print("Generating Calibration Curve... \n")
    plt_cal = plot_calibration_curve(y_test_long, y_proba, "XGBoost", n_bins=10)
    plt_cal.savefig(results_dir / "xgb_calibration_curve.png", dpi=300, bbox_inches='tight')
    print(f"Saved to results/xgb_calibration_curve.png")
    plt_cal.close()
    
    print("Generating Feature Importance Plot... \n")
    try:
        plt_imp = plot_feature_importance(xgb_model, feature_names, max_features=15)
        plt_imp.savefig(results_dir / "xgb_feature_importance.png", dpi=300, bbox_inches='tight')
        print(f"Saved to results/xgb_feature_importance.png")
        plt_imp.close()
    except Exception as e:
        print(f"{e} \n")
    
    print("Generating SHAP Summary Plot... \n")
    try:
        n_samples = min(100, len(X_test_long_trf))
        X_test_shap = X_test_long_trf[:n_samples]
        shap_feature_names = feature_names[:X_test_shap.shape[1]] if len(feature_names) >= X_test_shap.shape[1] else [f"Feature_{i}" for i in range(X_test_shap.shape[1])]
        plt_shap = plot_shap_summary(xgb_model, X_test_shap, shap_feature_names, class_idx=1, max_display=15)
        if plt_shap is not None:
            plt_shap.savefig(results_dir / "xgb_shap_summary.png", dpi=300, bbox_inches='tight')
            print(f"Saved to results/xgb_shap_summary.png")
            plt_shap.close()
    except Exception as e:
        print(f"{e} \n")
    
    print("Visualizations Complete! \n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate model visualizations")
    parser.add_argument('--model', choices=['cross', 'long', 'all'], default='all',
                       help='Which model visualizations to generate')
    
    args = parser.parse_args()
    
    if args.model in ['cross', 'all']:
        generate_cross_sectional_visualizations()
    
    if args.model in ['long', 'all']:
        generate_longitudinal_visualizations()


if __name__ == "__main__":
    main()

