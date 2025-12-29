import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.load_data import load_longitudinal_data
from src.data.preprocessing import create_longitudinal_preprocessing_pipeline
from src.features.feature_engineering import prepare_longitudinal_train_test_split
from src.models.xgboost_model import optimize_xgboost
from src.evaluation.metrics import evaluate_model, print_evaluation_results
import joblib


def main(cdr_model=None, cdr_preprocessor=None, cdr_encoder=None):
    
    data_path = "data/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"
    model_save_path = "models/xgb_longitudinal.pkl"
    preprocessor_save_path = "models/preprocessor_longitudinal.pkl"
    
    print("Training Longitudinal XGBoost Model")
    
    ldf = load_longitudinal_data(data_path)
    print(f"Loaded {len(ldf)} visit records")
    
    X_train_long, X_test_long, y_train_long, y_test_long = prepare_longitudinal_train_test_split(
        ldf,
        cdr_predictor=cdr_model,
        preprocessor=cdr_preprocessor,
        label_encoder=cdr_encoder
    )
    print(f"Train set: {len(X_train_long)} subjects")
    print(f"Test set: {len(X_test_long)} subjects")
    
    preprocessor, numeric_cols, binary_cols = create_longitudinal_preprocessing_pipeline(X_train_long)
    X_train_long_trf = preprocessor.fit_transform(X_train_long)
    X_test_long_trf = preprocessor.transform(X_test_long)

    study, best_model = optimize_xgboost(
        X_train_long_trf,
        y_train_long,
        n_trials=300
    )
    
    metrics = evaluate_model(best_model, X_test_long_trf, y_test_long)
    print_evaluation_results(metrics)
    
    joblib.dump(best_model, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Preprocessor saved to {preprocessor_save_path}")
    
    # Get feature names for later use
    feature_names = X_train_long.drop(columns=['Hand'], errors='ignore').columns.tolist()
    
    return best_model, preprocessor, feature_names, metrics


if __name__ == "__main__":
    # Optionally load cross-sectional model for CDR prediction
    try:
        import joblib
        cdr_model = joblib.load("models/rf_cross_sectional.pkl")
        cdr_preprocessor = joblib.load("models/preprocessor_cross_sectional.pkl")
        cdr_encoder = joblib.load("models/label_encoder.pkl")
        print("Loaded cross-sectional model for CDR prediction")
    except FileNotFoundError:
        print("Cross-sectional model not found. Training without CDR prediction.")
        cdr_model = None
        cdr_preprocessor = None
        cdr_encoder = None
    
    main(cdr_model, cdr_preprocessor, cdr_encoder)


