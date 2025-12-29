import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.load_data import load_cross_sectional_data, prepare_train_test_split
from src.data.preprocessing import create_preprocessing_pipeline, encode_labels
from src.models.random_forest import optimize_random_forest
from src.evaluation.metrics import evaluate_model, print_evaluation_results
import joblib


def main():
    # Configuration
    data_path = "data/oasis_cross-sectional-5708aa0a98d82080.xlsx"
    model_save_path = "models/rf_cross_sectional.pkl"
    preprocessor_save_path = "models/preprocessor_cross_sectional.pkl"
    encoder_save_path = "models/label_encoder.pkl"
    
    print("Training Cross-Sectional Random Forest Model")
    
    df = load_cross_sectional_data(data_path)
    print(f"Loaded {len(df)} samples")
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    preprocessor, num_cols, bin_cols = create_preprocessing_pipeline(X_train)
    X_train_trf = preprocessor.fit_transform(X_train)
    X_test_trf = preprocessor.transform(X_test)
    
    y_train_trf, y_test_trf, label_encoder = encode_labels(y_train, y_test)
    
    study_rf, best_rf = optimize_random_forest(
        X_train_trf, 
        y_train_trf, 
        n_trials=75
    )
    
    metrics = evaluate_model(best_rf, X_test_trf, y_test_trf)
    print_evaluation_results(metrics)
    
    joblib.dump(best_rf, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)
    joblib.dump(label_encoder, encoder_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Preprocessor saved to {preprocessor_save_path}")
    print(f"Label encoder saved to {encoder_save_path}")
    
    return best_rf, preprocessor, label_encoder, metrics


if __name__ == "__main__":
    main()


