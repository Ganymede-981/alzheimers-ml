import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
import xgboost


def plot_calibration_curve(y_true, y_proba, model_name="Model", n_bins=5):

    prob_true, prob_pred = calibration_curve(
        y_true, y_proba,
        n_bins=n_bins,
        strategy="quantile"
    )
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed probability')
    plt.legend()
    plt.title('Calibration Curve')
    plt.grid(True, alpha=0.3)
    return plt


def plot_roc_curve(y_true, y_proba, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt


def plot_feature_importance(model, feature_names, max_features=10):
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        booster.feature_names = feature_names
        xgboost.plot_importance(booster, max_num_features=max_features)
    else:
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        feat_imp.head(max_features).plot(kind='barh')
        plt.xlabel('Importance')
        plt.title('Feature Importance')
    
    return plt


def plot_shap_summary(model, X_test, feature_names, class_idx=1, max_display=10):
    try:
        import shap
        shap.initjs()
        
        # Limit to number of features in X_test
        n_features = min(len(feature_names), X_test.shape[1])
        feature_names_actual = feature_names[:n_features]
        X_test_actual = X_test[:, :n_features] if X_test.ndim > 1 else X_test
        
        X_test_df = pd.DataFrame(X_test_actual, columns=feature_names_actual)
        
        # Use TreeExplainer for tree-based models
        if hasattr(model, 'get_booster') or hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[class_idx]
        else:
            # For other models, use Explainer
            explainer = shap.Explainer(
                model.predict_proba,
                X_test_df,
                feature_names=feature_names_actual
            )
            shap_values = explainer(X_test_df)
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values[:, :, class_idx]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, max_display=max_display, show=False)
        return plt
        
    except ImportError:
        print("SHAP library not installed. Install with pip install shap")
        return None
    except Exception as e:
        print(f"{e}")
        return None


