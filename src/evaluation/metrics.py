import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    f1_score,
    brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_model(model, X_test, y_test, y_pred=None, y_proba=None):

    if y_pred is None:
        y_pred = model.predict(X_test)
    
    if y_proba is None:
        y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'brier_score': brier_score_loss(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics


def print_evaluation_results(metrics):
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print("Classification Report:")
    print(metrics['classification_report'])
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])


def cross_validate_model(model, X, y, cv_folds=10, scoring='f1'):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return scores.mean(), scores.std(), scores


