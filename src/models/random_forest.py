"""
Random Forest model with Optuna hyperparameter optimization
"""
import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score


def rf_objective(X_train, y_train, trial):
    """
    Objective function for Random Forest hyperparameter optimization
    
    Args:
        X_train: Training features
        y_train: Training labels
        trial: Optuna trial object
        
    Returns:
        Cross-validation ROC-AUC score
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }
    
    rf = RandomForestClassifier(**params)
    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    score = cross_val_score(
        rf,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    ).mean()
    
    return score


def optimize_random_forest(X_train, y_train, n_trials=75, random_state=42):
    """
    Optimize Random Forest hyperparameters using Optuna
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_trials: Number of optimization trials
        random_state: Random seed
        
    Returns:
        Optuna study object and best model
    """
    study_rf = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    study_rf.optimize(
        lambda trial: rf_objective(X_train, y_train, trial), 
        n_trials=n_trials
    )
    
    print(f"Best RF ROC-AUC: {study_rf.best_value}")
    print(f"Best RF params: {study_rf.best_params}")
    
    # Create best model
    best_rf = RandomForestClassifier(
        **study_rf.best_params,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    
    best_rf.fit(X_train, y_train)
    
    return study_rf, best_rf


