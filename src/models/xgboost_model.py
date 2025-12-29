"""
XGBoost model with Optuna hyperparameter optimization
"""
import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score


def xgb_objective(X_train, y_train, trial):
    """
    Objective function for XGBoost hyperparameter optimization
    
    Args:
        X_train: Training features
        y_train: Training labels
        trial: Optuna trial object
        
    Returns:
        Cross-validation ROC-AUC score
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5),
        "eval_metric": "logloss",
        "random_state": 42
    }
    
    model = XGBClassifier(**params)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    ).mean()
    
    return score


def optimize_xgboost(X_train, y_train, n_trials=300, random_state=42):
    """
    Optimize XGBoost hyperparameters using Optuna
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_trials: Number of optimization trials
        random_state: Random seed
        
    Returns:
        Optuna study object and best model
    """
    study = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    study.optimize(
        lambda trial: xgb_objective(X_train, y_train, trial), 
        n_trials=n_trials
    )
    
    print(f"Best ROC-AUC: {study.best_value}")
    print(f"Best params: {study.best_params}")
    
    # Create best model
    best_model = XGBClassifier(
        **study.best_params,
        eval_metric="logloss",
        random_state=random_state
    )
    
    best_model.fit(X_train, y_train)
    
    return study, best_model


