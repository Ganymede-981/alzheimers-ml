import pandas as pd
import numpy as np
from typing import Optional


def extract_longitudinal_features(df: pd.DataFrame, 
                                  is_training: bool = True,
                                  cdr_predictor=None,
                                  preprocessor=None,
                                  label_encoder=None) -> pd.DataFrame:
    rows = []
    
    for sid, g in df.groupby("Subject ID"):
        g = g.sort_values("Visit")
        
        first = g.iloc[0]
        last = g.iloc[-1]
        
        row = {}
        row["Subject ID"] = sid
        
        # Static demographic features
        row["Sex"] = first["M/F"]
        row["Hand"] = first["Hand"]
        row["Educ"] = first["EDUC"]
        row["SES"] = first["SES"]
        
        # Baseline features
        row["MMSE_baseline"] = first["MMSE"]
        row["nWBV_baseline"] = first["nWBV"]
        row["Age_baseline"] = first["Age"]
        row["eTIV"] = first["eTIV"]
        row["ASF"] = first["ASF"]
        
        # Predict CDR from baseline if model provided
        if cdr_predictor is not None and preprocessor is not None:
            cdr_features = ['M/F', 'Hand', 'Age', 'EDUC', 'SES',
                           'MMSE', 'eTIV', 'nWBV', 'ASF']
            X_cdr = first[cdr_features].to_frame().T
            X_cdr = X_cdr.rename(columns={'EDUC': 'Educ'})
            X_cdr_trf = preprocessor.transform(X_cdr)
            row["CDR_pred"] = cdr_predictor.predict(X_cdr_trf)[0]
        else:
            row["CDR_pred"] = first.get("CDR_pred", 0)
        
        # Delta features
        row["MMSE_delta"] = last["MMSE"] - first["MMSE"]
        row["nWBV_delta"] = last["nWBV"] - first["nWBV"]
        
        # Time features
        followup_days = g["MR Delay"].max()
        followup_years = followup_days / 365.0
        
        row["n_visits"] = g.shape[0]
        row["followup_years"] = followup_years
        
        # Slope features
        if followup_years > 0:
            row["MMSE_slope"] = row["MMSE_delta"] / followup_years
            row["nWBV_slope"] = row["nWBV_delta"] / followup_years
        else:
            row["MMSE_slope"] = 0.0
            row["nWBV_slope"] = 0.0
        
        # Visit indicator
        row["has_multiple_visits"] = int(g.shape[0] > 1)
        
        # Target (only for training)
        if is_training:
            row["target_group"] = last["Group_binary"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def prepare_longitudinal_train_test_split(ldf: pd.DataFrame,
                                         test_size: float = 0.2,
                                         random_state: int = 42,
                                         cdr_predictor=None,
                                         preprocessor=None,
                                         label_encoder=None):
    
    from sklearn.model_selection import train_test_split
    
    subjects = ldf["Subject ID"].unique()
    
    subject_level_labels = (
        ldf
        .sort_values("Visit")
        .groupby("Subject ID")["Group_binary"]
        .last()
        .values
    )
    
    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_size,
        stratify=subject_level_labels,
        random_state=random_state
    )
    
    train_df = ldf[ldf["Subject ID"].isin(train_subjects)]
    test_df = ldf[ldf["Subject ID"].isin(test_subjects)]
    
    X_train_long = extract_longitudinal_features(
        train_df, 
        is_training=True,
        cdr_predictor=cdr_predictor,
        preprocessor=preprocessor,
        label_encoder=label_encoder
    ).drop(columns=["target_group", "Subject ID"])
    
    y_train_long = extract_longitudinal_features(
        train_df, 
        is_training=True,
        cdr_predictor=cdr_predictor,
        preprocessor=preprocessor,
        label_encoder=label_encoder
    )["target_group"]
    
    X_test_long = extract_longitudinal_features(
        test_df, 
        is_training=True,
        cdr_predictor=cdr_predictor,
        preprocessor=preprocessor,
        label_encoder=label_encoder
    ).drop(columns=["target_group", "Subject ID"])
    
    y_test_long = extract_longitudinal_features(
        test_df, 
        is_training=True,
        cdr_predictor=cdr_predictor,
        preprocessor=preprocessor,
        label_encoder=label_encoder
    )["target_group"]
    
    return X_train_long, X_test_long, y_train_long, y_test_long


