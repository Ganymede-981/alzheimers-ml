import pandas as pd
import numpy as np
from pathlib import Path


def load_cross_sectional_data(data_path: str) -> pd.DataFrame:
    df = pd.read_excel(data_path)
    df = df.dropna(subset=["CDR"])
    df['CDR'] = (df['CDR'] > 0).astype(int)
    return df


def load_longitudinal_data(data_path: str) -> pd.DataFrame:
    ldf = pd.read_excel(data_path)
    ldf["Group_binary"] = (ldf["Group"] == "Demented").astype(int)
    ldf['CDR'] = (ldf['CDR'] > 0).astype(int)
    
    # Filling missing MR Delay values
    ldf["MR Delay"] = ldf["MR Delay"].fillna(0)
    
    # Convert numeric columns
    num_cols = ["MMSE", "Age", "eTIV", "nWBV", "ASF", "MR Delay"]
    for col in num_cols:
        ldf[col] = pd.to_numeric(ldf[col], errors="coerce")
    
    return ldf


def prepare_train_test_split(df: pd.DataFrame, 
                            target_col: str = 'CDR',
                            test_size: float = 0.2,
                            random_state: int = 42,
                            drop_cols: list = None):
    
    # Prepare train-test split for cross-sectional data    
    from sklearn.model_selection import train_test_split
    
    if drop_cols is None:
        drop_cols = ['ID', 'CDR', 'Delay']
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )


