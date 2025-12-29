import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def create_preprocessing_pipeline(X_train: pd.DataFrame):
    # Identify column types
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    bin_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Create pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    
    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("OneHotEncoder", OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocess = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("bin", bin_pipeline, bin_cols)
    ])
    
    return preprocess, num_cols, bin_cols


def create_longitudinal_preprocessing_pipeline(X_train: pd.DataFrame):
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    binary_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("OneHotEncoder", OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("bin", binary_pipeline, binary_cols)
    ])
    
    return preprocessor, numeric_cols, binary_cols


def encode_labels(y_train, y_test=None):
    le = LabelEncoder()
    y_train_trf = le.fit_transform(y_train)
    
    if y_test is not None:
        y_test_trf = le.transform(y_test)
        return y_train_trf, y_test_trf, le
    
    return y_train_trf, le


