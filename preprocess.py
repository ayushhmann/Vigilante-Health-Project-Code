import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_preprocess(df):
    # Fill numeric missing with median and categorical with mode
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if len(num_cols)>0:
        num_imp = SimpleImputer(strategy='median')
        df[num_cols] = num_imp.fit_transform(df[num_cols])

    if len(cat_cols)>0:
        cat_imp = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    return df

def scale_features(df, feature_columns):
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler
