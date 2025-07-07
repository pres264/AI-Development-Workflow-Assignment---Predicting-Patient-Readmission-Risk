import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Handle missing values for both numerical and categorical columns.
    Args:
        df (pd.DataFrame): Raw data.
    Returns:
        pd.DataFrame: Cleaned data.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=[object]).columns
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_and_scale(df, drop_cols=None):
    """
    One-hot encode categorical features and scale numerical features.
    Args:
        df (pd.DataFrame): Cleaned data.
        drop_cols (list): Columns to drop before encoding (e.g., IDs).
    Returns:
        pd.DataFrame: Processed features.
    """
    if drop_cols:
        df = df.drop(columns=drop_cols)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=[object]).columns
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_num = scaler.fit_transform(df[num_cols])
    X_cat = encoder.fit_transform(df[cat_cols]) if len(cat_cols) > 0 else np.empty((len(df),0))
    X = np.hstack([X_num, X_cat])
    feature_names = list(num_cols) + list(encoder.get_feature_names_out(cat_cols)) if len(cat_cols) > 0 else list(num_cols)
    return pd.DataFrame(X, columns=feature_names)

def preprocess_and_save(input_path, output_path, drop_cols=None):
    """
    Full preprocessing pipeline: load, clean, encode, scale, and save.
    Args:
        input_path (str): Path to raw data.
        output_path (str): Path to save cleaned data.
        drop_cols (list): Columns to drop (e.g., IDs).
    Returns:
        pd.DataFrame: Processed features.
    """
    df = load_data(input_path)
    df = clean_data(df)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    X = encode_and_scale(df, drop_cols=drop_cols)
    return X

if __name__ == "__main__":
    # Example usage
    input_path = '../data/sample_data.csv'
    output_path = '../data/cleaned_data.csv'
    preprocess_and_save(input_path, output_path, drop_cols=['patient_id'])