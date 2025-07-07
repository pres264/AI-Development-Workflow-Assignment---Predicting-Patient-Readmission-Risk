# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

def load_cleaned_data(file_path, target_col):
    """
    Load cleaned data and split into features and target.
    Args:
        file_path (str): Path to cleaned CSV.
        target_col (str): Name of target column.
    Returns:
        X (pd.DataFrame), y (pd.Series)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col, 'patient_id']) if 'patient_id' in df.columns else df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def build_pipeline():
    """
    Build a pipeline with a RandomForestClassifier.
    """
    pipeline = Pipeline([
        ('clf', RandomForestClassifier(random_state=42))
    ])
    return pipeline

def train_and_tune(X_train, y_train):
    """
    Train and tune the model using GridSearchCV.
    """
    pipeline = build_pipeline()
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_split': [2, 5]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    return grid.best_estimator_

def save_model(model, file_path):
    """
    Save the trained model to disk.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

if __name__ == "__main__":
    # Example usage
    X, y = load_cleaned_data('../data/cleaned_data.csv', 'readmission')
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_and_tune(X_train, y_train)
    save_model(model, '../data/patient_readmission_model.pkl')