import pandas as pd
import joblib

def print_section(title):
    """
    Print a formatted section title.
    """
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def load_csv(file_path):
    """
    Load a CSV file into a DataFrame.
    """
    return pd.read_csv(file_path)

def save_model(model, file_path):
    """
    Save a trained model to disk.
    """
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    Load a trained model from disk.
    """
    return joblib.load(file_path)

def split_data(data, target_column, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets.

    Parameters:
    data (DataFrame): The dataset to split.
    target_column (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: A tuple containing the training and testing sets (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)