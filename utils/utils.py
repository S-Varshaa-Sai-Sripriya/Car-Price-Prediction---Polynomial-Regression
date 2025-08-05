import os
import numpy as np
import pandas as pd
import joblib


def save_data(X: np.ndarray, y: np.ndarray, filepath: str) -> None:
    """
    Saves the features and labels to a .npz file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, X=X, y=y)


def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads features and labels from a .npz file.
    """
    data = np.load(filepath)
    return data['X'], data['y']


def save_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves a pandas DataFrame to a CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a CSV file.
    """
    return pd.read_csv(filepath)


def save_model(model, filepath: str) -> None:
    """
    Saves a trained model using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """
    Loads a model saved using joblib.
    """
    return joblib.load(filepath)
