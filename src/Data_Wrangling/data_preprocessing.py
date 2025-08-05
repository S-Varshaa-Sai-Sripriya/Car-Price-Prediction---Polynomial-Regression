import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from utils.logger import logging
from utils.exception import CustomException


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Preprocess the dataset by scaling and splitting.

        Time Complexity: O(n), where n is number of rows
        Space Complexity: O(n)

        Args:
            df (pd.DataFrame): Raw dataframe
            target_column (str): Name of the target column

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        try:
            logging.info("Starting preprocessing.")
            X = df.drop(target_column, axis=1)
            y = df[target_column]

            logging.info("Splitting dataset.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info("Applying scaling.")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            return pd.DataFrame(X_train_scaled, columns=X.columns), pd.DataFrame(X_test_scaled, columns=X.columns), y_train, y_test
        except Exception as e:
            raise CustomException(f"Error in preprocessing: {e}")
