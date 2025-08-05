import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from typing import Tuple
from utils.logger import logging
from utils.exception import CustomException


class DataTransformer:
    def __init__(self):
        self.le = LabelEncoder()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)

    def label_encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Encodes categorical column using Label Encoding.

        Time Complexity: O(n), where n is number of rows in df
        Space Complexity: O(n)

        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column to encode

        Returns:
            pd.DataFrame: Encoded dataframe
        """
        try:
            logging.info(f"Label encoding column: {column}")
            df[column] = self.le.fit_transform(df[column])
            return df
        except Exception as e:
            raise CustomException(f"Error encoding column {column}: {e}")

    def polynomial_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies polynomial feature transformation.

        Time Complexity: O(n * d^k), where d is number of features, k is degree
        Space Complexity: O(n * d^k)

        Args:
            X (pd.DataFrame): Features

        Returns:
            pd.DataFrame: Transformed features
        """
        try:
            logging.info("Applying polynomial feature transformation.")
            X_poly = self.poly.fit_transform(X)
            return pd.DataFrame(X_poly, columns=self.poly.get_feature_names_out(X.columns))
        except Exception as e:
            raise CustomException(f"Error in polynomial transformation: {e}")