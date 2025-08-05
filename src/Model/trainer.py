import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from utils.logger import logger
from utils.exception import CustomException
from src.Data_Wrangling.data_transformation import DataTransformer
from src.Data_Wrangling.data_preprocessing import DataPreprocessor
from src.Evaluation.evaluate import evaluate_model

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.degree = 2  # Degree of polynomial

    def train_model(self, data_path: str) -> None:
        try:
            logging.info("Reading dataset...")
            df = pd.read_csv(data_path)

            logging.info("Applying transformation...")
            df = DataTransformer().transform(df)

            logging.info("Applying preprocessing...")
            X_train, X_test, y_train, y_test = DataPreprocessor().preprocess(df)

            logging.info("Initializing polynomial regression pipeline...")
            pipeline = Pipeline([
                ('poly_features', PolynomialFeatures(degree=self.degree, include_bias=False)),
                ('linear_regression', LinearRegression())
            ])

            logging.info("Training the model...")
            pipeline.fit(X_train, y_train)

            logging.info("Evaluating the model...")
            y_pred = pipeline.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"RMSE: {rmse}")
            logging.info(f"R² Score: {r2}")

            logging.info("Saving the trained model...")
            joblib.dump(pipeline, self.model_path)
            logging.info(f"Model saved at: {self.model_path}")
            
            evaluate_model(y_test, y_pred)
            print("Model training and evaluation completed successfully.")
            
        except Exception as e:
            raise CustomException(e)

