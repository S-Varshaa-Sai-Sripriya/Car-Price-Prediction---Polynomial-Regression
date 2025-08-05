import numpy as np
from typing import Tuple

# Polynomial Feature Expansion
class PolynomialFeatures:
    def __init__(self, degree: int):
        self.degree = degree

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[:, np.newaxis]

        n_samples = X.shape[0]
        X_poly = np.ones((n_samples, self.degree + 1))  # [1, x, x^2, ..., x^d]

        for d in range(1, self.degree + 1):
            X_poly[:, d] = X[:, 0] ** d

        return X_poly

# Mean Squared Error Loss
class MeanSquaredError:
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def gradient(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = len(y_true)
        return (-2 / n) * X.T @ (y_true - y_pred)

# Core Polynomial Regression Model
class PolynomialRegression:
    def __init__(self, degree: int, lr: float = 0.0001, n_iters: int = 1000):
        self.degree = degree
        self.lr = lr
        self.n_iters = n_iters
        self.theta = None
        self.loss_fn = MeanSquaredError()
        self.poly = PolynomialFeatures(degree)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_poly = self.poly.transform(X)
        n_samples, n_features = X_poly.shape
        self.theta = np.zeros(n_features)

        for i in range(self.n_iters):
            y_pred = X_poly @ self.theta
            gradient = self.loss_fn.gradient(X_poly, y, y_pred)
            self.theta -= self.lr * gradient

            if i % 100 == 0:
                loss = self.loss_fn.compute(y, y_pred)
                print(f"Epoch {i}: MSE = {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_poly = self.poly.transform(X)
        return X_poly @ self.theta

# Custom Evaluation Metrics
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total)