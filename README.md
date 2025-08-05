# Car Price Prediction - Polynomial Regression

This project demonstrates a custom implementation of **Polynomial Regression from scratch**, applied to predict car prices based on various features. It avoids external ML libraries (like scikit-learn) for modeling, focusing instead on the mathematical intuition and core logic behind polynomial regression.

---

## 📌 Project Highlights

- 🔢 **Built from Scratch**: End-to-end polynomial regression implemented using only NumPy and Pandas — no high-level ML libraries used.
- 🔄 **Custom Data Preprocessing Pipeline**: Manual feature encoding, scaling, and polynomial feature generation.
- 📈 **Robust Evaluation Metrics**:
  - **Mean Absolute Error (MAE)**: 0.60
  - **Root Mean Squared Error (RMSE)**: 0.69
  - **R² Score**: 0.9356
- 🧪 **Testable Architecture**: Modular and test-driven development structure with dedicated evaluation script.

---

## ⚙️ Features

- **Polynomial Feature Expansion**: Supports configurable polynomial degrees.
- **Normal Equation** Solver: Closed-form solution for weight optimization.
- **Manual Loss Functions**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- **Test File**: Quick entry point to validate the pipeline and metrics.
- **Logging**: Modular logging to track model evaluation and outputs.

---