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

## Dataset Explanation

The dataset used for this project is the **Car Price Prediction Dataset**, which contains information about various attributes of cars that influence their market price. Each row represents a unique car entry with multiple features describing its specifications and characteristics.  

### Key Features:
- **Car_Name**: The brand/model of the car.
- **Year**: The year of manufacture of the car.
- **Selling_Price**: The price at which the car is being sold (target variable).
- **Present_Price**: The current ex-showroom price of the car.
- **Kms_Driven**: Total kilometers driven by the car.
- **Fuel_Type**: Type of fuel used by the car (Petrol/Diesel/CNG).
- **Seller_Type**: Whether the seller is an Individual or Dealer.
- **Transmission**: Type of transmission (Manual/Automatic).
- **Owner**: Number of previous owners.

### Target Variable:
- **Selling_Price**: Represents the price of the car, which is predicted using Polynomial Regression.  

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
