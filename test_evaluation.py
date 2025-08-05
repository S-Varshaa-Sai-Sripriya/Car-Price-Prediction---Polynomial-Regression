import sys
import os
sys.path.append(os.path.abspath('./'))

from src.Evaluation.evaluate import evaluate_model
import numpy as np

# Dummy test data
y_true = np.array([3.5, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

# Run the evaluation
evaluate_model(y_true, y_pred)