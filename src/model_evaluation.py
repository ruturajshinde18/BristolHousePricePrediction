import pandas as pd
import yaml
import pickle
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def evaluate_model():
    """Evaluate the trained model"""
    params = load_params()

    # Load test data
    X_test = pd.read_csv('data/split/x_test.csv')
    y_test = pd.read_csv('data/split/y_test.csv').squeeze()

    # Load trained model
    with open('models/catboost_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'mse': float(mean_squared_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': float(r2_score(y_test, y_pred)),
        'mape': float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    }

    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print results
    print("Model Evaluation Results:")
    print(f"MAE: £{metrics['mae']:,.0f}")
    print(f"RMSE: £{metrics['rmse']:,.0f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")


if __name__ == "__main__":
    evaluate_model()