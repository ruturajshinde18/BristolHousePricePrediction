import pandas as pd
import yaml
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor


def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def build_model():
    """Build and train the ML model"""
    params = load_params()

    # Load training data
    X_train = pd.read_csv('data/split/x_train.csv')
    y_train = pd.read_csv('data/split/y_train.csv').squeeze()

    # Define column groups
    cat_cols = params['features']['categorical']
    num_cols = params['features']['numerical']

    # Build preprocessing pipeline
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median"))
            ]), num_cols)
        ]
    )

    # Full model pipeline
    model = Pipeline([
        ("prep", preprocess),
        ("catb", CatBoostRegressor(
            iterations=params['model']['catboost']['iterations'],
            learning_rate=params['model']['catboost']['learning_rate'],
            depth=params['model']['catboost']['depth'],
            loss_function=params['model']['catboost']['loss_function'],
            random_state=params['model']['catboost']['random_state'],
            verbose=0
        ))
    ])

    # Train model
    print("Training model...")
    model.fit(X_train, y_train)

    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/catboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model training complete and saved.")


if __name__ == "__main__":
    build_model()