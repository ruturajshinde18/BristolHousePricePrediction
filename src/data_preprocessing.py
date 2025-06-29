import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split


def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def preprocess_data():
    """Clean and prepare data for modeling"""
    params = load_params()

    # Load processed data
    house = pd.read_csv(params['data']['processed_data'])

    # Drop secondary address (lots of missing values)
    house.drop("secondary_address", axis=1, inplace=True, errors='ignore')

    # Extract date features
    house["Year"] = house["date_of_transfer"].apply(lambda x: pd.to_datetime(x).year)
    house.drop("date_of_transfer", axis=1, inplace=True)

    # Select features and target
    feature_cols = params['features']['selected_features']
    target_col = params['features']['target']

    X = house[feature_cols]
    y = house[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['split']['test_size'],
        random_state=params['split']['random_state']
    )

    # Save split data
    os.makedirs('data/split', exist_ok=True)
    X_train.to_csv('data/split/x_train.csv', index=False)
    X_test.to_csv('data/split/x_test.csv', index=False)
    y_train.to_csv('data/split/y_train.csv', index=False)
    y_test.to_csv('data/split/y_test.csv', index=False)

    print(f"Data preprocessing complete.")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")


if __name__ == "__main__":
    preprocess_data()