import pandas as pd
import os
import yaml


def load_params():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def ingest_data():
    """Load raw data and perform initial filtering"""
    params = load_params()

    # Load house price data
    house = pd.read_csv(params['data']['raw_house_data'], names=[
        "transaction_id", "price", "date_of_transfer", "postcode",
        "property_type", "new_build", "tenure", "primary_address",
        "secondary_address", "street", "locality", "town_city",
        "district", "county", "ppd_category_type", "record_status"
    ])

    # Load ONS postcode data
    onspd = pd.read_csv(params['data']['raw_postcode_data'],
                        encoding='latin1', dtype={'pcds': str})

    # Clean postcodes for merging
    house['postcode'] = house['postcode'].str.strip().str.upper()
    onspd['pcds'] = onspd['pcds'].str.strip().str.upper()

    # Merge to add lat/lon coordinates
    house = house.merge(onspd[['pcds', 'lat', 'long']],
                        left_on='postcode', right_on='pcds', how='left')
    house.drop(columns='pcds', inplace=True)

    # Filter for Bristol only
    house = house[house["town_city"] == params['data']['target_city']]

    # Save raw processed data
    os.makedirs('data/processed', exist_ok=True)
    house.to_csv(params['data']['processed_data'], index=False)

    print(f"Data ingestion complete. Shape: {house.shape}")
    return house


if __name__ == "__main__":
    ingest_data()