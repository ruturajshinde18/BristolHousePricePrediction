import pandas as pd
import numpy as np
from typing import Dict, Any


def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate if coordinates are within reasonable Bristol bounds"""
    # Bristol approximate bounds
    bristol_bounds = {
        'lat_min': 51.3,
        'lat_max': 51.6,
        'lon_min': -2.8,
        'lon_max': -2.3
    }

    return (bristol_bounds['lat_min'] <= lat <= bristol_bounds['lat_max'] and
            bristol_bounds['lon_min'] <= lon <= bristol_bounds['lon_max'])


def format_property_details(details: Dict[str, Any]) -> str:
    """Format property details for display"""
    property_type_map = {
        "D": "Detached",
        "S": "Semi-Detached",
        "T": "Terraced",
        "F": "Flat/Maisonette",
        "O": "Other"
    }

    tenure_map = {
        "F": "Freehold",
        "L": "Leasehold"
    }

    new_build_map = {
        "Y": "Yes",
        "N": "No"
    }

    return f"""
    Property Type: {property_type_map.get(details.get('property_type', ''), 'Unknown')}
    Tenure: {tenure_map.get(details.get('tenure', ''), 'Unknown')}
    New Build: {new_build_map.get(details.get('new_build', ''), 'Unknown')}
    Date: {details.get('day', 1)}/{details.get('month', 1)}/{details.get('year', 2024)}
    """


def create_sample_predictions():
    """Create sample data for testing"""
    return pd.DataFrame({
        'location': ['Clifton', 'Redland', 'Southville', 'Bedminster'],
        'lat': [51.4641, 51.4711, 51.4398, 51.4325],
        'lon': [-2.6103, -2.6037, -2.6205, -2.5889],
        'predicted_price': [450000, 380000, 320000, 280000]
    })