import math
import random
import pandas as pd

def clean_nans(data: dict):
    """Recursively replace NaN values with None"""
    for key, value in data.items():
        if isinstance(value, float) and math.isnan(value):
            data[key] = None
    return data

# Generate a random customer profile
def generate_random_customer_profile(df: pd.DataFrame) -> dict:
    rand_index = random.randint(0, df.shape[0] - 1)
    return df.iloc[rand_index].to_dict()