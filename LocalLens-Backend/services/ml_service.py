"""
Loads all city models once at startup via joblib.
predict_price_per_sqft() builds a DataFrame with columns in the exact same order as X during training: ["bhk", "area", "type", "region", "status", "age"]
"""

import joblib
import pandas as pd
from pathlib import Path
from config import MODEL_PATHS, SUPPORTED_CITIES

_models: dict = {}

# X = df[["bhk", "area", "type", "region", "status", "age"]]
FEATURE_COLUMNS = ["bhk", "area", "type", "region", "status", "age"] # schema 


def load_all_models():
    """Called once from main.py lifespan. Loads all .pkl files into memory."""
    for city in SUPPORTED_CITIES:
        path: Path = MODEL_PATHS[city]
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found for '{city}' at: {path}\n"
                "Make sure the models/ folder is at the project root."
            )
        _models[city] = joblib.load(path) 
        print(f"[ML] Loaded model for '{city}' ✓")


def get_model(city: str):
    if city not in _models:
        raise KeyError(f"Model for '{city}' not loaded. Was load_all_models() called?")
    return _models[city]


def predict_price_per_sqft(city: str, features: dict) -> float:
    """
    features dict must contain these keys:
        bhk (int), area (float),
        type (str), region (str), status (str), age (str)

    Returns predicted price_per_sqft as ₹/sqft.
    """
    model = get_model(city)

    row = {col: [features[col]] for col in FEATURE_COLUMNS}
    df  = pd.DataFrame(row)

    result = model.predict(df)
    return round(float(result[0]), 2)
