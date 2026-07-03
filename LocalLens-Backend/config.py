import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "mumbai":    BASE_DIR / "models" / "mumbai_price_model.pkl",
    "bengaluru": BASE_DIR / "models" / "bengaluru_price_model.pkl",
    "pune":      BASE_DIR / "models" / "pune_price_model.pkl",
}

RAG_DATA_PATHS = {
    "mumbai":    BASE_DIR / "rag_dfs" / "mumbai_cleaned_dataset.csv",
    "bengaluru": BASE_DIR / "rag_dfs" / "bengaluru_cleaned_dataset.csv",
    "pune":      BASE_DIR / "rag_dfs" / "pune_cleaned_dataset.csv",
}

SUPPORTED_CITIES = list(MODEL_PATHS.keys())

# All 3 datasets are from 2022 → inflate 3 years to reach 2025
DATASET_AGE_YEARS = 3

GROWTH_RATES = {
    "mumbai":    0.08,
    "bengaluru": 0.12,
    "pune":      0.06,
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "llama-3.3-70b-versatile"
